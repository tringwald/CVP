import termcolor
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import config
import util as u
from feature_extractors import resnet
from typing import Optional, Union, Dict, List
from torch.cuda.amp import autocast
from feature_extractors import dsbn
import math
from torch.nn import ModuleList


class SamplingLayer(nn.Module):
    def __init__(self, feature_size, num_samples, thresh):
        super().__init__()
        self.feature_size = feature_size
        self.num_samples = num_samples
        self.thresh = thresh
        self.sigma_projection = nn.Sequential(
            nn.Linear(feature_size, feature_size, bias=True),
            nn.ReLU(),
            nn.Linear(feature_size, 1, bias=True)
        )

    def forward(self, x):
        mu = x
        unbound_sigma = self.sigma_projection(x)
        sigma = F.softplus(unbound_sigma).expand_as(mu)

        # Sampling
        n, d = mu.shape
        exp_mu = mu.unsqueeze(1).expand(n, self.num_samples, d)
        exp_sig = sigma.unsqueeze(1).expand_as(exp_mu)
        eps = T.randn_like(exp_mu)
        samples = exp_mu + eps * exp_sig
        return mu, sigma, samples


class UtilBase(nn.Module):
    """ Base functionality for all networks. """

    def __init__(self):
        super(UtilBase, self).__init__()
        self.c = config.get_global_config()
        self.classifiers: ModuleList[nn.Module] = ModuleList()
        self.feature_extractor: Optional[nn.Module] = None
        self.sampling_layer: Optional[nn.Module] = None
        self.embedding_size: Optional[int] = None
        self.num_classifiers = 2

    def get_parameters(self, phase: u.Phase) -> List[Dict]:
        """ Enables different learning rates/WD per param group.
        :param phase: Phase object indexing into the global_config keys of the optimizer part.
        :return: List of parameter groups as required by PyTorch.
        """
        c = config.get_global_config()
        all_parameter_names = {n for n, p in self.named_parameters()}
        ignored_names = set(x for x in all_parameter_names if x.split('.')[1] in c.ignore_params) if c.ignore_params else set()
        for name, tensor in self.named_parameters():
            if name in ignored_names:
                tensor.requires_grad = False
            else:
                tensor.requires_grad = True
        feature_params = {n: p for n, p in self.named_parameters() if 'feature_extractor' in n and n not in ignored_names}
        sampling_params = {n: p for n, p in self.named_parameters() if 'sampling' in n and n not in ignored_names}
        classifier_params = {n: p for n, p in self.named_parameters() if 'classifier' in n and n not in ignored_names}
        used_params = set(list(feature_params.keys()) + list(sampling_params.keys()) + list(classifier_params.keys()))
        unused_params = all_parameter_names.difference(used_params)
        print("Ignored parameters: ", ignored_names)
        print(termcolor.colored(f"Unused params: {str(unused_params)}", on_color='on_red' if len(unused_params) > 0 else None))
        phase_params = c.optimizer_params[phase.get_phase()]
        return [{'params': list(feature_params.values()), 'lr': 1. * phase_params['lr'], 'weight_decay': 1. * phase_params['weight_decay']},
                {'params': list(sampling_params.values()), 'lr': 1. * phase_params['lr'], 'weight_decay': 1. * phase_params['weight_decay']},
                {'params': list(classifier_params.values()), 'lr': 1. * phase_params['lr'], 'weight_decay': 1. * phase_params['weight_decay']}]

    def setup(self, feature_extr):
        self.feature_extractor = feature_extr
        self.embedding_size = feature_extr.embedding_size
        if self.c.use_cvp:
            self.sampling_layer = SamplingLayer(self.embedding_size, self.c.num_samples, self.c.upper_bound)

        # Classifiers
        for i in range(self.num_classifiers - 1):
            _classifier = nn.Linear(self.embedding_size, self.embedding_size, bias=True)
            nn.init.eye_(_classifier.weight)
            nn.init.constant_(_classifier.bias, 0.)
            self.classifiers.append(_classifier)
        self.classifiers.append(nn.Linear(self.embedding_size, self.c.current.num_classes, bias=True))

    def classify(self, x, dropout=0.):
        for i in range(self.num_classifiers - 1):
            x = F.dropout(x, p=dropout, training=True)
            x = self.classifiers[i](x)
            x = F.relu(x)
        x = F.dropout(x, p=dropout, training=True)
        x = self.classifiers[-1](x)
        return x

    def apply_mcd(self, x):
        n, f = x.shape
        x = x.unsqueeze(1).expand(n, self.c.uncertain_mc_iters, f).reshape(-1, f)
        x = self.classify(x, dropout=self.c.uncertain_mc_dropout)
        return x.view(n, self.c.uncertain_mc_iters, self.c.current.num_classes)

    def classify_samples(self, samples, dropout: float):
        # Unravel embeddings to 2 dimensions, classify samples, ravel back
        res = self.classify(samples.view(-1, self.embedding_size), dropout=dropout)
        return res.view(-1, self.c.num_samples, self.c.current.num_classes)

    @autocast(enabled=config.get_global_config().use_amp)
    def forward(self, img, phase):
        mu_logits, sigma, samples, samples_logits, mcd_mu_logits = [], [], [], [], []

        # Extract features with base network, then pass to sampling layer
        features = self.feature_extractor(img)
        if self.c.use_cvp:
            mu, sigma, samples = self.sampling_layer(features)
        else:
            mu = features
        mcd_mu_logits = self.apply_mcd(mu)

        # Classify features, set dropout depending on phase
        if phase in [u.Phase(u.Phase.ADAPTATION_TRAIN), u.Phase(u.Phase.SOURCE_ONLY_TRAIN)]:
            mu_logits = self.classify(mu, dropout=self.c.base_dropout)
            if self.c.use_cvp:
                samples_logits = self.classify_samples(samples, dropout=self.c.base_dropout)
        elif phase in [u.Phase(u.Phase.MC_DROPOUT)]:
            mu_logits = self.classify(mu, dropout=0.)
        else:
            mu_logits = self.classify(mu, dropout=0.)
            if self.c.use_cvp:
                samples_logits = self.classify_samples(samples, dropout=0.)

        return {'mu': mu,
                'sigma': sigma if sigma != [] else T.zeros_like(mu),
                'samples': samples,
                'normal': {'mu_logits': mu_logits,
                           'samples_logits': samples_logits},
                'mcd': {'mu_logits': mcd_mu_logits}
                }


################################################################################################################################################################
# Available networks
class Resnet50(UtilBase):
    def __init__(self):
        super().__init__()
        self.setup(resnet.resnet50(pretrained=self.c.pretrained, norm_layer=dsbn.DomainSpecificBatchNorm if self.c.use_dsbn else None))


class Resnet101(UtilBase):
    def __init__(self):
        super().__init__()
        self.setup(resnet.resnet101(pretrained=self.c.pretrained, norm_layer=dsbn.DomainSpecificBatchNorm if self.c.use_dsbn else None))


class Resnext50_32x4d(UtilBase):
    def __init__(self):
        super().__init__()
        self.setup(resnet.resnext50_32x4d(pretrained=self.c.pretrained, norm_layer=dsbn.DomainSpecificBatchNorm if self.c.use_dsbn else None))


################################################################################################################################################################
# Util functions for listing and getting available models
def get_model(model_name, gpus):
    # Look for all defined subclasses of nn.Module
    defined_models = {v.__name__: v for k, v in globals().items() if type(v) == type and issubclass(v, nn.Module)}
    chosen_model = defined_models[model_name]()
    chosen_model = nn.DataParallel(chosen_model, device_ids=gpus).cuda()
    # Calculate parameter count
    num_params, num_backbone_params = 0, 0
    for p in chosen_model.parameters():
        num_params += p.numel()
    for p in chosen_model.module.feature_extractor.parameters():
        num_backbone_params += p.numel()
    print(f"Loaded model {termcolor.colored(chosen_model.module.__class__.__name__, on_color='on_blue', attrs=['bold'])} "
          f"with {num_params:,} parameters (backbone {num_backbone_params:,})")
    return chosen_model


def get_available():
    return [v.__name__ for k, v in globals().items() if type(v) == type and issubclass(v, nn.Module)]
