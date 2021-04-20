import datetime
import json
import math
import os
import os.path as osp
import random
import shutil
import socket
import time
import numpy as np
import termcolor
import torch
import torch.nn.functional as F
import torch.optim
import torchvision.transforms as TF
from PIL import Image
import config
from torchvision.transforms.functional import InterpolationMode


def persist_sigma(cycle, batch, sigma, domain):
    c = config.get_global_config()
    with open(c.sigma_file, 'a+') as f:
        src_sig = sigma.mean(dim=-1)[domain == Domain.source()]
        tar_sig = sigma.mean(dim=-1)[domain == Domain.target()]
        both_sig = sigma.mean(dim=-1)
        f.write(f"{cycle},{batch},"
                f"{src_sig.min().item()},{src_sig.mean().item()},{src_sig.max().item()},"
                f"{tar_sig.min().item()},{tar_sig.mean().item()},{tar_sig.max().item()},"
                f"{both_sig.min().item()},{both_sig.mean().item()},{both_sig.max().item()}\n"
                )


def save_setup():
    c = config.get_global_config()
    os.makedirs(c['task_dir'], exist_ok=True)

    with open('/proc/self/cmdline', 'r') as f:
        cmd_line = f.read().replace('\x00', ' ').strip()
    with open(osp.join(c['task_dir'], 'cmd.sh'), 'w+') as f:
        f.write(cmd_line)
    with open(osp.join(c['task_dir'], 'info.json'), 'w+') as f:
        f.write(json.dumps({'host': socket.gethostname(), 'date': str(datetime.datetime.now())}, indent=4))
    with open(osp.join(c['task_dir'], 'executed_config.json'), 'w+') as f:
        f.write(str(config.get_global_config()))

    shutil.copytree('./src', osp.join(c['task_dir'], 'src'))
    shutil.copytree('./configs', osp.join(c['task_dir'], 'configs'))


def save_snapshot(model, optimizer, phase, extra_data=None):
    c = config.get_global_config()
    save_dict = {'model': model.state_dict(),
                 'optimizer': optimizer.state_dict() if optimizer is not None else None,
                 'RNG': {
                     'python': random.getstate(),
                     'numpy': np.random.get_state(),
                     'pytorch': torch.random.get_rng_state(),
                     'cuda': torch.cuda.get_rng_state_all(),
                     'seed': c.seed
                 },
                 'config': str(config.get_global_config()),
                 'data': extra_data,
                 'active_phase': phase.get_phase(),
                 }
    torch.save(save_dict, c.snapshot_path)


class FreezeRNGState:
    def __init__(self):
        self.store = None

    def __enter__(self):
        self.store = {'python': random.getstate(),
                      'numpy': np.random.get_state(),
                      'pytorch': torch.random.get_rng_state(),
                      'cuda': torch.cuda.get_rng_state_all(),
                      }

    def __exit__(self, exc_type, exc_val, exc_tb):
        random.setstate(self.store['python'])
        torch.random.set_rng_state(self.store['pytorch'])
        torch.cuda.set_rng_state_all(self.store['cuda'])
        np.random.set_state(self.store['numpy'])


class AutoSnapshotter:
    def __init__(self, model, optimizer, current_phase, next_phase, extra_data=None):
        self.model = model
        self.optimizer = optimizer
        self.current_phase = current_phase
        self.next_phase = next_phase
        self.extra_data = extra_data

    def __enter__(self):
        c = config.get_global_config()
        import datasets
        d_name, _ = datasets.parse_specifier(c.source_dataset, return_d_name=True)
        s_d, t_d = ','.join(map(str.title, datasets.parse_specifier(c.source_dataset))), ','.join(map(str.title, datasets.parse_specifier(c.target_dataset)))
        print(termcolor.colored(f"\n\n\nEntering phase [{self.current_phase.get_phase()}] "
                                f"[{d_name}] "
                                f"[{s_d}->{t_d}] "
                                f"[{c.comment}]:\n{c.task_dir} ",
                                on_color=f"on_{self.current_phase.get_color()}"))

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Do not snapshot if error occurs within the context
        if exc_type is None:
            save_snapshot(self.model, self.optimizer, self.next_phase, self.extra_data)
            self.current_phase.inplace_update(self.next_phase.get_phase())


def load_snapshot(path, model):
    print(f"Restoring snapshot from {path}!")
    c = config.get_global_config()
    save_dict = torch.load(path)
    assert c.seed == save_dict['RNG']['seed'], f"Current seed {c.seed} does not match snapshot seed {save_dict['RNG']['seed']}."
    try:
        model.load_state_dict(save_dict['model'])
    except RuntimeError as e:
        print(termcolor.colored(e, on_color='on_red'))
        model.load_state_dict(save_dict['model'], strict=False)
    random.setstate(save_dict['RNG']['python'])
    torch.random.set_rng_state(save_dict['RNG']['pytorch'])
    torch.cuda.set_rng_state_all(save_dict['RNG']['cuda'])
    np.random.set_state(save_dict['RNG']['numpy'])
    return save_dict['data'], save_dict['config'], Phase(save_dict['active_phase'])


def get_optimizer(m, phase):
    c = config.get_global_config()
    params = m.module.get_parameters(phase)
    optimizer_name = c.optimizer_params[phase.get_phase()]['optimizer']
    return getattr(torch.optim, optimizer_name)(params, **{k: v for k, v in c.optimizer_params[phase.get_phase()].items() if k not in {'optimizer'}})


class Domain:
    SOURCE = 'SOURCE'
    TARGET = 'TARGET'

    def __init__(self, name):
        assert name in [self.SOURCE, self.TARGET]
        self.domain = name

    def as_int(self):
        return {self.SOURCE: 0, self.TARGET: 1}[self.domain]

    @staticmethod
    def source():
        return Domain(Domain.SOURCE)

    @staticmethod
    def target():
        return Domain(Domain.TARGET)

    def get_text(self):
        return self.domain

    def __eq__(self, other):
        if isinstance(other, str):
            return self.domain == other
        elif isinstance(other, (int, float)):
            return self.as_int() == other
        elif isinstance(other, Domain):
            return self.domain == other.domain
        elif isinstance(other, list):
            return [self.__eq__(x) for x in other]
        elif isinstance(other, torch.Tensor):
            return (other == self.as_int()).bool()
        elif isinstance(other, np.ndarray):
            return np.array([self.__eq__(x.item()) for x in other], dtype=np.bool)
        else:
            raise ValueError

    def __str__(self):
        return f"<Domain {self.domain}>"


class PseudoLabelType:
    MCD = 'MCD'
    NORMAL = 'NORMAL'


class CorrelationType:
    PEARSON = 'PEARSON'
    SPEARMAN = 'SPEARMAN'
    LIN = 'LIN'
    NONE = 'NONE'


class Transforms:
    TRAIN = TF.Compose([
        TF.RandomResizedCrop((256, 256), scale=(0.75, 1.25), interpolation=InterpolationMode.BICUBIC),
        TF.RandomHorizontalFlip(p=0.5),
        TF.RandomAffine(10, translate=(0.025, 0.025), scale=(0.975, 1.025), shear=10, interpolation=InterpolationMode.BICUBIC),
        TF.RandomChoice([
            TF.RandomGrayscale(p=0.2),
            TF.ColorJitter(brightness=0.05, saturation=0.05, contrast=0.05, hue=0.05)
        ]),
        TF.RandomCrop((224, 224)),
        TF.ToTensor(),
        TF.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])
    TEST = TF.Compose([
        TF.Resize((256, 256)),
        TF.CenterCrop((224, 224)),
        TF.ToTensor(),
        TF.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225])
    ])
    EXTRACT = TF.Compose([
        TF.RandomResizedCrop((256, 256), scale=(0.75, 1.25), interpolation=InterpolationMode.BICUBIC),
        TF.RandomHorizontalFlip(p=0.5),
        TF.RandomAffine(10, translate=(0.025, 0.025), scale=(0.975, 1.025), shear=10, interpolation=InterpolationMode.BICUBIC),
        TF.RandomCrop((224, 224)),
        TF.ToTensor(),
        TF.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])


class Phase:
    TRAIN = 'TRAIN'
    TEST = 'TEST'
    ADAPTATION_TRAIN = 'ADAPTATION_TRAIN'
    FEAT_ADAPTATION_TRAIN = 'FEAT_ADAPTATION_TRAIN'
    SOURCE_ONLY_TRAIN = 'SOURCE_ONLY_TRAIN'
    METRIC_LEARNING = 'METRIC_LEARNING'
    FINAL_TEST = 'FINAL_TEST'
    FEATURE_EXTRACTION = 'FEATURE_EXTRACTION'
    MC_DROPOUT = 'MC_DROPOUT'

    def __init__(self, phase_name: str):
        assert phase_name.upper() in Phase.__dict__
        self.phase = phase_name.upper()
        self._phases = {Phase.TRAIN: 0,
                        Phase.TEST: 2,
                        Phase.ADAPTATION_TRAIN: 4,
                        Phase.FEAT_ADAPTATION_TRAIN: 5,
                        Phase.FINAL_TEST: 6,
                        Phase.FEATURE_EXTRACTION: 10,
                        Phase.MC_DROPOUT: 11,
                        Phase.METRIC_LEARNING: 12}

    def inplace_update(self, new_phase: str):
        assert new_phase.upper() in Phase.__dict__
        self.phase = new_phase

    def as_int(self) -> int:
        return self._phases[self.phase]

    def is_train(self):
        return self.phase in [self.TRAIN, self.ADAPTATION_TRAIN, self.SOURCE_ONLY_TRAIN, self.FEAT_ADAPTATION_TRAIN, self.METRIC_LEARNING]

    def is_test(self):
        return self.phase in [self.TEST, self.FINAL_TEST, self.FEATURE_EXTRACTION, self.MC_DROPOUT]

    def get_phase(self):
        return self.phase

    def __eq__(self, other):
        if isinstance(other, str):
            return self.phase == other.upper()
        elif isinstance(other, Phase):
            return self.phase == other.phase
        else:
            raise TypeError("<other> needs to be of type <str> or <Phase>")

    def get_color(self):
        return {self.TRAIN: None,
                self.TEST: 'cyan',
                self.ADAPTATION_TRAIN: 'blue',
                self.METRIC_LEARNING: 'yellow',
                self.FINAL_TEST: 'magenta'}.get(self.phase, 'grey')


class Metric:
    def __init__(self, mapping=None):
        self.logit_buf = None
        self.target_buf = None
        self.mapping = mapping
        self.reset()

    def reset(self):
        self.logit_buf = torch.empty(0)
        self.target_buf = torch.empty(0).long()

    def update_logits(self, logits, target):
        logits = logits.cpu().detach()
        target = target.cpu().detach()
        self.logit_buf = torch.cat([self.logit_buf, logits], dim=0)
        self.target_buf = torch.cat([self.target_buf, target.view(-1, 1)], dim=0)

    def get_accuracy(self):
        return float((self.logit_buf.argmax(dim=1) == self.target_buf.squeeze()).float().sum() / self.target_buf.shape[0])

    def get_top_k_accuracy(self, k=1):
        top_k_indices = self.logit_buf.topk(dim=1, k=k).indices
        exp_targets = self.target_buf.expand_as(top_k_indices)
        correct_preds = (top_k_indices == exp_targets).float().sum(dim=1).clamp(0, 1)
        top_k_acc = float(correct_preds.sum() / correct_preds.shape[0])
        return top_k_acc

    def get_element_count(self):
        assert self.logit_buf.shape[0] == self.target_buf.shape[0]
        return self.target_buf.shape[0]

    def get_per_class_acc(self, mapping=None):
        if mapping is None:
            mapping = self.mapping
        accs = {}
        sorted_items = sorted(list(mapping.items()), key=lambda x: x[1])
        assert len(sorted_items) == config.get_global_config().current.num_classes
        ctr = 0
        for class_idx, class_name in sorted_items:
            indicies = (self.target_buf.squeeze() == class_idx)
            ctr += int(indicies.float().sum())
            accs[class_name] = float((self.logit_buf[indicies].argmax(dim=1) == self.target_buf[indicies].squeeze()).float().sum() / indicies.float().sum())
        assert len(accs.keys()) == config.get_global_config().current.num_classes
        assert ctr == self.get_element_count()
        return accs

    def get_mean_class_acc(self, mapping=None):
        if mapping is None:
            mapping = self.mapping
        vals = list(self.get_per_class_acc(mapping).values())
        assert len(vals) == config.get_global_config().current.num_classes
        return np.mean(vals).item()

    def get_change_percentage(self, other):
        if other is None:
            return 1.
        assert self.get_element_count() == other.get_element_count()
        agreement = self.logit_buf.argmax(dim=1).squeeze() == other.logit_buf.argmax(dim=1).squeeze()
        return (1. - (agreement.float().sum() / self.get_element_count()).item()) * 100.


def sm_xent_logits(logits, target, reduction='mean', weight=None, temp=1.):
    """ TF-like crossentropy implementation, as PyTorch only accepts the class index and not a one hot encoded label.
    :param logits: NxC matrix of logits.
    :param target: NxC matrix of probability distribution.
    :param temp: Temperature for softmax.
    :return: Average loss over all batch elements.
    """
    loss = torch.sum(- target * F.log_softmax(logits / temp, dim=-1), dim=-1)
    if weight is not None:
        loss *= weight.cuda()
    if reduction == 'mean':
        loss = loss.mean()
    return loss


def label_smoothing(class_label, eps, n, smooth_at=None):
    num_samples = class_label.shape[0]
    smoothed_target = torch.zeros(num_samples, n).fill_(eps / (n - 1)).cuda()
    smoothed_target.scatter_(1, class_label.unsqueeze(1), 1. - eps)

    if smooth_at is not None:
        hard_target = torch.zeros(num_samples, n).cuda().scatter_(1, class_label.unsqueeze(1), 1.)
        indices = ~(smooth_at.bool())
        smoothed_target[indices] = hard_target[indices]
    return smoothed_target


def sample_avoid_dupes(arr, num: int):
    num = int(num)
    if num == 0:
        return []
    elif len(arr) >= num:
        return np.random.choice(arr, size=num, replace=False).tolist()
    else:
        buffer = []
        while len(buffer) != num:
            buffer.extend(np.random.choice(arr, size=min([len(arr), num - len(buffer)]), replace=False).tolist())
            assert len(buffer) <= num, (len(buffer), num)
        return buffer


def clear_line():
    print("\033[K", end='')


def print_header(cycle, phase):
    import datasets
    c = config.get_global_config()
    d_name, _ = datasets.parse_specifier(c.source_dataset, return_d_name=True)
    s_d, t_d = ','.join(map(str.title, datasets.parse_specifier(c.source_dataset))), ','.join(map(str.title, datasets.parse_specifier(c.target_dataset)))
    print(termcolor.colored(f"[{phase.get_phase()}] "
                            f"[Cycle {cycle}/{c.adaptation_cycles}] "
                            f"[{d_name.title()}] "
                            f"[{s_d}–>{t_d}] "
                            f"[{c.comment}] "
                            f"[GPU {','.join(map(str, c.real_gpus))}]", on_color=f"on_{phase.get_color()}"))


def loss_dict_to_str(l):
    return '[' + ' | '.join(f"{k.title()}: {float(v):.3f}" for k, v in l.items()) + ']'


def get_smoothing_positions(domain_flag, domains: list):
    smoothing_positions = torch.zeros_like(domain_flag).bool()
    if domains is not None and len(domains) > 0:
        for d in domains:
            smoothing_positions |= (domain_flag == Domain(d))
    return smoothing_positions


def worker_init_fn(worker_id):
    return np.random.seed((torch.initial_seed() + worker_id) % (2 ** 32 - 1))


class Timer:
    def __init__(self, cycles):
        self.cycles = cycles
        self.current_cycle = 0
        self.__instantiation_time = time.time()
        self.starting_time = time.time()
        self.history = []

    def trigger(self):
        cycle_time = time.time() - self.starting_time
        self.current_cycle += 1
        self.starting_time = time.time()
        self.history.append(cycle_time)
        return cycle_time

    def get_estimate(self):
        if len(self.history) == 0:
            return "??h:??m"

        mean_cycle_duration = float(np.mean(self.history))
        remaining_time = mean_cycle_duration * (self.cycles - self.current_cycle)
        return self.format_time(remaining_time)

    def format_time(self, t: float):
        hours, rest = divmod(t, 3600)
        minutes, seconds = divmod(rest, 60)
        hours, minutes, seconds = map(int, [hours, minutes, seconds])
        return f"{hours:0>2}h:{minutes:0>2}m"

    def get_passed_time(self):
        return self.format_time(time.time() - self.__instantiation_time)


def log_to_tb(d, phase, n_iter, infix=None):
    c = config.get_global_config()
    for k, v in d.items():
        c.tb.add_scalar(f"{phase.get_phase()}/{infix.upper() + '/' if infix is not None else ''}{k.upper()}", v, n_iter)
