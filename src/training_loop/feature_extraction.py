from functools import partial

import numpy as np
import torch as T
from easydict import EasyDict as edict
from feature_extractors import dsbn

import config
import util as u
from feature_extractors.dsbn import DomainSpecificBatchNorm
from torch.cuda.amp import autocast


def extract_features(model, loader):
    model.eval()
    total_len = len(loader)
    c = config.get_global_config()

    output_buf = {}
    with T.no_grad():
        for name, trans, phase, repeats in [('uncertainty',
                                             u.Transforms.EXTRACT,
                                             u.Phase(u.Phase.MC_DROPOUT),
                                             c.uncertain_repeats)]:
            # Buffer to be filled
            output_buf[name] = {'paths': [np.empty(0)] * repeats,
                                'mu': [T.empty(0)] * repeats,
                                'mu_logits': [T.empty(0)] * repeats,
                                'features': [T.empty(0)] * repeats,
                                'sigma': [T.empty(0)] * repeats,
                                'mu_mcd_logits': [T.empty(0)] * repeats}
            # Set data augmentations
            loader.dataset.set_transforms(trans)

            dataset_size = 0
            for i_repeat in range(repeats):
                for batch_idx, data in enumerate(loader, start=1):
                    with autocast():
                        u.clear_line()
                        print(f"\r[{name.upper()}]: [{i_repeat + 1:>2}/{repeats:>2}] Extracting features [{batch_idx:>5}/{total_len:>5}] ...", end='')
                        dataset_size += data['image'].shape[0]
                        img = data['image'].cuda().requires_grad_(False)
                        if c.use_dsbn:
                            fn = partial(dsbn.inform_batchnorm, domain_mapping=data['domain'], phase=phase)
                            model.apply(fn)
                        o = edict(model(img, phase))
                    output_buf[name]['mu'][i_repeat] = T.cat([output_buf[name]['mu'][i_repeat], o.mu.detach().cpu()], dim=0)
                    output_buf[name]['mu_logits'][i_repeat] = T.cat([output_buf[name]['mu_logits'][i_repeat], o.normal.mu_logits.detach().cpu()], dim=0)
                    output_buf[name]['mu_mcd_logits'][i_repeat] = T.cat([output_buf[name]['mu_mcd_logits'][i_repeat], o.mcd.mu_logits.detach().cpu()], dim=0)
                    output_buf[name]['sigma'][i_repeat] = T.cat([output_buf[name]['sigma'][i_repeat], o.sigma.detach().cpu()], dim=0)
                    output_buf[name]['paths'][i_repeat] = np.hstack([output_buf[name]['paths'][i_repeat], np.array(data['path'])])

            # Concat the results from multiple runs, output shape is <Datset Size x Repeat (x MC iters) x Classes/Feature size>
            output_buf[name]['mu'] = T.cat([x.unsqueeze(1) for x in output_buf[name]['mu']], dim=1)
            output_buf[name]['mu_logits'] = T.cat([x.unsqueeze(1) for x in output_buf[name]['mu_logits']], dim=1)
            output_buf[name]['mu_mcd_logits'] = T.cat([x.unsqueeze(1) for x in output_buf[name]['mu_mcd_logits']], dim=1)
            output_buf[name]['paths'] = np.concatenate([x.reshape(-1, 1) for x in output_buf[name]['paths']], axis=1)
    u.clear_line()
    print(f"\rFeature extraction done (repeats={c.uncertain_repeats}, size={dataset_size // repeats}). {' ' * 100}", end='')
    return output_buf
