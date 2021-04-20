from functools import partial

import torch as T
from easydict import EasyDict as edict
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
import numpy as np

import util
from feature_extractors import dsbn

import config
import datasets
import util as u
import losses.losses as L
from training_loop.evaluation import evaluate
from training_loop.feature_extraction import extract_features


def adapt(model, loader, phase, optimizer, test_loader):
    c = config.get_global_config()
    optimizer.zero_grad()
    scheduler = T.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1.)
    scaler = GradScaler(enabled=c.use_amp)

    # Augmented loader
    feature_extract_dataset = datasets.load_dataset(c.target_dataset, transforms=None, keep_gt_label=False, domain=u.Domain.target(),
                                                    usage='feature extraction')
    # noinspection PyArgumentList
    feature_extract_loader = DataLoader(feature_extract_dataset, batch_size=c.test_batch_size, shuffle=False, num_workers=c.workers * 2, pin_memory=True,
                                        drop_last=False, persistent_workers=True, worker_init_fn=util.worker_init_fn)

    timer = u.Timer(c.adaptation_cycles)
    for ada_cycle in range(1, c.adaptation_cycles + 1):
        u.print_header(ada_cycle, phase)

        # Extract features and setup batch sampler for the current cycle
        extracted_tar_data = extract_features(model, feature_extract_loader)
        loader.batch_sampler.set_values(extracted_tar_data, domain=u.Domain.target())

        model.train()
        model.zero_grad()
        for batch_idx, data in enumerate(loader, start=1):
            with autocast(enabled=c.use_amp):
                # Prepare input data for model and move to GPU
                img = data['image'].cuda(non_blocking=True).requires_grad_(False)
                label = data['label'].cuda(non_blocking=True).requires_grad_(False)
                domain = data['domain'].cuda(non_blocking=True).requires_grad_(False)
                src_mask = domain == u.Domain.source()
                tar_mask = domain == u.Domain.target()
                if c.use_dsbn:
                    fn = partial(dsbn.inform_batchnorm, domain_mapping=data['domain'], phase=phase)
                    model.apply(fn)

                # Actual forward pass
                o = edict(model(img, phase))

                # Gather losses
                losses: dict = L.default_losses(o, label, data)
                if c.use_cvp:
                    losses = L.merge(losses, L.cvp_losses(o, label, data))
                loss = sum([v for k, v in losses.items()])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            u.log_to_tb(losses, phase, (ada_cycle - 1) * c.adaptation_cycle_length + batch_idx - 1, infix='loss')
            u.clear_line()
            print(f"\r[{batch_idx:>4}/{len(loader):>4} @{img.shape[0]}]: "
                  f"LR {optimizer.param_groups[0]['lr']}, "
                  f"{u.loss_dict_to_str(losses)}, "
                  f"Sig-S: {float(o.sigma[src_mask].mean(dim=-1).min()):.2f}|{float(o.sigma[src_mask].mean(dim=-1).max()):.2f}, "
                  f"Sig-T: {float(o.sigma[tar_mask].mean(dim=-1).min()):.2f}|{float(o.sigma[tar_mask].mean(dim=-1).max()):.2f}",
                  end='')
            u.persist_sigma(ada_cycle, batch_idx, o.sigma, data['domain'])

        # Reset cursor
        print('')

        # Forward the scheduler after every cycle
        scheduler.step()
        if ada_cycle % 10 in [0, 5] and ada_cycle != c.adaptation_cycles:
            del label, img
            m = evaluate(model, test_loader, ada_cycle, show_per_class_stats=False)
            u.log_to_tb({'gt_acc': m.get_accuracy(),
                         'gt_mean_acc': m.get_mean_class_acc()}, phase, ada_cycle, infix='metric')
        timer.trigger()
