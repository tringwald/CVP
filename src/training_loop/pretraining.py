from functools import partial

import torch as T
from easydict import EasyDict as edict

import config
import util as u
from training_loop.evaluation import evaluate
import losses.losses as L
from torch.cuda.amp import autocast, GradScaler
from feature_extractors import dsbn


def train(model, loader, phase, iterations, optimizer, test_loader):
    c = config.get_global_config()
    metric = u.Metric()
    scheduler = T.optim.lr_scheduler.LambdaLR(optimizer, lambda x: 1.)
    scaler = GradScaler(enabled=c.use_amp)

    # Pre-training loop
    for batch_idx, data in enumerate(loader, start=1):
        model.train()
        model.zero_grad()
        with autocast(enabled=c.use_amp):
            img = data['image'].cuda(non_blocking=True).requires_grad_(False)
            label = data['label'].cuda(non_blocking=True).requires_grad_(False)
            if c.use_dsbn:
                fn = partial(dsbn.inform_batchnorm, domain_mapping=data['domain'], phase=phase)
                model.apply(fn)
            o = edict(model(img, phase))
            losses = L.default_losses(o, label, data)
            if c.use_cvp:
                losses = L.merge(losses, L.cvp_losses(o, label, data))
            loss = sum([v for k, v in losses.items()])

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        metric.update_logits(o.normal.mu_logits, label)
        u.log_to_tb(losses, phase, batch_idx - 1, infix='loss')
        u.clear_line()
        print(f"\r[{batch_idx:>5}/{iterations:>5} @{img.shape[0]}], Acc: {metric.get_accuracy():.3f} ({metric.get_element_count()} elements), "
              f"{u.loss_dict_to_str(losses)}, "
              f"Sig-S: {float(o.sigma.mean(dim=-1).min()):.2f}|{float(o.sigma.mean(dim=-1).max()):.2f}",
              end='')

        if batch_idx % c.test_every == 0:
            del label, img
            m = evaluate(model, test_loader, batch_idx, persist=batch_idx == len(loader), persist_suffix=phase.get_phase(),
                         show_per_class_stats=(batch_idx == len(loader)) and c.show_per_class_acc)
            u.log_to_tb({'gt_acc': m.get_accuracy(), 'gt_mean_acc': m.get_mean_class_acc()}, phase, batch_idx, infix='metric')
    # Flush to new line
    print('')
