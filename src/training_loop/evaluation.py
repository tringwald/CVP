from functools import partial

import termcolor
import torch as T
import torch.nn.functional as F
from easydict import EasyDict as edict

import config
from losses import correlation
import util as u
from torch.cuda.amp import autocast
from feature_extractors import dsbn


def evaluate(model, test_loader, iteration, show_per_class_stats=False, persist=False, persist_suffix=None, show_top_k=False):
    with u.FreezeRNGState():
        model.eval()
        metric = u.Metric(mapping=test_loader.dataset.idx_to_class)
        total_len = len(test_loader)
        c = config.get_global_config()
        phase = u.Phase(u.Phase.TEST)

        sigs = T.empty(0)
        logits = T.empty(0)
        gts = T.empty(0)

        with T.no_grad():
            for batch_idx, data in enumerate(test_loader, start=1):
                with autocast(enabled=c.use_amp):
                    label = data['label'].cuda(non_blocking=True).requires_grad_(False)
                    img = data['image'].cuda().requires_grad_(False)
                    domain = data['domain'].cuda()
                    if c.use_dsbn:
                        fn = partial(dsbn.inform_batchnorm, domain_mapping=data['domain'], phase=phase)
                        model.apply(fn)
                    assert bool(T.all(domain == u.Domain.target()))
                    u.clear_line()
                    print(f"\rEvaluating [{batch_idx:>5}/{total_len:>5} @{img.shape[0]}]", end='')
                    o = edict(model(img, phase))
                    sigs = T.cat([sigs, o.sigma.cpu().detach()], dim=0)
                    logits = T.cat([logits, o.normal.mu_logits.cpu().detach()], dim=0)
                    gts = T.cat([gts, data['__gt'].cpu().detach().squeeze()], dim=0)
                    metric.update_logits(o.normal.mu_logits, label)

        # Correlation
        _pearson_score_gt = correlation.pearsonr(sigs.mean(dim=-1).squeeze(), logits.gather(1, gts.view(-1, 1).long()).squeeze()).item()
        _pearson_score = correlation.pearsonr(sigs.mean(dim=-1).squeeze(), logits.max(dim=-1).values.squeeze()).item()
        _pearson_score_sm = correlation.pearsonr(sigs.mean(dim=-1).squeeze(), logits.softmax(dim=-1).max(dim=-1).values.squeeze()).item()
        _sorted = F.softmax(logits, dim=-1).sort(dim=-1, descending=True).values
        _sorted_logits = logits.sort(dim=-1, descending=True).values
        _pearson_score_diff = correlation.pearsonr(sigs.mean(dim=-1).squeeze(),
                                                   (_sorted[:, 0] - _sorted[:, 1]).squeeze()).item()
        _pearson_score_logits_diff = correlation.pearsonr(sigs.mean(dim=-1).squeeze(),
                                                          (_sorted_logits[:, 0] - _sorted_logits[:, 1]).squeeze()).item()
        u.clear_line()
        print(f"\r┃ Pearson MaxL: "
              f"L_GT={_pearson_score_gt * 100:.1f} "
              f"L={_pearson_score * 100:.1f}, "
              f"SM={_pearson_score_sm * 100:.1f}, "
              f"Diff_SM={_pearson_score_diff * 100:.1f}, "
              f"Diff_Logits={_pearson_score_logits_diff * 100:.1f}, "
              )

        # Standard accuracy and per-class accuracy
        formatted_metric = termcolor.colored(f"{metric.get_accuracy() * 100:>6.3f}", on_color='on_grey')
        print(f"\r┃ Eval acc over {metric.get_element_count()} elements: {formatted_metric}")
        if show_per_class_stats:
            items = metric.get_per_class_acc(test_loader.dataset.idx_to_class).items()
            for idx, (k, v) in enumerate(items):
                print(f"┃   {k:<20}:   {v * 100:>6.3f}")

        # Print top-k acc
        if show_top_k:
            output_buf = []
            for k in range(1, min(6, config.get_global_config().current.num_classes + 1)):
                output_buf.append(f"Top-{k} acc: {metric.get_top_k_accuracy(k) * 100:>6.3f}")
            print('┃', ' | '.join(output_buf))

        # Output mean class accuracy
        formatted_class_acc = termcolor.colored(f"{metric.get_mean_class_acc() * 100:.3f}", on_color='on_grey')
        if c.show_mean_class_acc:
            print(f"┃ Mean over class accuracy: {formatted_class_acc}")

        # Persist results to filesystem
        if persist:
            with open(c.results_file.format(persist_suffix), 'a+') as f:
                f.write(f"{c.source_dataset},"
                        f"{c.target_dataset},"
                        f"Acc: {metric.get_accuracy() * 100},"
                        f"MAcc: {metric.get_mean_class_acc() * 100},"
                        f"P: L={_pearson_score:.5f} SM={_pearson_score_sm:.5f} Diff={_pearson_score_diff:.5f}, "
                        f"{c.comment}\n")
        return metric
