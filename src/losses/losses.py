import config
import util as u
import torch as T
import torch.nn.functional as F
from typing import Dict


def default_losses(o, label, data) -> Dict[str, T.Tensor]:
    c = config.get_global_config()
    losses = {}
    losses['xent'] = u.sm_xent_logits(o.normal.mu_logits,
                                      u.label_smoothing(label,
                                                        eps=c.lsm_eps,
                                                        n=c.current.num_classes,
                                                        smooth_at=u.get_smoothing_positions(data['domain'], c.smooth_domains)
                                                        )
                                      )
    return losses


def antagonist_loss(samples_xent, sigma, upper_bound, num_samples):
    xent = samples_xent.clone().detach().view(-1, num_samples)
    loss_means = xent.mean(dim=-1)
    capped = F.relu(upper_bound - loss_means)
    return F.smooth_l1_loss(sigma.mean(dim=-1), capped)


def cvp_losses(o, label, data) -> Dict[str, T.Tensor]:
    assert len(o.sigma.squeeze().shape) == 2
    c = config.get_global_config()
    losses = {}

    # Xent loss for samples
    smoothing_mask = u.get_smoothing_positions(data['domain'].view(-1, 1).expand(label.shape[0], c.num_samples).flatten(), domains=c.smooth_domains)
    nonreduced_loss = u.sm_xent_logits(o.normal.samples_logits.view(-1, c.current.num_classes),
                                       u.label_smoothing(label.view(-1, 1).expand(label.shape[0], c.num_samples).flatten(),
                                                         eps=c.lsm_eps,
                                                         n=c.current.num_classes,
                                                         smooth_at=smoothing_mask
                                                         ),
                                       reduction=None
                                       )
    losses['samples_xent'] = c.samples_loss_weight * nonreduced_loss.mean()

    # Antagonist loss
    losses['Antag'] = antagonist_loss(nonreduced_loss, o.sigma, c.upper_bound, c.num_samples)
    return losses


def merge(*args: Dict[str, T.Tensor]):
    output = {}
    for arg in args:
        for k, v in arg.items():
            assert k not in output, f"Duplicated key {k} found!"
            output[k] = v
    return output
