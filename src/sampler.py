import copy
from collections import defaultdict
from functools import partial
import random

import numpy as np
import torch as T
from easydict import EasyDict as edict
from typing import Union, Dict
from torch.utils.data.sampler import BatchSampler as PTBatchSampler

import config
import util as u


class BatchSampler(PTBatchSampler):
    def __init__(self, concat_dataset):
        self.c = config.get_global_config()
        self.concat_dataset = concat_dataset
        self.multi_source_ptr = 0

        # Create source structure
        self.source_structure = defaultdict(partial(defaultdict, list))
        for inst in self.concat_dataset.source_dataset:
            self.source_structure[inst.dataset_subdomain][inst.class_index].append(inst)

        # Uncertain results
        self.o = None

    def __len__(self) -> int:
        return self.c.adaptation_cycle_length

    def __iter__(self):
        c = config.get_global_config()

        # Prepare uncertainty, shape is Datset Size x Repeat (x MC iters) x Classes
        assert len(self.o.uncertainty['mu_logits'].shape) == 3, self.o.uncertainty['mu_logits'].shape
        assert len(self.o.uncertainty['mu_mcd_logits'].shape) == 4, self.o.uncertainty['mu_mcd_logits'].shape
        if c.pseudo_label_type.upper() == u.PseudoLabelType.NORMAL:
            softmaxed_logits = self.o.uncertainty['mu_logits'].softmax(dim=-1)
            mean_probs = softmaxed_logits.mean(dim=1)
            std_probs = T.zeros_like(mean_probs)
            argmax_class = mean_probs.argmax(dim=-1).squeeze()
        elif c.pseudo_label_type.upper() == u.PseudoLabelType.MCD:
            softmaxed_logits = self.o.uncertainty['mu_mcd_logits'].softmax(dim=-1)
            mean_probs = softmaxed_logits.mean(dim=1).mean(dim=1)
            std_probs = softmaxed_logits.mean(dim=1).std(dim=1)
            argmax_class = mean_probs.argmax(dim=-1).squeeze()
        else:
            raise AttributeError

        # Create target structure for current cycle
        target_structure = defaultdict(list)
        for idx, inst in enumerate(self.concat_dataset.get_domain(u.Domain.target())):
            assert not inst.tainted and inst.class_index is None
            _inst = copy.deepcopy(inst)
            _inst.tainted = True

            _inst.class_index = int(argmax_class[idx])
            _inst.distribution_mean = mean_probs[idx]
            _inst.distribution_std = std_probs[idx]
            _inst.chosen_prob = float(_inst.distribution_mean[_inst.class_index])
            target_structure[_inst.class_index].append(_inst)
        yield from self.sample_batches(self.source_structure, target_structure, num_batches=c.adaptation_cycle_length)

    def sample_batches(self, s_struct: Dict, t_struct: Dict, num_batches: int):
        c = config.get_global_config()
        src_subdomain = self.get_subdomain(self.source_structure)

        prepared_classes = u.sample_avoid_dupes(np.arange(c.current.num_classes, dtype=np.int32), c.sample_num_classes * num_batches)

        # Actually sample and interleave batch
        for batch_num in range(num_batches):
            current_batch = []
            sampled_classes = prepared_classes[batch_num * c.sample_num_classes:(batch_num + 1) * c.sample_num_classes]
            per_domain_samples = c.batch_size / (2 * c.sample_num_classes)

            for cl in sampled_classes:
                tmp_source, tmp_target = [], []
                # Sample source
                tmp_source.extend(u.sample_avoid_dupes(s_struct[src_subdomain][cl], per_domain_samples))
                # Sample target
                if cl in t_struct:
                    tmp_target.extend(u.sample_avoid_dupes(t_struct[cl], per_domain_samples))
                else:
                    # Use source samples instead if no image in the target domain was classified as <cl>
                    tmp_target.extend(u.sample_avoid_dupes(s_struct[src_subdomain][cl], per_domain_samples))
                for s, t in zip(tmp_source, tmp_target):
                    current_batch.extend((s, t))
            yield current_batch

    def get_subdomain(self, s_struct: Dict) -> str:
        """ Round robin sampling of a domain name in case of multiple source domains, otherwise just return the single available domain.
        :param s_struct: Source domain mapping: domain name -> class -> DatasetInstances
        :return: The sampled domain name.
        """
        num_source_domains = list(sorted(s_struct.keys()))
        if len(num_source_domains) > 1:
            sampled_d_subdomain = num_source_domains[self.multi_source_ptr]
            self.multi_source_ptr = (self.multi_source_ptr + 1) % len(num_source_domains)
        else:
            sampled_d_subdomain = list(s_struct.keys())[0]
        return sampled_d_subdomain

    def set_values(self, output_buf, domain):
        # features, logits, paths, certain_logits, certain_features,
        if domain == u.Domain.target():
            self.o = edict(output_buf)
        else:
            raise AttributeError()


class EndlessBatchSampler(PTBatchSampler):
    """Fixes suboptimal PyTorch dataloader behavior for small datasets. PyTorch used to kill all worker processes at the end of iteration; for small datasets
    this resulted in a constant cycle of killing and respawning worker processes. In newer PyTorch versions, one can use the <persistent_workers> flag instead.
    """

    def __init__(self, d):
        self.c = config.get_global_config()
        self.d = d

    def __len__(self):
        return self.c.source_iterations

    def __iter__(self):
        batch_size = self.c.batch_size
        ind_buf = []
        while len(ind_buf) < len(self) * batch_size:
            ind_buf.extend(T.randperm(len(self.d)).tolist())

        # Sample one batch
        for batch_idx in range(len(self)):
            yield ind_buf[batch_idx * batch_size: (batch_idx + 1) * batch_size]
