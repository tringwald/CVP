from itertools import chain
import torch
import torch.nn as nn
import torch.optim
from util import Domain


def inform_batchnorm(m, domain_mapping, phase):
    num_devices = torch.cuda.device_count()
    chunked = domain_mapping.chunk(num_devices)
    if hasattr(m, '_inform_bn'):
        m._inform_bn(chunked, phase)


class DomainSpecificBatchNorm(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.bn_dict = nn.ModuleDict()
        self.bn_dict[Domain.SOURCE] = nn.BatchNorm2d(*args, **kwargs)
        self.bn_dict[Domain.TARGET] = nn.BatchNorm2d(*args, **kwargs)
        # Domains of input samples
        self.domain_mapping = None
        self.phase = None

    def forward(self, input_tensor):
        # Support multi-GPU training and evaluation
        cur_replica_domains = self.domain_mapping[input_tensor.device.index]
        assert cur_replica_domains.size(0) == input_tensor.size(0), f"{cur_replica_domains.size(0)} domains doesn't match {input_tensor.size(0)} inputs."

        source_indices = cur_replica_domains == Domain.source().as_int()
        target_indices = cur_replica_domains == Domain.target().as_int()
        input_tensor[source_indices] = self.bn_dict[Domain.SOURCE](input_tensor[source_indices])
        input_tensor[target_indices] = self.bn_dict[Domain.TARGET](input_tensor[target_indices])
        return input_tensor

    def _inform_bn(self, domain_mapping, phase):
        self.domain_mapping = domain_mapping
        self.phase = phase

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)
        if version is None or version < 2:
            # at version 2: added num_batches_tracked buffer
            #               this should have a default value of 0
            num_batches_tracked_key = prefix + 'num_batches_tracked'
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)
        # Load from standard BN
        state_params = dict((n, p) for n, p in state_dict.items() if n.startswith(prefix))

        for state_n, state_p in state_params.items():
            for n, p in chain(self.named_parameters(), self.named_buffers()):
                if n.split('.')[-1] == state_n.split('.')[-1]:
                    p.data.copy_(state_p)
                if n == state_n:
                    p.data.copy_(state_p)

    def _check_input_dim(self, inp):
        if inp.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(inp.dim()))
