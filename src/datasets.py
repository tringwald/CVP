import threading
import numpy as np
import random
import config
from functools import lru_cache
import os
import os.path as osp
import re
import torch
from torch.utils.data import Dataset as PTDataset
from PIL import Image
import util as u
from typing import List
import matplotlib.pyplot as plt

DATASET_SEPARATOR = '/'


class DatasetInstance:
    def __init__(self, path, dataset_index, class_index, domain: u.Domain, subdomain: str, keep_gt: bool, labelmap: dict = None):
        self.path = path
        self.dataset_subdomain = subdomain
        self.dataset_index = dataset_index
        self.labelmap = labelmap
        self.domain = domain
        self.__gt_loaded = keep_gt
        self.__gt_class_index = class_index
        self.class_index = class_index if keep_gt else None
        self.tainted = False

        self.distribution_mean = None
        self.distribution_std = None

        # Make sure to not leak GT labels
        assert (self.class_index is None) ^ keep_gt

    def _get_ground_truth(self):
        return self.__gt_class_index

    def get_current_class_name(self):
        return self.labelmap[int(self.class_index)]

    def __getstate__(self):
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, torch.Tensor):
                v.share_memory_()
                d[k] = v
            elif isinstance(v, np.ndarray):
                d[k] = {'tag': 'numpy', 'data': v.tolist(), 'dtype': str(v.dtype)}
            else:
                d[k] = v
        return d

    def __setstate__(self, state):
        for k, v in state.items():
            if isinstance(v, dict) and 'tag' in v and 'data' in v and 'dtype' in v:
                if v['tag'] == 'numpy':
                    state[k] = np.array(v['data'], dtype=getattr(np, v['dtype']))
                else:
                    raise AttributeError
        self.__dict__.update(state)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"<DSI: of class {self.get_current_class_name()} (GT{' not ' if self.__gt_loaded else ' '}loaded)>"


class Dataset(PTDataset):
    DEFAULT_EXTENSIONS = ['jpeg', 'jpg', 'png', 'bmp', 'gif', 'tiff']

    def __init__(self, dataset_name, dataset_subdomains, domain, keep_gt_label, transforms=u.Transforms.TEST):
        c = config.get_global_config()
        self.name = dataset_name
        self.dataset_subdomains = dataset_subdomains
        self.__had_gt = keep_gt_label
        self.transforms = transforms
        self.domain: u.Domain = domain
        self.dataset: List[DatasetInstance] = []
        self.merged = c.multisource_merged and domain == u.Domain.source()

        # Iterate over all subdomains:
        for subdomain in self.dataset_subdomains:
            # Parse file pattern
            pattern = c['current']['pattern']
            self.dataset_domain_root = osp.join(pattern[:pattern.find('<domain>')], subdomain)
            self.dataset_domain_class_root = pattern[:pattern.find('<class>')].replace('<domain>', subdomain)
            self.class_names = list(x for x in sorted(os.listdir(self.dataset_domain_class_root)) if osp.isdir(osp.join(self.dataset_domain_class_root, x)))
            self.class_to_idx = {k: v for k, v in zip(self.class_names, range(len(self.class_names)))}
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
            image_pattern = re.compile(pattern.replace('<domain>', subdomain)
                                       .replace('<class>', """(?P<class>.*?)""")
                                       .replace('<image>', """(?P<image>.*)"""),
                                       re.DOTALL | re.UNICODE)

            # Walk through dataset root, check against regex pattern
            for root, dirs, files in os.walk(self.dataset_domain_root):
                dirs[:] = list(sorted(dirs))
                if files:
                    for file in sorted(files):
                        full_path = osp.join(root, file)
                        m = image_pattern.match(full_path)
                        if m is not None:
                            if full_path.split('.')[-1].lower() in c['current'].get('file_extensions', self.DEFAULT_EXTENSIONS):
                                class_name = m.groupdict()['class']
                                self.dataset.append(DatasetInstance(path=full_path,
                                                                    dataset_index=float('nan'),
                                                                    class_index=self.class_to_idx[class_name],
                                                                    domain=self.domain,
                                                                    subdomain=str(self.dataset_subdomains) if self.merged else subdomain,
                                                                    keep_gt=keep_gt_label,
                                                                    labelmap=self.idx_to_class))
                            else:
                                print(f"Found questionable file {full_path}!")

        # Dataset is loaded in order, this could leak information ==> shuffle it in a deterministic way
        _state = random.getstate()
        random.seed(0)
        random.shuffle(self.dataset)
        random.setstate(_state)
        # Renumber
        for idx, inst in enumerate(self.dataset):
            inst.dataset_index = idx
        assert self.num_classes == c['current']['num_classes']

    def __iter__(self):
        yield from self.dataset

    def __len__(self):
        return len(self.dataset)

    @property
    def num_classes(self):
        return len(self.class_names)

    def __str__(self):
        return f"<DS: {self.name}, domain {self.dataset_subdomains}_[Merged={self.merged}] with {len(self.dataset)} instances (GT" \
               f"{' not ' if not self.__had_gt else ' '}loaded)>"

    def __getitem__(self, index):
        if isinstance(index, (int, np.int64, np.int32, np.int)):
            inst = self.dataset[index]
        elif isinstance(index, DatasetInstance):
            inst = index
        else:
            raise AttributeError()
        return self.make_response(inst)

    def set_transforms(self, state):
        self.transforms = state

    def apply_transforms(self, img):
        trans_img = self.transforms(img)
        return trans_img

    def get_instance(self, index):
        return self.dataset[index]

    def make_response(self, inst):
        img = Image.open(inst.path).convert('RGB')
        trans_img = self.apply_transforms(img)
        if config.get_global_config()['debug'] and (threading.current_thread() is threading.main_thread()):
            # Unnormalize and display image
            unnorm_img = Image.fromarray((np.clip((trans_img.cpu().numpy() *
                                                   np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1) +
                                                   np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)) * 255., 0., 255.).astype(np.uint8)).transpose(1, 2, 0))
            plt.imshow(unnorm_img)
            plt.show()
            plt.clf()
        _tmp = {'image': trans_img,
                'label': inst.class_index if inst.class_index is not None else -1,
                'domain': inst.domain.as_int(),
                '__gt': inst._get_ground_truth(),
                }
        self.add_variable(_tmp, inst, 'path', "N/A")
        if inst.distribution_mean is None:
            _placeholder_mean = torch.zeros(len(self.class_to_idx))
            _placeholder_mean[inst.class_index] = 1.
            _placeholder_std = torch.zeros(len(self.class_to_idx))
        else:
            _placeholder_mean = float('nan')
            _placeholder_std = float('nan')
        self.add_variable(_tmp, inst, 'distribution_mean', _placeholder_mean)
        self.add_variable(_tmp, inst, 'distribution_std', _placeholder_std)
        return _tmp

    def add_variable(self, d, inst, v, default=None):
        try:
            tmp = getattr(inst, v)
            d[v] = tmp if tmp is not None else default
        except:
            d[v] = default


class ConcatDataset(PTDataset):
    def __init__(self, source, target):
        self.source_dataset = source
        self.target_dataset = target

    def get_domain(self, d):
        if d == u.Domain.target():
            return self.target_dataset
        elif d == u.Domain.source():
            return self.source_dataset
        else:
            raise ValueError()

    def __len__(self):
        return len(self.source_dataset) + len(self.target_dataset)

    def __getitem__(self, item):
        assert isinstance(item, DatasetInstance)
        if item.domain == u.Domain.source():
            return self.source_dataset.make_response(item)
        elif item.domain == u.Domain.target():
            return self.target_dataset.make_response(item)
        else:
            raise ValueError()


def load_dataset(dataset_specifier, domain, keep_gt_label=True, transforms=u.Transforms.TEST, usage: str = None):
    dataset_name, d_subdomains = parse_specifier(dataset_specifier, return_d_name=True)
    print(f"Loading dataset {dataset_name}, domain {d_subdomains} as {str(domain)}, GT={keep_gt_label}{f', usage={usage.upper()}' if usage else ''}.")
    d = Dataset(dataset_name, d_subdomains, domain, keep_gt_label, transforms)
    print(f"    Found {len(d)} samples from {d.num_classes} classes.")
    if domain == u.Domain.source():
        print(f"    Source domains are {'merged' if d.merged else 'separate'}.")
    return d


def parse_specifier(spec, return_d_name=False):
    d_names = []
    d_subdomains = []
    for ds in spec:
        dataset_name, dataset_subdomain = ds.split(DATASET_SEPARATOR)
        d_names.append(dataset_name)
        d_subdomains.append(dataset_subdomain)
    if return_d_name:
        return d_names[0], d_subdomains
    else:
        return d_subdomains


@lru_cache(maxsize=None)
def get_available():
    gc = config.get_global_config()
    datasets = []
    for k, v in gc['datasets'].items():
        datasets.extend((f"{k}{DATASET_SEPARATOR}{domain}" for domain in v['domains']))
    return datasets
