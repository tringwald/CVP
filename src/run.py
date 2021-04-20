import random
import warnings
from pprint import pprint

import numpy as np
import torch as T
from torch.utils.data import DataLoader

import architectures
import argument_parser
import config
import datasets
import sampler
import util
import util as u
from training_loop.adaptation import adapt
from training_loop.evaluation import evaluate
from training_loop.pretraining import train


def get_loaders():
    c = config.get_global_config()
    # Load datasets
    source_dataset = datasets.load_dataset(c.source_dataset, transforms=u.Transforms.TRAIN, domain=u.Domain.source(), keep_gt_label=True)
    source_loader = DataLoader(source_dataset, num_workers=c.workers, pin_memory=True, batch_sampler=sampler.EndlessBatchSampler(source_dataset),
                               worker_init_fn=util.worker_init_fn)
    evaluation_dataset = datasets.load_dataset(c.target_dataset, transforms=u.Transforms.TEST, domain=u.Domain.target(), keep_gt_label=True, usage='evaluation')
    evaluation_loader = DataLoader(evaluation_dataset, batch_size=c.test_batch_size, shuffle=False, num_workers=c.workers * 2, pin_memory=True, drop_last=False,
                                   worker_init_fn=util.worker_init_fn)
    return source_dataset, source_loader, evaluation_dataset, evaluation_loader


def create_adaptation_loaders():
    c = config.get_global_config()
    source_dataset, _, _, evaluation_loader = get_loaders()
    # Adapt model on target
    target_dataset = datasets.load_dataset(c.target_dataset, transforms=u.Transforms.TRAIN, keep_gt_label=False, domain=u.Domain.target(),
                                           usage='concat target')
    concat_dataset = datasets.ConcatDataset(source=source_dataset, target=target_dataset)
    # noinspection PyArgumentList
    concat_loader = DataLoader(concat_dataset,
                               batch_sampler=sampler.BatchSampler(concat_dataset),
                               num_workers=c.workers,
                               pin_memory=True,
                               worker_init_fn=util.worker_init_fn)
    return concat_loader, evaluation_loader


def main():
    # Fix for PyTorch warning that is not actually correct
    warnings.filterwarnings('ignore', category=UserWarning)

    # Load config and set initial seeds for reproducibility
    c = argument_parser.parse()
    random.seed(c.seed)
    T.manual_seed(c.seed)
    T.cuda.manual_seed(c.seed)
    T.cuda.manual_seed_all(c.seed)
    np.random.seed(c.seed)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False

    # Print current config
    pprint(config.get_minimal_config())

    model = architectures.get_model(c.architecture, c.gpus)
    # If snapshot is set, overwrite current_phase and restore weights and RNG states
    if c.snapshot is not None:
        loaded_data, loaded_config, current_phase = u.load_snapshot(c.snapshot, model)
    else:
        current_phase = u.Phase(u.Phase.SOURCE_ONLY_TRAIN)

    # Training phases
    if current_phase == u.Phase(u.Phase.SOURCE_ONLY_TRAIN):
        optimizer = u.get_optimizer(model, current_phase)
        _, source_loader, _, evaluation_loader = get_loaders()
        with u.AutoSnapshotter(model, optimizer, current_phase, next_phase=u.Phase(u.Phase.ADAPTATION_TRAIN)):
            # Train on source only
            train(model, source_loader,
                  phase=current_phase,
                  iterations=c.source_iterations,
                  optimizer=optimizer,
                  test_loader=evaluation_loader)
        del source_loader, evaluation_loader
        if c.source_only:
            return

    if current_phase == u.Phase(u.Phase.ADAPTATION_TRAIN):
        optimizer = u.get_optimizer(model, current_phase)
        concat_loader, evaluation_loader = create_adaptation_loaders()
        with u.AutoSnapshotter(model, optimizer, current_phase, next_phase=u.Phase(u.Phase.FINAL_TEST)):
            # noinspection PyTypeChecker
            adapt(model, concat_loader, phase=current_phase, optimizer=optimizer, test_loader=evaluation_loader)
        del concat_loader, evaluation_loader

    if current_phase == u.Phase(u.Phase.FINAL_TEST):
        _, _, _, evaluation_loader = get_loaders()
        with u.AutoSnapshotter(model, None, current_phase, next_phase=u.Phase(u.Phase.FINAL_TEST)):
            m = evaluate(model, evaluation_loader, -1, show_per_class_stats=c.show_per_class_acc, persist=True, persist_suffix=current_phase.get_phase())
            u.log_to_tb({'gt_acc': m.get_accuracy(), 'gt_mean_acc': m.get_mean_class_acc()}, u.Phase(u.Phase.FINAL_TEST), 1, infix='metric')
            return m


if __name__ == '__main__':
    main()
