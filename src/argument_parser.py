import datetime
import socket
import argparse
import datasets
import config
import os
import math
import util as u
import os
import os.path as osp
import torch
import torch.utils.tensorboard as tb
import warnings


def remove_extension(s, ext=('yaml',)):
    for ex in ext:
        if s.endswith(ex):
            return s[:-len(ex) - 1]
    return s


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-dataset', type=str, choices=datasets.get_available(), nargs='+')
    parser.add_argument('--target-dataset', type=str, choices=datasets.get_available(), nargs='+')
    parser.add_argument('--configs', type=str, default=[], nargs='+')
    parser.add_argument('--sub-dir', type=str, default='')
    parser.add_argument('--task-dir', type=str, default=None)
    parser.add_argument('--exp-dir', type=str, default=None)
    parser.add_argument('--snapshot', type=str, default=None)
    parser.add_argument('--comment', type=str, required=True)
    args = parser.parse_args()
    config.set_cli_args(args)
    for sub_config in args.configs:
        config.merge_into_global_from_file(sub_config)
    c = config.get_global_config()

    # Check GPU setup
    try:
        c.real_gpus = list(map(lambda x: int(x.strip()), os.environ['CUDA_VISIBLE_DEVICES'].split(',')))
        c.gpus = list(range(torch.cuda.device_count()))
    except Exception as e:
        raise ValueError('Please set CUDA_VISIBLE_DEVICES.')
    c.workers = 1 if c.debug else 8 * len(c.gpus)
    if len(c.gpus) != 2:
        warnings.warn(f"Please use 2 GPUs for reproducibility, chosen {len(c.gpus)} instead.")

    # Setup datasets
    source_datasets = [x.split(datasets.DATASET_SEPARATOR)[0] for x in args.source_dataset]
    source_subdomains = [x.split(datasets.DATASET_SEPARATOR)[1] for x in args.source_dataset]
    target_subdomains = [x.split(datasets.DATASET_SEPARATOR)[1] for x in args.target_dataset]
    assert len(set(source_datasets)) == 1, "Can only load from same datasets!"
    dataset_name = args.source_dataset[0].split(datasets.DATASET_SEPARATOR)[0]
    c['current'] = c['datasets'][dataset_name]

    c.adaptation_cycle_length = int(math.ceil(c.current.num_classes / c.sample_num_classes) * 5)
    c.upper_bound = math.log(c.current.num_classes)

    # Make logdir
    args.exp_dir = osp.join(c['paths']['log_dir'],
                            dataset_name,
                            c['architecture'],
                            args.sub_dir,
                            remove_extension(args.comment.split('/')[-1]),
                            )
    args.results_file = osp.join(args.exp_dir, 'results_{}.csv')
    args.task_dir = osp.join(args.exp_dir,
                             "{}-{}_{}_{}".format(','.join(source_subdomains),
                                                  ','.join(target_subdomains),
                                                  str(datetime.datetime.now()).replace(' ', '_'),
                                                  socket.gethostname(),
                                                  ),
                             )
    args.feature_file = osp.join(args.task_dir, 'misc', 'features_{}.pth')
    args.sigma_file = osp.join(args.task_dir, 'misc', 'sigmas.csv')
    os.makedirs(osp.dirname(args.feature_file), exist_ok=True, mode=0o700)

    args.snapshot_path = osp.join(args.task_dir, 'snapshots', 'snapshot.pth')
    os.makedirs(osp.dirname(args.snapshot_path), exist_ok=True, mode=0o700)
    args.tb_path = osp.join(args.task_dir, 'tb')
    os.makedirs(osp.dirname(args.tb_path), exist_ok=True, mode=0o700)
    args.tb = tb.SummaryWriter(log_dir=args.tb_path, max_queue=1, flush_secs=1)

    # Fix path when loading from snapshot
    if args.snapshot is not None and not args.snapshot.endswith('.pth'):
        args.snapshot = osp.join(args.snapshot, 'snapshots', 'snapshot.pth')

    config.merge_into_global_from_dict(args.__dict__)
    u.save_setup()
    return c
