import yaml
from easydict import EasyDict as edict

GLOBAL_CONFIG_PATH = "configs/global_config.yaml"
CLI_ARGS = None

with open(GLOBAL_CONFIG_PATH, 'r') as f:
    global_config = edict(yaml.safe_load(f))


def set_cli_args(args):
    global CLI_ARGS
    CLI_ARGS = args


def get_cli_args():
    return CLI_ARGS


def merge_into_global_from_file(file_path):
    with open(file_path, 'r') as f:
        new_values = yaml.safe_load(f)
    global_config.update(new_values)
    return global_config


def merge_into_global_from_dict(d: dict):
    global_config.update(d)
    return global_config


def get_global_config():
    return global_config


def get_minimal_config():
    return {k: v for k, v in get_global_config().items() if k not in {'datasets', 'optimizer_params', 'paths', 'snapshot_path', 'feature_file', 'task_dir',
                                                                      'results_file', 'exp_dir', 'tb', 'tb_path', 'current'}}
