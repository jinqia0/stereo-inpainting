import os
import json
import argparse
import subprocess

from shutil import copyfile
import torch.distributed as dist

import torch
import torch.multiprocessing as mp

import core
import core.trainer
import core.trainer_flow_w_edge


# import warnings
# warnings.filterwarnings("ignore")

from core.dist import (
    get_world_size,
    get_local_rank,
    get_global_rank,
    get_master_ip,
)

parser = argparse.ArgumentParser()
parser.add_argument('-c',
                    '--config',
                    default='ProPainter/configs/train_propainter.json',
                    type=str)
parser.add_argument('-p', '--port', default='23490', type=str)
args = parser.parse_args()


def main_worker(rank, config):
    if 'local_rank' not in config:
        config['local_rank'] = config['global_rank'] = rank
    if config['distributed']:
        torch.cuda.set_device(int(config['local_rank']))
        torch.distributed.init_process_group(
            backend='nccl', init_method=config['init_method'],
            world_size=config['world_size'], rank=config['global_rank'],
            group_name='mtorch'
        )
        print(
            f"using GPU {int(config['global_rank'])}-{int(config['local_rank'])} for training"
        )


    config['save_dir'] = os.path.join(
        config['save_dir'],
        f"{config['model']['net']}_{os.path.basename(args.config).split('.')[0]}",
    )

    config['save_metric_dir'] = os.path.join(
        './scores',
        f"{config['model']['net']}_{os.path.basename(args.config).split('.')[0]}",
    )

    if torch.cuda.is_available():
        config['device'] = torch.device(f"cuda:{config['local_rank']}")
    else:
        config['device'] = 'cpu'

    if (not config['distributed']) or config['global_rank'] == 0:
        os.makedirs(config['save_dir'], exist_ok=True)
        config_path = os.path.join(config['save_dir'], args.config.split('/')[-1])
        if not os.path.isfile(config_path):
            copyfile(args.config, config_path)
        print(f"[**] create folder {config['save_dir']}")

    trainer_version = config['trainer']['version']
    trainer = core.__dict__[trainer_version].__dict__['TrainerStereo'](config)
    # Trainer(config)
    trainer.train()


if __name__ == "__main__":

    torch.backends.cudnn.benchmark = True

    mp.set_sharing_strategy('file_system')

    # loading configs
    config = json.load(open(args.config))

    # setting distributed configurations
    # config['world_size'] = get_world_size()
    config['world_size'] = torch.cuda.device_count()
    config['init_method'] = f"tcp://{get_master_ip()}:{args.port}"
    config['distributed'] = config['world_size'] > 1
    print('world_size:', config['world_size'])
    # setup distributed parallel training environments

    # if get_master_ip() == "127.0.0.X":
    #     # manually launch distributed processes
    #     mp.spawn(main_worker, nprocs=config['world_size'], args=(config, ))
    # else:
    #     # multiple processes have been launched by openmpi
    #     config['local_rank'] = get_local_rank()
    #     config['global_rank'] = get_global_rank()
    #     main_worker(-1, config)

    mp.spawn(main_worker, nprocs=torch.cuda.device_count(), args=(config, ))