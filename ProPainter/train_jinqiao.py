import os
import json
import argparse

from shutil import copyfile
import torch.distributed as dist

import torch
import torch.multiprocessing as mp

import core
import core.trainer
import core.trainer_flow_w_edge


def main(config):
    # 设置设备
    torch.cuda.set_device(0)  # 使用第一个 GPU
    print("Using single GPU for training.")

    config['save_dir'] = os.path.join(
        config['save_dir'],
        f"{config['model']['net']}_{os.path.basename(args.config).split('.')[0]}",
    )

    config['save_metric_dir'] = os.path.join(
        './scores',
        f"{config['model']['net']}_{os.path.basename(args.config).split('.')[0]}",
    )

    config['device'] = torch.device("cuda:0") if torch.cuda.is_available() else "cpu"

    os.makedirs(config['save_dir'], exist_ok=True)
    config_path = os.path.join(config['save_dir'], args.config.split('/')[-1])
    if not os.path.isfile(config_path):
        copyfile(args.config, config_path)
    print(f"[**] create folder {config['save_dir']}")

    trainer_version = config['trainer']['version']
    trainer = core.__dict__[trainer_version].__dict__['Trainer'](config)
    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--config',
                        default='/root/autodl-tmp/stereo-inpainting/ProPainter/configs/train_propainter.json',
                        type=str)
    args = parser.parse_args()
    
    torch.backends.cudnn.benchmark = True

    # 加载配置
    config = json.load(open(args.config))
    config['world_size'] = torch.cuda.device_count()

    main(config)
