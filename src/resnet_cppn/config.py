import os
import time
import argparse
import json


def get_config():
    parser = argparse.ArgumentParser()

    # mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])

    # dataset
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--dataset_name', type=str, default='mnist')
    parser.add_argument('--with_label', action='store_true')
    parser.add_argument('--img_size', type=int, default=28)

    # training
    parser.add_argument('--max_itr', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr_g', type=float, default=0.002)
    parser.add_argument('--lr_d', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--n_gen', type=int, default=3)
    parser.add_argument('--checkpoint_path', type=str, default='')

    # model
    parser.add_argument('--dim_z', type=int, default=32)
    parser.add_argument('--dim_x', type=int, default=1024, help='Used for generate large scale image after training')
    parser.add_argument('--dim_y', type=int, default=1024, help='Used for generate large scale image after training')
    parser.add_argument('--dim_c', type=int, default=1)
    parser.add_argument('--ngf', type=int, default=6)
    parser.add_argument('--n_resblock', type=int, default=24)
    parser.add_argument('--n_sub_resblock', type=int, default=4)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--scale', type=float, default=1.0)

    # misc
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint')
    parser.add_argument('--checkpoint_interval', type=int, default=100)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--log_interval', type=int, default=20)
    parser.add_argument('--sample_dir', type=str, default='samples')
    parser.add_argument('--sample_interval', type=int, default=100)

    args = parser.parse_args()

    time_str = time.strftime("%Y%m%d-%H%M%S")
    config_name = f'{time_str}-resnet-{args.dataset_name}-s{args.img_size}-b{args.batch_size}'

    runs_path = os.path.join('runs', config_name)
    args.log_dir = os.path.join(runs_path, args.log_dir)
    args.checkpoint_dir = os.path.join(runs_path, args.checkpoint_dir)
    args.sample_dir = os.path.join(runs_path, args.sample_dir)

    if args.mode == 'train':
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.sample_dir, exist_ok=True)
        with open(os.path.join(runs_path, 'config.txt'), 'w') as f:
            json.dump(args.__dict__, f, indent=4)

    return args
