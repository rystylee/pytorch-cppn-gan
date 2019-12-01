import argparse


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_z', type=int, default=32)
    parser.add_argument('--dim_x', type=int, default=1080)
    parser.add_argument('--dim_y', type=int, default=1080)
    parser.add_argument('--dim_c', type=int, default=1)
    parser.add_argument('--ch', type=int, default=32)
    parser.add_argument('--scale', type=float, default=10.0)
    return parser.parse_args()
