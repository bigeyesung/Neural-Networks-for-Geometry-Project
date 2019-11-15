# Import python dependencies
import argparse


arg_lists = []
parser = argparse.ArgumentParser()


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# -----------------------------------------------------------------------------
# Network
net_arg = add_argument_group("Network")
net_arg.add_argument("--run_mode", type=str, default="test",
                     help='run_mode')
net_arg.add_argument('--input_dim', type=int, default=4096,
                     help='the dimension of the input features')
net_arg.add_argument('--output_dim', type=int, default=32,
                     help='the dimension of the learned local descriptor')
net_arg.add_argument('--log_path', type=str, default='./logs',
                     help='path to the directory with the tensorboard logs')