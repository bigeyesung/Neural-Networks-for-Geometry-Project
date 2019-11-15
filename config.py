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

                     # Test
test_arg = add_argument_group("Evaluate")
test_arg.add_argument("--evaluate_input_folder", type=str, default="./data/evaluate/input_data/",
                          help='prefix for the input folder locations')
test_arg.add_argument("--evaluate_output_folder", type=str, default="./data/evaluate/output_data/",
                          help='prefix for the output folder locations')
test_arg.add_argument('--evaluation_batch_size', type=int, default=1000,
                          help='the number of examples for each iteration of inference')
test_arg.add_argument('--saved_model_dir', type=str, default='./models/',
                     help='the directory of the pre-trained model')
test_arg.add_argument('--saved_model_evaluate', type=str, default='3DSmoothNet',
                     help='file name of the model to load')