import argparse


def get_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data_path', type=str, default='/data/xpf/datasets')
    parser.add_argument('--weight_path', type=str, default='./weights')
    parser.add_argument('--result_path', type=str, default='./results')
    parser.add_argument('--figure_path', type=str, default='./figures')
    parser.add_argument('--data_name', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--model_name', type=str, default='vgg11', choices=['vgg11', 'vgg13', 'vgg16', 'vgg19'])
    parser.add_argument('--fgr_lamb', type=float, default=0)
    parser.add_argument('--rs_sigma', type=float, default=0.01)
    parser.add_argument('--disable_bar', type=bool, default=False)
    opts = parser.parse_args()
    return opts
