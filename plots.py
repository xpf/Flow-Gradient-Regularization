from opts import get_opts
from utils.utils import get_name
import matplotlib.pyplot as plt
import os, json


def plot_acc(opts, fgr_lambs=[]):
    names = [get_name(opts) for opts.fgr_lamb in fgr_lambs]
    results = []
    for name in names:
        with open(os.path.join(opts.result_path, '{}.json'.format(name)), 'r') as fp:
            results.append(json.load(fp))
    pass


if __name__ == '__main__':
    opts = get_opts()
    plot_acc(opts, fgr_lambs=[0, 100, 200, 300])
