from opts import get_opts
from utils.utils import get_name
import matplotlib.pyplot as plt
import os, json


def plot_acc(opts, fgr_lambs=[]):
    if not os.path.isdir(opts.figure_path): os.mkdir(opts.figure_path)
    names = [get_name(opts) for opts.fgr_lamb in fgr_lambs]
    results = []
    for name in names:
        with open(os.path.join(opts.result_path, '{}.json'.format(name)), 'r') as fp:
            results.append(json.load(fp))
    val_accs, adv_accs = [result['val_acc'] for result in results], [result['adv_acc'] for result in results]
    plt.figure(figsize=(4.8, 3.2))
    for val_acc, adv_acc in zip(val_accs, adv_accs):
        plt.scatter(val_acc, adv_acc)
    legends = [r'$\lambda$={}'.format(fgr_lamb) for fgr_lamb in fgr_lambs]
    plt.legend(legends)
    plt.title(r'Multi-step, $\epsilon$=0.01, K=100')
    plt.xlabel('Clean accuracy'), plt.ylabel('Adversarial accuracy')
    plt.tight_layout()
    plt.savefig(os.path.join(opts.figure_path, '{}_{}.png'.format(opts.data_name, opts.model_name)))
    plt.show()


if __name__ == '__main__':
    opts = get_opts()
    plot_acc(opts, fgr_lambs=[0, 100, 200, 300])
