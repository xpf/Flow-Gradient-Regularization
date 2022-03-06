from opts import get_opts
from utils.utils import get_name
from utils.settings import DATASETTINGS
from models import build_model
from datasets import build_transform, build_data
from torch.utils.data import DataLoader
from attacks.gradient import multi_step
import tqdm, torch, os, json


def evaluate(opts):
    name = get_name(opts)
    print('evaluate', name)
    if not os.path.isdir(opts.result_path): os.mkdir(opts.result_path)
    DSET = DATASETTINGS[opts.data_name]

    model = build_model(opts.model_name, DSET['num_classes']).to(opts.device).eval()
    model.load_state_dict(torch.load(os.path.join(opts.weight_path, '{}.pt'.format(name))))
    val_transform = build_transform(False, DSET['img_size'], DSET['crop_pad'], DSET['flip'])
    val_data = build_data(opts.data_name, opts.data_path, False, val_transform)
    val_loader = DataLoader(val_data, DSET['batch_size'], shuffle=False, num_workers=2)

    correct, total = 0, 0
    desc = 'val   - acc: {:.3f}'
    run_tqdm = tqdm.tqdm(val_loader, desc=desc.format(0), disable=opts.disable_bar)
    for x, y in run_tqdm:
        x, y = x.to(opts.device), y.to(opts.device)
        with torch.no_grad():
            p = model(x)
        _, p = torch.max(p, dim=1)
        correct += (p == y).sum().item()
        total += y.shape[0]
        run_tqdm.set_description(desc.format(correct / total))
    val_acc = correct / total

    if opts.disable_bar:
        print(desc.format(val_acc))

    correct, total = 0, 0
    desc = 'adv   - acc: {:.3f}'
    run_tqdm = tqdm.tqdm(val_loader, desc=desc.format(0), disable=opts.disable_bar)
    for x, y in run_tqdm:
        x, y = x.to(opts.device), y.to(opts.device)
        x = multi_step(model, x, y, iters=100)
        with torch.no_grad():
            p = model(x)
        _, p = torch.max(p, dim=1)
        correct += (p == y).sum().item()
        total += y.shape[0]
        run_tqdm.set_description(desc.format(correct / total))
    adv_acc = correct / total

    if opts.disable_bar:
        print(desc.format(adv_acc))

    with open(os.path.join(opts.result_path, '{}.json'.format(name)), 'w') as fp:
        json.dump({'val_acc': val_acc, 'adv_acc': adv_acc}, fp)


if __name__ == '__main__':
    opts = get_opts()
    evaluate(opts)
