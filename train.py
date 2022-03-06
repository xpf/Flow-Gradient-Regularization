from opts import get_opts
from utils.utils import get_name
from utils.settings import DATASETTINGS
from models import build_model
from datasets import build_transform, build_data
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from regularizers.fgr import FlowGradientReg
import tqdm, torch, os


def train(opts):
    name = get_name(opts)
    print('train', name)
    if not os.path.isdir(opts.weight_path): os.mkdir(opts.weight_path)
    DSET = DATASETTINGS[opts.data_name]

    model = build_model(opts.model_name, DSET['num_classes']).to(opts.device)
    train_transform = build_transform(True, DSET['img_size'], DSET['crop_pad'], DSET['flip'])
    val_transform = build_transform(False, DSET['img_size'], DSET['crop_pad'], DSET['flip'])
    train_data = build_data(opts.data_name, opts.data_path, True, train_transform)
    val_data = build_data(opts.data_name, opts.data_path, False, val_transform)
    train_loader = DataLoader(train_data, DSET['batch_size'], shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, DSET['batch_size'], shuffle=False, num_workers=2)

    optimizer = optim.SGD(model.parameters(), lr=DSET['learning_rate'], weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, DSET['decay_steps'], 0.1)
    criterion = nn.CrossEntropyLoss().to(opts.device)

    if opts.fgr_lamb > 0:
        regularizer = FlowGradientReg().to(opts.device)

    best_val_acc = 0
    for epoch in range(DSET['epochs']):
        model.train()
        correct, total = 0, 0
        desc = 'train - epoch: {:3d}, acc: {:.3f}'
        run_tqdm = tqdm.tqdm(train_loader, desc=desc.format(0, 0), disable=opts.disable_bar)
        for x, y in run_tqdm:
            x, y = x.to(opts.device), y.to(opts.device)
            if opts.fgr_lamb > 0 and opts.rs_sigma > 0:
                x = regularizer.random_start(x, opts.rs_sigma)
            if opts.fgr_lamb > 0:
                x.requires_grad_(True)
                x = regularizer(x)
            p = model(x)
            loss = criterion(p, y)
            if opts.fgr_lamb > 0:
                loss = loss + opts.fgr_lamb * regularizer.calculate(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, p = torch.max(p, dim=1)
            correct += (p == y).sum().item()
            total += y.shape[0]
            run_tqdm.set_description(desc.format(epoch, correct / total))
        train_acc = correct / total
        scheduler.step()

        if opts.disable_bar:
            print(desc.format(epoch, train_acc))

        model.eval()
        correct, total = 0, 0
        desc = 'val   - epoch: {:3d}, acc: {:.3f}'
        run_tqdm = tqdm.tqdm(val_loader, desc=desc.format(0, 0), disable=opts.disable_bar)
        for x, y in run_tqdm:
            x, y = x.to(opts.device), y.to(opts.device)
            with torch.no_grad():
                p = model(x)
            _, p = torch.max(p, dim=1)
            correct += (p == y).sum().item()
            total += y.shape[0]
            run_tqdm.set_description(desc.format(epoch, correct / total))
        val_acc = correct / total

        if opts.disable_bar:
            print(desc.format(epoch, val_acc))

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(opts.weight_path, '{}.pt'.format(name)))


if __name__ == '__main__':
    opts = get_opts()
    train(opts)
