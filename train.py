import argparse, torch, tqdm, os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from regularizer import FlowGradient
from regularizer import random_start

from utils import get_name, build_dataset, build_model

def train(opts):
    folder, name = get_name(opts=opts)
    print('folder: {}, name: {}'.format(folder, name))
    if not os.path.isdir(os.path.join(opts.weight_path, folder)): os.mkdir(os.path.join(opts.weight_path, folder))

    model = build_model(model_name=opts.model_name, num_classes=int(opts.dataset.replace('cifar', '')))
    model = model.to(opts.device)

    train_data, val_data = build_dataset(dataset=opts.dataset, data_path=opts.data_path)
    train_loader = DataLoader(dataset=train_data, batch_size=opts.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(dataset=val_data, batch_size=opts.batch_size, shuffle=False, num_workers=2)

    optimizer = optim.SGD(model.parameters(), lr=opts.learning_rate, weight_decay=5e-4, momentum=0.9)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, opts.decay_steps, 0.1)
    criterion = nn.CrossEntropyLoss().to(opts.device)

    regularizer = FlowGradient(lambd=opts.lamb)
    best_acc = 0

    for epoch in range(opts.epochs):
        model.train()
        correct, total = 0, 0
        desc = 'train - epoch: {:3d}, acc: {:.3f}'
        run_tqdm = tqdm.tqdm(train_loader, desc=desc.format(0, 0), disable=opts.disable_bar)
        for x, y in run_tqdm:
            x, y = x.to(opts.device), y.to(opts.device)
            x = random_start(x, 0.01)

            x.requires_grad_(True)
            x = regularizer(x)
            p = model(x)
            loss = criterion(p, y)
            loss += regularizer.calculate(loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, p = torch.max(p, dim=1)
            correct += (p == y).sum().item()
            total += y.shape[0]
            run_tqdm.set_description(desc.format(epoch, correct / total))
        train_acc = correct / total
        scheduler.step()
        if opts.disable_bar: print(desc.format(epoch, train_acc))

        model.eval()
        correct, total = 0, 0
        desc = 'val   - epoch: {:3d}, acc: {:.3f}'
        run_tqdm = tqdm.tqdm(val_loader, desc=desc.format(0, 0), disable=opts.disable_bar)
        for x, y in run_tqdm:
            x, y = x.to(opts.device), y.to(opts.device)
            p = model(x)
            _, p = torch.max(p, dim=1)
            correct += (p == y).sum().item()
            total += y.shape[0]
            run_tqdm.set_description(desc.format(epoch, correct / total))
        val_acc = correct / total
        if opts.disable_bar: print(desc.format(epoch, val_acc))

        if best_acc < val_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(opts.weight_path, folder, name + '_weight.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./datasets')
    parser.add_argument('--weight_path', type=str, default='./weights')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'])
    parser.add_argument('--model_name', type=str, default='vgg13_bn')
    parser.add_argument('--lamb', type=float, default=100)

    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--decay_steps', type=list, default=[40, 60])

    parser.add_argument('--device', type=str, default='cuda:1' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--disable_bar', type=bool, default=True)

    opts = parser.parse_args()

    train(opts=opts)
