from torchvision import datasets, transforms

from models.vgg import vgg13_bn
from models.resnet import resnet20

def get_name(opts):
    folder = '{}_{}'.format(opts.dataset, opts.model_name)
    name = 'fgr_{}'.format(opts.lamb)
    return folder, name

DATASETS = {
    'cifar10': datasets.CIFAR10, 'cifar100': datasets.CIFAR100,
}

def build_dataset(dataset='cifar10', data_path=''):
    assert dataset in DATASETS.keys()
    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ])
    train_data = DATASETS[dataset](root='{}/{}/'.format(data_path, dataset), transform=train_transforms, train=True)
    val_data = DATASETS[dataset](root='{}/{}/'.format(data_path, dataset), transform=val_transforms, train=False)
    return train_data, val_data

MODELS = {
    'resnet20': resnet20, 'vgg13_bn': vgg13_bn,
}

def build_model(model_name='vgg16_bn', num_classes=10):
    model = MODELS[model_name](num_classes=num_classes)
    return model