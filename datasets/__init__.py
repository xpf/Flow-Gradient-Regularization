from os.path import join
from torchvision import transforms
from datasets.cifar10 import CIFAR10
from datasets.cifar100 import CIFAR100

DATASETS = {
    'cifar10': CIFAR10,
    'cifar100': CIFAR100,
}


def build_transform(train, img_size, crop_pad, flip):
    transform = []
    transform.append(transforms.Resize((img_size + crop_pad, img_size + crop_pad)))
    if train:
        transform.append(transforms.RandomCrop((img_size, img_size)))
        if flip:
            transform.append(transforms.RandomHorizontalFlip())
    else:
        transform.append(transforms.CenterCrop((img_size, img_size)))
    transform = transforms.Compose(transform)
    return transform


def build_data(data_name, data_path, train, transform):
    assert data_name in DATASETS.keys()
    data = DATASETS[data_name](join(data_path, data_name), train, transform)
    return data
