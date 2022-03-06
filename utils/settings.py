DATASETTINGS = {
    'cifar10': {
        'num_classes': 10,
        'img_size': 32,
        'crop_pad': 4,
        'flip': True,

        'epochs': 80,
        'batch_size': 128,
        'learning_rate': 0.01,
        'decay_steps': [40, 60],
    },
    'cifar100': {
        'num_classes': 100,
        'img_size': 32,
        'crop_pad': 4,
        'flip': True,

        'epochs': 80,
        'batch_size': 128,
        'learning_rate': 0.01,
        'decay_steps': [40, 60],
    }
}
