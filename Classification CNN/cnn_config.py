
# base128_lr0.001_mom0.9_VGG3_img50_bs64_SGD_epoch20_dropout_augmentation
# ### VGG1 Model
models_config = [

    {
        'augmentation': True,
        'lr': 0.001,
        'momentum': 0.9,
        'target_size': (224, 224),
        'batch_size': 64,
        'optimizer': 'SGD',
        'epochs': 10,
        'layers': [],
        'base': {
            'units': 128,
            'dropout': None
        }
    },

    {
        'augmentation': True,
        'lr': 0.001,
        'momentum': None,
        'target_size': (224, 224),
        'batch_size': 64,
        'optimizer': 'Adamax',
        'epochs': 150,
        'layers': [
            {
                'filters': 32,
                'kernel_size': (3, 3),
                'pooling': (2, 2),
                'dropout': 0.1
            },
            {
                'filters': 64,
                'kernel_size': (3, 3),
                'pooling': (2, 2),
                'dropout': 0.1
            },
            {
                'filters': 128,
                'kernel_size': (3, 3),
                'pooling': (2, 2),
                'dropout': 0.1
            },
        ],
        'base': {
            'units': 300,
            'dropout': 0.3
        }
    },

    {
        'augmentation': True,
        'lr': 0.001,
        'momentum': None,
        'target_size': (224, 224),
        'batch_size': 64,
        'optimizer': 'RMSprop',
        'epochs': 150,
        'layers': [
            {
                'filters': 32,
                'kernel_size': (3, 3),
                'pooling': (2, 2),
                'dropout': 0.1
            },
            {
                'filters': 64,
                'kernel_size': (3, 3),
                'pooling': (2, 2),
                'dropout': 0.1
            },
            {
                'filters': 128,
                'kernel_size': (3, 3),
                'pooling': (2, 2),
                'dropout': 0.1
            },
        ],
        'base': {
            'units': 300,
            'dropout': 0.3
        }
    },

    {
        'augmentation': True,
        'lr': 0.001,
        'momentum': None,
        'target_size': (224, 224),
        'batch_size': 64,
        'optimizer': 'Nadam',
        'epochs': 150,
        'layers': [
            {
                'filters': 32,
                'kernel_size': (3, 3),
                'pooling': (2, 2),
                'dropout': 0.1
            },
            {
                'filters': 64,
                'kernel_size': (3, 3),
                'pooling': (2, 2),
                'dropout': 0.1
            },
            {
                'filters': 128,
                'kernel_size': (3, 3),
                'pooling': (2, 2),
                'dropout': 0.1
            },
        ],
        'base': {
            'units': 300,
            'dropout': 0.3
        }
    },

    {
        'augmentation': True,
        'lr': 0.001,
        'momentum': 0.9,
        'target_size': (224, 224),
        'batch_size': 64,
        'optimizer': 'SGD',
        'epochs': 150,
        'layers': [
            {
                'filters': 32,
                'kernel_size': (3, 3),
                'pooling': (2, 2),
                'dropout': 0.1
            },
            {
                'filters': 64,
                'kernel_size': (3, 3),
                'pooling': (2, 2),
                'dropout': 0.1
            },
            {
                'filters': 128,
                'kernel_size': (3, 3),
                'pooling': (2, 2),
                'dropout': 0.1
            },
        ],
        'base': {
            'units': 300,
            'dropout': 0.3
        }
    },

    {
        'augmentation': True,
        'lr': 0.001,
        'momentum': None,
        'target_size': (224, 224),
        'batch_size': 64,
        'optimizer': 'Adam',
        'epochs': 150,
        'layers': [
            {
                'filters': 32,
                'kernel_size': (3, 3),
                'pooling': (2, 2),
                'dropout': 0.1
            },
            {
                'filters': 64,
                'kernel_size': (3, 3),
                'pooling': (2, 2),
                'dropout': 0.1
            },
            {
                'filters': 128,
                'kernel_size': (3, 3),
                'pooling': (2, 2),
                'dropout': 0.1
            },
        ],
        'base': {
            'units': 300,
            'dropout': 0.3
        }
    },

    {
        'augmentation': True,
        'lr': 0.001,
        'momentum': None,
        'target_size': (224, 224),
        'batch_size': 64,
        'optimizer': 'Adagrad',
        'epochs': 150,
        'layers': [
            {
                'filters': 32,
                'kernel_size': (3, 3),
                'pooling': (2, 2),
                'dropout': 0.1
            },
            {
                'filters': 64,
                'kernel_size': (3, 3),
                'pooling': (2, 2),
                'dropout': 0.1
            },
            {
                'filters': 128,
                'kernel_size': (3, 3),
                'pooling': (2, 2),
                'dropout': 0.1
            },
        ],
        'base': {
            'units': 300,
            'dropout': 0.3
        }
    },
    {
        'augmentation': True,
        'lr': 0.001,
        'momentum': None,
        'target_size': (224, 224),
        'batch_size': 64,
        'optimizer': 'Adadelta',
        'epochs': 150,
        'layers': [
            {
                'filters': 32,
                'kernel_size': (3, 3),
                'pooling': (2, 2),
                'dropout': 0.1
            },
            {
                'filters': 64,
                'kernel_size': (3, 3),
                'pooling': (2, 2),
                'dropout': 0.1
            },
            {
                'filters': 128,
                'kernel_size': (3, 3),
                'pooling': (2, 2),
                'dropout': 0.1
            },
        ],
        'base': {
            'units': 300,
            'dropout': 0.3
        }
    },
    {
        'augmentation': True,
        'lr': 0.001,
        'momentum': None,
        'target_size': (224, 224),
        'batch_size': 64,
        'optimizer': 'Ftrl',
        'epochs': 150,
        'layers': [
            {
                'filters': 32,
                'kernel_size': (3, 3),
                'pooling': (2, 2),
                'dropout': 0.1
            },
            {
                'filters': 64,
                'kernel_size': (3, 3),
                'pooling': (2, 2),
                'dropout': 0.1
            },
            {
                'filters': 128,
                'kernel_size': (3, 3),
                'pooling': (2, 2),
                'dropout': 0.1
            },
        ],
        'base': {
            'units': 300,
            'dropout': 0.3
        }
    },
]
