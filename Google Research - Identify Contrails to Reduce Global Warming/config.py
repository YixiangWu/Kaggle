import torch


class Path:
    __PATH = '/Users/yixiang/ML_Data'
    ORI_DATA_PATH = __PATH + '/google-research-identify-contrails-reduce-global-warming/'
    DATA_PATH = __PATH + '/google-research-identify-contrails/'


DEVICE = torch.device('cuda') if torch.cuda.is_available() else \
    (torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))

GLOBAL_DICE_COEFFICIENT_SAVING_THRESHOLD = 0.5

MODELS = {
    'unet_resnet34': {
        'name': 'unet_resnet34',
        'epoch': 20,
        'batch_size': 32,
        'network': 'unet',
        'network_kwargs': {
            'encoder_name': 'resnet34',
            'encoder_weights': 'imagenet',
            'activation': None
        },
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'scheduler_kwargs': {
            'patience': 2
        }
    },

    'unet_efficientnet_b0': {
        'name': 'unet_efficientnet_b0',
        'epoch': 20,
        'batch_size': 32,
        'network': 'unet',
        'network_kwargs': {
            'encoder_name': 'efficientnet-b0',
            'encoder_weights': 'imagenet',
            'activation': None
        },
        'optimizer': 'adam',
        'learning_rate': 0.001,
        'scheduler_kwargs': {
            'factor': 0.25,
            'patience': 3,
            'threshold': 0.02,
            'threshold_mode': 'rel'
        }
    }
}
