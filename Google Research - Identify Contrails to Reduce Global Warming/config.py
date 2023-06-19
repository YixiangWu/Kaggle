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
        'lr_patience': 2
    }
}
