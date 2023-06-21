import json
import os
import torch


class Path:
    __PATH = '/Users/yixiang/ML_Data'
    ORI_DATA_PATH = __PATH + '/google-research-identify-contrails-reduce-global-warming/'
    DATA_PATH = __PATH + '/google-research-identify-contrails/'


DEVICE = torch.device('cuda') if torch.cuda.is_available() else \
    (torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))

GLOBAL_DICE_COEFFICIENT_SAVING_THRESHOLD = 0.5

with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models.json')) as file:
    MODELS = json.loads(file.read())
