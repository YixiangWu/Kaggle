import numpy as np
import os
import pandas as pd
import torch
import tqdm

from config import DEVICE, MODELS, Path
from preprocess import load_image
from utilities import ensemble


def rle_encode(mask):
    """Provided by Kaggle"""
    dots = np.where(mask.T.flatten() == 1)[0]
    run_lengths, prev = [], -2
    for b in dots:
        if b > prev + 1:
            run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return str(run_lengths).replace('[', '').replace(']', '').replace(',', '') if run_lengths else '-'


def submit(model_name):
    network_name = MODELS[model_name][0]['name']
    k_fold = MODELS[model_name][0]['k_fold']
    additional_channel = MODELS[model_name][0]['additional_channel']
    voting_ensemble = MODELS[model_name][1]['voting_ensemble']
    network_filenames = os.listdir(os.path.join(Path.DATA_PATH, 'network', network_name))

    # load networks
    networks = list()
    if not k_fold:
        best = sorted(network_filenames)[-1]
        networks.append(torch.load(os.path.join(Path.DATA_PATH, 'network', network_name, best), map_location=DEVICE))
    else:
        for name in network_filenames:
            if name[:len(network_name) + 6] == f'{network_name}_fold_':
                networks.append(torch.load(os.path.join(Path.DATA_PATH, 'network', network_name, name), map_location=DEVICE))

    submission = pd.read_csv(os.path.join(Path.ORI_DATA_PATH, 'sample_submission.csv'), index_col='record_id')
    for record_id in tqdm.tqdm(os.listdir(os.path.join(Path.ORI_DATA_PATH, 'test')), desc='Testing'):
        image = torch.from_numpy(load_image(os.path.join(Path.ORI_DATA_PATH, 'test', record_id))).float().to(DEVICE)
        image = image[:] if additional_channel else image[:3]
        prediction = ensemble(networks, image, voting_ensemble=voting_ensemble)
        submission.loc[int(record_id), 'encoded_pixels'] = rle_encode(prediction)
    submission.to_csv('submission.csv')


if __name__ == '__main__':
    submit('unet_resnet34')
