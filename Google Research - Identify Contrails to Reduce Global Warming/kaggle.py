import numpy as np
import os
import pandas as pd
import torch
import tqdm

from config import DEVICE, MODELS, Path
from model import Model, NETWORKS
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


def submit(model):
    # load networks
    networks = list()
    network_filenames = os.listdir(os.path.join(Path.DATA_PATH, 'network', model.name))
    filenames_to_load = [filename for filename in network_filenames if filename[:len(model.name) + 6] == f'{model.name}_fold_'] if \
        model.k_fold else [sorted(network_filenames)[-1]]  # [sorted(network_filenames)[-1]] <- best among all in the directory
    for filename in filenames_to_load:
        network = NETWORKS[model.network](**model.network_kwargs)
        network.load_state_dict(torch.load(os.path.join(Path.DATA_PATH, 'network', model.name, filename), map_location=DEVICE))
        networks.append(network.to(DEVICE))

    submission = pd.read_csv(os.path.join(Path.ORI_DATA_PATH, 'sample_submission.csv'), index_col='record_id')
    for record_id in tqdm.tqdm(os.listdir(os.path.join(Path.ORI_DATA_PATH, 'test')), desc='Testing'):
        image = torch.from_numpy(load_image(os.path.join(Path.ORI_DATA_PATH, 'test', record_id))).float().to(DEVICE)
        image = image[:] if model.additional_channel else image[:3]
        prediction = ensemble(networks, image, voting_ensemble=model.voting_ensemble)
        submission.loc[int(record_id), 'encoded_pixels'] = rle_encode(prediction)
    submission.to_csv('submission.csv')


if __name__ == '__main__':
    submit(Model(**MODELS['Early Exploration']['unet_resnet34']))
