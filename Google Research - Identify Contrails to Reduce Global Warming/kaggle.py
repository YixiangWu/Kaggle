import numpy as np
import os
import pandas as pd
import torch
import tqdm

from config import DEVICE, MODELS, Path
from model import Model
from preprocess import load_image
from utilities import ensemble, load_networks


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


def submit(models):
    if len(models) == 2:  # classification + segmentation
        assert models[0].input_size == models[1].input_size
        assert models[0].additional_channel == models[1].additional_channel

    # load networks
    networks = list()
    for model in models:
        networks.extend(load_networks(model.name, model.k_fold, model.network, model.network_kwargs))

    submission = pd.read_csv(os.path.join(Path.ORI_DATA_PATH, 'sample_submission.csv'), index_col='record_id')
    for record_id in tqdm.tqdm(os.listdir(os.path.join(Path.ORI_DATA_PATH, 'test')), desc='Testing'):
        image = torch.from_numpy(load_image(os.path.join(Path.ORI_DATA_PATH, 'test', record_id))).float().to(DEVICE)
        image = image[:] if models[0].additional_channel else image[:3]
        prediction = None

        if len(models) == 1:
            prediction = ensemble(networks, image, models[0].resize, voting_ensemble=models[0].voting_ensemble)
        elif len(models) == 2:  # networks[0]: classification, networks[1:]: segmentation
            networks[0].eval()
            torch.set_grad_enabled(False)
            prediction = networks[0]((models[0].resize[0](image) if models[0].resize else image).unsqueeze(0))[0]
            prediction = prediction.cpu().numpy()
            prediction = np.zeros(models[0].image_size) if prediction[0] < prediction[1] else \
                ensemble(networks[1:], image, models[1].resize, voting_ensemble=models[1].voting_ensemble)

        # from dataset import display_image
        # display_image(np.moveaxis(np.array(image.cpu()), 0, -1), prediction)
        submission.loc[int(record_id), 'encoded_pixels'] = rle_encode(prediction)
    submission.to_csv('submission.csv')


if __name__ == '__main__':
    submit([Model(**MODELS['Early Exploration']['unet_resnet34'])])
