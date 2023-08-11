import numpy as np
import os
import pandas as pd
import torch
import tqdm

from config import DEVICE, MODELS, Path
from model import Model
from preprocess import load_image


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


def __ensemble(predictions, ensemble_method='averaging'):
    if ensemble_method == 'voting':
        for i in range(len(predictions)):
            predictions[i][predictions[i] > 0] = 1
            predictions[i][predictions[i] <= 0] = -1
    return np.average(predictions, axis=0)


def ensemble(networks, image, resize, ensemble_method='averaging'):
    if resize:
        image = resize[0](image)
    predictions = list()
    for network in networks:
        network.eval()
        torch.set_grad_enabled(False)
        prediction = network.predict(image.unsqueeze(0))[0]
        prediction = (resize[1](prediction) if resize else prediction)[0].cpu().numpy()
        predictions.append(prediction)
    return __ensemble(predictions, ensemble_method=ensemble_method)


def submit(models, cls_threshold=0, seg_threshold=0, seg_ensemble_method='averaging', seg_weights=None):
    if len(models) == 2:  # classification + segmentation
        assert models[0].input_size[1:] == models[1].input_size[1:]
        assert models[0].additional_channel == models[1].additional_channel

    # load networks
    networks = list()
    for model in models:
        networks.append(model.load_networks())

    submission = pd.read_csv(os.path.join(Path.ORI_DATA_PATH, 'sample_submission.csv'), index_col='record_id')
    for record_id in tqdm.tqdm(os.listdir(os.path.join(Path.ORI_DATA_PATH, 'test')), desc='Testing'):
        image = torch.from_numpy(load_image(os.path.join(Path.ORI_DATA_PATH, 'test', record_id))).float().to(DEVICE)
        image = image[:] if models[0].additional_channel else image[:3]

        if len(models) == 1:
            prediction = ensemble(networks[0], image, models[0].resize, ensemble_method=models[0].voting_ensemble)
        else:  # networks[0]: classification, networks[1:]: segmentation
            networks[0][0].eval()
            torch.set_grad_enabled(False)
            prediction = networks[0][0]((models[0].resize[0](image) if models[0].resize else image).unsqueeze(0))[0]
            prediction = prediction.cpu().numpy()
            if prediction[0] + cls_threshold < prediction[1]:
                prediction = np.zeros(models[0].image_size)
            else:
                predictions = list()
                for i, network in enumerate(networks[1:]):
                    predictions.append(ensemble(network, image, models[i + 1].resize, ensemble_method=models[i + 1].voting_ensemble))
                    if seg_weights:
                        predictions[-1] *= seg_weights[i]
                prediction = __ensemble(predictions, ensemble_method=seg_ensemble_method)
                prediction[prediction > seg_threshold] = 1
                prediction[prediction <= seg_threshold] = 0

        # from dataset import display_image
        # display_image(np.moveaxis(np.array(image.cpu()), 0, -1), prediction)
        submission.loc[int(record_id), 'encoded_pixels'] = rle_encode(prediction)
    submission.to_csv('submission.csv')


if __name__ == '__main__':
    submit([
        Model(**MODELS['Classification']['classification_backbone_ensemble']),
        Model(**MODELS['Segmentation']['segmentation_unet_efficientnet_b4_kfold_5']),
        Model(**MODELS['Segmentation']['segmentation_unet_resnest50d_kfold_5']),
        Model(**MODELS['Segmentation']['segmentation_unet_resnest101e_kfold_5'])
    ],
        cls_threshold=0, seg_threshold=-5, seg_ensemble_method='averaging', seg_weights=(1.1, 0.8, 1.1)
    )
