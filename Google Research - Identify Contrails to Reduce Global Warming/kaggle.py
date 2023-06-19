import numpy as np
import os
import pandas as pd
import torch
import tqdm

from config import DEVICE, Path
from preprocess import load_image_in_ash_rgb


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


def submit(network_name):
    best = sorted(os.listdir(os.path.join(Path.DATA_PATH, 'network', network_name)))[-1]
    network = torch.load(os.path.join(Path.DATA_PATH, 'network', network_name, best), map_location=DEVICE)
    network.eval()
    torch.set_grad_enabled(False)

    submission = pd.read_csv(os.path.join(Path.ORI_DATA_PATH, 'sample_submission.csv'), index_col='record_id')
    for record_id in tqdm.tqdm(os.listdir(os.path.join(Path.ORI_DATA_PATH, 'test')), desc='Testing'):
        image = load_image_in_ash_rgb(os.path.join(Path.ORI_DATA_PATH, 'test', record_id))
        image = torch.moveaxis(torch.from_numpy(image), -1, 0).float().to(DEVICE)

        prediction = network.predict(image.unsqueeze(0))[0][0]
        prediction = prediction.cpu().numpy()
        prediction[prediction > 0] = 1
        prediction[prediction <= 0] = 0
        submission.loc[int(record_id), 'encoded_pixels'] = rle_encode(prediction)
    submission.to_csv('submission.csv')


if __name__ == '__main__':
    submit('unet_resnet34')
