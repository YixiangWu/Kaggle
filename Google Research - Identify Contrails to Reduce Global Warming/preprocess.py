import numpy as np
import os
import tqdm

from config import Path


def load_image_in_ash_rgb(path):
    """
    Reference:
    L. Kulik, “Satellite-based detection of contrails using deep learning,”
        Ph.D. dissertation, Massachusetts Institute of Technology, 2019.
    """
    data = dict()
    for key in ['band_11', 'band_14', 'band_15']:
        data[key] = np.load(os.path.join(path, key + '.npy'))

    normalize_range = lambda data, bounds: (data - bounds[0]) / (bounds[1] - bounds[0])
    r = normalize_range(data['band_15'] - data['band_14'], (-4, 2))
    g = normalize_range(data['band_14'] - data['band_11'], (-4, 5))
    b = normalize_range(data['band_14'], (243, 303))
    return np.clip(np.stack([r, g, b], axis=2), 0, 1)[..., 4]


def data_preprocess(train_data=True):
    data_set_type = 'train' if train_data else 'validation'
    for record_id in tqdm.tqdm(
            os.listdir(os.path.join(Path.ORI_DATA_PATH, data_set_type)),
            desc=f'Loading {data_set_type.title()} Data'
    ):
        image = load_image_in_ash_rgb(os.path.join(Path.ORI_DATA_PATH, data_set_type, record_id))
        target = np.load(os.path.join(Path.ORI_DATA_PATH, data_set_type, record_id, 'human_pixel_masks.npy'))

        if not os.path.exists(os.path.join(Path.DATA_PATH, data_set_type, record_id)):
            os.makedirs(os.path.join(Path.DATA_PATH, data_set_type, record_id))
        np.save(os.path.join(Path.DATA_PATH, data_set_type, record_id, 'image.npy'), image)
        np.save(os.path.join(Path.DATA_PATH, data_set_type, record_id, 'target.npy'), target)


if __name__ == '__main__':
    data_preprocess(train_data=True)
    data_preprocess(train_data=False)
