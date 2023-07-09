import json
import numpy as np
import os
import tqdm

from config import Path


def find_duplicates(difference_threshold=2401):
    with open(os.path.join(Path.ORI_DATA_PATH, 'train_metadata.json')) as file:
        raw_train_metadata = json.loads(file.read())
    with open(os.path.join(Path.ORI_DATA_PATH, 'validation_metadata.json')) as file:
        raw_test_metadata = json.loads(file.read())

    # store timestamps that share the same geographic information together
    metadata = dict()
    for raw_metadata in [raw_train_metadata, raw_test_metadata]:
        for data in raw_metadata:
            geographic_info = (data['row_min'], data['row_size'], data['col_min'], data['col_size'])
            if geographic_info not in metadata:
                metadata[geographic_info] = [list(), dict()]
            metadata[geographic_info][0].append(data['timestamp'])
            metadata[geographic_info][1][data['timestamp']] = data['record_id']

    # find duplicates (timestamps difference below a certain threshold)
    duplicates = list()
    for geographic_info, data in metadata.items():
        if len(data[0]) > 1:
            data[0].sort()
            for i, timestamp in enumerate(data[0][1:]):
                if timestamp - data[0][i] < difference_threshold:  # timestamp: data[0][i + 1]
                    if not duplicates or metadata[geographic_info][1][data[0][i]] not in duplicates[-1]:
                        duplicates.append([metadata[geographic_info][1][data[0][i]]])
                    duplicates[-1].append(metadata[geographic_info][1][timestamp])
    return duplicates


def load_image(path):
    data = dict()
    for key in ['band_11', 'band_13', 'band_14', 'band_15']:
        data[key] = np.load(os.path.join(path, key + '.npy'))[..., 4]

    # Infra-Red False Color (Ash RGB)
    # Reference:
    # L. Kulik, “Satellite-based detection of contrails using deep learning,”
    #     Ph.D. dissertation, Massachusetts Institute of Technology, 2019.
    rgb_normalize = lambda data, bounds: (data - bounds[0]) / (bounds[1] - bounds[0])
    r = rgb_normalize(data['band_15'] - data['band_14'], (-4, 2))
    g = rgb_normalize(data['band_14'] - data['band_11'], (-4, 5))
    b = rgb_normalize(data['band_14'], (243, 303))

    # additional channel
    swd = (lambda data: data / np.linalg.norm(data, ord=2))(data['band_13'] - data['band_15'])  # Split Window Difference

    return np.clip(np.stack([r, g, b, swd], axis=0), 0, 1)


def data_preprocess(train_data=True):
    data_set_type = 'train' if train_data else 'validation'

    duplicates = find_duplicates()
    record_ids_to_ignore = set()
    for duplicate in duplicates:
        record_ids_to_ignore.update(duplicate[1:])

    for record_id in tqdm.tqdm(
            os.listdir(os.path.join(Path.ORI_DATA_PATH, data_set_type)),
            desc=f'Loading {data_set_type.title()} Data'
    ):
        if record_id in record_ids_to_ignore:
            continue

        image = load_image(os.path.join(Path.ORI_DATA_PATH, data_set_type, record_id))
        target = np.moveaxis(np.load(os.path.join(Path.ORI_DATA_PATH, data_set_type, record_id, 'human_pixel_masks.npy')), -1, 0)

        if not os.path.exists(os.path.join(Path.DATA_PATH, 'data', record_id)):
            os.makedirs(os.path.join(Path.DATA_PATH, 'data', record_id))
        np.save(os.path.join(Path.DATA_PATH, 'data', record_id, 'image.npy'), image)
        np.save(os.path.join(Path.DATA_PATH, 'data', record_id, 'target.npy'), target)


if __name__ == '__main__':
    data_preprocess(train_data=True)
    data_preprocess(train_data=False)
