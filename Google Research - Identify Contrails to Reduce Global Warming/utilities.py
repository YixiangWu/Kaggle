import numpy as np
import os
import torch

from config import DEVICE, Path
from network import NETWORKS


def ensemble(networks, image, voting_ensemble=True):
    predictions = list()
    for network in networks:
        network.eval()
        torch.set_grad_enabled(False)
        prediction = network.predict(image.unsqueeze(0))[0][0]
        prediction = prediction.cpu().numpy()
        predictions.append(prediction)

    if voting_ensemble:  # True: Voting Ensemble; False: Averaging Ensemble
        for i in range(len(predictions)):
            predictions[i][predictions[i] > 0] = 1
            predictions[i][predictions[i] <= 0] = -1
    prediction = np.average(predictions, axis=0)
    prediction[prediction > 0] = 1
    prediction[prediction <= 0] = 0
    return prediction


def load_networks(name, k_fold, network, network_kwargs):
    networks = list()
    network_filenames = os.listdir(os.path.join(Path.DATA_PATH, 'network', name))
    filenames_to_load = [filename for filename in network_filenames if filename[:len(name) + 6] == f'{name}_fold_'] if \
        k_fold else [sorted(network_filenames)[-1]]  # [sorted(network_filenames)[-1]] <- best among all in the directory
    for filename in filenames_to_load:
        network = NETWORKS[network](**network_kwargs)
        network.load_state_dict(torch.load(os.path.join(Path.DATA_PATH, 'network', name, filename), map_location=DEVICE))
        networks.append(network.to(DEVICE))
    return networks
