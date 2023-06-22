import numpy as np
import torch


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
