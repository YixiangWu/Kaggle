import torch
import tqdm

from config import DEVICE, MODELS
from model import Model


def cls(model, threshold=0):
    network = model.load_networks()[-1]
    network.to(DEVICE)
    network.eval()

    recall = 0
    metrics = dict(true_positive=0, false_positive=0, true_negative=0, false_negative=0, count=0)
    dataloader = torch.utils.data.DataLoader(model.validation_dataset, sampler=torch.utils.data.SubsetRandomSampler(model.validation_indices))
    pbar = tqdm.tqdm(dataloader, total=len(dataloader), desc='threshold ' + str(threshold))
    for i, (image, target) in enumerate(pbar):
        output = network(model.resize[0](image.to(DEVICE)) if model.resize else image.to(DEVICE)).cpu().detach()
        if output[0][0] + threshold > output[0][1]:  # predict [1, 0] (positive)
            metric_to_increment = 'true_positive' if target[0][0] == 1 else 'false_positive'
        else:  # predict [0, 1] (negative)
            metric_to_increment = 'true_negative' if target[0][1] == 1 else 'false_negative'
        metrics[metric_to_increment] += 1
        recall = metrics['true_positive'] / (metrics['true_positive'] + metrics['false_negative']) if \
            (metrics['true_positive'] + metrics['false_negative']) != 0 else 0
        pbar.set_postfix(
            accuracy=(metrics['true_positive'] + metrics['true_negative']) / (i + 1), recall=recall,
            true_positive=metrics['true_positive'], false_positive=metrics['false_positive'],
            true_negative=metrics['true_negative'], false_negative=metrics['false_negative']
        )
    return recall


def find_cls_threshold(target_recall, threshold_list=None):
    if not threshold_list:
        threshold_list = [-0.25, 0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2]
    threshold_list.sort()
    threshold, recall = 0, 0
    while recall < target_recall:
        threshold = threshold_list.pop(0) if threshold_list else threshold * 2
        recall = cls(model, threshold=threshold)
    return threshold


if __name__ == '__main__':
    model = Model(**MODELS['Classification']['classification_backbone_ensemble'])
    print(find_cls_threshold(0.95))
