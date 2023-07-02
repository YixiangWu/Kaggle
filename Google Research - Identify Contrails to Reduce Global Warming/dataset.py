import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torchvision

from config import Path


class Dataset(torch.utils.data.Dataset):
    def __init__(self, additional_channel=False, augmentations=None, variant=None):
        self.path = os.path.join(Path.DATA_PATH, 'data')
        self.record_ids = os.listdir(self.path)
        self.additional_channel = additional_channel
        self.augmentations = augmentations
        self.variant = variant

    def __len__(self):
        return len(self.record_ids)

    def __getitem__(self, idx):
        image = np.load(os.path.join(self.path, self.record_ids[idx], 'image.npy'))
        target = np.load(os.path.join(self.path, self.record_ids[idx], 'target.npy'))

        if not self.additional_channel:
            image = image[:3]

        image, target = torch.from_numpy(image).float(), torch.from_numpy(target).float()

        if self.augmentations:
            if 'horizontal_flip' in self.augmentations and random.random() < self.augmentations['horizontal_flip']:
                image = torchvision.transforms.functional.hflip(image)
                target = torchvision.transforms.functional.hflip(target)
            if 'vertical_flip' in self.augmentations and random.random() < self.augmentations['vertical_flip']:
                image = torchvision.transforms.functional.vflip(image)
                target = torchvision.transforms.functional.vflip(target)
            if 'rotate' in self.augmentations and random.random() < self.augmentations['rotate']:
                angle = random.choice([-90, 90, 180])
                image = torchvision.transforms.functional.rotate(image, angle)
                target = torchvision.transforms.functional.rotate(target, angle)

        if self.variant == 'classes':
            target = torch.tensor([0, 1]).float() if len(torch.unique(target)) == 1 else torch.tensor([1, 0]).float()

        return image, target


def display_image(image, mask):
    plt.figure(figsize=(18, 6))

    ax = plt.subplot(1, 3, 1)
    ax.imshow(image)
    ax.set_title('Image')

    ax = plt.subplot(1, 3, 2)
    ax.imshow(mask)
    ax.set_title('Mask')

    ax = plt.subplot(1, 3, 3)
    ax.imshow(image)
    ax.imshow(mask, cmap='Reds', alpha=.4)
    ax.set_title('Image + Contrail Mask')

    plt.show()


if __name__ == '__main__':
    # show random image from the training data set
    train_dataloader = torch.utils.data.DataLoader(Dataset(), shuffle=True)
    image, target = next(iter(train_dataloader))
    display_image(torch.moveaxis(image[0], 0, -1), target[0][0])
