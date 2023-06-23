import matplotlib.pyplot as plt
import numpy as np
import os
import torch

from config import Path


class Dataset(torch.utils.data.Dataset):
    def __init__(self, additional_channel=False):
        self.path = os.path.join(Path.DATA_PATH, 'data')
        self.record_ids = os.listdir(self.path)
        self.additional_channel = additional_channel

    def __len__(self):
        return len(self.record_ids)

    def __getitem__(self, idx):
        image = torch.from_numpy(np.load(os.path.join(self.path, self.record_ids[idx], 'image.npy')))
        target = torch.from_numpy(np.load(os.path.join(self.path, self.record_ids[idx], 'target.npy')))
        return image.float() if self.additional_channel else image[:3].float(), target.float()


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
