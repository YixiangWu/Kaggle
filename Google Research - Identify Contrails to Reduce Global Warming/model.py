import matplotlib.pyplot as plt
import os
import segmentation_models_pytorch as smp
import sklearn.model_selection
import torch
import tqdm

from config import DEVICE, MODELS, Path
from dataset import Dataset


NETWORKS = {
    'unet': smp.Unet
}

OPTIMIZERS = {
    'adam': torch.optim.Adam
}


class Model:
    def __init__(
            self,
            name='-',
            epoch=20,
            batch_size=32,
            k_fold=5,
            network='unet',
            network_kwargs=None,
            optimizer='adam',
            learning_rate=0.001,
            scheduler_kwargs=None,
            additional_channel=False
    ):
        self.name = name
        self.epoch = epoch
        self.batch_size = batch_size
        self.k_fold = k_fold
        self.channel_size = 4 if additional_channel else 3
        self.image_size = (256, 256)
        self.input_size = (self.batch_size, self.channel_size, *self.image_size)
        self.network = network
        self.network_kwargs = network_kwargs
        if self.network.lower() == 'unet':
            self.network_kwargs['in_channels'] = self.channel_size
            self.network_kwargs['classes'] = 1
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.scheduler_kwargs = scheduler_kwargs
        self.additional_channel = additional_channel

        self.train_dataloader = None
        self.validation_dataloader = None

        self.criterion = smp.losses.DiceLoss(mode='binary')

    def __train(self, fold=0):
        network = NETWORKS[self.network.lower()](**self.network_kwargs)
        optimizer = OPTIMIZERS[self.optimizer.lower()](network.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', verbose=True, **self.scheduler_kwargs
        )

        network.to(DEVICE)
        train_global_dice_coefficients, validation_global_dice_coefficients = list(), list()
        for epoch in range(1, self.epoch + 1):
            # train
            network.train()
            torch.set_grad_enabled(True)
            pbar = tqdm.tqdm(
                enumerate(self.train_dataloader),
                total=len(self.train_dataloader),
                desc=f'Fold-{fold}-Epoch-{epoch}-Train' if fold else f'Epoch-{epoch}-Train'
            )
            global_dice_coefficient = 0
            for i, (image, target) in pbar:
                image, target = image.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()  # clear gradients for every batch
                prediction = network(image)  # feedforward
                loss = self.criterion(prediction, target)  # calculate loss
                loss.backward()  # backpropagation
                optimizer.step()  # update parameters
                global_dice_coefficient += 1 - loss.item()
                pbar.set_postfix(global_dice_coefficient=round(global_dice_coefficient / (i + 1), 3))
            train_global_dice_coefficients.append(global_dice_coefficient / len(self.train_dataloader))

            # validation
            network.eval()
            torch.set_grad_enabled(False)
            pbar = tqdm.tqdm(
                enumerate(self.validation_dataloader),
                total=len(self.validation_dataloader),
                desc=f'Fold-{fold}-Epoch-{epoch}-Validation' if fold else f'Epoch-{epoch}-Validation'
            )
            global_dice_coefficient = 0
            for i, (image, target) in pbar:
                image, target = image.to(DEVICE), target.to(DEVICE)
                prediction = network(image)
                global_dice_coefficient += 1 - self.criterion(prediction, target).item()
                pbar.set_postfix(global_dice_coefficient=round(global_dice_coefficient / (i + 1), 3))
            validation_global_dice_coefficients.append(global_dice_coefficient / len(self.validation_dataloader))
            scheduler.step(1 - validation_global_dice_coefficients[-1])  # adjust learning rate

        info = f'{self.name}_fold_{fold}' if fold else self.name
        if not os.path.exists(os.path.join(Path.DATA_PATH, 'network', self.name)):
            os.makedirs(os.path.join(Path.DATA_PATH, 'network', self.name))

        plt.plot(train_global_dice_coefficients, label='Train')
        plt.plot(validation_global_dice_coefficients, label='Validation')
        plt.title(info)
        plt.legend()
        plt.savefig(os.path.join(Path.DATA_PATH, 'network', self.name, f'_{info}.png'))
        plt.clf()

        filename = f'{info}_global_dice_coefficient_{int(validation_global_dice_coefficients[-1] * 1e5 // 1)}.pt'
        torch.save(network, os.path.join(Path.DATA_PATH, 'network', self.name, filename))
        print('Model Saved')

    def train(self):
        if not self.k_fold:
            self.train_dataloader = torch.utils.data.DataLoader(
                Dataset(train_data=True, additional_channel=self.additional_channel),
                batch_size=self.batch_size, shuffle=True
            )
            self.validation_dataloader = torch.utils.data.DataLoader(
                Dataset(train_data=False, additional_channel=self.additional_channel),
                batch_size=self.batch_size, shuffle=False
            )
            self.__train()
        else:  # K-Fold Cross Validation
            dataset = torch.utils.data.ConcatDataset([
                Dataset(train_data=True, additional_channel=self.additional_channel),
                Dataset(train_data=False, additional_channel=self.additional_channel)
            ])
            k_fold = sklearn.model_selection.KFold(n_splits=self.k_fold, shuffle=True)

            for fold, (train_indices, validation_indices) in enumerate(k_fold.split(dataset), start=1):
                self.train_dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=self.batch_size,
                    sampler=torch.utils.data.SubsetRandomSampler(train_indices)
                )
                self.validation_dataloader = torch.utils.data.DataLoader(
                    dataset, batch_size=self.batch_size,
                    sampler=torch.utils.data.SubsetRandomSampler(validation_indices)
                )
                self.__train(fold=fold)


if __name__ == '__main__':
    # for model in MODELS:
    #     model = Model(**MODELS[model][0])
    #     model.train()

    model = Model(**MODELS['unet_resnet34'][0])
    model.train()
