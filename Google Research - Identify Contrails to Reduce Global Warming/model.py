import matplotlib.pyplot as plt
import os
import segmentation_models_pytorch as smp
import torch
import tqdm

from config import DEVICE, GLOBAL_DICE_COEFFICIENT_SAVING_THRESHOLD, MODELS, Path
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
            network='unet',
            network_kwargs=None,
            optimizer='adam',
            learning_rate=0.001,
            lr_patience=2
    ):
        self.name = name
        self.epoch = epoch
        self.batch_size = batch_size
        self.channel_size = 3
        self.image_size = (256, 256)
        self.input_size = (self.batch_size, self.channel_size, *self.image_size)

        self.train_dataloader = torch.utils.data.DataLoader(
            Dataset(train_data=True),
            batch_size=self.batch_size,
            shuffle=True
        )
        self.validation_dataloader = torch.utils.data.DataLoader(
            Dataset(train_data=False),
            batch_size=self.batch_size,
            shuffle=False
        )

        if network.lower() == 'unet':
            network_kwargs['in_channels'] = self.channel_size
            network_kwargs['classes'] = 1
        self.network = NETWORKS[network.lower()](**network_kwargs)
        self.network.to(DEVICE)

        self.criterion = smp.losses.DiceLoss(mode='binary')
        self.optimizer = OPTIMIZERS[optimizer.lower()](self.network.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=lr_patience)
        self.global_dice_coefficient_saving_threshold = GLOBAL_DICE_COEFFICIENT_SAVING_THRESHOLD

    def train(self, plot=True):
        train_global_dice_coefficients, validation_global_dice_coefficients = list(), list()
        for epoch in range(1, self.epoch + 1):
            # train
            self.network.train()
            torch.set_grad_enabled(True)
            pbar = tqdm.tqdm(
                enumerate(self.train_dataloader),
                total=len(self.train_dataloader),
                desc=f'Epoch-{epoch}-Train'
            )
            global_dice_coefficient = 0
            for i, (image, target) in pbar:
                pbar.set_description(f'Epoch-{epoch}-Train (Learning_Rate-{self.optimizer.param_groups[0]["lr"]})')
                image, target = image.to(DEVICE), target.to(DEVICE)
                self.optimizer.zero_grad()  # clear gradients for every batch
                prediction = self.network(image)  # feedforward
                loss = self.criterion(prediction, target)  # calculate loss
                loss.backward()  # backpropagation
                self.optimizer.step()  # update parameters
                global_dice_coefficient += 1 - loss.item()
                pbar.set_postfix(global_dice_coefficient=round(global_dice_coefficient / (i + 1), 3))
            train_global_dice_coefficients.append(global_dice_coefficient / len(self.train_dataloader))

            # validation
            self.network.eval()
            torch.set_grad_enabled(False)
            pbar = tqdm.tqdm(
                enumerate(self.validation_dataloader),
                total=len(self.validation_dataloader),
                desc=f'Epoch-{epoch}-Validation'
            )
            global_dice_coefficient = 0
            for i, (image, target) in pbar:
                image, target = image.to(DEVICE), target.to(DEVICE)
                prediction = self.network(image)
                global_dice_coefficient += 1 - self.criterion(prediction, target).item()
                pbar.set_postfix(global_dice_coefficient=round(global_dice_coefficient / (i + 1), 3))
            validation_global_dice_coefficients.append(global_dice_coefficient / len(self.validation_dataloader))
            self.scheduler.step(validation_global_dice_coefficients[-1])  # adjust learning rate

            if validation_global_dice_coefficients[-1] > self.global_dice_coefficient_saving_threshold:  # save model
                if not os.path.exists(os.path.join(Path.DATA_PATH, 'network', self.name)):
                    os.makedirs(os.path.join(Path.DATA_PATH, 'network', self.name))
                filename = f'{self.name}_global_dice_coefficient_{int(validation_global_dice_coefficients[-1] * 1e5 // 1)}.pt'
                torch.save(self.network, os.path.join(Path.DATA_PATH, 'network', self.name, filename))
                print('Model Saved')
                self.global_dice_coefficient_saving_threshold = validation_global_dice_coefficients[-1]

        if plot:
            plt.plot(train_global_dice_coefficients, label='Train')
            plt.plot(validation_global_dice_coefficients, label='Validation')
            plt.title(self.name)
            plt.legend()
            if not os.path.exists(os.path.join(Path.DATA_PATH, 'network', self.name)):
                os.makedirs(os.path.join(Path.DATA_PATH, 'network', self.name))
            plt.savefig(os.path.join(Path.DATA_PATH, 'network', self.name, f'_{self.name}.png'))


if __name__ == '__main__':
    # for model_kwargs in MODELS.values():
    #     model = Model(**model_kwargs)
    #     model.train()

    model = Model(**MODELS['unet_resnet34'])
    model.train()
