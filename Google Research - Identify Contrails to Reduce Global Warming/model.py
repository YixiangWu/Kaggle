import matplotlib.pyplot as plt
import os
import segmentation_models_pytorch as smp
import sigfig
import sklearn.model_selection
import torch
import tqdm

from config import DEVICE, MODELS, Path
from dataset import Dataset


NETWORKS = {
    'unet': smp.Unet
}

CRITERION = {
    'bce_loss': torch.nn.BCEWithLogitsLoss(),
    'dice_loss': smp.losses.DiceLoss(mode='binary')
}

OPTIMIZERS = {
    'adam': torch.optim.Adam
}

SCHEDULER = {
    'reduce_on_plateau': torch.optim.lr_scheduler.ReduceLROnPlateau
}


class Model:
    def __init__(
            self,
            name='-',
            epoch=20,
            batch_size=32,
            k_fold=5,
            voting_ensemble=True,
            network='unet',
            network_kwargs=None,
            criterion='dice_loss',
            optimizer='adam',
            learning_rate=0.001,
            scheduler='reduce_on_plateau',
            scheduler_kwargs=None,
            additional_channel=False
    ):
        self.name = name
        self.epoch = epoch
        self.batch_size = batch_size
        self.k_fold = k_fold
        self.voting_ensemble = voting_ensemble
        self.channel_size = 4 if additional_channel else 3
        self.image_size = (256, 256)
        self.input_size = (self.batch_size, self.channel_size, *self.image_size)
        self.network = network.lower()
        self.network_kwargs = network_kwargs
        if self.network == 'unet':
            self.network_kwargs['in_channels'] = self.channel_size
            self.network_kwargs['classes'] = 1
        self.criterion = criterion.lower()
        self.optimizer = optimizer.lower()
        self.learning_rate = learning_rate
        self.scheduler = scheduler.lower()
        self.scheduler_kwargs = scheduler_kwargs
        if self.scheduler == 'reduce_on_plateau' and not self.scheduler_kwargs:
            self.scheduler_kwargs = dict(patience=3)
        self.additional_channel = additional_channel
        self.dataset = Dataset(additional_channel=self.additional_channel)

        self.network_archives = list()
        self.path_to_save = os.path.join(Path.DATA_PATH, 'network', self.name)
        if not os.path.exists(self.path_to_save):
            os.makedirs(self.path_to_save)

    def __train(self, train_dataloader, validation_dataloader, fold=0):
        network = NETWORKS[self.network](**self.network_kwargs)
        criterion = CRITERION[self.criterion]
        optimizer = OPTIMIZERS[self.optimizer](network.parameters(), lr=self.learning_rate)
        scheduler = SCHEDULER[self.scheduler](optimizer, verbose=True, **self.scheduler_kwargs)

        network.to(DEVICE)
        cumulative_global_dice_coefficient = 0
        network_archive = dict([('name', self.name), ('k fold', self.k_fold), ('train losses', list()), ('validation losses', list())])
        for epoch in range(1, self.epoch + 1):
            # train
            network.train()
            torch.set_grad_enabled(True)
            pbar = tqdm.tqdm(
                enumerate(train_dataloader), total=len(train_dataloader),
                desc=f'Fold-{fold}-Epoch-{epoch}-Train' if fold else f'Epoch-{epoch}-Train'
            )
            cumulative_loss = 0
            for i, (image, target) in pbar:
                image, target = image.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()  # clear gradients for every batch
                prediction = network(image)  # feedforward
                loss = criterion(prediction, target)  # calculate loss
                loss.backward()  # backpropagation
                optimizer.step()  # update parameters
                cumulative_loss += loss.item()
                pbar.set_postfix(dict([(self.criterion, sigfig.round(cumulative_loss / (i + 1), sigfigs=3))]))
            network_archive['train losses'].append(cumulative_loss / len(train_dataloader))

            # validation
            network.eval()
            torch.set_grad_enabled(False)
            pbar = tqdm.tqdm(
                enumerate(validation_dataloader), total=len(validation_dataloader),
                desc=f'Fold-{fold}-Epoch-{epoch}-Validation' if fold else f'Epoch-{epoch}-Validation'
            )
            cumulative_loss, cumulative_global_dice_coefficient = 0, 0
            for i, (image, target) in pbar:
                image, target = image.to(DEVICE), target.to(DEVICE)
                prediction = network(image)
                cumulative_loss += criterion(prediction, target).item()
                cumulative_global_dice_coefficient += 1 - CRITERION['dice_loss'](prediction, target).item()
                pbar.set_postfix(dict([
                    (self.criterion, sigfig.round(cumulative_loss / (i + 1), sigfigs=3)),
                    ('global_dice_coefficient', sigfig.round(cumulative_global_dice_coefficient / (i + 1), sigfigs=3))
                ]))
            network_archive['validation losses'].append(cumulative_loss / len(validation_dataloader))
            scheduler.step(network_archive['validation losses'][-1])  # adjust learning rate

        network_archive['network'] = network.state_dict()
        network_archive['criterion'], network_archive['loss'] = self.criterion, network_archive['validation losses'][-1]
        network_archive['global dice coefficient'] = cumulative_global_dice_coefficient / len(validation_dataloader)
        self.network_archives.append(network_archive)

    def plot(self, fold=1):
        title = self.network_archives[fold - 1]['name'] + (f'_fold_{fold}' if self.network_archives[fold - 1]['k fold'] else '')
        plt.plot(self.network_archives[fold - 1]['train losses'], label='Loss (Train)')
        plt.plot(self.network_archives[fold - 1]['validation losses'], label='Loss (Validation)')
        plt.title(title)
        plt.legend()
        plt.savefig(os.path.join(self.path_to_save, f'_{title}.png'))
        plt.clf()

    def save_network(self, fold=1):
        filename = '{}_global_dice_coefficient_{:05d}_{}_{:05d}.pt'.format(
            self.network_archives[fold - 1]['name'] + (f'_fold_{fold}' if self.network_archives[fold - 1]['k fold'] else ''),
            int(self.network_archives[fold - 1]['global dice coefficient'] * 1e5 // 1),
            self.network_archives[fold - 1]['criterion'], int(self.network_archives[fold - 1]['loss'] * 1e5 // 1)
        )
        torch.save(self.network_archives[fold - 1]['network'], os.path.join(self.path_to_save, filename))
        print('Model Saved')

    def train(self):
        if not self.k_fold:
            train_dataset, validation_dataset = torch.utils.data.random_split(self.dataset, [0.8, 0.2])
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)
            self.__train(train_dataloader, validation_dataloader)
            self.plot()
            self.save_network()
        else:  # K-Fold Cross Validation
            k_fold = sklearn.model_selection.KFold(n_splits=self.k_fold, shuffle=True)
            for fold, (train_indices, validation_indices) in enumerate(k_fold.split(self.dataset), start=1):
                train_dataloader = torch.utils.data.DataLoader(
                    self.dataset, batch_size=self.batch_size,
                    sampler=torch.utils.data.SubsetRandomSampler(train_indices)
                )
                validation_dataloader = torch.utils.data.DataLoader(
                    self.dataset, batch_size=self.batch_size,
                    sampler=torch.utils.data.SubsetRandomSampler(validation_indices)
                )
                self.__train(train_dataloader, validation_dataloader, fold=fold)
                self.plot(fold)
                self.save_network(fold)


if __name__ == '__main__':
    # for model in MODELS['Early Exploration']:
    #     model = Model(**MODELS['Early Exploration'][model][0])
    #     model.train()

    model = Model(**MODELS['Early Exploration']['unet_resnet34'])
    model.train()
