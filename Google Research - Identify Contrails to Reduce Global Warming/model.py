import matplotlib.pyplot as plt
import numpy as np
import os
import segmentation_models_pytorch as smp
import sigfig
import sklearn.model_selection
import torch
import tqdm

from config import DEVICE, MODELS, NUM_WORKERS, Path
from dataset import Dataset
from network import NETWORKS
from utilities import load_networks


CRITERION = {
    'bce_loss': torch.nn.BCEWithLogitsLoss(),
    'cross_entropy_loss': torch.nn.CrossEntropyLoss(),
    'dice_loss': smp.losses.DiceLoss(mode='binary')
}

OPTIMIZERS = {
    'adam': torch.optim.Adam
}

SCHEDULER = {
    'exponential': torch.optim.lr_scheduler.ExponentialLR,
    'multi_step': torch.optim.lr_scheduler.MultiStepLR,
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
            train_test_ratio=0.8,
            additional_channel=False,
            filter=False,
            backbone=None,
            metric='global_dice_coefficient'
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
        self.network_kwargs = self.__update_network_kwargs(self.network, network_kwargs, additional_channel)
        self.criterion = criterion.lower()
        self.optimizer = optimizer.lower()
        self.learning_rate = learning_rate
        self.scheduler = scheduler.lower()
        self.scheduler_kwargs = scheduler_kwargs
        if not self.scheduler_kwargs:
            if self.scheduler == 'exponential':
                self.scheduler_kwargs = dict(gamma=0.95)
            elif self.scheduler == 'multi_step':
                self.scheduler_kwargs = dict(milestones=list(range(5, self.epoch + 1, 5)))
            elif self.scheduler == 'reduce_on_plateau':
                self.scheduler_kwargs = dict(patience=3)
        self.train_test_ratio = train_test_ratio
        self.additional_channel = additional_channel
        self.filter = filter

        self.backbone = True if backbone else False
        if self.backbone:
            # load feature extractor
            feature_extractor_info = MODELS[backbone['feature_extractor']['category']][backbone['feature_extractor']['name']]
            self.feature_extractor_network_kwargs = self.__update_network_kwargs(
                feature_extractor_info['network'].lower(),
                feature_extractor_info['network_kwargs'],
                feature_extractor_info['additional_channel']
            )
            self.feature_extractor = load_networks(
                feature_extractor_info['name'], feature_extractor_info['k_fold'],
                feature_extractor_info['network'].lower(), self.feature_extractor_network_kwargs
            )[0].__dict__['_modules'][backbone['feature_extractor']['module']]
            self.feature_extractor.to(DEVICE)
            self.feature_extractor.eval()

            # initialize head
            self.head_network = backbone['head']['network'].lower()
            self.head_network_kwargs = backbone['head']['network_kwargs']
            if self.head_network == 'cls_head':
                if not self.head_network_kwargs:
                    self.head_network_kwargs = dict()

                sample = torch.zeros(self.input_size).to(DEVICE)
                self.head_network_kwargs['channels'] = self.feature_extractor(sample)[-1].shape[1]

            self.network_kwargs['feature_extractor'] = self.feature_extractor
            self.network_kwargs['head'] = NETWORKS[self.head_network](**self.head_network_kwargs)

        self.dataset = Dataset(additional_channel=self.additional_channel, variant='classes' if backbone else None)
        self.dataloader_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=True if DEVICE == torch.device('cuda') else False)
        if self.filter:
            indices_to_keep = list()
            for i, (_, target) in enumerate(self.dataset):
                if len(torch.unique(target)) == 2:
                    indices_to_keep.append(i)
            self.dataset = torch.utils.data.Subset(self.dataset, indices_to_keep)

        self.metric = metric
        self.network_archives = list()
        self.path_to_save = os.path.join(Path.DATA_PATH, 'network', self.name)
        if not os.path.exists(self.path_to_save):
            os.makedirs(self.path_to_save)

    @staticmethod
    def __update_network_kwargs(network, network_kwargs, additional_channel):
        if not network_kwargs:
            network_kwargs = dict()

        if network == 'unet':
            network_kwargs['in_channels'] = 4 if additional_channel else 3
            network_kwargs['classes'] = 1
        return network_kwargs

    def __train(self, train_dataloader, validation_dataloader, fold=0):
        network = NETWORKS[self.network](**self.network_kwargs) if not self.backbone else \
            NETWORKS[self.head_network](**self.head_network_kwargs)
        criterion = CRITERION[self.criterion]
        optimizer = OPTIMIZERS[self.optimizer](network.parameters(), lr=self.learning_rate)
        scheduler = SCHEDULER[self.scheduler](optimizer, verbose=True, **self.scheduler_kwargs)

        network.to(DEVICE)
        metric = 0
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
                network_input = image if not self.backbone else self.feature_extractor(image)[-1]
                prediction = network(network_input)  # feedforward
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
            cumulative_loss, metric = 0, 0
            for i, (image, target) in pbar:
                image, target = image.to(DEVICE), target.to(DEVICE)
                network_input = image if not self.backbone else self.feature_extractor(image)[-1]
                prediction = network(network_input)
                cumulative_loss += criterion(prediction, target).item()
                if self.metric == 'global_dice_coefficient':
                    metric = (metric * i + 1 - CRITERION['dice_loss'](prediction, target).item()) / (i + 1)
                elif self.metric == 'accuracy':
                    correct_classes = np.count_nonzero(np.array([
                        1 if (prediction[i][0] > prediction[i][1] and target[i][0] == 1) or \
                             (prediction[i][0] < prediction[i][1] and target[i][1] == 1) else 0 \
                        for i in range(len(image))
                    ]))
                    metric = (metric * i * self.batch_size + correct_classes) / ((i + 1) * self.batch_size)
                pbar.set_postfix(dict([(self.criterion, sigfig.round(cumulative_loss / (i + 1), sigfigs=3)), (self.metric, metric)]))
            network_archive['validation losses'].append(cumulative_loss / len(validation_dataloader))
            scheduler.step(network_archive['validation losses'][-1]) if self.scheduler == 'reduce_on_plateau' else scheduler.step()  # adjust learning rate

        if self.backbone:
            self.network_kwargs['feature_extractor'] = self.feature_extractor
            self.network_kwargs['head'] = network
            network = NETWORKS[self.network](**self.network_kwargs)
        network_archive['network'] = network.state_dict()
        network_archive['criterion'], network_archive['loss'] = self.criterion, network_archive['validation losses'][-1]
        network_archive['metric name'], network_archive['metric'] = self.metric, metric
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
        filename = '{}_{}_{:05d}_{}_{:05d}.pt'.format(
            self.network_archives[fold - 1]['name'] + (f'_fold_{fold}' if self.network_archives[fold - 1]['k fold'] else ''),
            self.network_archives[fold - 1]['metric name'], int(self.network_archives[fold - 1]['metric'] * 1e5 // 1),
            self.network_archives[fold - 1]['criterion'], int(self.network_archives[fold - 1]['loss'] * 1e5 // 1)
        )
        torch.save(self.network_archives[fold - 1]['network'], os.path.join(self.path_to_save, filename))
        print('Model Saved')

    def train(self):
        if not self.k_fold:
            train_dataset, validation_dataset = torch.utils.data.random_split(self.dataset, [self.train_test_ratio, 1 - self.train_test_ratio])
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, **self.dataloader_kwargs)
            validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False, **self.dataloader_kwargs)
            self.__train(train_dataloader, validation_dataloader)
            self.plot()
            self.save_network()
        else:  # K-Fold Cross Validation
            k_fold = sklearn.model_selection.KFold(n_splits=self.k_fold, shuffle=True)
            for fold, (train_indices, validation_indices) in enumerate(k_fold.split(self.dataset), start=1):
                train_dataloader = torch.utils.data.DataLoader(
                    self.dataset, batch_size=self.batch_size,
                    sampler=torch.utils.data.SubsetRandomSampler(train_indices), **self.dataloader_kwargs
                )
                validation_dataloader = torch.utils.data.DataLoader(
                    self.dataset, batch_size=self.batch_size,
                    sampler=torch.utils.data.SubsetRandomSampler(validation_indices), **self.dataloader_kwargs
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
