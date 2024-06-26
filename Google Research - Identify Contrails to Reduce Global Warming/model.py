import matplotlib.pyplot as plt
import numpy as np
import os
import segmentation_models_pytorch as smp
import sigfig
import sklearn.model_selection
import torch
import torchvision
import tqdm

from config import DEVICE, MODELS, NUM_WORKERS, SEED, Path
from dataset import Dataset
from network import NETWORKS


CRITERION = {
    'bce_loss': torch.nn.BCEWithLogitsLoss(),
    'cross_entropy_loss': torch.nn.CrossEntropyLoss(),
    'dice_loss': smp.losses.DiceLoss(mode='binary')
}

OPTIMIZERS = {
    'adam': torch.optim.Adam
}

SCHEDULER = {
    'cosine_annealing': torch.optim.lr_scheduler.CosineAnnealingLR,
    'cosine_annealing_warm_restarts': torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    'exponential': torch.optim.lr_scheduler.ExponentialLR,
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
            augmentations=False,
            resize_factor=1,
            filter=False,
            backbone=None,
            checkpoint=None,
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
        self.network_kwargs = network_kwargs if network_kwargs else dict()
        if self.network == 'unet' or self.network == 'unet++':
            self.network_kwargs['in_channels'] = self.channel_size
            self.network_kwargs['classes'] = 1
        self.criterion = criterion.lower()
        self.optimizer = optimizer.lower()
        self.learning_rate = learning_rate
        self.scheduler = scheduler.lower()
        self.scheduler_kwargs = scheduler_kwargs
        if not self.scheduler_kwargs:
            if self.scheduler == 'cosine_annealing':
                self.scheduler_kwargs = dict(T_max=20, eta_min=1e-6)
            elif self.scheduler == 'cosine_annealing_warm_restarts':
                self.scheduler_kwargs = dict(T_0=6, T_mult=2, eta_min=1e-6)
            elif self.scheduler == 'exponential':
                self.scheduler_kwargs = dict(gamma=0.95)
            elif self.scheduler == 'reduce_on_plateau':
                self.scheduler_kwargs = dict(patience=3)
        self.train_test_ratio = train_test_ratio
        self.additional_channel = additional_channel
        self.augmentations = augmentations
        self.dataset_variant = 'classes' if self.network == 'cls' else None
        self.resize_factor = resize_factor
        self.filter = None if not filter else lambda dataset, indices: [index for index in indices if len(torch.unique(dataset[index][1])) == 2]

        self.resize = None
        if self.resize_factor != 1:
            self.resize = list()
            self.resize.append(torchvision.transforms.Resize(
                tuple(int(size * self.resize_factor) for size in self.image_size),
                interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR, antialias=True
            ))
            self.resize.append(torchvision.transforms.Resize(
                self.image_size, interpolation=torchvision.transforms.functional.InterpolationMode.BILINEAR, antialias=True
            ))

        self.backbone = True if backbone else False
        if self.backbone:
            self.feature_extractors = list()
            # load feature extractor
            for feature_extractor in backbone['feature_extractors']:
                feature_extractor_info = MODELS[feature_extractor['category']][feature_extractor['name']]
                assert feature_extractor_info['resize_factor'] == self.resize_factor
                for network in Model(**feature_extractor_info).load_networks():
                    self.feature_extractors.append(network.__dict__['_modules'][feature_extractor['module']])
                    self.feature_extractors[-1].to(DEVICE)
                    self.feature_extractors[-1].eval()

            # initialize head
            self.head_network = backbone['head']['network'].lower()
            self.head_network_kwargs = backbone['head']['network_kwargs']
            if self.head_network == 'cls_head':
                if not self.head_network_kwargs:
                    self.head_network_kwargs = dict()

                sample = torch.zeros((*self.input_size[:2], *(int(size * self.resize_factor) for size in self.image_size)), device=DEVICE)
                self.head_network_kwargs['channels'] = 0
                for feature_extractor in self.feature_extractors:
                    self.head_network_kwargs['channels'] += feature_extractor(sample)[-1].shape[1]

            self.network_kwargs['feature_extractors'] = self.feature_extractors
            self.network_kwargs['head'] = NETWORKS[self.head_network](**self.head_network_kwargs)

        self.checkpoint_paths = None
        if checkpoint:
            checkpoint_model = Model(**MODELS[checkpoint['category']][checkpoint['name']])
            assert checkpoint_model.k_fold == self.k_fold
            self.checkpoint_paths = checkpoint_model.__load_network_paths()

        self.train_dataset = Dataset(additional_channel=self.additional_channel, augmentations=self.augmentations, variant=self.dataset_variant)
        self.validation_dataset = Dataset(additional_channel=self.additional_channel, augmentations=False, variant=self.dataset_variant)
        self.dataloader_kwargs = dict(num_workers=NUM_WORKERS, pin_memory=True if DEVICE == torch.device('cuda') else False)
        self.train_indices, self.validation_indices = sklearn.model_selection.train_test_split(
            range(len(self.train_dataset)), test_size=1 - self.train_test_ratio,
            train_size=self.train_test_ratio, random_state=SEED
        )

        self.metric = metric
        self.network_archives = list()
        self.path_to_save = os.path.join(Path.DATA_PATH, 'network', self.name)
        if not os.path.exists(self.path_to_save):
            os.makedirs(self.path_to_save)

        torch.backends.cudnn.benchmark = True if DEVICE == torch.device('cuda') else False

    @staticmethod
    def __cuda_automatic_mixed_precision_autocast(func):
        if DEVICE != torch.device('cuda'):
            return func()
        else:  # if CUDA, use automatic mixed precision
            with torch.cuda.amp.autocast():
                return func()

    def __train(self, train_indices, validation_indices, fold=0):
        if self.filter:
            train_indices = self.filter(self.train_dataset, train_indices)
            validation_indices = self.filter(self.validation_dataset, validation_indices)

        train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=self.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(train_indices),
            **self.dataloader_kwargs
        )
        validation_dataloader = torch.utils.data.DataLoader(
            self.validation_dataset, batch_size=self.batch_size,
            sampler=torch.utils.data.SubsetRandomSampler(validation_indices),
            **self.dataloader_kwargs
        )

        network = NETWORKS[self.network](**self.network_kwargs) if not self.backbone else \
            NETWORKS[self.head_network](**self.head_network_kwargs)
        criterion = CRITERION[self.criterion]
        optimizer = OPTIMIZERS[self.optimizer](network.parameters(), lr=self.learning_rate)
        scheduler = SCHEDULER[self.scheduler](optimizer, verbose=True, **self.scheduler_kwargs)
        scaler = torch.cuda.amp.GradScaler() if DEVICE == torch.device('cuda') else None

        if self.checkpoint_paths:  # load checkpoint
            network.load_state_dict(torch.load(sorted(self.checkpoint_paths)[fold], map_location=DEVICE))

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
                if self.resize:
                    image = self.resize[0](image[:])
                network_input = image
                if self.backbone:
                    network_input = torch.tensor([], device=DEVICE)
                    for feature_extractor in self.feature_extractors:
                        feature_volume = self.__cuda_automatic_mixed_precision_autocast(lambda: feature_extractor(image)[-1])
                        network_input = torch.cat((network_input, feature_volume), dim=1) if network_input.shape[0] != 0 else feature_volume
                optimizer.zero_grad()  # clear gradients for every batch
                if DEVICE != torch.device('cuda'):
                    prediction = network(network_input)  # feedforward
                    if self.resize and self.dataset_variant != 'classes':
                        prediction = self.resize[1](prediction[:])
                    loss = criterion(prediction, target)  # calculate loss
                    loss.backward()  # backpropagation
                    optimizer.step()  # update parameters
                else:  # if CUDA, use automatic mixed precision
                    with torch.cuda.amp.autocast():
                        prediction = network(network_input)
                        if self.resize and self.dataset_variant != 'classes':
                            prediction = self.resize[1](prediction[:])
                        loss = criterion(prediction, target)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
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
                if self.resize:
                    image = self.resize[0](image[:])
                network_input = image
                if self.backbone:
                    network_input = torch.tensor([], device=DEVICE)
                    for feature_extractor in self.feature_extractors:
                        feature_volume = self.__cuda_automatic_mixed_precision_autocast(lambda: feature_extractor(image)[-1])
                        network_input = torch.cat((network_input, feature_volume), dim=1) if network_input.shape[0] != 0 else feature_volume
                prediction = self.__cuda_automatic_mixed_precision_autocast(
                    lambda: network(network_input) if not self.resize or self.dataset_variant == 'classes' else self.resize[1](network(network_input)[:])
                )
                cumulative_loss += criterion(prediction, target).item()
                if self.metric == 'global_dice_coefficient':
                    metric = (metric * i + 1 - CRITERION['dice_loss'](prediction, target).item()) / (i + 1)
                elif self.metric == 'accuracy':
                    correct_classes = np.count_nonzero(np.array([
                        1 if (prediction[i][0] > prediction[i][1] and target[i][0] == 1) or \
                             (prediction[i][0] < prediction[i][1] and target[i][1] == 1) else 0 \
                        for i in range(len(image))
                    ]))
                    metric = (metric * i * self.batch_size + correct_classes) / (i * self.batch_size + len(image))
                pbar.set_postfix(dict([(self.criterion, sigfig.round(cumulative_loss / (i + 1), sigfigs=3)), (self.metric, metric)]))
            network_archive['validation losses'].append(cumulative_loss / len(validation_dataloader))
            scheduler.step(network_archive['validation losses'][-1]) if self.scheduler == 'reduce_on_plateau' else scheduler.step()  # adjust learning rate

        if self.backbone:
            self.network_kwargs['feature_extractors'] = self.feature_extractors
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

    def __load_network_paths(self):
        network_filenames = os.listdir(self.path_to_save)
        filenames_to_load = [filename for filename in network_filenames if filename[:len(self.name) + 6] == f'{self.name}_fold_'] if \
            self.k_fold else [sorted(network_filenames)[-1]]  # [sorted(network_filenames)[-1]] <- best among all in the directory
        return [os.path.join(self.path_to_save, filename) for filename in filenames_to_load]

    def load_networks(self):
        networks = list()
        network = NETWORKS[self.network](**self.network_kwargs)
        for path in self.__load_network_paths():
            network.load_state_dict(torch.load(path, map_location=DEVICE))
            networks.append(network)
        return networks

    def train(self):
        if not self.k_fold:
            self.__train(self.train_indices, self.validation_indices)
            self.plot()
            self.save_network()
        else:  # K-Fold Cross Validation
            k_fold = sklearn.model_selection.KFold(n_splits=self.k_fold, shuffle=True, random_state=SEED)
            k_folds = k_fold.split(self.train_indices)
            for fold, (train_indices_indices, validation_indices_indices) in enumerate(k_folds, start=1):
                train_indices = [self.train_indices[train_indices_index] for train_indices_index in train_indices_indices]
                validation_indices = [self.train_indices[validation_indices_index] for validation_indices_index in validation_indices_indices]
                self.__train(train_indices, validation_indices, fold=fold)
                self.plot(fold)
                self.save_network(fold)


if __name__ == '__main__':
    # for model in MODELS['Early Exploration']:
    #     model = Model(**MODELS['Early Exploration'][model][0])
    #     model.train()

    model = Model(**MODELS['Early Exploration']['unet_resnet34'])
    model.train()
