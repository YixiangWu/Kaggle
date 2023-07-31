import math
import segmentation_models_pytorch as smp
import torch


class Classification(torch.nn.Module):
    def __init__(self, feature_extractors, head):
        super().__init__()
        self.num_extractors = len(feature_extractors)
        for i, feature_extractor in enumerate(feature_extractors):
            setattr(self, 'feature_extractor_' + str(i), feature_extractor)
        self.head = head

    def forward(self, input):
        feature_volumes = torch.tensor([], device=torch.device('cuda') if torch.cuda.is_available() else \
            (torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')))
        for i in range(self.num_extractors):
            feature_volume = self.__dict__['_modules']['feature_extractor_' + str(i)](input)[-1]
            feature_volumes = torch.cat((feature_volumes, feature_volume), dim=1) if feature_volumes.shape[0] != 0 else feature_volume
        return self.head(feature_volumes)


class ClassificationHead(torch.nn.Module):
    def __init__(self, channels, factor=4, linear_layers=0):
        super().__init__()
        if not linear_layers:
            linear_layers = int(math.log(channels // (256 // factor // factor), factor))

        layers = list()
        in_channels = channels
        for i in range(linear_layers):
            out_channels = in_channels // factor if i < linear_layers - 1 else 2
            layers.append(torch.nn.Linear(in_channels, out_channels, bias=True))
            if i != linear_layers - 1:
                torch.nn.init.constant_(layers[-1].bias, 0)
                torch.nn.init.kaiming_normal_(layers[-1].weight, nonlinearity='relu')
                layers.append(torch.nn.BatchNorm1d(out_channels))
                layers.append(torch.nn.ReLU(inplace=True))
                in_channels = out_channels
            else:
                torch.nn.init.constant_(layers[-1].bias, 0)
                torch.nn.init.xavier_normal_(layers[-1].weight, gain=torch.nn.init.calculate_gain('tanh'))
                layers.append(torch.nn.Tanh())

        self.head = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), *layers, torch.nn.Tanh())

    def forward(self, feature_volume):
        return self.head(feature_volume)


NETWORKS = {
    'cls': Classification,
    'cls_head': ClassificationHead,
    'unet': smp.Unet,
    'unet++': smp.UnetPlusPlus
}
