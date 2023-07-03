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
        feature_volumes = torch.tensor([]).to(
            torch.device('cuda') if torch.cuda.is_available() else
            (torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'))
        )
        for i in range(self.num_extractors):
            feature_volume = self.__dict__['_modules']['feature_extractor_' + str(i)](input)[-1]
            feature_volumes = torch.cat((feature_volumes, feature_volume), dim=1) if feature_volumes.shape[0] != 0 else feature_volume
        return self.head(feature_volumes)


class ClassificationHead(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(channels, 2, bias=True),
            torch.nn.ReLU()
        )

    def forward(self, feature_volume):
        return self.head(feature_volume)


NETWORKS = {
    'cls': Classification,
    'cls_head': ClassificationHead,
    'unet': smp.Unet,
    'unet++': smp.UnetPlusPlus
}
