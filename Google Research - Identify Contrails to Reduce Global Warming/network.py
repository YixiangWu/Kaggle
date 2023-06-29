import segmentation_models_pytorch as smp
import torch


class Classification(torch.nn.Module):
    def __init__(self, feature_extractor, head):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.head = head

    def forward(self, input):
        feature_volume = self.feature_extractor(input)[-1]
        return self.head(feature_volume)


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
