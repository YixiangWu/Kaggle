# [Google Research - Identify Contrails to Reduce Global Warming](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming)

## Early Exploration

| # | Model |    Encoder     | Weights  | Optimizer | K Folds | Additional Channel | Global Dice Coefficient |
|:-:|:-----:|:--------------:|:--------:|:---------:|:-------:|:------------------:|:-----------------------:|
| 5 | U-Net | EfficientNetB0 | ImageNet |   Adam    |    5    |        Off         |          0.606          |
| 4 | U-Net | EfficientNetB0 | ImageNet |   Adam    |    3    |        Off         |          0.602          |
| 3 | U-Net | EfficientNetB0 | ImageNet |   Adam    |    1    |         On         |          0.600          |
| 2 | U-Net | EfficientNetB0 | ImageNet |   Adam    |    1    |        Off         |          0.593          |
| 1 | U-Net |    ResNet34    | ImageNet |   Adam    |    1    |        Off         |          0.587          |

## Classification Stage

### Backbone

| # |    Backbone    | Parent Module | Weights  | Criterion |  Loss   | Global Dice Coefficient |
|:-:|:--------------:|:-------------:|:--------:|:---------:|:-------:|:-----------------------:|
| 1 | EfficientNetB0 |     U-Net     | ImageNet | BCE Loss  | 0.00905 |          0.520          |

### Classification

| # |  Backbone  |                    Head                    |     Criterion      | Loss  | Accuracy |
|:-:|:----------:|:------------------------------------------:|:------------------:|:-----:|:--------:|
| 1 | Backbone#1 | Adaptive Pooling + One Linear Layer + ReLU | Cross Entropy Loss | 0.366 |  0.818   |
