# [Google Research - Identify Contrails to Reduce Global Warming](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming)

## Classification Stage

### Backbone

| # |    Backbone     | Parent Module | Weights  | Criterion |  Loss   | Global Dice Coefficient |
|:-:|:---------------:|:-------------:|:--------:|:---------:|:-------:|:-----------------------:|
| 2 | EfficientNetB0  |     U-Net     | ImageNet | BCE Loss  | 0.00855 |          0.517          |
| 1 | ResNext50 32x4D |     U-Net     | ImageNet | BCE Loss  | 0.00871 |          0.533          |

### Classification

| # |        Backbone         |                    Head                    |     Criterion      | Loss  | Accuracy |
|:-:|:-----------------------:|:------------------------------------------:|:------------------:|:-----:|:--------:|
| 3 | Backbone#1 + Backbone#2 | Adaptive Pooling + One Linear Layer + ReLU | Cross Entropy Loss | 0.182 |  0.927   |
| 2 |       Backbone#2        | Adaptive Pooling + One Linear Layer + ReLU | Cross Entropy Loss | 0.207 |  0.910   |
| 1 |       Backbone#1        | Adaptive Pooling + One Linear Layer + ReLU | Cross Entropy Loss | 0.247 |  0.899   |

## Segmentation Stage

### Segmentation

| # |  Model  |    Encoder     | Weights  | Filter | Criterion | Loss  | Global Dice Coefficient |
|:-:|:-------:|:--------------:|:--------:|:------:|:---------:|:-----:|:-----------------------:|
| 6 | U-Net++ | EfficientNetB4 | ImageNet |   On   | Dice Loss | 0.327 |          0.673          |
| 5 | U-Net++ | EfficientNetB3 | ImageNet |   On   | Dice Loss | 0.333 |          0.667          |
| 4 | U-Net++ | EfficientNetB2 | ImageNet |   On   | Dice Loss | 0.333 |          0.667          |
| 3 | U-Net++ | EfficientNetB1 | ImageNet |   On   | Dice Loss | 0.329 |          0.671          |
| 2 | U-Net++ | EfficientNetB0 | ImageNet |   On   | Dice Loss | 0.338 |          0.662          |
| 1 |  U-Net  | EfficientNetB0 | ImageNet |   On   | Dice Loss | 0.347 |          0.653          |

## Classification Stage + Segmentation Stage

| # |      Classification      |      Segmentation      | Global Dice Coefficient (Kaggle Public Score) |
|:-:|:------------------------:|:----------------------:|:---------------------------------------------:|
| * | Classification#3 (0.927) | Segmentation#3 (0.671) |                     0.631                     |
| 2 | Classification#2 (0.910) | Segmentation#3 (0.671) |                     0.633                     |
| 1 | Classification#1 (0.899) | Segmentation#2 (0.662) |                     0.616                     |
