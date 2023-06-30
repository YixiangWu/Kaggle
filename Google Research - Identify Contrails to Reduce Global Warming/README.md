# [Google Research - Identify Contrails to Reduce Global Warming](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming)

## Classification Stage

### Backbone

| # |    Backbone    | Parent Module | Weights  | Criterion |  Loss   | Global Dice Coefficient |
|:-:|:--------------:|:-------------:|:--------:|:---------:|:-------:|:-----------------------:|
| 4 | EfficientNetB3 |     U-Net     | ImageNet | BCE Loss  | 0.00876 |          0.524          |
| 3 | EfficientNetB2 |     U-Net     | ImageNet | BCE Loss  | 0.00875 |          0.515          |
| 2 | EfficientNetB1 |     U-Net     | ImageNet | BCE Loss  | 0.00896 |          0.508          |
| 1 | EfficientNetB0 |     U-Net     | ImageNet | BCE Loss  | 0.00895 |          0.503          |

### Classification

| # |  Backbone  |                    Head                    |     Criterion      | Loss  | Accuracy |
|:-:|:----------:|:------------------------------------------:|:------------------:|:-----:|:--------:|
| 4 | Backbone#4 | Adaptive Pooling + One Linear Layer + ReLU | Cross Entropy Loss | 0.295 |  0.881   |
| 3 | Backbone#3 | Adaptive Pooling + One Linear Layer + ReLU | Cross Entropy Loss | 0.272 |  0.895   |
| 2 | Backbone#2 | Adaptive Pooling + One Linear Layer + ReLU | Cross Entropy Loss | 0.278 |  0.871   |
| 1 | Backbone#1 | Adaptive Pooling + One Linear Layer + ReLU | Cross Entropy Loss | 0.319 |  0.874   |

## Segmentation Stage

### Segmentation

| # |  Model  |    Encoder     | Weights  | Filter | Criterion | Loss  | Global Dice Coefficient |
|:-:|:-------:|:--------------:|:--------:|:------:|:---------:|:-----:|:-----------------------:|
| 6 | U-Net++ | EfficientNetB4 | ImageNet |   On   | Dice Loss | 0.328 |          0.672          |
| 5 | U-Net++ | EfficientNetB3 | ImageNet |   On   | Dice Loss | 0.333 |          0.667          |
| 4 | U-Net++ | EfficientNetB2 | ImageNet |   On   | Dice Loss | 0.334 |          0.666          |
| 3 | U-Net++ | EfficientNetB1 | ImageNet |   On   | Dice Loss | 0.338 |          0.662          |
| 2 | U-Net++ | EfficientNetB0 | ImageNet |   On   | Dice Loss | 0.338 |          0.662          |
| 1 |  U-Net  | EfficientNetB0 | ImageNet |   On   | Dice Loss | 0.342 |          0.658          |

## Classification Stage + Segmentation Stage

| # |      Classification      |      Segmentation      | Global Dice Coefficient (Kaggle Public Score) |
|:-:|:------------------------:|:----------------------:|:---------------------------------------------:|
| 3 | Classification#3 (0.895) | Segmentation#6 (0.672) |                     0.641                     |
| 2 | Classification#3 (0.895) | Segmentation#4 (0.666) |                     0.626                     |
| 1 | Classification#1 (0.874) | Segmentation#1 (0.658) |                     0.605                     |
