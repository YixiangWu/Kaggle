# [Google Research - Identify Contrails to Reduce Global Warming](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming)

## Classification Stage

### Backbone

| # |    Backbone     | Parent Module | Weights  | Criterion |  Loss   | Global Dice Coefficient |
|:-:|:---------------:|:-------------:|:--------:|:---------:|:-------:|:-----------------------:|
| 6 |    ResNet50     |     U-Net     | ImageNet | BCE Loss  | 0.00838 |          0.519          |
| 5 |    ResNet34     |     U-Net     | ImageNet | BCE Loss  | 0.00852 |          0.515          |
| 4 |    ResNet18     |     U-Net     | ImageNet | BCE Loss  | 0.00872 |          0.506          |
| 3 | EfficientNetB1  |     U-Net     | ImageNet | BCE Loss  | 0.00834 |          0.532          |
| 2 | EfficientNetB0  |     U-Net     | ImageNet | BCE Loss  | 0.00855 |          0.517          |
| 1 | ResNext50 32x4D |     U-Net     | ImageNet | BCE Loss  | 0.00871 |          0.533          |

### Classification

| # |  Backbone  |     Criterion      | Loss  | Accuracy |
|:-:|:----------:|:------------------:|:-----:|:--------:|
| 6 | Backbone#6 | Cross Entropy Loss | 0.219 |  0.929   |
| 5 | Backbone#5 | Cross Entropy Loss | 0.223 |  0.920   |
| 4 | Backbone#4 | Cross Entropy Loss | 0.244 |  0.910   |
| 3 | Backbone#3 | Cross Entropy Loss | 0.210 |  0.925   |
| 2 | Backbone#2 | Cross Entropy Loss | 0.211 |  0.925   |
| 1 | Backbone#1 | Cross Entropy Loss | 0.223 |  0.918   |

### Classification Ensemble

| # |        Backbone        |  Ensemble Method   |     Criterion      | Loss  | Accuracy |
|:-:|:----------------------:|:------------------:|:------------------:|:-----:|:--------:|
| 3 | Backbone#2, #3, #5, #6 | Stacking (Encoder) | Cross Entropy Loss | 0.191 |  0.945   |
| 2 |      Backbone#1-3      | Stacking (Encoder) | Cross Entropy Loss | 0.187 |  0.941   |
| 1 |     Backbone#1, #2     | Stacking (Encoder) | Cross Entropy Loss | 0.199 |  0.931   |

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

| # |          Classification           |      Segmentation      | Global Dice Coefficient (Kaggle Public Score) |
|:-:|:---------------------------------:|:----------------------:|:---------------------------------------------:|
| 3 | Classification Ensemble#2 (0.941) | Segmentation#3 (0.671) |                     0.638                     |
| 2 |     Classification#2 (0.925)      | Segmentation#3 (0.671) |                     0.628                     |
| 1 |     Classification#1 (0.918)      | Segmentation#2 (0.662) |                     0.610                     |
