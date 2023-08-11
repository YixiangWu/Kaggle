# [Google Research - Identify Contrails to Reduce Global Warming](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming)

## Classification Stage

### Backbone

| #  |     Backbone     | Parent Module | Weights  | Criterion |  Loss   | Global Dice Coefficient |
|:--:|:----------------:|:-------------:|:--------:|:---------:|:-------:|:-----------------------:|
| 10 |   DenseNet169    |     U-Net     | ImageNet | BCE Loss  | 0.00849 |          0.529          |
| 9  |   DenseNet121    |     U-Net     | ImageNet | BCE Loss  | 0.00853 |          0.535          |
| 8  |    ResNest26d    |     U-Net     | ImageNet | BCE Loss  | 0.00859 |          0.538          |
| 7  |    ResNest14d    |     U-Net     | ImageNet | BCE Loss  | 0.00865 |          0.534          |
| 6  | ResNext101 32x8D |     U-Net     | ImageNet | BCE Loss  | 0.00847 |          0.541          |
| 5  | ResNext50 32x4D  |     U-Net     | ImageNet | BCE Loss  | 0.00871 |          0.528          |
| 4  |    ResNet152     |     U-Net     | ImageNet | BCE Loss  | 0.00854 |          0.536          |
| 3  |    ResNet101     |     U-Net     | ImageNet | BCE Loss  | 0.00882 |          0.518          |
| 2  |     ResNet50     |     U-Net     | ImageNet | BCE Loss  | 0.00883 |          0.523          |
| 1  |     ResNet34     |     U-Net     | ImageNet | BCE Loss  | 0.00882 |          0.524          |

### Classification

| #  |  Backbone   |     Criterion      | Loss  | Accuracy |
|:--:|:-----------:|:------------------:|:-----:|:--------:|
| 10 | Backbone#10 | Cross Entropy Loss | 0.285 |  0.909   |
| 9  | Backbone#9  | Cross Entropy Loss | 0.283 |  0.908   |
| 8  | Backbone#8  | Cross Entropy Loss | 0.270 |  0.921   |
| 7  | Backbone#7  | Cross Entropy Loss | 0.275 |  0.915   |
| 6  | Backbone#6  | Cross Entropy Loss | 0.278 |  0.913   |
| 5  | Backbone#5  | Cross Entropy Loss | 0.273 |  0.917   |
| 4  | Backbone#4  | Cross Entropy Loss | 0.283 |  0.910   |
| 3  | Backbone#3  | Cross Entropy Loss | 0.270 |  0.916   |
| 2  | Backbone#2  | Cross Entropy Loss | 0.280 |  0.908   |
| 1  | Backbone#1  | Cross Entropy Loss | 0.284 |  0.907   |

### Classification Ensemble

| # |   Backbone   |  Ensemble Method   |     Criterion      | Loss  | Accuracy |
|:-:|:------------:|:------------------:|:------------------:|:-----:|:--------:|
| * | Backbone#3-8 | Stacking (Encoder) | Cross Entropy Loss | 0.263 |  0.924   |

## Segmentation Stage

### Segmentation

| # |  Model  |    Encoder     | Weights  | Criterion | Loss  | Global Dice Coefficient |
|:-:|:-------:|:--------------:|:--------:|:---------:|:-----:|:-----------------------:|
| 4 | U-Net++ | EfficientNetB1 | ImageNet | Dice Loss | 0.333 |          0.667          |
| 3 | U-Net++ | EfficientNetB0 | ImageNet | Dice Loss | 0.338 |          0.662          |
| 2 | U-Net++ |    ResNet34    | ImageNet | Dice Loss | 0.345 |          0.655          |
| 1 | U-Net++ |    ResNet18    | ImageNet | Dice Loss | 0.352 |          0.648          |

### K-Fold Segmentation

| # |  Model  |     Encoder      | Weights  | Criterion |       Global Dice Coefficient       |
|:-:|:-------:|:----------------:|:--------:|:---------:|:-----------------------------------:|
| 6 |  U-Net  |   ResNest101e    | ImageNet | Dice Loss | (0.663, 0.660, 0.663, 0.667, 0.663) |
| 5 |  U-Net  |    ResNest50d    | ImageNet | Dice Loss | (0.664, 0.664, 0.662, 0.657, 0.663) |
| 4 |  U-Net  |   InceptionV4    | ImageNet | Dice Loss | (0.660, 0.651, 0.653, 0.659, 0.655) |
| 3 |  U-Net  |  EfficientnetB4  | ImageNet | Dice Loss | (0.665, 0.662, 0.665, 0.667, 0.663) |
| 2 | U-Net++ | EfficientnetV2S  | ImageNet | Dice Loss | (0.657, 0.651, 0.656, 0.658, 0.652) |
| 1 | U-Net++ | EfficientnetV2B3 | ImageNet | Dice Loss | (0.657, 0.658, 0.661, 0.657, 0.650) |

### Segmentation Ensemble

| # |            Network            |                           Ensemble Method                           |
|:-:|:-----------------------------:|:-------------------------------------------------------------------:|
| 2 | K-Fold Segmentation#3, #5, #6 |        Weighted Averaging Ensemble (Weights: 1.1, 0.8, 1.1)         |
| 1 |    K-Fold Segmentation#1-6    | Weighted Averaging Ensemble (Weights: 0.8, 0.2, 1.5, 0.8, 1.2, 1.5) |

## Classification Stage + Segmentation Stage (Final Submission)

| # |                  Classification                   |         Segmentation          | Global Dice Coefficient (Kaggle Score) |
|:-:|:-------------------------------------------------:|:-----------------------------:|:--------------------------------------:|
| 2 |  Classification Ensemble* (0.924) (Threshold=0)   | Segmentation#2 (Threshold=-5) |    Public: 66831, Private: 0.66062     |
| 1 | Classification Ensemble* (0.924) (Threshold=1.75) | Segmentation#2 (Threshold=0)  |    Public: 66468, Private: 0.65499     |

Final Submissions#2 ranks 158/954(**Top 17%**)
