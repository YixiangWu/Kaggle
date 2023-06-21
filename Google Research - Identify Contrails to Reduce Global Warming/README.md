# [Google Research - Identify Contrails to Reduce Global Warming](https://www.kaggle.com/competitions/google-research-identify-contrails-reduce-global-warming)

| # | Model |    Encoder     | Weights  | Optimizer | Additional Channel | Global Dice Coefficient |
|:-:|:-----:|:--------------:|:--------:|:---------:|:------------------:|:-----------------------:|
| 3 | U-Net | EfficientNetB0 | ImageNet |   Adam    |         On         |          0.600          |
| 2 | U-Net | EfficientNetB0 | ImageNet |   Adam    |        Off         |          0.593          |
| 1 | U-Net |    ResNet34    | ImageNet |   Adam    |        Off         |          0.587          |
