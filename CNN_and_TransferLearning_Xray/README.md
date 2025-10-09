# Introduction

Chest X-ray imaging is a critical diagnostic tool in detecting various pulmonary diseases. However, manual interpretation of chest X-rays can be time-consuming and prone to variability between radiologists. To address these challenges, deep learning techniques - particularly **Convolutional Neural Networks (CNNs)** - have been widely adopted for automated medical image analysis due to their ability to learn hierarchical visual features directly from data.

In this project, a CNN was developed to classify chest X-ray images using relatively small 224x224 and also 64×64 pixel inputs, aiming to explore whether lightweight models can still retain sufficient diagnostic power for image-based classification tasks. Beyond training a basic CNN from scratch, **transfer learning** was employed by leveraging pretrained convolutional backbones such as ResNet, to assess whether pretrained models can further enhance classification performance when applied to chest X-ray images.

The objective was to compare the effectiveness of a simple CNN versus a transfer learning approach, focusing on model generalization, convergence speed, and performance on a limited dataset.

## Convolutional neural networks

![CNN representation from https://stanford.edu/~shervine/teaching/cs-230/](https://stanford.edu/~shervine/teaching/cs-230/illustrations/architecture-cnn-en.jpeg?3b7fccd728e29dc619e1bd8022bf71cf)

For this example, I will use the [chest X-ray data-set](https://drive.google.com/file/d/1Y9iTkRrfh_2UfoG9o8CRjZc_3rj73nap/view)
from [Kermany et al. 2018](https://www.sciencedirect.com/science/article/pii/S0092867418301545?via%3Dihub).

| normal | pneumonia |
| --- | --- |
| <img alt='a sample normal chest x-ray from Kermany et al. 2018' src='data/NORMAL-1003233-0001.jpeg' width="400"> | <img alt='a pneumonia normal chest x-ray from Kermany et al. 2018' src='data/BACTERIA-1008087-0001.jpeg' width="400"> |
 


Here is a refresher on CNN: [CNN cheatsheet](https://stanford.edu/~shervine/teaching/cs-230/cheatsheet-convolutional-neural-networks).

## Transfer Learning


Here, I will continue working on the chest x-ray dataset, from [Kermany et al. 2018](https://www.sciencedirect.com/science/article/pii/S0092867418301545?via%3Dihub), but this time I will use transfer learning described in the original paper.

![fig 1 of "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning" by Kermany et al.](data/tranfer_learning_xray.jpg)

The base model I will re-use is [ResNet50](https://pytorch.org/vision/main/models/generated/torchvision.models.resnet50.html) from [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). With inspirations from [this github repo](https://github.com/liyu95/Deep_learning_examples/blob/master/4.ResNet_X-ray_classification/Densenet_fine_tune.ipynb) and [this kaggle thread](https://www.kaggle.com/code/iamsdt/transferlearning-pytorch-resnet-50).


# Methods


## Set-up

Project structure:
```{bash}
├── config.yaml
├── data/
│   ├── chest_xray_224/
│   │   ├── train/ NORMAL/ PNEUMONIA/
│   │   └── test/  NORMAL/ PNEUMONIA/
│   └── chest_xray_63/      # optional smaller dataset
└── src/
    ├── data.py             # dataset + dataloaders
    ├── model.py            # CNN model definition
    ├── train.py            # training loop
    └── eval.py             # evaluation loop
```


Create and activate an environment (e.g., via conda or venv):
```{bash}
conda env create -f environment.yml
conda activate xray_cnn
```

Install dependencies::
```{bash}
pip install -r requirements.txt
```



## Quickstart

### Train

Train the CNN using the default configuration (config.yaml):

```{bash}
python -m src.train
```

Example output:
```{bash}
(xray-cnn) katwre@katwre-XPS-13-9350:~/projects/ML-projects/CNN_and_TransferLearning_Xray$ python -m src.train
[train] cfg=config.yaml data_dir=data/chest_xray_224 device=cpu
[train] batches: train=82 val=10

============================================================
[INFO] Model architecture:

CNN(
  (conv): Sequential(
    (0): Conv2d(1, 2, kernel_size=(16, 16), stride=(4, 4))
    (1): ReLU(inplace=True)
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv2d(2, 4, kernel_size=(5, 5), stride=(1, 1))
    (4): ReLU(inplace=True)
    (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  )
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (classifier): Sequential(
    (0): Linear(in_features=484, out_features=16, bias=True)
    (1): Linear(in_features=16, out_features=8, bias=True)
    (2): Linear(in_features=8, out_features=1, bias=True)
    (3): Sigmoid()
  )
)

[INFO] Detailed summary:
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
CNN                                      [1, 1]                    --
├─Sequential: 1-1                        [1, 4, 11, 11]            --
│    └─Conv2d: 2-1                       [1, 2, 53, 53]            514
│    └─ReLU: 2-2                         [1, 2, 53, 53]            --
│    └─MaxPool2d: 2-3                    [1, 2, 26, 26]            --
│    └─Conv2d: 2-4                       [1, 4, 22, 22]            204
│    └─ReLU: 2-5                         [1, 4, 22, 22]            --
│    └─MaxPool2d: 2-6                    [1, 4, 11, 11]            --
├─Flatten: 1-2                           [1, 484]                  --
├─Sequential: 1-3                        [1, 1]                    --
│    └─Linear: 2-7                       [1, 16]                   7,760
│    └─Linear: 2-8                       [1, 8]                    136
│    └─Linear: 2-9                       [1, 1]                    9
│    └─Sigmoid: 2-10                     [1, 1]                    --
==========================================================================================
Total params: 8,623
Trainable params: 8,623
Non-trainable params: 0
Total mult-adds (M): 1.55
==========================================================================================
Input size (MB): 0.20
Forward/backward pass size (MB): 0.06
Params size (MB): 0.03
Estimated Total Size (MB): 0.30
==========================================================================================

[INFO] Total parameters: 8,623
[INFO] Trainable parameters: 8,623
============================================================

Epoch 01: train_loss=0.5752 acc=0.742 | val_loss=0.6902 acc=0.625                                                                                                                            
Epoch 02: train_loss=0.5716 acc=0.742 | val_loss=0.6988 acc=0.625                                                                                                                            
Epoch 03: train_loss=0.5712 acc=0.742 | val_loss=0.6909 acc=0.625                                                                                                                            
Epoch 04: train_loss=0.5718 acc=0.742 | val_loss=0.6921 acc=0.625                                                                                                                            
Early stopping.
Training finished. Best val loss: 0.690
```

You can override settings without editing config.yaml:
```{bash}

# use smaller 63px dataset
python -m src.train --data_dir data/chest_xray_63

# use more workers / bigger batch size
python -m src.train --num_workers 4 --batch_size 64
```

Checkpoints & outputs:

- Best model: outputs/cnn_best.pt
- Metrics printed at the end of each epoch
- Early stopping based on validation loss


### Evaluate

Evaluate the best saved checkpoint (default: outputs/cnn_best.pt):
```
python -m src.eval
```


## Change backbone
Set `model.name` in `config.yaml` to `resnet50` or `efficientnet_b0` (if supported by your torchvision).


# Results

The first CNN trained on 64×64 resized chest X-ray images achieved a training accuracy of approximately 92% and a validation accuracy of around 82–83%. The model demonstrated good generalization, with early stopping triggered after 10-12 epochs to prevent overfitting. The BCE loss curve showed stable convergence, with validation loss increasing slightly after early stopping, suggesting that the model benefited from regularization. Overall, this baseline network performed well using relatively small input images, confirming that 64×64 resizing preserves enough relevant features for accurate classification.


To further improve performance, transfer learning was applied using a pretrained ResNet backbone combined with a custom classifier head (Deep_LR). This approach allowed the model to converge faster and generalize better than training from scratch. Initial evaluation at the default decision threshold of 0.5 produced a strong ROC-AUC of 0.834 but misclassified all NORMAL cases as PNEUMONIA due to probability outputs being skewed above 0.5. By tuning the classification threshold using Youden’s J statistic (optimal cutoff ≈0.6865), the model achieved balanced performance:

- **NORMAL**: precision 0.681, recall 0.739, F1 0.709  
- **PNEUMONIA**: precision 0.835, recall 0.792, F1 0.813  
- **Overall accuracy**: 77.2%  

Transfer learning not only improved generalization but, after threshold adjustment, provided clinically meaningful sensitivity and specificity, demonstrating the utility of pretrained visual representations for chest X-ray classification.
