# SAP

Official Code Repository for "SAP: : Corrective Machine Unlearning with Scaled Activation Projection for Label Noise Robustness", Proceedings of the AAAI Conference on Artificial Intelligence, 2025

[[ArXiv Paper](https://arxiv.org/abs/2403.08618)]
[[AAAI  Proceedings]()]

## Abstract 
Label corruption, where training samples are mislabeled due to non-expert annotation or adversarial attacks, significantly degrades model performance. Acquiring large, perfectly labeled datasets is costly, and retraining models from scratch is computationally expensive. To address this, we introduce Scaled Activation Projection (SAP), a novel SVD (Singular Value Decomposition)-based corrective machine unlearning algorithm. SAP mitigates label noise by identifying a small subset of trusted samples using cross-entropy loss and projecting model weights onto a clean activation space estimated using SVD on these trusted samples. This process suppresses the noise introduced in activations due to the mislabeled samples. In our experiments, we demonstrate SAP’s effectiveness on synthetic noise with different settings and real-world label noise. SAP applied to the CIFAR dataset with 25% synthetic corruption show upto 6% generalization improvements. Additionally, SAP can improve the generalization over noise robust training approaches on CIFAR dataset by ∼ 3.2% on average. Further, we observe generalization improvements of 2.31% for a Vision Transformer model trained on naturally corrupted Clothing1M
## Authors 
Sangamesh Kodge, Deepak Ravikumar, Gobinda Saha, Kaushik Roy 

## Dependency Installation
To set up the environment and install dependencies, follow these steps:
### Installation using conda
Install the packages either manually or use the environment.yml file with conda. 
- Installation using yml file
    ```bash
    conda env create -f environment.yml
    ```
    OR
- Manual Installation with conda environment 
    ```bash    
    ### Create Envirornment (Optional, but recommended)
        conda create --name sap python=3.11.4
        conda activate sap

        ### Install Packages
        pip install wandb 
        pip install argparse 
        pip install scikit-learn
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        pip install matplotlib
        pip install seaborn
        pip install SciencePlots
    ```



## Supported Datasets
### Real-world noisy dataset
The real-world noisy dataset used in this project is the WebVision dataset, designed to facilitate research on learning visual representation from noisy web data. 

1. [WebVision 1.0](https://data.vision.ee.ethz.ch/cvl/webvision/dataset2017.html)- The WebVision dataset is designed to facilitate the research on learning visual representation from noisy web data. See ```data/WebVision1.0/``` to download the data directory and process data in supported format (Refer repository [WebVision1.0](https://github.com/sangamesh-kodge/WebVision1.0)). 

2. [Mini-WebVision](https://arxiv.org/abs/1911.09781)- is a subset of the first 50 classes of Goole partition of WebVision 1.0 dataset (contains about 61234 training images). See ```data/MiniWebVision/``` to download the data directory and process data in supported format (Refer repository [Mini-WebVision](https://github.com/sangamesh-kodge/Mini-WebVision)).


3. [Clothing1M](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Xiao_Learning_From_Massive_2015_CVPR_paper.pdf)- is a 14 class dataset containing clothes (dataset has 1000000 training images with noisy labels).  See ```data/Clothing1M/``` to download the data directory and process data in supported format (Refer repository [Clothing1M](https://github.com/sangamesh-kodge/Clothing1M)).


### Synthetic Noise in standard dataset. 
In addition to the real-world noisy dataset, synthetic noise is introduced into standard datasets for further analysis and evaluation of label noise robustness. Set the ```--percentage-mislabeled``` command line argument to desired level of label noise percentage for adding synthetic uniform noise to standard dataset. The following standard datasets are used with synthetic noise (We use torchvision and hence do not require any preprocessing for these datasets):
- [MNIST](https://ieeexplore.ieee.org/document/6296535)
- [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [CIFAR100](https://www.cs.toronto.edu/~kriz/cifar.html)
- [ImageNet](https://www.image-net.org/)


## Supported Network Architectures
- [LeNet5](https://ieeexplore.ieee.org/document/726791) - for MNIST
- [ResNets](https://arxiv.org/pdf/1512.03385.pdf) - for CIFAR, ImageNet , WebVision and Clothing1M
- [VGGs](https://arxiv.org/pdf/1409.1556.pdf) - for CIFAR, ImageNet and WebVision
- [InceptionResNetv2](https://arxiv.org/pdf/1602.07261.pdf) - for CIFAR, ImageNet , WebVision and Clothing1M
- [Vision Transformers (ViTs)](https://arxiv.org/pdf/2010.11929.pdf) - for CIFAR, ImageNet, WebVision and Clothing1M

## Supported Noise-Robust Training Algorithms 
Refer repository [LabelNoiseRobustness](https://github.com/sangamesh-kodge/LabelNoiseRobustness/) 

1. Vanilla SGD - Standard Stochastic Gradient Descent algorithm. 
2. [Mixup](https://arxiv.org/pdf/1806.05236.pdf)- enhances model robustness by linearly interpolating between pairs of training examples and their corresponding labels. Specifically, it generates augmented training samples by blending two input samples and their labels. This process introduces beneficial noise during training, which helps the model learn more effectively even when the training data contains noisy labels. To use mixup add the cli argument ```--mixup-alpha <value-of-hyperparameter-alpha>```. For example, ```--mixup-alpha 0.2``` means the alpha hyperparameter is set to 0.1.

2. [SAM (Sharpness-Aware Minimization)](https://arxiv.org/pdf/2010.01412.pdf)- Instead of solely minimizing the loss value, SAM aims to find a balance between low loss and smoothness. It encourages the model to explore regions with uniformly low loss, avoiding sharp spikes that might lead to overfitting. SAM exhibits remarkable resilience to noisy labels. To use SAM add the cli argument ```--sam-rho <value-of-hyperparameter-rho>```. For example, ```--sam-rho 0.1``` means the rho hyperparameter is set to 0.1.


4. [MentorMix](https://arxiv.org/pdf/1911.09781.pdf) develops on the idea of MentorNet and Mixup. To use MentorMix, you can add the cli argument ```--mnet-gamma-p <value-of-hyperparameter-gamma-p> --mmix-alpha <value-of-hyperparameter-alpha >```. For example, ```--mnet-gamma-p 0.85 --mmix-alpha  0.2``` means using gamma-p  of 0.85 and alpha 0.2.



# DEMO
To run the demo script, run the following command form terminal:
```bash
mkdir -p images
mkdir -p pretrained_models/2DSpiral
python demo_spiral.py
```

# Results
Check the examples scripts in ```./example_scripts``` to get the results below. We use seeds - 12484, 32087 and 35416.
## Synthetic Noise in standard dataset
Test accuracy for CIFAR10 and CIFAR100 dataset averaged over 3 randomly chosen seeds. 
We show the Baseline Accuracy and Accuracy when sap is applied to the baseline model.

###  Corrective Unlearning on CIFAR Dataset.

|                     | Method   | Retain samples | Forget samples | VGG11_BN η=0.1 | VGG11_BN η=0.25 | ResNet18 η=0.1 | ResNet18 η=0.25 | ResNet18 η=0.1 | ResNet18 η=0.25 | ResNet18 η=0.1 | ResNet18 η=0.25 | Average |
|---------------------|----------|----------------|----------------|-----------------|------------------|-----------------|------------------|-----------------|------------------|-----------------|------------------|---------|
|         | Retrain  | -              | -              | 90.18±0.14      | 89.48±0.04       | 93.21±0.22      | 92.45±0.06       | 93.38±0.07      | 92.75±0.23       | 93.47±0.21      | 93.14±0.23       | 92.26   |
|                     | Vanilla  | 0              | 0              | 86.04±0.17      | 76.68±0.48       | 88.55±0.16      | 79.47±0.46       | 88.53±0.34      | 79.79±1.53       | 91.42±0.33      | 86.42±0.38       | 84.61   |
|                     | Finetune | 5000           | 0              | 85.47±0.13      | 80.94±0.76       | 88.28±0.30      | 85.16±0.12       | 87.31±0.81      | 82.82±0.98       | 91.42±0.33      | **88.23±0.73**   | 86.20   |
|  **CIFAR100**          | SSD      | 5000           | 1000           | 86.00±0.21      | 76.77±0.58       | 88.54±0.17      | 79.48±0.50       | 88.52±0.35      | 80.36±1.45       | 91.42±0.33      | 86.39±0.42       | 84.68   |
|                     | SCRUB    | 1000           | 200            | 85.88±0.35      | 78.90±0.25       | 89.50±0.17      | 83.77±0.44       | 89.50±0.22      | 83.60±0.14       | 91.65±0.12      | 88.00±0.58       | 86.35   |
|                     | SAP      | 0 | 0              | **87.25±0.16**  | **82.27±0.15**   | **90.12±0.11**  | **85.49±0.39**   | **90.03±0.25**  | **86.32±0.66**   | **91.87±0.22**  | 87.92±0.69       | **87.66**|
|                     |          |   |                |   |    |   |    |   |    |   |        | |
|   | Retrain  | -              | -              | 65.76±0.23      | 63.66±0.47       | 72.43±0.42      | 71±0.15         | 73.76±0.46      | 71.62±0.47       | 72.73±0.19      | 71.16±0.39       | 70.26   |
|                     | Vanilla  | 0              | 0              | 60.41±0.14      | 50.64±0.60       | 65.84±0.33      | 54.75±0.45       | 72.98±0.13      | 61.86±0.60       | 67.45±0.12      | 57.82±0.21       | 61.47   |
|                     | Finetune | 5000           | 0              | 60.26±0.11      | 52.50±0.31       | 65.97±0.35      | 57.33±0.40       | 72.98±0.13      | 61.29±1.13       | 67.55±0.15      | 59.98±1.11       | 62.23   |
|   **CIFAR100**      | SSD      | 5000           | 1000           | 60.38±0.16      | 50.62±0.60       | 65.84±0.33      | 54.77±0.42       | 72.99±0.11      | 61.67±0.55       | 67.43±0.21      | 57.83±0.21       | 61.44   |
|                     | SCRUB    | 1000           | 200            | 60.93±0.09      | 52.11±0.63       | **67.02±0.29**  | 57.36±0.40       | **73.12±0.18**  | 63.37±0.69       | 68.20±0.13      | 60.24±0.33       | 62.79   |
|                     | SAP      | 0 | 0              | **61.10±0.23**  | **53.31±0.78**   | 66.82±0.17      | **58.74±0.61**   | 72.92±0.30      | **63.57±0.49**   | **68.24±0.17**  | **60.76±0.50**   | **63.18**|




### Noise Robust Algorithm.
|                     | Method      | Baseline      | SAP           | Improvements |
|---------------------|-------------|---------------|---------------|-------------|
|                     | Vanilla     | 79.47±0.46    | 85.46±0.41    | **5.99**    |
|                     | Logit Clip  | 82.91±0.32    | 85.99±0.67    | **3.08**    |
|**CIFAR10**          | MixUp       | 83.12±0.44    | 86.45±0.52    | **3.33**    |
|                     | SAM         | 83.29±0.28    | 87.29±0.08    | **4.0**     |
|                     | MentorMix   | 89.64±0.32    | 90.51±0.17    | **0.87**    |
|                     | **Average** | 83.69         | 87.14         | **3.45**    |
|                     |             |               |               |             |
|                     | Vanilla     | 54.75±0.45    | 58.69±0.68    | **3.94**    |
|                     | Logit Clip  | 56.41±0.43    | 59.90±0.84    | **3.49**    |
|  **CIFAR100**       | SAM         | 56.49±0.93    | 59.34±0.76    | **2.85**    |
|                     | MixUp       | 58.25±0.65    | 62.32±0.91    | **4.07**    |
|                     | MentorMix   | 68.53±0.35    | 68.98±0.45    | **0.45**    |
|                     | **Average** | 58.89         | 61.85         | **2.98**    |
|                     |             |               |               |             |


## Real-world noisy dataset
Test/Val accuracy for Mini-WebVision dataset and Clothing1M dataset averaged over 3 randomly chosen seeds. We show the Baseline Accuracy and Accuracy when SAP is applied to the baseline model. 

| Dataset        | Architecture | Vanilla       | SAP           | Improvement   |
|----------------|--------------|---------------|---------------|---------------|
| Mini-WebVision | IRV2         | 63.81±0.38    | 64.73±0.53    | **0.92**      |
| Clothing1M     | ResNet50     | 67.48±0.64    | 69.64±0.57    | **2.16**      |
| Clothing1M     | ViT_B_16     | 69.12±0.45    | 71.43±0.60    | **2.31**      |
| **Average**    |              | 66.80         | 68.60         | **1.80**      |


## License

This project is licensed under the [Apache 2.0 License](LICENSE).




# Citation
Kindly cite the [paper](https://arxiv.org/abs/2403.08618) if you use the code. Thanks!

### APA
```
Kodge, S., Ravikumar, D., Saha, G., & Roy, K. (2025). SAP: Corrective Machine Unlearning with Scaled Activation Projection for Label Noise Robustness 	. https://arxiv.org/abs/2403.08618
```
or 
### Bibtex
```
@misc{kodge2025sap,
      title={SAP: Corrective Machine Unlearning with Scaled Activation Projection for Label Noise Robustness }, 
      author={Sangamesh Kodge and Deepak Ravikumar and Gobinda Saha and Kaushik Roy},
      year={2024},
      eprint={2403.08618},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```