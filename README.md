## Introduction 
The code is a pytorch implementation of our work "Exploring Feature Representation Learning for Semi-supervised Medical Image Segmentation". 
Unlike prior state-of-the-art semi-supervised segmentation methods that predominantly rely on pseudo supervision directly 
on predictions, such as consistency regularization and pseudo labeling, our key insight is to explore the feature representation to regularize a more compact and better-separated feature space, 
which paves the way for low-density decision boundary learning and therefore enhances the segmentation performance.
A stage-adaptive contrastive learning method is proposed, containing a 
boundary-aware contrastive loss that takes advantage of the 
labeled images in the ﬁrst stage, as well as a prototype-aware 
contrastive loss to optimize both labeled and pseudo labeled 
images in the second stage. To obtain more accurate prototype estimation, which plays a critical rule in prototype-aware 
contrastive learning, we present an aleatoric uncertainty-aware 
method, namely AUA, to generate higher quality pseudo labels. 
AUA adaptively regularizes prediction consistency by taking 
adavantage of image ambiguity, which, given its signiﬁcance, is 
under-explored by existing works. Our method achieves the best 
results on three public medical image segmentation benchmarks.

## Installation
This repository is based on PyTorch 1.6.0.

## Usage
Please clone the repository and refer to run.sh for training and testing scripts.
Our [trained models](https://github.com/Huiimin5/AUA/tree/main/model) are released.
## Citation
If this work is helpful in your research, please consider citing
```
@article{wu2021exploring,
  title={Exploring Feature Representation Learning for Semi-supervised Medical Image Segmentation},
  author={Wu, Huimin and Li, Xiaomeng and Cheng, Kwang-Ting},
  journal={arXiv preprint arXiv:2111.10989},
  year={2021}
}
```

## Acknowledgement
We would like to thank following open-source projects:
[UA-MT](https://github.com/yulequan/UA-MT),
[stochastic\_segmentation\_networks](stochastic_segmentation_networks),
[DenseCL](https://github.com/WXinlong/DenseCL),
[SDCA](https://github.com/BIT-DA/SDCA).
