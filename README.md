# SRDH-Net & DHSR-Net

<!-- ![](https://img.shields.io/badge/pytorch-0.4.0-blue.svg) ![](https://img.shields.io/badge/python-3.6.5-brightgreen.svg) -->

## Introducion

This project is a *pytorch implementation* of merging "Super Resolution" and "Image Dehazing" into an End-to-End Network.

AOD-Net : All-in-One Network for Dehazing.

## Preparation
1. First of all, clone the code
```
git clone https://github.com/walsvid/AOD-Net.pytorch.git
```
2. Then, install prerequisites
```
pip install -r requirements.txt
```
### Data Preparation
Please download the `training images` and `original images` from [author's web page(NYU2_synthesis)](https://sites.google.com/site/boyilics/website-builder/project-page).

Then make a directory for data, change the parameters about data directories.
## Train
```
python train_dhsr.py
```
You can change the parameter in train bash script to satisfied your project.
## Test
```
python evaluate.py
```
## Demo
This is the dehazing result image comparison. Left image is haze image, right image is clean image processed by DHSR-Net.
![](samples\20_2.jpg)


## Credicts
AOD-Net : All-in-One Network for Dehazing. [walsvid/AOD-Net-PyTorch](https://github.com/walsvid/AOD-Net.pytorch.git)  
Image Super-Resolution Using Deep Convolutional Networks [yjn870/SRCNN-pytorch](https://github.com/yjn870/SRCNN-pytorch)