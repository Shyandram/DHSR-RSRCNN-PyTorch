# SRDH-Net & DHSR-Net

## Introducion
"NOT COMPLETE YET"  
This project is a *pytorch implementation* of merging "Super Resolution" and "Image Dehazing" into an End-to-End Network.  
[Reducing Computational Requirements of Image Dehazing Using Super-Resolution Networks](https://ieeexplore.ieee.org/document/10219494)
```
@INPROCEEDINGS{10219494,
  author={Weng, Shyang-En and Ye, Yan-Gu and Lin, Ying-Cheng and Miaou, Shaou-Gang},
  booktitle={2023 Sixth International Symposium on Computer, Consumer and Control (IS3C)}, 
  title={Reducing Computational Requirements of Image Dehazing Using Super-Resolution Networks}, 
  year={2023},
  volume={},
  number={},
  pages={326-329},
  keywords={Training;Image quality;Interpolation;Computational modeling;Superresolution;Focusing;Computational efficiency;Image Dehazing;Super-Resolution;Advanced Driver Assistance Systems;Computational Requirement},
  doi={10.1109/IS3C57901.2023.00094}}
```
AOD-Net : All-in-One Network for Dehazing.

## Preparation


### Data Preparation


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
