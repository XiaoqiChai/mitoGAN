# Synthetic microscopy image generation with pix2pix and CycleGAN
Yile Feng, Xiaoqi Chai, Qinle Ba, and Ge Yang  
International Symposium on Biomedical Imaging (ISBI), 2019

## Prerequisite to use this code:
1. Python >= v2.7 or v3.6
2. Keras v2.2.0
3. Tensorflow v1.9.0
4. PIL v1.1.7
5. numpy v1.14.5
6. openCV v3.4.2
7. matplotlib v1.5.1
8. Matlab R2018a

## Acknowledgement
The codes for traing CycleGAN and pix2pix are modified from https://github.com/tjwei/GANotebooks.

## Steps to execute the codes:
1. Download the codes
```
git clone https://github.com/XiaoqiChai/mitoGAN
cd mitoGAN
```
2. Download and extract the dataset 
```
wget https://cmu.box.com/shared/static/zzen200cng2ymdgu6rukgwukflrhyr4r.tar
tar -xvf zzen200cng2ymdgu6rukgwukflrhyr4r.tar
```

3. Run `make_data.m`
4. Train CycleGAN or pix2pix by 
```
cd CycleGAN
python CycleGAN-keras.py
```
or
```
cd pix2pix
python pix2pix-keras.py
```
5. To predict the test data, in either folder, type
```
python predict.py
```
6. To make the dataset for Unet segmentation, in either folder, run `make_Unet_set.m`
7. To characterize the images, copy and paste 
*CycleGAN/CycleGAN/test_predicted_cgan*, 
*pix2pix/pix2pix/test_predicted_pgan*, 
*sim_images_test* 
and *real_images_test* 
to *characterization/*. 
Then run `SSIM.m` or `background_chara.m`.


Codes for Unet segmentation and IOU calculation is concluded in `unet_seg.py`.
