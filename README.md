# Synthetic microscopy image generation with pix2pix and CycleGAN

## Prerequisite to use this code:
1. Python >= v2.7 or v3.6
2. Keras v2.2.0
3. Tensorflow v1.9.0
4. PIL v1.1.7
5. numpy v1.14.5
6. openCV v3.4.2
7. matplotlib v1.5.1
8. Matlab R2018a

## Steps to execute the codes:
1. Download the codes
```
git clone https://github.com/XiaoqiChai/mitoGAN
cd mitoGAN
```
2. Download the dataset 
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
*CycleGAN/CycleGAN/test_predicted_cgan, 
pix2pix/pix2pix/test_predicted_pgan, 
sim_images_test* 
and *real_images_test* 
to *characterization/*. 
Then run `SSIM.m` or `background_chara.m`.
8. To perform Unet segmentation, 
