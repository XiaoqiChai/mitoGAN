clear
clc

mkdir('Unet_set_CycleGAN')
mkdir('Unet_set_CycleGAN/images')
mkdir('Unet_set_CycleGAN/masks')

image_dir = dir(fullfile('CycleGAN/candidate_predicted','*.tif'));
mask_dir = dir(fullfile('CycleGAN/candidateA','*.tif'));

image_names = {image_dir.name};
mask_names = {mask_dir.name};

for i = 1:length(image_names)
    image = imread(fullfile('CycleGAN/candidate_predicted',image_names{i}));
    mask = imread(fullfile('CycleGAN/candidateA',mask_names{i}));
    
    mask(mask~=0) = 1;
    
    mask = logical(mask);
    
    imwrite(mask,strcat('Unet_set_CycleGAN/masks/',mask_names{i}))
    imwrite(image,strcat('Unet_set_CycleGAN/images/',image_names{i}))
end
