clear
clc

mkdir('Unet_set_pix2pix')
mkdir('Unet_set_pix2pix/images')
mkdir('Unet_set_pix2pix/masks')

image_dir = dir(fullfile('pix2pix/candidate_predicted','*.tif'));
mask_dir = dir(fullfile('pix2pix/candidateA','*.tif'));

image_names = {image_dir.name};
mask_names = {mask_dir.name};

for i = 1:length(image_names)
    image = imread(fullfile('pix2pix/candidate_predicted',image_names{i}));
    mask = imread(fullfile('pix2pix/candidateA',mask_names{i}));
    
    mask(mask~=0) = 1;
    
    mask = logical(mask);
    
    imwrite(mask,strcat('pix2pix/masks/',mask_names{i}))
    imwrite(image,strcat('pix2pix/images/',image_names{i}))
end
