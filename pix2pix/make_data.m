% Script to make the dataset for training pix2pix

clear
clc

image_dir = dir(fullfile('images','*.tif'));
mask_dir = dir(fullfile('masks','*.tif'));

image_names = {image_dir.name};
mask_names = {mask_dir.name};

mkdir('pix2pix')

mkdir('pix2pix/train')
mkdir('pix2pix/val')
mkdir('pix2pix/candidateA')
mkdir('pix2pix/candidateB')
mkdir('pix2pix/testA')
mkdir('pix2pix/testB')

for i = 1:500
    
    image = imread(fullfile('images',image_names{i}));
    
    mask = imread(fullfile('masks',mask_names{i}));
    mask = uint16(mask);
    mask(mask == 1) = 1500;
    
    imwrite([image,mask],strcat('pix2pix/train/',image_names{i}))
    
end

for j = 501:1100
    
    image = imread(fullfile('images',image_names{j}));
    
    mask = imread(fullfile('masks',mask_names{j}));
    mask = uint16(mask);
    mask(mask == 1) = 1500;
    
    imwrite(mask,strcat('pix2pix/candidateA/',image_names{j}))
    imwrite(image,strcat('pix2pix/candidateB/',image_names{j}))
    
end

for k = 1101:1200
    
    image = imread(fullfile('images',image_names{k}));
    
    mask = imread(fullfile('masks',mask_names{k}));
    mask = uint16(mask);
    mask(mask == 1) = 1500;
    
    imwrite([image,mask],strcat('pix2pix/val/',image_names{k}))
end

for h = 1201:length(image_names)
    
    image = imread(fullfile('images',image_names{h}));
    
    mask = imread(fullfile('masks',mask_names{h}));
    mask = uint16(mask);
    mask(mask == 1) = 1500;
    
    imwrite(mask,strcat('pix2pix/testA/',image_names{h}))
    imwrite(image,strcat('pix2pix/testB/',image_names{h}))
    
end