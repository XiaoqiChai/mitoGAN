% Script to make the dataset for training CycleGAN

clear
clc

image_dir = dir(fullfile('images','*.tif'));
mask_dir = dir(fullfile('masks','*.tif'));

image_names = {image_dir.name};
mask_names = {mask_dir.name};

mkdir('CycleGAN')

mkdir('CycleGAN/trainA')
mkdir('CycleGAN/trainB')

mkdir('CycleGAN/testA')
mkdir('CycleGAN/testB')

mkdir('CycleGAN/candidateA')
mkdir('CycleGAN/candidateB')

mkdir('CycleGAN/valA')
mkdir('CycleGAN/valB')


for i = 1:500
    
    image = imread(fullfile('images',image_names{i}));
    
    mask = imread(fullfile('masks',mask_names{i}));
    mask = uint16(mask);
    mask(mask == 1) = 1500;
    
     imwrite(mask,strcat('CycleGAN/trainA/',image_names{i}))
     imwrite(image,strcat('CycleGAN/trainB/',image_names{i}))
    
end

for j = 501:1100
    
    image = imread(fullfile('images',image_names{j}));
    
    mask = imread(fullfile('masks',mask_names{j}));
    mask = uint16(mask);
    mask(mask == 1) = 1500;
    
    imwrite(mask,strcat('CycleGAN/candidateA/',image_names{j}))
    imwrite(image,strcat('CycleGAN/candidateB/',image_names{j}))
end

for k = 1101:1200
    image = imread(fullfile('images',image_names{k}));
    
    mask = imread(fullfile('masks',mask_names{k}));
    mask = uint16(mask);
    mask(mask == 1) = 1500;
    
    imwrite(mask,strcat('CycleGAN/valA/',image_names{k}))
    imwrite(image,strcat('CycleGAN/valB/',image_names{k}))
end

for h = 1201:length(image_names)
    image = imread(fullfile('images',image_names{h}));
    
    mask = imread(fullfile('masks',mask_names{h}));
    mask = uint16(mask);
    mask(mask == 1) = 1500;
    
    imwrite(mask,strcat('CycleGAN/testA/',image_names{h}))
    imwrite(image,strcat('CycleGAN/testB/',image_names{h}))
end