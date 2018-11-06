% Script to make the dataset for training CycleGAN

%% Make the dataset for CycleGAN
clear
clc

image_dir = dir(fullfile('dataset','images','*.tif'));
mask_dir = dir(fullfile('dataset','masks','*.tif'));

image_names = {image_dir.name};
mask_names = {mask_dir.name};

mkdir('CycleGAN/CycleGAN')

mkdir('CycleGAN/CycleGAN/trainA')
mkdir('CycleGAN/CycleGAN/trainB')

mkdir('CycleGAN/CycleGAN/testA')
mkdir('CycleGAN/CycleGAN/testB')

mkdir('CycleGAN/CycleGAN/candidateA')
mkdir('CycleGAN/CycleGAN/candidateB')

mkdir('CycleGAN/CycleGAN/valA')
mkdir('CycleGAN/CycleGAN/valB')


for i = 1:500
    
    image = imread(fullfile('dataset','images',image_names{i}));
    
    mask = imread(fullfile('dataset','masks',mask_names{i}));
    mask = uint16(mask);
    mask(mask == 1) = 1500;
    
     imwrite(mask,strcat('CycleGAN/CycleGAN/trainA/',image_names{i}))
     imwrite(image,strcat('CycleGAN/CycleGAN/trainB/',image_names{i}))
    
end

for j = 501:1100
    
    image = imread(fullfile('dataset','images',image_names{j}));
    
    mask = imread(fullfile('dataset','masks',mask_names{j}));
    mask = uint16(mask);
    mask(mask == 1) = 1500;
    
    imwrite(mask,strcat('CycleGAN/CycleGAN/candidateA/',image_names{j}))
    imwrite(image,strcat('CycleGAN/CycleGAN/candidateB/',image_names{j}))
end

for k = 1101:1200
    image = imread(fullfile('dataset','images',image_names{k}));
    
    mask = imread(fullfile('dataset','masks',mask_names{k}));
    mask = uint16(mask);
    mask(mask == 1) = 1500;
    
    imwrite(mask,strcat('CycleGAN/CycleGAN/valA/',image_names{k}))
    imwrite(image,strcat('CycleGAN/CycleGAN/valB/',image_names{k}))
end

for h = 1201:length(image_names)
    image = imread(fullfile('dataset','images',image_names{h}));
    
    mask = imread(fullfile('dataset','masks',mask_names{h}));
    mask = uint16(mask);
    mask(mask == 1) = 1500;
    
    imwrite(mask,strcat('CycleGAN/CycleGAN/testA/',image_names{h}))
    imwrite(image,strcat('CycleGAN/CycleGAN/testB/',image_names{h}))
end

%% Make the dataset for pix2pix
clear
clc

image_dir = dir(fullfile('dataset','images','*.tif'));
mask_dir = dir(fullfile('dataset','masks','*.tif'));

image_names = {image_dir.name};
mask_names = {mask_dir.name};

mkdir('pix2pix/pix2pix')

mkdir('pix2pix/pix2pix/train')
mkdir('pix2pix/pix2pix/val')
mkdir('pix2pix/pix2pix/candidateA')
mkdir('pix2pix/pix2pix/candidateB')
mkdir('pix2pix/pix2pix/testA')
mkdir('pix2pix/pix2pix/testB')

for i = 1:500
    
    image = imread(fullfile('dataset','images',image_names{i}));
    
    mask = imread(fullfile('dataset','masks',mask_names{i}));
    mask = uint16(mask);
    mask(mask == 1) = 1500;
    
    imwrite([image,mask],strcat('pix2pix/pix2pix/train/',image_names{i}))
    
end

for j = 501:1100
    
    image = imread(fullfile('dataset','images',image_names{j}));
    
    mask = imread(fullfile('dataset','masks',mask_names{j}));
    mask = uint16(mask);
    mask(mask == 1) = 1500;
    
    imwrite(mask,strcat('pix2pix/pix2pix/candidateA/',image_names{j}))
    imwrite(image,strcat('pix2pix/pix2pix/candidateB/',image_names{j}))
    
end

for k = 1101:1200
    
    image = imread(fullfile('dataset','images',image_names{k}));
    
    mask = imread(fullfile('dataset','masks',mask_names{k}));
    mask = uint16(mask);
    mask(mask == 1) = 1500;
    
    imwrite([image,mask],strcat('pix2pix/pix2pix/val/',image_names{k}))
end

for h = 1201:length(image_names)
    
    image = imread(fullfile('dataset','images',image_names{h}));
    
    mask = imread(fullfile('dataset','masks',mask_names{h}));
    mask = uint16(mask);
    mask(mask == 1) = 1500;
    
    imwrite(mask,strcat('pix2pix/pix2pix/testA/',image_names{h}))
    imwrite(image,strcat('pix2pix/pix2pix/testB/',image_names{h}))
    
end