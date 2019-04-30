% Script to calculate error matrices
% 3 different criteria are taken into consideration: SSIM, PSNR and NCC
% To use this script, make sure that corresponding images in different folders have the same filename

clear
clc
close all

%% Section to calculate NFM
load background_coordinates.mat backgrounds inds
gt_dir = dir(fullfile('real_images_test','*.tif'));

image_names = {gt_dir.name};

distance_cgan_all = zeros(length(inds),1);
distance_pgan_all = zeros(length(inds),1);
distance_sim_all = zeros(length(inds),1);

% Choose the backgrounds and characterize them
for i = 1:length(inds)
    
    % Read the images
    gt = imread(fullfile('real_images_test',image_names{inds(i)}));
    pred_cgan = imread(fullfile('test_predicted_cgan',image_names{inds(i)}));
    pred_pgan = imread(fullfile('test_predicted_pgan',image_names{inds(i)}));
    sim = imread(fullfile('sim_images_test',image_names{inds(i)}));
    
    % Crop the background
    back_gt = imcrop(gt,backgrounds(i,:));
    back_pred_cgan = imcrop(pred_cgan,backgrounds(i,:));
    back_pred_pgan = imcrop(pred_pgan,backgrounds(i,:));
    back_sim = imcrop(sim,backgrounds(i,:));
    
    back_gt = double(back_gt);
    back_pred_cgan = double(back_pred_cgan);
    back_pred_pgan = double(back_pred_pgan);
    back_sim = double(back_sim);
    
    % Subtract Sorensen distance of the background
    distance_cgan = calculatePdfAreaNonOverlap(back_gt(:),back_pred_cgan(:));
    distance_pgan = calculatePdfAreaNonOverlap(back_gt(:),back_pred_pgan(:));
    distance_sim = calculatePdfAreaNonOverlap(back_gt(:),back_sim(:));
    
    distance_cgan_all(i) = distance_cgan;
    distance_pgan_all(i) = distance_pgan;
    distance_sim_all(i) = distance_sim;
end

% Calculate the p-values
[~,pCGnfm] = kstest2(distance_cgan_all,distance_pgan_all);
[~,pCSnfm] = kstest2(distance_cgan_all,distance_sim_all);
[~,pGSnfm] = kstest2(distance_pgan_all,distance_sim_all);

%% Section to calculate BFM
sharp_gt_all = zeros(length(image_names),1);
sharp_cgan_all = zeros(length(image_names),1);
sharp_pgan_all = zeros(length(image_names),1);
sharp_sim_all = zeros(length(image_names),1);

for i = 1:length(image_names)
    
    % Read the images
    im_gt = imread(fullfile('real_images_test',image_names{i}));
    im_pre_cgan = imread(fullfile('test_predicted_cgan',image_names{i}));
    im_pre_pgan = imread(fullfile('test_predicted_pgan',image_names{i}));
    im_pre_sim = imread(fullfile('sim_images_test',image_names{i}));
    
    img_size = size(im_gt);
    
    % Extract image sharpness of each individual images
    [~,sharp_gt] = sharpMeasure(im_gt);
    [~,sharp_cgan] = sharpMeasure(im_pre_cgan);
    [~,sharp_pgan] = sharpMeasure(im_pre_pgan);
    [~,sharp_sim] = sharpMeasure(im_pre_sim);
    
    sharp_gt_all(i) = sharp_gt; %#ok<*SAGROW>
    sharp_cgan_all(i) = sharp_cgan;
    sharp_pgan_all(i) = sharp_pgan;
    sharp_sim_all(i) = sharp_sim;
    
end

% Calculate the ratio between the sharpness of simulation images and ground truth
relative_sharp_cgan = sharp_cgan_all./sharp_gt_all;
relative_sharp_pgan = sharp_pgan_all./sharp_gt_all;
relative_sharp_sim = sharp_sim_all./sharp_gt_all;

% Calculate the p-values
[~,pCGbfm] = kstest2(relative_sharp_cgan,relative_sharp_pgan);
[~,pCSbfm] = kstest2(relative_sharp_cgan,relative_sharp_sim);
[~,pGSbfm] = kstest2(relative_sharp_pgan,relative_sharp_sim);


%% Calculate basic statistics
%{
results_cgan = [ssim_test_cgan';psnr_test_cgan';cross_test_cgan'];
stats_cgan = [mean(ssim_test_cgan),std(ssim_test_cgan),mean(psnr_test_cgan),std(psnr_test_cgan),...
    mean(cross_test_cgan),std(cross_test_cgan)];

results_pgan = [ssim_test_pgan';psnr_test_pgan';cross_test_pgan'];
stats_pgan = [mean(ssim_test_pgan),std(ssim_test_pgan),mean(psnr_test_pgan),std(psnr_test_pgan),...
    mean(cross_test_pgan),std(cross_test_pgan)];

results_sim = [ssim_test_sim';psnr_test_sim';cross_test_sim'];
stats_sim = [mean(ssim_test_sim),std(ssim_test_sim),mean(psnr_test_sim),std(psnr_test_sim),...
    mean(cross_test_sim),std(cross_test_sim)];
%}

xlswrite('Sorensen_distance.xlsx',[distance_cgan_all,distance_pgan_all,distance_sim_all])
xlswrite('Blurring_sharpness.xlsx',[relative_sharp_cgan,relative_sharp_pgan,relative_sharp_sim])
