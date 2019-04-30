% Script to calculate error matrices
% 3 different criteria are taken into consideration: SSIM, PSNR and NCC
% To use this script, make sure that corresponding images in different folders have the same filename

clear
clc
close all

warning('off')

% load files
pre_dir_test_cgan = dir(fullfile('test_predicted_cgan','*.tif'));
pre_dir_test_pgan = dir(fullfile('test_predicted_pgan','*.tif'));
sim_dir = dir(fullfile('sim_images_test','*.tif'));
ref_dir_test = dir(fullfile('real_images_test','*.tif'));
test_names = {pre_dir_test_cgan.name};

% initialize matrices
ssim_test_cgan = zeros(length(test_names),1);
ssim_test_pgan = zeros(length(test_names),1);
ssim_test_sim = zeros(length(test_names),1);

psnr_test_cgan = zeros(length(test_names),1);
psnr_test_pgan = zeros(length(test_names),1);
psnr_test_sim = zeros(length(test_names),1);

cross_test_cgan = zeros(length(test_names),1);
cross_test_pgan = zeros(length(test_names),1);
cross_test_sim = zeros(length(test_names),1);

% read test image and calculate matrices
for i = 1:length(test_names)
    
    im_gt = imread(fullfile('real_images_test',test_names{i}));
    im_pre_cgan = imread(fullfile('test_predicted_cgan',test_names{i}));
    im_pre_pgan = imread(fullfile('test_predicted_pgan',test_names{i}));
    im_pre_sim = imread(fullfile('sim_images_test',test_names{i}));
    
    img_size = size(im_gt);
    
    [ssimval,peaksnr,corr] = compMat(im_gt,im_pre_cgan);
    
    ssim_test_cgan(i) = ssimval;
    psnr_test_cgan(i) = peaksnr;
    cross_test_cgan(i) = corr;
    
    [ssimval,peaksnr,corr] = compMat(im_gt,im_pre_pgan);
    
    ssim_test_pgan(i) = ssimval;
    psnr_test_pgan(i) = peaksnr;
    cross_test_pgan(i) = corr;
    
    [ssimval,peaksnr,corr] = compMat(im_gt,im_pre_sim);
    
    ssim_test_sim(i) = ssimval;
    psnr_test_sim(i) = peaksnr;
    cross_test_sim(i) = corr;
    
end

% calculate the p-values
[~,pssim1] = ttest2(ssim_test_cgan,ssim_test_sim);
[~,pssim2] = ttest2(ssim_test_pgan,ssim_test_sim);
[~,pssim3] = ttest2(ssim_test_pgan,ssim_test_cgan);

[~,ppsnr1] = ttest2(psnr_test_cgan,psnr_test_sim);
[~,ppsnr2] = ttest2(psnr_test_pgan,psnr_test_sim);
[~,ppsnr3] = ttest2(psnr_test_pgan,psnr_test_cgan);

[~,pcorr1] = ttest2(cross_test_cgan,cross_test_sim);
[~,pcorr2] = ttest2(cross_test_pgan,cross_test_sim);
[~,pcorr3] = ttest2(cross_test_pgan,cross_test_cgan);

p_values = [pssim1,pssim2,pssim3;ppsnr1,ppsnr2,ppsnr3;pcorr1,pcorr2,pcorr3];

% organize the results and save them into csv files
results_cgan = [ssim_test_cgan';psnr_test_cgan';cross_test_cgan'];
stats_cgan = [mean(ssim_test_cgan),std(ssim_test_cgan),mean(psnr_test_cgan),std(psnr_test_cgan),...
    mean(cross_test_cgan),std(cross_test_cgan)];

results_pgan = [ssim_test_pgan';psnr_test_pgan';cross_test_pgan'];
stats_pgan = [mean(ssim_test_pgan),std(ssim_test_pgan),mean(psnr_test_pgan),std(psnr_test_pgan),...
    mean(cross_test_pgan),std(cross_test_pgan)];

results_sim = [ssim_test_sim';psnr_test_sim';cross_test_sim'];
stats_sim = [mean(ssim_test_sim),std(ssim_test_sim),mean(psnr_test_sim),std(psnr_test_sim),...
    mean(cross_test_sim),std(cross_test_sim)];

% csvwrite('matric_cgan_v2.csv',results_cgan)
% csvwrite('matric_pgan_v2.csv',results_pgan)
% csvwrite('matric_sim_v2.csv',results_sim)
% csvwrite('stats_v2.csv',[stats_cgan;stats_pgan;stats_sim]);