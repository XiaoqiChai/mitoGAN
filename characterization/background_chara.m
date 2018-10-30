% Script to characterize the background noise with autocorrelation
% The coordinates of the backgrounds have been defined in advance
% The size of each background is 101*101
% To use this script, make sure that corresponding images in different folders have the same filename

clear
clc
close all

load background_coordinates.mat backgrounds inds

gt_dir = dir(fullfile('real_images_test','*.tif'));

B = [];

image_names = {gt_dir.name};

% Choose the backgrounds and characterize them
for i = [3,4,13]
    
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
    
    % Subtract mean and normalize the data
    back_gt = back_gt - mean(back_gt(:));
    back_pred_cgan = back_pred_cgan - mean(back_pred_cgan(:));
    back_pred_pgan = back_pred_pgan - mean(back_pred_pgan(:));
    back_sim = back_sim - mean(back_sim(:));
    
    back_gt = back_gt/norm(back_gt,'fro');
    back_pred_cgan = back_pred_cgan/norm(back_pred_cgan,'fro');
    back_pred_pgan = back_pred_pgan/norm(back_pred_pgan,'fro');
    back_sim = back_sim/norm(back_sim,'fro');
    
    % Perform correlation analysis
    b1 = xcorr2(back_gt);
    b2 = xcorr2(back_pred_cgan);
    b3 = xcorr2(back_pred_pgan);
    b4 = xcorr2(back_sim);
    
    sizes = size(b1);
    
    % Concatenate the matrices for visualization
    b = [b1,1*ones(sizes(1),15),b4,1*ones(sizes(1),15),b3,1*ones(sizes(1),15),b2];
    
    B = [B;1*ones(15,4*sizes(1)+45);b]; %#ok<AGROW>
end

% Visualize the result
B = B(16:end,:);
imagesc(B)
colorbar
caxis([min(B(:)),0.5*max(B(:))])
axis('equal')
axis off
colormap('hot')