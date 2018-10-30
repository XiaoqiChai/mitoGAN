% Function to compute image similarity
% Takes in two arguments: gt as the reference image and img as the image to be tested
% Returns SSIM, PSNR and NCC correspondingly

function [ssimval,peaksnr,corr] = compMat(gt,img)

% calculate SSIM and PSNR
[ssimval,~] = ssim(gt,img);
[peaksnr,~] = psnr(gt,img);

% calculate NCC
dgt = double(gt);
dimg = double(img);

dgt = dgt/norm(dgt,'fro');
dimg = dimg/norm(dimg,'fro');

corr = xcorr2(dgt,dimg);
corr = corr(256,256);

end