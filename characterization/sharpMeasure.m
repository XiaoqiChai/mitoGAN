% Function to calculate image sharpness over the whole iamge
% Takes in one arguments: img as the grayscale image
% Returns pixel-wise gradient map and overall image sharpness correspondingly

function [sharpnessGradimap,sharpness] = sharpMeasure(img)

siz = size(img,1)*size(img,2);

% Define Laplacian filter and extract gradient information
h = [[1,4,1];[4,-20,4];[1,4,1]];
gradimap = conv2(img,h,'same');

% Extract pixel-wise gradient information and overall image sharpness
sharpnessGradimap = double(abs(gradimap));
sharpness = sum(sharpnessGradimap(:))/siz;

end