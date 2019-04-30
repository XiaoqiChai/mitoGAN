% Function to compute area non-overlap of two distributions
% Takes in two arguments: dis1 as the first distribution and dis2 as the second distribution
% Returns the area of the non-overlap regions of two distributions

function areaNonOverlap = calculatePdfAreaNonOverlap(dis1,dis2)

% evaluate distribution (pdf) at 200 points
nKsDensityPts = 200; 
     
maxDistance = max(max(dis1),max(dis2));

% Points at which distributions are evaluated
ksDensityPts = linspace(0,maxDistance,nKsDensityPts); 
[pdf1,~] = ksdensity(dis1,ksDensityPts);
[pdf2,~] = ksdensity(dis2,ksDensityPts); 

% Calculate the area of the overlap between two distributions
areaNonOverlapPair = trapz(ksDensityPts,abs(pdf1-pdf2));
areaSum = trapz(ksDensityPts,pdf1) + trapz(ksDensityPts,pdf2);
areaNonOverlap = areaNonOverlapPair/(areaSum);
     
end