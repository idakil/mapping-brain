
% volume repetition 2.5s
%In each run, the subjects passively viewed greyscale
%images of eight object categories, grouped in 24s blocks separated by rest
%periods. Each image was shown for 500ms and was followed by a 1500ms
%inter-stimulus interval.  Full-brain fMRI data were recorded with a volume
%repetition time of 2.5s, thus, a stimulus block was covered by roughly 9
%volumes.

%% Read data

bold = niftiread("data\subj1\bold.nii.gz");
labels = readtable("data\subj1\labels.txt", "Delimiter", " "); %1452
hrf = load("hrf.mat");
vt = niftiread("data\subj1\mask4_vt.nii.gz");
face = niftiread("data\subj1\mask8_face_vt.nii.gz");
house = niftiread("data\subj1\mask8_house_vt.nii.gz");
faceb = niftiread("data\subj1\mask8b_face_vt.nii.gz");
houseb = niftiread("data\subj1\mask8b_house_vt.nii.gz");

%% 
% Get categories from labels
categories = unique(labels(:, "labels").labels, 'stable');
% ?? Remove 'rest' from categories
categories(strcmp(categories, "rest")) = []; 
% Transform to string array
categories = string(categories);
nCategories = length(categories);
% Number of timepoints (1452)
timePoints = size(bold, 4);
% Initialize design matrix of size 1452*9
designMatrix = zeros(timePoints, nCategories);

boxcar = zeros(timePoints,nCategories);
for c = 1:length(categories)
    idx = find(strcmp(labels.labels, categories(c))); % indexes of category
    for i = 1:length(idx)
        boxcar(idx(i), c) = 1; %set to one when this category is shown
    end
end
for i = 1:length(categories)
    b = conv(boxcar(:, i), hrf.hrf_sampled);
    designMatrix(:, i) = b(1:1452,:);
end
figure;
subplot(2,1, 1)
imagesc(boxcar)
title('Boxcar')
xticklabels(categories)
subplot(2,1, 2)
imagesc(designMatrix)
title('Design matrix')
colorbar
xticklabels(categories)

%% GLM for single voxel
% single = squeeze(bold(20,20,20,:));
% 
% %coefficient estimates, deviance of fit, model statistics
% [b, dev, stats] = glmfit(designMatrix, double(single));
% tValues = stats.t;
% imagesc(tValues);
% %% GLM voxel by voxel
% [betaMaps, tMaps] = fitData(nCategories, bold, designMatrix);
% %% Plot
% figure;
% title("T values, slice 20")
% for i = 1:nCategories
%     subplot(2, 4, i);                 
%     imagesc(squeeze(tMaps(:,:,40,i)));
%     colormap('hot');
%     title([categories(i)]);
% end
% %%
% figure;
% title("B values, slice 20")
% for i = 1:nCategories
%     subplot(2, 4, i);                 
%     imagesc(squeeze(betaMaps(:,:,20,i)));
%     colormap('hot');                  
%     title([categories(i)]);
% end

%% 
[~, tM, ~] = fitData(8, bold, designMatrix);
%% Plot T maps with no regressors
figure;
for i = 1:nCategories
    subplot(2, 4, i);                 
    imagesc(squeeze(tM(:,:,28,i)));
    title(categories(i))
end
colorbar
%% Extra regressors

% constant terms columns of ones and zeros?

blockMatrix = zeros(1452, 12);
blockSize = 121;
numBlocks = size(blockMatrix, 2); 

for i = 1:numBlocks
    rowStart = (i - 1) * blockSize + 1; 
    rowEnd = i * blockSize;       
    blockMatrix(rowStart:rowEnd, i) = 1;
end
designMatrixConst = [designMatrix, blockMatrix];
imagesc(designMatrixConst)
colormap('gray')
title('Design matrix')

%% GLM with extra regressors
[betaMaps, tMaps, varMap] = fitData(20, bold, designMatrixConst);
%% Plot beta maps
figure;
title(categories(3))
for i = 1:40
    subplot(6, 7, i);                 
    imagesc(squeeze(betaMaps(:,:,i,3)), [-20, 20]);
end

%% 
figure;
title(categories(3))
j = 1;
for i = 20:39
    subplot(4, 5, j);                 
    imagesc(squeeze(tMaps(:,:,i,3)), [-10,10]);
    title(i)
    colorbar
    j = j+1;
end
%% Plot T maps for all categories, slice 28
figure;
for i = 1:nCategories
    subplot(2, 4, i);                 
    imagesc(squeeze(tMaps(:,:,28,i)), [-10, 10]);
    title(categories(i))
end
colorbar

%% Contrast
cMap = zeros(size(bold));
c1 = categoryIndex(categories, "house");
c2 = categoryIndex(categories, "face");

for x = 1:size(bold, 1)
    for y = 1:size(bold, 2)
        for z = 1:size(bold, 3)
            voxelTimeSeries = squeeze(double(bold(x, y, z, :))); 
            c = zeros(1,20);
            c(c1) = -1;
            c(c2) = 1;
            cov = c*(voxelTimeSeries'*voxelTimeSeries)^-1*c';
            cMap(x,y,z)= c*squeeze(betaMaps(x,y,z,:))/sqrt(varMap(x,y,z)*cov);
        end
    end
end
%% Plot
figure;
j = 1;
for i = 20:39
    subplot(5, 4, j);
    imagesc(cMap(:,:,i), [-10,10]);
    %colormap('gray');   
    title(i)
    j = j+1;
    colorbar
end

%% ROI

roi_mask_expanded = repmat(house, [1, 1, 1, size(bold, 4)]); % [40, 64, 64, 1452]
masked_data = double(bold) .* double(roi_mask_expanded);

[betaMasked, tMasked, varMasked] = fitData(20, masked_data, designMatrixConst);
%% Plot tMaps for house mask vs face
cats = [categoryIndex(categories, "house"), categoryIndex(categories, "face")];
figure;
for i = 1:length(cats) 
    subplot(1,2,i)
    imagesc(squeeze(tMasked(:,:,28,cats(i))), [0, 1]);
    title(categories(cats(i)))
end

%%
c = categoryIndex(categories, "face");
for x = 1:size(bold, 1)
    for y = 1:size(bold, 2)
        for z = 1:size(bold, 3)
            changes = betaMaps(x,y,z,c)
        end
    end
end

%% Generate random masks
randomOut = randi([0,1], [40, 64, 64]);
roi_mask_random = repmat(randomOut, [1, 1, 1, size(bold, 4)]);
masked_random = double(bold) .* double(roi_mask_random);
[betaRandom, tRandom, varRandom] = fitData(20, masked_random, designMatrixConst);

%% GLM voxel by voxel
function [betaMaps, tMaps, varMap] = fitData(nCategories, bold, designMatrix)

betaMaps = zeros(size(bold, 1), size(bold, 2), size(bold, 3), nCategories);
tMaps = zeros(size(bold, 1), size(bold, 2), size(bold, 3), nCategories);
varMap = zeros(size(bold, 1), size(bold, 2), size(bold, 3));

% Loop through each voxel
for x = 1:size(bold, 1)
    for y = 1:size(bold, 2)
        for z = 1:size(bold, 3)
            voxelTimeSeries = squeeze(bold(x, y, z, :)); 
            [b, ~, stats] = glmfit(double(designMatrix), double(voxelTimeSeries), 'normal', 'constant', 'off'); 
            betaMaps(x, y, z,:) = b;
            tMaps(x, y, z,:) = stats.t;
            varMap(x,y,z) = var(stats.resid);
        end
    end
end
end

%% Find category index
function c = categoryIndex(categories, category)
c = find(strcmp(categories, category));
end