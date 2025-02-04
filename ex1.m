

%% Read data

bold = niftiread("data\subj1\bold.nii.gz");
labels = readtable("data\subj1\labels.txt", "Delimiter", " "); %1452
hrf = load("hrf.mat");
vt_mask = niftiread("data\subj1\mask4_vt.nii.gz");
face_mask = niftiread("data\subj1\mask8_face_vt.nii.gz");
house_mask = niftiread("data\subj1\mask8_house_vt.nii.gz");
faceb_mask = niftiread("data\subj1\mask8b_face_vt.nii.gz");
houseb_mask = niftiread("data\subj1\mask8b_house_vt.nii.gz");

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

roi_mask_expanded = repmat(house_mask, [1, 1, 1, size(bold, 4)]); % [40, 64, 64, 1452]
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
roi_centre = [28 30 40];
roi_radius = 5;

[X,Y,Z] = ndgrid(1:size(vt_mask,1),1:size(vt_mask,2),1:size(vt_mask,3));
mask_random_in = sqrt((X-roi_centre(1)).^2+(Y-roi_centre(2)).^2+(Z-roi_centre(3)).^2)<roi_radius;

figure;
for ind = 1:size(vt_mask,1)
    subplot(4,10,ind)
    imagesc(squeeze(mask_random_in(ind,:,:)))
end

roi_centre = [28 50 50];
roi_radius = 5;

[X,Y,Z] = ndgrid(1:size(vt_mask,1),1:size(vt_mask,2),1:size(vt_mask,3));
mask_random_out = sqrt((X-roi_centre(1)).^2+(Y-roi_centre(2)).^2+(Z-roi_centre(3)).^2)<roi_radius;

figure;
for ind = 1:size(vt_mask,1)
    subplot(4,10,ind)
    imagesc(squeeze(mask_random_out(ind,:,:)))
    title(ind)
end

%% 

figure
subplot(2,1,1);
for ind = 1:nCategories
    plot(designMatrixConst(:, ind)); hold on;
end
subplot(2,1,2);
%data = reshape(masked_data, [], 1452);
%y = mean(data, 1);
%plot(y, 'b');
beta_reshaped = reshape(betaMasked, [], 20)';
disp(size(beta_reshaped))
disp(size(designMatrixConst))
y_pred = designMatrixConst * beta_reshaped;
plot(mean(y_pred,1), 'r')
%% 
roi_mask_random = repmat(roi_outside, [1, 1, 1, size(bold, 4)]);
disp(size(roi_mask_random))
disp(size(bold))
masked_random = double(bold) .* double(roi_mask_random);
[betaRandom, tRandom, varRandom] = fitData(20, masked_random, designMatrixConst);
%%
figure;
for i = 1:20
    subplot(5, 4, i);                 
    imagesc(squeeze(tRandom(:,:,i,1)));
end
colorbar
%% PART 3

even_mask = mod(1:1452, 2) == 0; 
odd_mask = ~even_mask; 

% divide data to train and test sets
train = bold(:,:,:,even_mask);
test = bold(:,:,:,odd_mask);

% divide desing matrix
constants = zeros(726, 6);

% Logical masks for even and odd rows
even_mask = mod(1:size(designMatrix, 1), 2) == 0; 
odd_mask = ~even_mask; 

% Split the data
d_train = designMatrix(even_mask, :);
d_test = designMatrix(odd_mask, :); 

blockSize = 121;
numBlocks = size(constants, 2); 

for i = 1:numBlocks
    rowStart = (i - 1) * blockSize + 1; 
    rowEnd = i * blockSize;       
    constants(rowStart:rowEnd, i) = 1;
end
d_trainc = [d_train constants];
d_testc = [d_test constants];
imagesc(d_testc)

%% GLM train and test data
[bTrain, tTrain, ~] = fitData(14, train, d_trainc);
[bTest, tTest, ~] = fitData(14, test, d_testc);
%%
data = bold; 
total_runs = size(data, 4);

% Preallocate to store results

within_corrs = zeros(total_runs, 1); 
between_corrs = zeros(total_runs, 1); 
predictions = zeros(total_runs, 1);  


%     
%     test_mask = false(1, total_runs);
%     test_mask(test_run) = true;  % testing
    
    % Split the data
%     test_data = data(:, :, :, test_mask);  % Test set (one run)
%     test_label = lbs(test_mask);        % Category label for the test run
%     train_data = data(:, :, :, ~test_mask); % Training set (all other runs)
%     train_labels = lbs(~test_mask);     % Training labels
% 
%     test_vector = reshape(test_data, [], 1);  
%     train_vectors = reshape(train_data, [], sum(~test_mask));
disp(size(squeeze(tTest(:,:,:,1))))
d = reshape(tTrain, size(tTrain,4), []);
dt = reshape(tTest, size(tTest,4), []);



var_train = var(d,0,2);
disp(var_train)
corr_matrix = corr(t_map_odd_clean', t_map_even_clean')
figure;
imagesc(corr_matrix);
colorbar;
title('Pattern Correlation Between Odd and Even Runs');
xlabel('Stimulus Categories (Even)');
ylabel('Stimulus Categories (Odd)');
axis square;
% for ind= 1:nCategories
%     correlations = corr(squeeze(tTest(:,:,:,ind)), squeeze(tTrain(:,:,:,ind)));
% 
% end

% within_corr = mean(correlations(strcmp(train_labels,test_label))); 
% between_corr = mean(correlations(~strcmp(train_labels, test_label)));
% 
% within_corrs(test_run) = within_corr;
% between_corrs(test_run) = between_corr;
% 
% % Classification: Predict the category with the highest mean correlation
% pred_corrs = arrayfun(@(cat) mean(correlations(train_labels == cat)), categories);
% [~, pred_idx] = max(pred_corrs);  % Category with the highest correlation
% 
% % FIX! Gives only NaN

%%
accuracy = mean(strcmp(predictions,lbs));
figure;
bar([mean(within_corrs), mean(between_corrs)]);
set(gca, 'XTickLabel', {'Within-Category', 'Between-Category'});
ylabel('Mean Correlation');
title('Within vs. Between Category Correlations');
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