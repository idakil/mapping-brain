%% GLM voxel by voxel
function [betaMaps, tMaps] = glm(nCategories, bold, designMatrix)

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
            varMap(x,y,z,:) = var(stats.resid);
        end
    end
end
end