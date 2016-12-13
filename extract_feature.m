%% Example use of the DDet
addpath(genpath('/Users/Xu/tool/vlfeat-0.9.20/toolbox/'));
%addpath(genpath('/Users/Xu/tool/matconvnet-1.0-beta15/matlab/'));
setup();

%% Detect the features
dataset_name = 'vggAffineDataset';
subsets = { 'bikes', 'trees', 'graf',' boat', 'bark', 'wall', 'leuven', 'ubc'};
%subsets = {'bark'};
%dataset_name = 'EFDataset';
%subsets = {'notredame','obama','paintedladies','rushmore','yosemite'};
%dataset_name = 'WebcamDataset';
%subsets = {'Chamonix','Courbevoie','Frankfurt','Mexico','Panorama','StLouis'};
%subsets = {'notredame'};
dir_name = ['/Users/Xu/program/Image_Genealogy/code/Covariant_Feature_Detection/eval/' ...
        'vlbenchmakrs/vlbenchmakrs-1.0-beta/data/'];
    
net_name = 'detnet_s4.mat';
net = dagnn.DagNN.loadobj(load(fullfile('nets', net_name)));

% Uncomment the following lines to compute on a GPU
% (works only if MatConvNet compiled with GPU support)
% gpuDevice(1);  net.move('gpu');

detector = DDet(net, 'thr', 0);
radius = 10;
point_number = 500;
pyramid_level = 5;
for set_index = 1:numel(subsets)
    subset = subsets{set_index};
    disp(set_index);
    image_list = load_image_list([dir_name 'datasets/' dataset_name '/'], subset);
    [s, mess, messid] = mkdir([dir_name 'ddet_feature_point/' dataset_name '/' subset '/']);
    for i = 1:numel(image_list)
        image = imread([dir_name 'datasets/' dataset_name '/' subset '/' image_list(i).name]);
        if(size(image,3) == 1)
                image = repmat(image, [1 1 3]);
        end
        factor = 1;
        feature = [];
        for j = 1:pyramid_level
            [features, ~, info] = detector.detect(image,point_number/factor^2);

            feature_t = zeros(6,size(features,2));
            feature_t(1,:) = radius;
            feature_t(5,:) = radius;
            feature_t(3,:) = features(1,:);
            feature_t(6,:) = features(2,:);
            feature_t = feature_t'*factor;
            
            if isempty(feature)
                feature = feature_t;
            else
                feature = [feature;feature_t];
            end

            %# Create the gaussian filter with hsize = [5 5] and sigma = 2
            G = fspecial('gaussian',[5 5],sqrt(2));
            %# Filter it
            image = imfilter(image,G,'same');
            image = imresize(image, 1/sqrt(2));
            factor = factor*sqrt(2);
        end
        disp(size(feature,1));
        save([dir_name 'ddet_feature_point/' dataset_name '/' subset '/' image_list(i).name(1:end-4) '.mat'],'feature');
    end

end
