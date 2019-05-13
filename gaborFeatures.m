function featureVector = gaborFeatures(img,gaborArray,d1,d2)




if (nargin ~= 4)        % Check correct number of arguments
    error('Please use the correct number of input arguments!')
end

if size(img,3) == 3     % Check if the input image is grayscale
    warning('The input RGB image is converted to grayscale!')
    img = rgb2gray(img);
end

img = double(img);


%% Filter the image using the Gabor filter bank

% Filter input image by each Gabor filter
[u,v] = size(gaborArray);
gaborResult = cell(u,v);
for i = 1:u
    for j = 1:v
        gaborResult{i,j} = imfilter(img, gaborArray{i,j});
    end
end


%% Create feature vector

% Extract feature vector from input image
featureVector = [];
for i = 1:u
    for j = 1:v
        
        gaborAbs = abs(gaborResult{i,j});
        gaborAbs = downsample(gaborAbs,d1);
        gaborAbs = downsample(gaborAbs.',d2);
        gaborAbs = gaborAbs(:);
        
        % Normalized to zero mean and unit variance. (if not applicable, please comment this line)
        gaborAbs = (gaborAbs-mean(gaborAbs))/std(gaborAbs,1);
        
        featureVector =  [featureVector; gaborAbs];
        
    end
end


%% Show filtered images (Please comment this section if not needed!)

% % Show real parts of Gabor-filtered images
% figure('NumberTitle','Off','Name','Real parts of Gabor filters');
% for i = 1:u
%     for j = 1:v        
%         subplot(u,v,(i-1)*v+j)    
%         imshow(real(gaborResult{i,j}),[]);
%     end
% end
% 
% % Show magnitudes of Gabor-filtered images
% figure('NumberTitle','Off','Name','Magnitudes of Gabor filters');
% for i = 1:u
%     for j = 1:v        
%         subplot(u,v,(i-1)*v+j)    
%         imshow(abs(gaborResult{i,j}),[]);
%     end
% end
