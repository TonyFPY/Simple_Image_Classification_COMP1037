% Initialization
clear ; close all; clc

% load the address of file 'animal'
file_path = '.\animals\';

% use internal functions to get a list of adresses of images
img_path_list = dir(fullfile(file_path,'*.jpg'));
img_num = length(img_path_list); % 3000

% read the image one by one
% if-statement is used to detect whether the images are read
X = zeros(40000,3000); % declare the size of the matrix
if img_num > 0 
    for i = 1 : img_num
        image_name = img_path_list(i).name;
        image = imread(fullfile(file_path,image_name));
        % fprintf('%d %s\n',i,strcat(image_name)); % code for debugging: print the current processing image
        
        % convert all RGB images into grayscale images
        grey_image = rgb2gray(image); 

        % standardize the images
        resized_image = imresize(grey_image,[200,200]);
        % figure; imshow(image);         % code for debugging
        % figure; imshow(grey_image);    % code for debugging
        % figure; imshow(resized_image); % code for debugging

        % reshape the image from 2D into 1D array wit
        X(:,i) = reshape(resized_image,[],1); 

        % vectorize the corresponding labels data into 1D
        if contains(image_name,'cat')
            y(: , i) = [1;0;0];
        elseif contains(image_name,'dog')
            y(: , i) = [0;1;0];
        else
            y(: , i) = [0;0;1];
        end
    end
end

% save variables
fprintf("X and y are successfully saved!\n");
save('data_efficiency.mat', 'X', 'y');




