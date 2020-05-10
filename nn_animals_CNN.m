% Load data
fprintf('Loading and Visualizing Data ...\n')
load('data_efficiency.mat');

% Prepare data

X_4D = reshape(X',200,200,1,3000);
len = size(y, 2);
for i = 1 : len
    if y(: , i) == [1;0;0]
        new_y(i) = 1;
    elseif y(: , i) == [0;1;0]
        new_y(i) = 2;
    else
        new_y(i) = 3;
    end
end
rand_indices = randperm(size(X_4D, 4));

trainData = X_4D(:,:,:,rand_indices(1:1800));
valData = X_4D(:,:,:,rand_indices(1801:2400));
testData = X_4D(:,:,:,rand_indices(2401:end));
trainLabels = new_y(rand_indices(1:1800))';
valLabels = new_y(rand_indices(1801:2400))';
testLabels = new_y(rand_indices(2401:end))';

% Define network layers
layers = [...
            imageInputLayer([200 200 1])
            convolution2dLayer(3,16,'Padding',1)
            batchNormalizationLayer
            reluLayer    
            maxPooling2dLayer(2,'Stride',2) 

            convolution2dLayer(3,32,'Padding',1)
            batchNormalizationLayer
            reluLayer 

            fullyConnectedLayer(3)
            softmaxLayer
            classificationLayer,...
         ];

% Customize training option
options = trainingOptions('sgdm', ...
        'ValidationData',{valData,categorical(valLabels)},...
        'MaxEpochs',10,...
        'InitialLearnRate',1e-3, ...
        'Verbose',false, ...
        'ExecutionEnvironment','gpu',...
        'Plots','training-progress');

% Train
net = trainNetwork(trainData,categorical(trainLabels),layers,options);

% Test
preds = classify(net,testData);

% find percentage of correct classifications
accuracy = 100 * length(find(categorical(testLabels) == preds)) / length(preds);
fprintf('Accuracy rate is %.2f\n', accuracy);

% confusion matrix
plotconfusion(categorical(testLabels), preds);
save('cnn_model.mat','net');
