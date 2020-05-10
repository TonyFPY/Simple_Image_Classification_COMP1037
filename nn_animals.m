% Load Training Data
fprintf('Loading and Visualizing Data ...\n')
load('data.mat');

% create a neural network
net = patternnet(64);
net.trainParam.goal = 0.05;

% divided into training, validation and testing simulate
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 0.2;

rand_indices = randperm(size(X, 2));

trainData = X(:, rand_indices(1:2400));
trainLabels = y(:, rand_indices(1:2400));
testData = X(:, rand_indices(2401:end));
testLabels = y(:, rand_indices(2401:end));

% train a neural network
net = train(net, trainData, trainLabels);
save('nn_model.mat','net');

% show the network
view(net);

preds = net(testData);
est = vec2ind(preds);
tar = vec2ind(testLabels);

% find percentage of correct classifications
accuracy = 100 * length(find(est == tar)) / length(tar);
fprintf('Accuracy rate is %.2f\n', accuracy);

% confusion matrix
plotconfusion(testLabels, preds);

% Pause
fprintf('Program paused. Press enter to continue.\n');
pause;
    
%  To give you an idea of the network's output, you can also run
%  through the examples one at the a time to see what it is predicting.
%  Randomly permute examples

rp = randperm(size(X,2));

for i = 1 : size(X,2)
    % Display 
    fprintf('\nDisplaying Example Image\n');
    displayData(X(:, rp(i))');

    pred = vec2ind(net(X(:, rp(i))));
    lbl = vec2ind(y(:, rp(i)));
    txtPred = strcat('Neural Network Prediction: ', int2str(pred));
    txtLabl = strcat('Label: ', int2str(lbl));
    text(2, 5, txtPred, 'FontSize', 12, 'Color', 'red');
    text(2, 10, txtLabl, 'FontSize', 12, 'Color', 'green');
    % fprintf('\nNeural Network Prediction: %d\n', pred);
    
    % Pause
    fprintf('Program paused. Press enter to continue.\n');
    pause;
end