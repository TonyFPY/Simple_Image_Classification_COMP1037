% Load Training Data
fprintf('Loading and Visualizing Data ...\n')
load('data_efficiency.mat');

k = 1000;
[COEFF,STORE,latent] = pca(X','Rows','complete','Economy','off');
X_pca = STORE(:,1:k);

% create a neural network
net = patternnet(64);
net.trainParam.goal = 0.05;

% divided into training, validation and testing simulate
net.divideParam.trainRatio = 0.8;
net.divideParam.valRatio = 0;
net.divideParam.testRatio = 0.2;

rand_indices = randperm(size(X, 2));

trainData = X_pca(:, rand_indices(1:2400));
trainLabels = y(:, rand_indices(1:2400));
testData = X_pca(:, rand_indices(2401:end));
testLabels = y(:, rand_indices(2401:end));

% train a neural network
net = train(net, trainData, trainLabels);

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
