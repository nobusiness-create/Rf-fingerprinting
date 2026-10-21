
clc;
clear;
close all;

%% 1. Load extracted features
[file, folder] = uigetfile('*.csv', 'Select rf_features.csv');

if isequal(file, 0)
    disp('No file selected.');
    return;
end

T = readtable(fullfile(folder, file));

% Start with magnitude-based features only.
% Exclude CFO and SNR for this first baseline.
featureNames = {'MeanMagnitude', 'StdMagnitude', ...
    'SkewMagnitude', 'KurtMagnitude', 'CrestFactor', ...
    'CFO', 'SNR'};

X = T{:, featureNames};
labels = string(T.BSSID);

% Remove invalid rows
valid = all(isfinite(X), 2) & labels ~= "";
X = X(valid, :);
labels = labels(valid);

classes = unique(labels);
fprintf('Recordings: %d\n', size(X,1));
fprintf('Candidate classes: %d\n', numel(classes));

%% 2. Split recordings into training and test sets
% Random split within each class
rng(42);

trainIdx = false(size(labels));
testIdx = false(size(labels));

for c = 1:numel(classes)
    idx = find(labels == classes(c));
    idx = idx(randperm(numel(idx)));

    nTrain = floor(0.7 * numel(idx));

    trainIdx(idx(1:nTrain)) = true;
    testIdx(idx(nTrain+1:end)) = true;
end

XTrain = X(trainIdx, :);
yTrain = labels(trainIdx);
XTest = X(testIdx, :);
yTest = labels(testIdx);

%% 3. Standardize using training data only
mu = mean(XTrain, 1);
sigma = std(XTrain, 0, 1);
sigma(sigma == 0) = 1;

XTrain = (XTrain - mu) ./ sigma;
XTest = (XTest - mu) ./ sigma;

%% 4. Build one centroid per candidate class
centroids = zeros(numel(classes), size(XTrain, 2));

for c = 1:numel(classes)
    centroids(c, :) = mean(XTrain(yTrain == classes(c), :), 1);
end

%% 5. Predict the class of each test recording
predicted = strings(size(yTest));

for i = 1:size(XTest, 1)
    distances = sum((centroids - XTest(i, :)).^2, 2);
    [~, bestClass] = min(distances);
    predicted(i) = classes(bestClass);
end

accuracy = mean(predicted == yTest) * 100;

fprintf('\nTest recordings: %d\n', numel(yTest));
fprintf('Baseline accuracy: %.2f%%\n', accuracy);

%% 6. Display per-class results
fprintf('\nPer-class test results:\n');

for c = 1:numel(classes)
    idx = yTest == classes(c);
    classAccuracy = mean(predicted(idx) == yTest(idx)) * 100;

    fprintf('%s: %.2f%% (%d recordings)\n', ...
        classes(c), classAccuracy, sum(idx));
end

%% 7. Plot confusion matrix without additional toolboxes
confusion = zeros(numel(classes));

for i = 1:numel(yTest)
    actualClass = find(classes == yTest(i), 1);
    predictedClass = find(classes == predicted(i), 1);
    confusion(actualClass, predictedClass) = ...
        confusion(actualClass, predictedClass) + 1;
end

figure('Name', 'Baseline Confusion Matrix');
imagesc(confusion);
axis square;
colorbar;
xticks(1:numel(classes));
yticks(1:numel(classes));
xticklabels(cellstr(classes));
yticklabels(cellstr(classes));
xtickangle(45);
xlabel('Predicted BSSID');
ylabel('Actual BSSID');
title(sprintf('Nearest-Centroid Baseline: %.2f%% Accuracy', accuracy));

disp('Baseline evaluation completed.');
