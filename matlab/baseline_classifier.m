
clc; clear; close all;
[file, folder] = uigetfile('*.csv', 'Select rf_features.csv');
if isequal(file,0), return; end

T = readtable(fullfile(folder,file));

magNames = {'MeanMagnitude','StdMagnitude', ...
    'SkewMagnitude','KurtMagnitude','CrestFactor'};
metaNames = {'CFO','SNR'};
allNames = [magNames, metaNames];

Y = string(T.BSSID);
valid = Y ~= "" & all(isfinite(T{:,allNames}),2);
T = T(valid,:);
Y = string(T.BSSID);
classes = unique(Y);

% One reproducible split shared by all feature sets
rng(42);
train = false(height(T),1);
test = false(height(T),1);

for c = 1:numel(classes)
    idx = find(Y == classes(c));
    idx = idx(randperm(numel(idx)));
    ntr = floor(0.7*numel(idx));
    train(idx(1:ntr)) = true;
    test(idx(ntr+1:end)) = true;
end

featureSets = {magNames, metaNames, allNames};
setNames = ["Magnitude only"; "CFO + SNR only"; "All features"];
accuracies = zeros(3,1);

for s = 1:3
    X = T{:,featureSets{s}};
    A = X(train,:);
    B = X(test,:);

    mu = mean(A,1);
    sd = std(A,0,1);
    sd(sd == 0) = 1;
    A = (A-mu)./sd;
    B = (B-mu)./sd;

    centers = zeros(numel(classes),size(A,2));
    for c = 1:numel(classes)
        centers(c,:) = mean(A(Y(train)==classes(c),:),1);
    end

    pred = strings(sum(test),1);
    actual = Y(test);
    for i = 1:size(B,1)
        dist = sum((centers-B(i,:)).^2,2);
        [~,j] = min(dist);
        pred(i) = classes(j);
    end

    accuracies(s) = mean(pred == actual)*100;
    fprintf('%s: %.2f%%\n',setNames(s),accuracies(s));
end

results = table(setNames,accuracies, ...
    'VariableNames',{'FeatureSet','AccuracyPercent'});
disp(results);

figure;
bar(accuracies);
xticks(1:3);
xticklabels(cellstr(setNames));
ylabel('Test accuracy (%)');
title('Feature Set Comparison');
grid on;
