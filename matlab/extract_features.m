
clc;
clear;
close all;

%% 1. Select the dataset folder
dataFolder = uigetdir(pwd, ...
    'Select the anechoic_chamber dataset folder');

if isequal(dataFolder, 0)
    disp('No folder selected.');
    return;
end

files = dir(fullfile(dataFolder, '*.mat'));

if isempty(files)
    error('No MAT files found in the selected folder.');
end

%% 2. Initialize the feature table
n = numel(files);

BSSID = strings(n, 1);
MeanMagnitude = nan(n, 1);
StdMagnitude = nan(n, 1);
RMSMagnitude = nan(n, 1);
SkewMagnitude = nan(n, 1);
KurtMagnitude = nan(n, 1);
CrestFactor = nan(n, 1);
CFO = nan(n, 1);
SNR = nan(n, 1);

%% 3. Extract features from each recording
for k = 1:n

    filePath = fullfile(dataFolder, files(k).name);
    data = load(filePath);

    if isfield(data, 'BSSID')
        BSSID(k) = strtrim(string(data.BSSID));
    end

    if isfield(data, 'CFO')
        CFO(k) = double(data.CFO(1));
    end

    if isfield(data, 'SNR')
        SNR(k) = double(data.SNR(1));
    end

    if ~isfield(data, 'beaconDataDownSampled')
        continue;
    end

    x = data.beaconDataDownSampled(:);
    mag = abs(x);

    if isempty(mag) || any(~isfinite(mag))
        continue;
    end

    MeanMagnitude(k) = mean(mag);
    StdMagnitude(k) = std(mag);
    RMSMagnitude(k) = sqrt(mean(mag.^2));

    % Standardize magnitude before computing shape statistics.
    % These features describe the distribution shape.
    if std(mag) > 0
        z = (mag - mean(mag)) / std(mag);

        SkewMagnitude(k) = mean(z.^3);
        KurtMagnitude(k) = mean(z.^4);
    end

    if RMSMagnitude(k) > 0
        CrestFactor(k) = max(mag) / RMSMagnitude(k);
    end
end

%% 4. Create and save the feature table
T = table(BSSID, MeanMagnitude, StdMagnitude, ...
    RMSMagnitude, SkewMagnitude, KurtMagnitude, ...
    CrestFactor, CFO, SNR);

disp('First 10 rows of extracted features:');
disp(T(1:min(10, height(T)), :));

outputFolder = fullfile(fileparts(dataFolder), 'processed');

if ~exist(outputFolder, 'dir')
    mkdir(outputFolder);
end

outputFile = fullfile(outputFolder, 'rf_features.csv');
writetable(T, outputFile);

fprintf('\nSaved features to:\n%s\n', outputFile);

%% 5. Plot RMS magnitude by candidate BSSID
validRows = BSSID ~= "" & isfinite(RMSMagnitude);

figure('Name', 'RMS Magnitude by Candidate BSSID');

boxchart(categorical(BSSID(validRows)), ...
    RMSMagnitude(validRows));

grid on;
xlabel('Candidate BSSID');
ylabel('RMS magnitude');
title('RMS Magnitude Across Recordings');

disp('Feature extraction completed.');
