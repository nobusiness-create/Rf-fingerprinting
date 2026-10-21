
clc;
clear;
close all;

%% 1. Select the dataset folder
dataFolder = uigetdir(pwd, 'Select the anechoic_chamber dataset folder');

if isequal(dataFolder, 0)
    disp('No folder selected.');
    return;
end

files = dir(fullfile(dataFolder, '*.mat'));

if isempty(files)
    error('No MAT files found. Select the folder containing the recordings.');
end

fprintf('Total MAT files found: %d\n', numel(files));

%% 2. Read BSSID labels from all recordings
labels = strings(numel(files), 1);

for k = 1:numel(files)
    filePath = fullfile(dataFolder, files(k).name);
    temp = load(filePath, 'BSSID');

    if isfield(temp, 'BSSID')
        labels(k) = strtrim(string(temp.BSSID));
    else
        labels(k) = "";
    end
end

valid = labels ~= "";
files = files(valid);
labels = labels(valid);

devices = unique(labels);
fprintf('Unique candidate BSSIDs found: %d\n', numel(devices));

disp('Candidate BSSIDs:');
disp(devices);

if numel(devices) ~= 5
    warning('Expected five candidate BSSIDs. Check the dataset and labels.');
end

%% 3. Calculate a normalized PSD for each recording
% Use up to 10 recordings per candidate BSSID.
% Average the normalized PSD estimates for a more stable comparison.

maxRecordingsPerDevice = 10;
nfft = 4096;

psdByDevice = cell(numel(devices), 1);
magByDevice = cell(numel(devices), 1);
freqReference = [];

for d = 1:numel(devices)

    idx = find(labels == devices(d));
    idx = idx(1:min(numel(idx), maxRecordingsPerDevice));

    devicePSDs = [];
    deviceMagnitudes = [];

    for j = 1:numel(idx)

        filePath = fullfile(dataFolder, files(idx(j)).name);
        data = load(filePath, ...
            'beaconDataDownSampled', 'baseBandSampleRate');

        if ~isfield(data, 'beaconDataDownSampled') || ...
           ~isfield(data, 'baseBandSampleRate')
            continue;
        end

        x = data.beaconDataDownSampled(:);
        Fs = double(data.baseBandSampleRate);

        if isempty(x) || Fs <= 0
            continue;
        end

        % Normalize each recording's PSD to its own peak.
        nfftThis = min(nfft, 2^floor(log2(numel(x))));
        if nfftThis < 64
            continue;
        end

        [Pxx, f] = pwelch(x, [], [], nfftThis, Fs, 'centered');

        PxxDB = 10*log10(Pxx + eps);
        PxxDB = PxxDB - max(PxxDB);

        if isempty(freqReference)
            freqReference = f;
        end

        % Keep frequency grids consistent.
        if numel(f) == numel(freqReference) && ...
                max(abs(f - freqReference)) < 1e-6
            devicePSDs(:, end+1) = PxxDB; %#ok<SAGROW>
        end

        % Store normalized magnitude samples for comparison.
        mag = abs(x);
        rmsValue = sqrt(mean(mag.^2));

        if rmsValue > 0
            mag = mag / rmsValue;
            deviceMagnitudes = [deviceMagnitudes; mag]; %#ok<AGROW>
        end
    end

    if isempty(devicePSDs)
        warning('No valid PSDs calculated for BSSID %s.', devices(d));
        psdByDevice{d} = [];
    else
        psdByDevice{d} = median(devicePSDs, 2);
    end

    magByDevice{d} = deviceMagnitudes;

    fprintf('%s: used %d recordings for PSD comparison\n', ...
        devices(d), size(devicePSDs, 2));
end

%% 4. Plot the normalized PSD comparison
figure('Name', 'Candidate Transmitter PSD Comparison');
hold on;
grid on;

for d = 1:numel(devices)
    if ~isempty(psdByDevice{d})
        plot(freqReference/1e6, psdByDevice{d}, ...
            'DisplayName', char(devices(d)));
    end
end

xlabel('Baseband frequency (MHz)');
ylabel('PSD relative to each recording peak (dB)');
title('Normalized PSD Comparison Across Candidate BSSIDs');
legend('Location', 'best', 'Interpreter', 'none');
hold off;

%% 5. Compare normalized signal magnitude distributions
figure('Name', 'Candidate Transmitter Magnitude Comparison');
hold on;
grid on;

for d = 1:numel(devices)
    mag = magByDevice{d};

    if ~isempty(mag)
        % Plot a density-normalized histogram.
        histogram(mag, 60, ...
            'Normalization', 'probability', ...
            'DisplayStyle', 'stairs', ...
            'DisplayName', char(devices(d)));
    end
end

xlabel('Magnitude / RMS magnitude');
ylabel('Probability per bin');
title('Normalized Signal Magnitude Distributions');
legend('Location', 'best', 'Interpreter', 'none');
hold off;

disp('Candidate transmitter comparison completed.');
