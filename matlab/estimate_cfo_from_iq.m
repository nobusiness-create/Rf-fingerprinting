
clc;
clear;
close all;

%% 1. Locate the project and MATLAB recordings
projectRoot = fileparts(fileparts(mfilename('fullpath')));

dataFolder = fullfile( ...
    projectRoot, 'data', 'anechoic_chamber', 'mat_files');

resultsFolder = fullfile(projectRoot, 'results');

if ~exist(dataFolder, 'dir')
    error('MAT recording folder not found: %s', dataFolder);
end

if ~exist(resultsFolder, 'dir')
    mkdir(resultsFolder);
end

files = dir(fullfile(dataFolder, '*.mat'));

if isempty(files)
    error('No MAT recordings found.');
end

fprintf('MAT recordings found: %d\n', numel(files));

%% 2. Estimator configuration
% Assumption: the preamble contains a repeated 802.11b
% DSSS synchronization sequence at 1 Mbit/s.
%
% At 55 MHz, one 1-Mbit/s symbol spans 55 samples.
% We must verify this assumption using the results below.

SYMBOL_RATE = 1e6;
SYNC_DURATION = 128e-6;

nFiles = numel(files);

BSSID = strings(nFiles, 1);
Filename = strings(nFiles, 1);
MetadataCFO_Hz = nan(nFiles, 1);
EstimatedCFO_Hz = nan(nFiles, 1);
CorrelationCoherence = nan(nFiles, 1);

%% 3. Estimate CFO for every recording
for k = 1:nFiles

    filePath = fullfile(dataFolder, files(k).name);

    data = load(filePath, ...
        'BSSID', 'CFO', 'sampleRate', 'beaconPreamble');

    if ~isfield(data, 'beaconPreamble') || ...
       ~isfield(data, 'sampleRate') || ...
       ~isfield(data, 'CFO') || ...
       ~isfield(data, 'BSSID')

        warning('Required field missing in %s', files(k).name);
        continue;
    end

    x = data.beaconPreamble(:);
    Fs = double(data.sampleRate);
    metadataCFO = double(data.CFO);

    if isempty(x) || ~isfinite(Fs) || Fs <= 0
        warning('Invalid signal or sample rate: %s', files(k).name);
        continue;
    end

    % Lag corresponding to one 1-Mbit/s symbol.
    lag = round(Fs / SYMBOL_RATE);

    % Analyze only the initial synchronization interval.
    nUse = min(numel(x), floor(SYNC_DURATION * Fs));

    if nUse <= lag + 100
        warning('Preamble is too short: %s', files(k).name);
        continue;
    end

    % Correlate the signal with a lagged copy.
    % A stable repeated sequence with CFO has a phase rotation
    % proportional to the lag.
    earlier = x(1:nUse-lag);
    later = x(lag+1:nUse);

    products = later .* conj(earlier);
    R = sum(products);

    if abs(R) < 1e-12
        warning('Correlation is too weak: %s', files(k).name);
        continue;
    end

    % Normalized correlation magnitude, between 0 and 1.
    coherence = abs(R) / (sum(abs(products)) + eps);

    % Estimated frequency offset in Hz.
    cfoEstimate = angle(R) * Fs / (2*pi*lag);

    BSSID(k) = upper(strtrim(string(data.BSSID)));
    Filename(k) = string(files(k).name);
    MetadataCFO_Hz(k) = metadataCFO;
    EstimatedCFO_Hz(k) = cfoEstimate;
    CorrelationCoherence(k) = coherence;

    if k <= 10
        fprintf('\n%s\n', files(k).name);
        fprintf('  Metadata CFO: %.3f Hz\n', metadataCFO);
        fprintf('  Estimated CFO: %.3f Hz\n', cfoEstimate);
        fprintf('  Correlation coherence: %.4f\n', coherence);
    end
end

%% 4. Save results
T = table( ...
    BSSID, Filename, MetadataCFO_Hz, ...
    EstimatedCFO_Hz, CorrelationCoherence);

valid = isfinite(MetadataCFO_Hz) & ...
        isfinite(EstimatedCFO_Hz);

if ~any(valid)
    error('No CFO estimates could be calculated.');
end

outputPath = fullfile(resultsFolder, 'cfo_iq_estimate_check.csv');
writetable(T, outputPath);

%% 5. Compare the estimate against the provided metadata
metadata = MetadataCFO_Hz(valid);
estimated = EstimatedCFO_Hz(valid);

errorSameSign = median(abs(estimated - metadata));
errorReverseSign = median(abs(-estimated - metadata));

fprintf('\n========== CFO ESTIMATION SUMMARY ==========\n');
fprintf('Successfully processed recordings: %d / %d\n', ...
    sum(valid), nFiles);

fprintf('Median correlation coherence: %.4f\n', ...
    median(CorrelationCoherence(valid)));

fprintf('Median absolute error, same sign: %.3f Hz\n', ...
    errorSameSign);

fprintf('Median absolute error, reversed sign: %.3f Hz\n', ...
    errorReverseSign);

fprintf('Results saved to:\n%s\n', outputPath);

%% 6. Plot estimated versus supplied CFO
figure('Name', 'CFO Estimation Validation');

scatter(metadata, estimated, 35, 'filled');
hold on;
grid on;

low = min([metadata; estimated]);
high = max([metadata; estimated]);

plot([low high], [low high], 'k--', ...
    'DisplayName', 'Ideal agreement');

xlabel('Dataset CFO metadata (Hz)');
ylabel('CFO estimated from I/Q (Hz)');
title('CFO Estimator Validation');
legend('Estimated CFO', 'Ideal agreement', 'Location', 'best');

hold off;

disp('CFO estimation diagnostic completed.');
