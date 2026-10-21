
clc;
clear;
close all;

%% 1. Select a Wi-Fi RF recording
[file, folder] = uigetfile('*.mat', ...
    'Select a Wi-Fi RF recording');

if isequal(file, 0)
    disp('No file selected.');
    return;
end

filePath = fullfile(folder, file);
data = load(filePath);

fprintf('\n===== RF SIGNAL ANALYSIS =====\n');
fprintf('File: %s\n', file);

%% 2. Display available metadata
disp('Variables stored in the file:');
disp(fieldnames(data));

if isfield(data, 'BSSID')
    disp('BSSID (candidate transmitter label):');
    disp(data.BSSID);
end

if isfield(data, 'SSID')
    disp('Wi-Fi network name:');
    disp(data.SSID);
end

if isfield(data, 'sampleRate')
    fprintf('Original sample rate: %.2f MHz\n', ...
        double(data.sampleRate) / 1e6);
end

if isfield(data, 'baseBandSampleRate')
    fprintf('Baseband sample rate: %.2f MHz\n', ...
        double(data.baseBandSampleRate) / 1e6);
end

%% 3. Extract the downsampled complex signal
if ~isfield(data, 'beaconDataDownSampled')
    error('The selected file has no beaconDataDownSampled variable.');
end

if ~isfield(data, 'baseBandSampleRate')
    error('The selected file has no baseBandSampleRate variable.');
end

x = data.beaconDataDownSampled(:);
Fs = double(data.baseBandSampleRate);

if isempty(x) || ~isnumeric(x) || Fs <= 0
    error('The signal or sample rate is invalid.');
end

N = length(x);
duration = N / Fs;

fprintf('\nNumber of samples: %d\n', N);
fprintf('Signal duration: %.4f milliseconds\n', ...
    duration * 1000);

%% 4. Plot the I/Q waveform and magnitude
samplesToPlot = min(N, 1000);
t = (0:samplesToPlot-1)' / Fs;

figure('Name', 'RF Signal: I/Q and Magnitude');

subplot(3,1,1);
plot(t * 1e6, real(x(1:samplesToPlot)));
grid on;
xlabel('Time (\mus)');
ylabel('Amplitude');
title('In-phase component (I)');

subplot(3,1,2);
plot(t * 1e6, imag(x(1:samplesToPlot)));
grid on;
xlabel('Time (\mus)');
ylabel('Amplitude');
title('Quadrature component (Q)');

subplot(3,1,3);
plot(t * 1e6, abs(x(1:samplesToPlot)));
grid on;
xlabel('Time (\mus)');
ylabel('Magnitude');
title('Signal magnitude');

%% 5. Calculate and plot the FFT spectrum
X = fftshift(fft(x));
f = (-floor(N/2):ceil(N/2)-1)' * (Fs/N);

spectrumDB = 20 * log10(abs(X)/N + eps);

figure('Name', 'RF Signal: FFT Spectrum');
plot(f/1e6, spectrumDB);
grid on;
xlabel('Frequency (MHz)');
ylabel('Magnitude (dB)');
title('Centered FFT Spectrum');

%% 6. Plot power spectral density
nfft = min(4096, 2^floor(log2(N)));

if nfft >= 64
    [Pxx, fPSD] = pwelch(x, [], [], nfft, Fs, 'centered');

    figure('Name', 'RF Signal: Power Spectral Density');
    plot(fPSD/1e6, 10*log10(Pxx + eps));
    grid on;
    xlabel('Frequency (MHz)');
    ylabel('Power/Frequency (dB/Hz)');
    title('Power Spectral Density (Welch Method)');
else
    disp('Signal is too short for the selected PSD settings.');
end

fprintf('\nSignal analysis completed successfully.\n');

