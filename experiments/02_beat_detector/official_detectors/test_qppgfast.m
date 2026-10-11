pkg load signal

addpath(genpath('third_party/ppg-beats/source'));

sig = csvread(
    'experiments/02_beat_detector/test_ppg.csv'
);

fs = 125;

[peaks, onsets] = qppgfast_beat_detector(sig, fs);

csvwrite(
    'experiments/02_beat_detector/test_qppgfast_peaks.csv',
    peaks
);

fprintf('Detected peaks: %d\n', length(peaks));
