pkg load signal

addpath(genpath('third_party/ppg-beats/source'));

input_dir = 'experiments/02_beat_detector/octave_input';
output_dir = 'experiments/02_beat_detector/octave_output';

files = dir(fullfile(input_dir, '*.csv'));

fs = 125;

summary_file = fullfile(output_dir, 'runtime_summary.csv');
fid = fopen(summary_file, 'w');
fprintf(fid, 'record_id,method,n_peaks,runtime_s\n');

for i = 1:length(files)

    [~, rid, ~] = fileparts(files(i).name);

    fprintf('[%02d/%02d] %s\n', i, length(files), rid);

    sig = csvread(
        fullfile(input_dir, files(i).name)
    );

    % MSPTDfast v2
    tic;
    [peaks, ~] = msptdfastv2_beat_detector(sig, fs);
    runtime = toc;

    csvwrite(
        fullfile(
            output_dir,
            [rid '_msptdfastv2.csv']
        ),
        peaks
    );

    fprintf(
        fid,
        '%s,msptdfastv2,%d,%.8f\n',
        rid,
        length(peaks),
        runtime
    );

    % qPPGfast
    tic;
    [peaks, ~] = qppgfast_beat_detector(sig, fs);
    runtime = toc;

    csvwrite(
        fullfile(
            output_dir,
            [rid '_qppgfast.csv']
        ),
        peaks
    );

    fprintf(
        fid,
        '%s,qppgfast,%d,%.8f\n',
        rid,
        length(peaks),
        runtime
    );

end

fclose(fid);

fprintf('DONE: %d records\n', length(files));
