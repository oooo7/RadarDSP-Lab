function export_reference_results(output_filepath)
% EXPORT_REFERENCE_RESULTS Generates MATLAB golden reference datasets and exports to JSON.
%
%   Usage:
%       export_reference_results('experiments/results/matlab_reference_results.json')

    if nargin < 1 || isempty(output_filepath)
        output_filepath = 'experiments/results/matlab_reference_results.json';
    end

    addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'dsp'));
    addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'radar'));

    % 1. DSP Aliasing
    f_orig = 700; Fs = 1000;
    f_alias = sampling_alias(f_orig, Fs);

    % 2. DSP FFT
    [t, x] = generate_sine(100, 1.0, 1000, 1.0, 0);
    [f_axis, mag] = compute_spectrum(x, 1000, 'hann');
    [~, max_idx] = max(mag);
    f_dom = f_axis(max_idx);

    % 3. Single Target FMCW Range Sweeps
    ranges_test = [50, 100, 150, 200, 300, 400, 500];
    matlab_single_targets = struct('target_range_m', {}, 'estimated_range_m', {}, 'absolute_error_m', {});

    for r_val = ranges_test
        s_beat = simulate_single_target(r_val, 77e9, 150e6, 100e-6, 20e6);
        [est_r, abs_err] = range_estimation(s_beat, 150e6, 100e-6, 20e6, r_val);
        matlab_single_targets(end+1).target_range_m = r_val;
        matlab_single_targets(end).estimated_range_m = est_r;
        matlab_single_targets(end).absolute_error_m = abs_err;
    end

    % 4. Multi-Target Range-Doppler
    targets = struct('range_m', {50, 120, 200}, ...
                     'velocity_mps', {10, -5, 0}, ...
                     'amplitude', {1.0, 0.7, 0.5});
    [s_cube, ~, ~] = multi_target_demo(targets, 77e9, 150e6, 100e-6, 20e6, 64);
    [RD, r_axis, v_axis] = range_doppler_demo(s_cube, 77e9, 150e6, 100e-6, 20e6);

    % 5. CA-CFAR Alpha Calculation
    alpha_calc = 144 * ((1e-4)^(-1/144) - 1);

    % Construct Struct
    data = struct();
    data.generator = 'MATLAB Golden Reference';
    data.dsp_alias = struct('original_freq_hz', f_orig, 'sampling_rate_hz', Fs, 'aliased_freq_hz', f_alias);
    data.dsp_fft = struct('signal_freq_hz', 100, 'dominant_freq_hz', f_dom);
    data.single_target_range = matlab_single_targets;
    data.cfar_alpha = alpha_calc;

    % Write JSON file if jsonencode is available
    try
        json_str = jsonencode(data);
        fid = fopen(output_filepath, 'w');
        if fid ~= -1
            fwrite(fid, json_str, 'char');
            fclose(fid);
            fprintf('Successfully exported MATLAB reference results to %s\n', output_filepath);
        end
    catch ME
        fprintf('Failed to export JSON: %s\n', ME.message);
    end
end
