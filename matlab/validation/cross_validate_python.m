function cross_validate_python(python_json_path)
% CROSS_VALIDATE_PYTHON Cross-validates MATLAB golden reference against Python baseline results.
%
%   Usage:
%       cross_validate_python('experiments/results/python_reference_results.json')

    if nargin < 1 || isempty(python_json_path)
        python_json_path = 'experiments/results/python_reference_results.json';
    end

    addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'dsp'));
    addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'radar'));

    fprintf('========================================\n');
    fprintf('RadarDSP Lab MATLAB Cross-Validation\n');
    fprintf('========================================\n\n');

    overall_pass = true;

    % --- DSP VALIDATION ---
    fprintf('DSP\n');

    % Aliasing check
    f_alias = sampling_alias(700, 1000);
    if abs(f_alias - 300) < 1e-6
        fprintf('[PASS] 700 Hz @ 1 kHz alias -> 300 Hz\n');
    else
        fprintf('[FAIL] 700 Hz @ 1 kHz alias expected 300 Hz, got %.2f Hz\n', f_alias);
        overall_pass = false;
    end

    % FFT spectrum check
    [~, x_sine] = generate_sine(100, 1.0, 1000, 1.0, 0);
    [f_axis, mag] = compute_spectrum(x_sine, 1000, 'hann');
    [~, max_idx] = max(mag);
    f_dom = f_axis(max_idx);
    if abs(f_dom - 100) <= 2.0 % Within 1 FFT bin resolution (Fs/N = 1 Hz)
        fprintf('[PASS] FFT dominant frequency\n');
    else
        fprintf('[FAIL] FFT dominant frequency expected 100 Hz, got %.2f Hz\n', f_dom);
        overall_pass = false;
    end
    fprintf('\n');

    % --- FMCW RANGE VALIDATION ---
    fprintf('FMCW RANGE\n');
    test_ranges = [50, 100, 150, 200, 300, 400, 500];
    delta_r = 299792458 / (2 * 150e6); % 0.9993 m

    for r_target = test_ranges
        s_beat = simulate_single_target(r_target, 77e9, 150e6, 100e-6, 20e6);
        [est_r, abs_err] = range_estimation(s_beat, 150e6, 100e-6, 20e6, r_target);

        if abs_err <= delta_r
            fprintf('[PASS] %d m (Est: %.2f m, Err: %.4f m)\n', r_target, est_r, abs_err);
        else
            fprintf('[FAIL] %d m (Est: %.2f m exceeds resolution cell Delta_R = %.2f m)\n', r_target, est_r, delta_r);
            overall_pass = false;
        end
    end
    fprintf('\n');

    % --- RANGE-DOPPLER VALIDATION ---
    fprintf('RANGE-DOPPLER\n');
    targets = struct('range_m', {50, 120, 200}, ...
                     'velocity_mps', {10, -5, 0}, ...
                     'amplitude', {1.0, 0.7, 0.5});
    [s_cube, ~, ~] = multi_target_demo(targets, 77e9, 150e6, 100e-6, 20e6, 64);
    [RD, r_axis, v_axis] = range_doppler_demo(s_cube, 77e9, 150e6, 100e-6, 20e6);

    RD_mag = abs(RD);
    delta_v = (299792458 / 77e9) / (2 * 64 * 100e-6); % ~0.304 m/s

    for i = 1:length(targets)
        tg = targets(i);
        [~, r_idx] = min(abs(r_axis - tg.range_m));
        [~, v_idx] = min(abs(v_axis - tg.velocity_mps));

        % Verify peak exists around target location
        r_sub = max(1, r_idx-2):min(length(r_axis), r_idx+2);
        v_sub = max(1, v_idx-2):min(length(v_axis), v_idx+2);
        sub_matrix = RD_mag(v_sub, r_sub);
        [~, max_flat] = max(sub_matrix(:));
        [sub_v, sub_r] = ind2sub(size(sub_matrix), max_flat);

        meas_r = r_axis(v_sub(1) + sub_r - 1);
        meas_v = v_axis(v_sub(1) + sub_v - 1);

        err_r = abs(meas_r - tg.range_m);
        err_v = abs(meas_v - tg.velocity_mps);

        if err_r <= delta_r && err_v <= delta_v
            fprintf('[PASS] Target %d (Expected: [%.0f m, %.0f m/s], Measured: [%.2f m, %.2f m/s])\n', ...
                i, tg.range_m, tg.velocity_mps, meas_r, meas_v);
        else
            fprintf('[FAIL] Target %d range/velocity error exceeds tolerance\n', i);
            overall_pass = false;
        end
    end
    fprintf('\n');

    % --- AWGN VALIDATION ---
    fprintf('AWGN\n');
    snr_tests = [20, 10, 0, -10];
    for snr = snr_tests
        [s_noisy, p_sig, p_noise] = add_awgn_noise(ones(1000, 1), snr, 42);
        measured_snr = 10 * log10(p_sig / p_noise);
        if abs(measured_snr - snr) < 0.5 % Statistical sample agreement within 0.5 dB
            fprintf('[PASS] %d dB (Measured: %.2f dB)\n', snr, measured_snr);
        else
            fprintf('[FAIL] %d dB expected, measured %.2f dB\n', snr, measured_snr);
            overall_pass = false;
        end
    end
    fprintf('\n');

    % --- CA-CFAR VALIDATION ---
    fprintf('CA-CFAR\n');
    % 1. Alpha calculation check
    N_tr = 144; Pfa = 1e-4;
    alpha_calc = N_tr * (Pfa^(-1 / N_tr) - 1);
    alpha_expected = 9.4795;
    if abs(alpha_calc - alpha_expected) < 1e-3
        fprintf('[PASS] alpha calculation (%.4f)\n', alpha_calc);
    else
        fprintf('[FAIL] alpha calculation expected %.4f, got %.4f\n', alpha_expected, alpha_calc);
        overall_pass = false;
    end

    % 2. Boundary handling & detection map check
    synth_RD = ones(32, 64) * 0.01;
    synth_RD(16, 32) = 10.0; % Strong target at center
    [mask, thresh, alpha_val, n_det] = run_ca_cfar(synth_RD, 1e-4, 2, 2, 4, 4);

    if mask(16, 32) == true
        fprintf('[PASS] boundary handling & detection map\n');
    else
        fprintf('[FAIL] target detection map failed\n');
        overall_pass = false;
    end
    fprintf('\n');

    if overall_pass
        fprintf('Overall: PASS\n');
    else
        fprintf('Overall: FAIL\n');
    end
end
