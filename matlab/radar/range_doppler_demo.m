function [RD_complex, range_axis, velocity_axis] = range_doppler_demo(s_beat, fc, B, Tc, Fs)
% RANGE_DOPPLER_DEMO Computes 2D Range-Doppler matrix spectrum via 2D FFT.
%
%   Inputs:
%       s_beat: Complex matrix of shape [num_chirps, samples_per_chirp]
%       fc: Carrier frequency (Hz), default 77e9
%       B: Sweep bandwidth (Hz), default 150e6
%       Tc: Chirp duration (s), default 100e-6
%       Fs: Sampling rate (Hz), default 20e6
%
%   Outputs:
%       RD_complex: 2D complex matrix of shape [num_doppler_bins, num_range_bins]
%       range_axis: 1D vector of range bin values (m)
%       velocity_axis: 1D vector of velocity bin values (m/s)

    if nargin < 2 || isempty(fc), fc = 77e9; end
    if nargin < 3 || isempty(B), B = 150e6; end
    if nargin < 4 || isempty(Tc), Tc = 100e-6; end
    if nargin < 5 || isempty(Fs), Fs = 20e6; end

    c = 299792458; % m/s
    S = B / Tc;
    wavelength = c / fc;

    [M, N] = size(s_beat);

    % 1. Fast-time Range FFT along rows (axis 2 in MATLAB, columns of matrix)
    % Fast-time Hann window normalized by mean
    w_fast = hann(N, 'periodic');
    w_fast = w_fast / mean(w_fast);

    range_profiles = zeros(M, floor(N/2) + 1);

    for m = 1:M
        x_win = s_beat(m, :) .* w_fast.';
        X = fft(x_win, N);
        % One-sided positive frequency spectrum scaled by N
        % Match Python compute_spectrum scaling (norm='backward' / N)
        % For positive frequencies (2:end-1), multiply by 2 for one-sided power parity
        X_pos = X(1:floor(N/2) + 1) / N;
        if N > 2
            X_pos(2:end-1) = 2 * X_pos(2:end-1);
        end
        range_profiles(m, :) = X_pos;
    end

    % Range Axis
    freq_range = (0:floor(N/2)) * (Fs / N);
    range_axis = c * freq_range / (2 * S);

    % 2. Slow-time Doppler FFT along columns (axis 1 in MATLAB, rows of matrix)
    w_slow = hann(M, 'periodic');
    w_slow = w_slow / mean(w_slow);

    % Apply slow-time window across chirps
    windowed_rp = range_profiles .* repmat(w_slow, 1, size(range_profiles, 2));

    % 1D FFT along slow-time and fftshift
    RD_complex = fft(windowed_rp, M, 1);
    RD_complex = fftshift(RD_complex, 1) / M;

    % Velocity Axis
    % Doppler frequencies in Hz [-PRF/2, +PRF/2)
    PRF = 1 / Tc;
    if mod(M, 2) == 0
        k = (-M/2 : M/2 - 1);
    else
        k = (-(M-1)/2 : (M-1)/2);
    end
    f_doppler = k * (PRF / M);
    velocity_axis = f_doppler * wavelength / 2;
end
