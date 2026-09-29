function [s_beat, t_fast, t_slow] = multi_target_demo(targets, fc, B, Tc, Fs, num_chirps)
% MULTI_TARGET_DEMO Simulates multi-chirp, multi-target FMCW beat signal data cube.
%
%   Inputs:
%       targets: Struct array with fields:
%                   .range_m      - Initial range R0 (m)
%                   .velocity_mps - Radial velocity v (m/s)
%                   .amplitude    - Signal amplitude A
%       fc: Carrier frequency (Hz), default 77e9
%       B: Sweep bandwidth (Hz), default 150e6
%       Tc: Chirp duration (s), default 100e-6
%       Fs: Sampling frequency (Hz), default 20e6
%       num_chirps: Number of chirps M, default 64
%
%   Outputs:
%       s_beat: Complex matrix of shape [num_chirps, samples_per_chirp]
%       t_fast: 1D fast-time vector (1 x N)
%       t_slow: 1D slow-time vector (1 x M)

    if nargin < 2 || isempty(fc), fc = 77e9; end
    if nargin < 3 || isempty(B), B = 150e6; end
    if nargin < 4 || isempty(Tc), Tc = 100e-6; end
    if nargin < 5 || isempty(Fs), Fs = 20e6; end
    if nargin < 6 || isempty(num_chirps), num_chirps = 64; end

    c = 299792458; % m/s
    S = B / Tc;
    wavelength = c / fc;

    N = round(Fs * Tc);
    t_fast = (0:N-1) / Fs;
    t_slow = (0:num_chirps-1) * Tc;

    s_beat = zeros(num_chirps, N);

    for idx = 1:length(targets)
        R0 = targets(idx).range_m;
        v  = targets(idx).velocity_mps;
        A  = targets(idx).amplitude;

        fb = 2 * S * R0 / c;
        fd = 2 * v / wavelength;
        tau0 = 2 * R0 / c;
        phi_const = 2 * pi * fc * tau0 - pi * S * (tau0^2);

        % Vectorized 2D phase matrix across slow-time (rows) and fast-time (cols)
        % phase[m, n] = 2*pi*fb*t_fast[n] + 2*pi*fd*t_slow[m] + phi_const
        phase = 2 * pi * (t_slow.' * fd) + 2 * pi * (t_fast * fb) + phi_const;
        s_beat = s_beat + A * exp(1i * phase);
    end
end
