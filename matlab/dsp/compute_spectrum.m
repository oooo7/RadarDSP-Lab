function [freq_axis, mag_spectrum, dom_freq, peak_mag] = compute_spectrum(x, sampling_rate_hz)
% COMPUTE_SPECTRUM Computes 1D single-sided FFT magnitude spectrum & sub-bin peak estimation.
% Matches Python Phase 3 FFT scaling and parabolic sub-bin frequency recovery.
%
% Inputs:
%   x                - 1D real or complex signal vector
%   sampling_rate_hz - ADC sampling rate Fs in Hz
%
% Outputs:
%   freq_axis    - 1D frequency axis in Hz [0, Fs/2]
%   mag_spectrum - 1D single-sided magnitude spectrum array
%   dom_freq     - Measured dominant peak frequency in Hz (with sub-bin interpolation)
%   peak_mag     - Peak magnitude value

N = length(x);
X_fft = fft(x);
num_bins = floor(N/2) + 1;
mag_full = abs(X_fft) / N;

mag_spectrum = mag_full(1:num_bins);
if num_bins > 2
    mag_spectrum(2:end-1) = 2.0 * mag_spectrum(2:end-1);
end

freq_axis = (0:num_bins-1) * (sampling_rate_hz / N);

[peak_mag, idx] = max(mag_spectrum);
dom_freq = freq_axis(idx);

% 3-point parabolic sub-bin frequency interpolation
if idx > 1 && idx < num_bins
    alpha = log(max(mag_spectrum(idx-1), 1e-15));
    beta  = log(max(mag_spectrum(idx), 1e-15));
    gamma = log(max(mag_spectrum(idx+1), 1e-15));
    denom = 2.0 * (2.0 * beta - alpha - gamma);
    if abs(denom) > 1e-12
        delta = (gamma - alpha) / denom;
        if isfinite(delta) && abs(delta) <= 1.0
            dom_freq = (idx - 1 + delta) * (sampling_rate_hz / N);
        end
    end
end
end
