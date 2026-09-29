function [t, x] = generate_multitone(amplitudes, frequencies_hz, phases_rad, sampling_rate_hz, duration_sec)
% GENERATE_MULTITONE Synthesizes a sum of sinusoids matching Python DSP engine.
%
% Formula:
%   x(t) = sum_k ( A_k * sin(2*pi*f_k*t + phi_k) )
%
% Inputs:
%   amplitudes       - Vector of peak amplitudes A_k
%   frequencies_hz   - Vector of tone frequencies f_k in Hz
%   phases_rad       - Vector of phase offsets phi_k in radians
%   sampling_rate_hz - ADC sampling rate Fs in Hz
%   duration_sec     - Signal duration in seconds
%
% Outputs:
%   t - 1D float time vector in seconds
%   x - 1D float multitone signal array

num_samples = round(duration_sec * sampling_rate_hz);
t = (0:num_samples-1) / double(sampling_rate_hz);
x = zeros(1, num_samples);

for k = 1:length(frequencies_hz)
    x = x + amplitudes(k) * sin(2.0 * pi * frequencies_hz(k) * t + phases_rad(k));
end
end
