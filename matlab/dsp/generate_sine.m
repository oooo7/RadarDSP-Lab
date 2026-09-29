function [t, x] = generate_sine(amplitude, frequency_hz, phase_rad, sampling_rate_hz, duration_sec, dc_offset)
% GENERATE_SINE Synthesizes a discrete-time sine wave matching Python DSP engine.
%
% Formula:
%   x(t) = amplitude * sin(2*pi*frequency_hz*t + phase_rad) + dc_offset
%
% Inputs:
%   amplitude        - Peak amplitude A
%   frequency_hz     - Tone frequency f in Hz (>= 0)
%   phase_rad        - Initial phase offset in radians
%   sampling_rate_hz - ADC sampling rate Fs in Hz (> 0)
%   duration_sec     - Signal duration in seconds (> 0)
%   dc_offset        - Constant DC shift (default 0.0)
%
% Outputs:
%   t - 1D float time vector in seconds
%   x - 1D float signal array

if nargin < 6
    dc_offset = 0.0;
end

num_samples = round(duration_sec * sampling_rate_hz);
t = (0:num_samples-1) / float_or_double(sampling_rate_hz);
x = amplitude * sin(2.0 * pi * frequency_hz * t + phase_rad) + dc_offset;
end

function val = float_or_double(v)
    val = double(v);
end
