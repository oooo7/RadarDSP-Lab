function [r_est, fb_meas, abs_error_m] = range_estimation(s_beat, chirp_struct, r_true)
% RANGE_ESTIMATION Estimates target range via 1D Range FFT.
%
% Formula:
%   R_est = c * f_b_meas / (2 * S)
%
% Inputs:
%   s_beat       - 1D complex or real beat signal array
%   chirp_struct - Struct from generate_fmcw_chirp
%   r_true       - Ground truth range in meters for error calculation
%
% Outputs:
%   r_est       - Estimated target range in meters
%   fb_meas     - Measured beat frequency in Hz
%   abs_error_m - Absolute range error |r_est - r_true| in meters

addpath(fullfile(fileparts(mfilename('fullpath')), '..', 'dsp'));

c = 299792458.0;
S = chirp_struct.chirp_slope_hz_per_sec;
Fs = chirp_struct.sampling_rate_hz;

[freq_axis, mag_spectrum, fb_meas, ~] = compute_spectrum(real(s_beat), Fs);
r_est = c * fb_meas / (2.0 * S);
abs_error_m = abs(r_est - r_true);
end
