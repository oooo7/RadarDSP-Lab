function [s_beat, fb_true, tau] = simulate_single_target(chirp_struct, range_m, velocity_mps, amplitude)
% SIMULATE_SINGLE_TARGET Synthesizes de-chirped beat signal for a single monostatic target.
% Uses analog-equivalent stretch processing de-chirp architecture.
%
% Formulas:
%   tau = 2 * R / c
%   f_b = S * tau = 2 * S * R / c
%   phi_const = 2*pi*fc*tau - pi*S*tau^2
%   s_beat(t) = amplitude * exp( j * ( 2*pi*f_b*t + phi_const ) )
%
% Inputs:
%   chirp_struct - Struct from generate_fmcw_chirp
%   range_m      - Target range R in meters (> 0)
%   velocity_mps - Target radial velocity v in m/s (positive = receding)
%   amplitude    - Reflection amplitude scale factor alpha (default 1.0)
%
% Outputs:
%   s_beat  - 1D complex beat signal array
%   fb_true - Theoretical beat frequency in Hz
%   tau     - Round-trip delay in seconds

if nargin < 4
    amplitude = 1.0;
end

c = 299792458.0;
S = chirp_struct.chirp_slope_hz_per_sec;
fc = chirp_struct.carrier_frequency_hz;
t = chirp_struct.time_vector;

tau = 2.0 * range_m / c;
fb_true = S * tau;
phi_const = 2.0 * pi * fc * tau - pi * S * (tau^2);

s_beat = amplitude * exp(1i * (2.0 * pi * fb_true * t + phi_const));
end
