function [chirp_struct] = generate_fmcw_chirp(carrier_freq_hz, bandwidth_hz, chirp_duration_sec, sampling_rate_hz)
% GENERATE_FMCW_CHIRP Constructs transmit FMCW chirp parameters matching Python chirp.py.
%
% Inputs:
%   carrier_freq_hz    - Carrier frequency fc in Hz (e.g., 77 GHz)
%   bandwidth_hz       - Sweep bandwidth B in Hz (e.g., 150 MHz)
%   chirp_duration_sec - Chirp duration Tc in seconds (e.g., 50 us or 100 us)
%   sampling_rate_hz   - ADC sampling rate Fs in Hz (e.g., 20 MHz)
%
% Outputs:
%   chirp_struct - Struct payload holding slope S, time vector, and tx signal

c = 299792458.0;
S = bandwidth_hz / chirp_duration_sec;
num_samples = round(chirp_duration_sec * sampling_rate_hz);
t = (0:num_samples-1) / double(sampling_rate_hz);

% Complex baseband transmit chirp: s_tx(t) = exp(j * pi * S * t^2)
s_tx = exp(1i * pi * S * t.^2);

chirp_struct.carrier_frequency_hz = carrier_freq_hz;
chirp_struct.bandwidth_hz = bandwidth_hz;
chirp_struct.chirp_duration_sec = chirp_duration_sec;
chirp_struct.sampling_rate_hz = sampling_rate_hz;
chirp_struct.chirp_slope_hz_per_sec = S;
chirp_struct.wavelength_m = c / carrier_freq_hz;
chirp_struct.speed_of_light_m_per_s = c;
chirp_struct.time_vector = t;
chirp_struct.tx_signal = s_tx;
end
