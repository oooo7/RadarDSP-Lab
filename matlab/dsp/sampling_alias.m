function f_alias = sampling_alias(fundamental_freq_hz, sampling_rate_hz)
% SAMPLING_ALIAS Computes principal Nyquist zone alias frequency [0, Fs/2].
%
% Formula:
%   rem = fundamental_freq_hz mod sampling_rate_hz
%   f_alias = rem if rem <= Fs/2 else (Fs - rem)
%
% Inputs:
%   fundamental_freq_hz - Input signal frequency f in Hz (>= 0)
%   sampling_rate_hz    - Sampling rate Fs in Hz (> 0)
%
% Outputs:
%   f_alias - Theoretical apparent alias frequency in Hz

rem_val = mod(fundamental_freq_hz, sampling_rate_hz);
folding_freq = sampling_rate_hz / 2.0;

if rem_val <= folding_freq
    f_alias = rem_val;
else
    f_alias = sampling_rate_hz - rem_val;
end
end
