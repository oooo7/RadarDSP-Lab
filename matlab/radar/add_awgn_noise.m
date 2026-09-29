function [s_noisy, P_signal, P_noise] = add_awgn_noise(s_clean, snr_db, rng_seed)
% ADD_AWGN_NOISE Adds complex additive white Gaussian noise (AWGN) to signal.
%
%   Inputs:
%       s_clean: Input complex or real signal array
%       snr_db: Target Signal-to-Noise Ratio in dB (SNR = 10 log10(P_signal / P_noise))
%       rng_seed: Optional integer seed for MATLAB PRNG
%
%   Outputs:
%       s_noisy: Noisy output signal array
%       P_signal: Measured mean signal power
%       P_noise: Measured mean noise power added

    if nargin >= 3 && ~isempty(rng_seed)
        rng(rng_seed);
    end

    P_signal = mean(abs(s_clean(:)).^2);
    snr_linear = 10^(snr_db / 10);
    P_noise = P_signal / snr_linear;

    if isreal(s_clean)
        noise = sqrt(P_noise) * randn(size(s_clean));
    else
        % Complex AWGN with equal variance P_noise/2 per real and imaginary channel
        noise = sqrt(P_noise / 2) * (randn(size(s_clean)) + 1i * randn(size(s_clean)));
    end

    s_noisy = s_clean + noise;
end
