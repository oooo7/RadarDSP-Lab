function [detection_mask, threshold_power, alpha, num_detections] = run_ca_cfar(RD_matrix, Pfa, g_r, g_d, t_r, t_d)
% RUN_CA_CFAR 2D Cell-Averaging Constant False Alarm Rate (CA-CFAR) detection.
%
%   Inputs:
%       RD_matrix: 2D complex or magnitude Range-Doppler matrix [num_doppler, num_range]
%       Pfa: Target false alarm probability (default 1e-4)
%       g_r: Guard cells per side in Range direction (default 2)
%       g_d: Guard cells per side in Doppler direction (default 2)
%       t_r: Training cells per side in Range direction (default 4)
%       t_d: Training cells per side in Doppler direction (default 4)
%
%   Outputs:
%       detection_mask: 2D logical matrix (true for detected cells)
%       threshold_power: 2D adaptive threshold power matrix
%       alpha: CFAR multiplier alpha
%       num_detections: Total count of detected cells

    if nargin < 2 || isempty(Pfa), Pfa = 1e-4; end
    if nargin < 3 || isempty(g_r), g_r = 2; end
    if nargin < 4 || isempty(g_d), g_d = 2; end
    if nargin < 5 || isempty(t_r), t_r = 4; end
    if nargin < 6 || isempty(t_d), t_d = 4; end

    [num_d, num_r] = size(RD_matrix);

    size_outer_d = 2 * (t_d + g_d) + 1;
    size_outer_r = 2 * (t_r + g_r) + 1;
    size_inner_d = 2 * g_d + 1;
    size_inner_r = 2 * g_r + 1;

    n_outer = size_outer_d * size_outer_r;
    n_inner = size_inner_d * size_inner_r;
    n_train = n_outer - n_inner;

    % Calculate threshold multiplier alpha
    alpha = n_train * (Pfa^(-1 / n_train) - 1);

    % Power domain processing
    if ~isreal(RD_matrix)
        power_matrix = abs(RD_matrix).^2;
    else
        power_matrix = RD_matrix.^2;
    end

    % 2D window sum using 2D convolution
    k_outer = ones(size_outer_d, size_outer_r);
    k_inner = zeros(size_outer_d, size_outer_r);
    offset_d = t_d + g_d + 1 - (g_d + 1);
    offset_r = t_r + g_r + 1 - (g_r + 1);
    k_inner(offset_d + (1:size_inner_d), offset_r + (1:size_inner_r)) = 1;

    k_train = k_outer - k_inner;

    sum_train = conv2(power_matrix, k_train, 'same');
    local_noise_power = sum_train / n_train;
    threshold_power = alpha * max(local_noise_power, 1e-15);

    raw_mask = power_matrix > threshold_power;

    % Boundary handling: zero out edges where window extends beyond map
    border_d = t_d + g_d;
    border_r = t_r + g_r;

    detection_mask = false(num_d, num_r);
    detection_mask(border_d+1 : num_d-border_d, border_r+1 : num_r-border_r) = ...
        raw_mask(border_d+1 : num_d-border_d, border_r+1 : num_r-border_r);

    num_detections = sum(detection_mask(:));
end
