clear; clc; close all;

%% ========================================================================
%  WR90: fminsearch — 3 frequencies
%% ========================================================================

%% Physical constants
c0  = 299792458;    % [m/s]
mu0 = 4*pi*1e-7;    % [H/m]

%% WR90 waveguide
a  = 22.86e-3;      % [m]
kc = pi / a;        % [rad/m]

%% Material
d = 5e-3;           % [m]

%% Permeabilities and permittivities
mu1 = 1; mu2 = 1; mu3 = 1;
eps1 = 1; eps3 = 1;

%% Search grid — 200x200 for better resolution
N = 200;                             % number of points per axis
eps_real_vec = linspace(1, 10, N);   % real part of epsilon, over [1,10]
eps_img_vec  = linspace(-5, 5,  N);   % imaginary part of epsilon, over [0,1]

%% Reference values (true values to recover)
eps_real_ref = 4.2;
eps_img_ref  = 4.2 * 0.058;                    % = 0.1125
eps2_ref     = eps_real_ref - 1j*eps_img_ref;

%% Threshold — change this value to show more or fewer points
threshold = 1e-5;   % ← modify here to change the threshold

%% 3 frequencies to analyze
f_choix = [8.2e9, 10.3e9, 12.4e9];   % [Hz]

%% ========================================================================
%  MAIN LOOP — one iteration per frequency
%% ========================================================================

for k = 1:3

    %% Parameters for this frequency
    fk  = f_choix(k);       % [Hz]
    wk  = 2*pi*fk;          % [rad/s]
    k0k = wk / c0;          % [rad/m]

    %% Reference S-parameters at this frequency
    [S11_ref, S21_ref] = calcul_S(eps2_ref, fk, wk, k0k, kc, mu0, mu1, mu2, mu3, eps1, eps3, d);

    %% Compute delta over the full NxN grid
    DELTA = zeros(N, N);

    for i = 1:N
        for j = 1:N
            eps_test = eps_real_vec(i) - 1j*eps_img_vec(j);
            [S11_c, S21_c] = calcul_S(eps_test, fk, wk, k0k, kc, mu0, mu1, mu2, mu3, eps1, eps3, d);
            DELTA(j,i) = abs(S11_c - S11_ref)^2 + abs(S21_c - S21_ref)^2;
        end
    end

    %% Find grid minimum (starting point for fminsearch)
    [delta_min, idx_min] = min(DELTA(:));
    [row_min, col_min]   = ind2sub(size(DELTA), idx_min);
    eps_real_min         = eps_real_vec(col_min);
    eps_img_min          = eps_img_vec(row_min);

    %% fminsearch from the best grid point
    x0_fk    = [eps_real_min, eps_img_min];   % starting point
    x_opt_fk = fminsearch(@(x) delta_func(x, S11_ref, S21_ref, fk, wk, k0k, ...
                kc, mu0, mu1, mu2, mu3, eps1, eps3, d), x0_fk);
    delta_exact = delta_func(x_opt_fk, S11_ref, S21_ref, fk, wk, k0k, ...
                kc, mu0, mu1, mu2, mu3, eps1, eps3, d);

    %% Find all grid points with delta < threshold
    [rows_s, cols_s] = find(DELTA < threshold);
    nb_pts           = length(rows_s);

    %% Command Window
    fprintf('\n=============================================================\n');
    fprintf('  Frequency : %.1f GHz\n', fk/1e9);
    fprintf('=============================================================\n');
    fprintf('  All points with delta < %.0e :\n', threshold);
    fprintf('  %-12s | %-12s | %-12s | %-12s\n', ...
            'eps_real', 'eps_img', 'delta', 'Status');
    fprintf('  -----------------------------------------------------------\n');

    if nb_pts == 0
        fprintf('  No point found — increase the threshold\n');
    else
        % Extract delta values for these points
        delta_pts = zeros(nb_pts, 1);
        for m = 1:nb_pts
            delta_pts(m) = DELTA(rows_s(m), cols_s(m));
        end

        % Sort by ascending delta — smallest first
        [delta_pts_sort, idx_sort] = sort(delta_pts);
        rows_sort = rows_s(idx_sort);
        cols_sort = cols_s(idx_sort);

        % Display each point
        for m = 1:nb_pts
            eps_r = eps_real_vec(cols_sort(m));
            eps_i = eps_img_vec(rows_sort(m));
            d_val = delta_pts_sort(m);

            if m == 1
                statut = '* MINIMUM';
            else
                statut = '';
            end
            fprintf('  %-12.4f | %-12.4f | %-12.2e | %s\n', ...
                eps_r, eps_i, d_val, statut);
        end
    end

    fprintf('  -----------------------------------------------------------\n');
    fprintf('  Total : %d point(s) with delta < %.0e\n', nb_pts, threshold);
    fprintf('  fminsearch exact : eps_real=%.6f  eps_img=%.6f  delta=%.2e\n', ...
        x_opt_fk(1), x_opt_fk(2), delta_exact);
    fprintf('  reference        : eps_real=%.4f   eps_img=%.4f\n', ...
        eps_real_ref, eps_img_ref);

    %% Plot for this frequency
    figure('Name', sprintf('f = %.1f GHz', fk/1e9), 'NumberTitle', 'off');
    imagesc(eps_real_vec, eps_img_vec, log10(DELTA));
    colormap('jet');
    set(gca, 'YDir', 'normal');
    colorbar;
    clim([-2 2]);
    hold on;

    % All points with delta < threshold — small red dots
    if nb_pts > 0
        scatter(eps_real_vec(cols_s), eps_img_vec(rows_s),10, 'red', 'filled', ...
            'DisplayName', sprintf('\\Delta < %.0e (%d pt)', threshold, nb_pts));
    end

    % Exact fminsearch minimum — black cross
    scatter(x_opt_fk(1), x_opt_fk(2), ...
        80, 'green', 'x', 'LineWidth', 3, ...
        'DisplayName', sprintf('fminsearch \\Delta=%.1e', delta_exact));

    % Annotation with exact values
    text(x_opt_fk(1) + 0.2, x_opt_fk(2) + 0.04, ...
        sprintf('\\epsilon''=%.4f\n\\epsilon''''=%.4f\n\\Delta=%.1e', ...
            x_opt_fk(1), x_opt_fk(2), delta_exact), ...
        'Color', 'b', 'FontSize', 8, 'FontWeight', 'bold', ...
        'BackgroundColor', 'white', 'EdgeColor', 'black');

    xlabel('\epsilon''');
    ylabel('\epsilon''''');
    title(sprintf('f = %.1f GHz', fk/1e9));
    legend('Location', 'northeast', 'FontSize', 10);
    grid on;

end

fprintf('\nDone!\n');

%% ========================================================================
%  FUNCTION : delta function for a single frequency
%% ========================================================================

function delta = delta_func(x, S11_ref, S21_ref, f, w, k0, kc, mu0, mu1, mu2, mu3, eps1, eps3, d)
    eps_real = x(1);
    eps_img  = x(2);

    % Penalty if outside search interval
    if eps_real < 1 || eps_real > 10 || eps_img < 0 || eps_img > 1
        delta = 1e10;
        return;
    end

    eps_test = eps_real - 1j*eps_img;
    [S11_c, S21_c] = calcul_S(eps_test, f, w, k0, kc, mu0, mu1, mu2, mu3, eps1, eps3, d);
    delta = abs(S11_c - S11_ref)^2 + abs(S21_c - S21_ref)^2;
end
%% ========================================================================
%  FUNCTION : compute S11 and S21
%% ========================================================================

function [S11, S21] = calcul_S(eps_r, f, w, k0, kc, mu0, mu1, mu2, mu3, eps1, eps3, d)

    f  = f(:);
    w  = w(:);
    k0 = k0(:);

    kz1 = sqrt( k0.^2 .* (mu1*eps1) - kc^2 );   % [rad/m]
    kz2 = sqrt( k0.^2 .* (mu2*eps_r) - kc^2 );  % [rad/m]
    kz3 = sqrt( k0.^2 .* (mu3*eps3) - kc^2 );   % [rad/m]

    C1  = 1;
    S11 = zeros(length(f), 1);
    S21 = zeros(length(f), 1);

    for n = 1:length(f)

        Y1 = kz1(n) / (w(n) * mu0 * mu1);   % [S/m]
        Y2 = kz2(n) / (w(n) * mu0 * mu2);   % [S/m]
        Y3 = kz3(n) / (w(n) * mu0 * mu3);   % [S/m]

        e2p = exp(-1j * kz2(n) * d);
        e2m = exp(+1j * kz2(n) * d);
        e3  = exp(-1j * kz3(n) * d);

        A = [ -1,      +1,      +1,      0      ; ...
              +Y1,     +Y2,     -Y2,     0      ; ...
               0,       e2p,     e2m,   -e3     ; ...
               0,   Y2*e2p, -Y2*e2m, -Y3*e3    ];

        b = [C1 ; Y1*C1 ; 0 ; 0];

        C      = A \ b;
        S11(n) = C(1);
        S21(n) = C(4);
    end
end