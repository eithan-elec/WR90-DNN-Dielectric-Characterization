clear; clc; close all;
 
%% ========================================================================
%  WR90: Compare measured S21 vs simulated S21 (eps=3.5, tan_delta=0.05)
%  + Extract eps' and tan_delta at each frequency
%% ========================================================================
 
%% Physical constants
c0  = 299792458;    % [m/s]
mu0 = 4*pi*1e-7;    % [H/m]
 
%% WR90 waveguide
a  = 22.86e-3;      % [m]
kc = pi / a;        % [rad/m]
 
%% Material thickness
d = 5e-3;           % [m]
 
%% Permeabilities and permittivities
mu1 = 1; mu2 = 1; mu3 = 1;
eps1 = 1; eps3 = 1;
 
%% Material properties true values
eps_real_ref = 3.5;
eps_img_ref  = 3.5 * 0.05;         % = 0.175
eps_mat      = eps_real_ref - 1j*eps_img_ref;
 
%% ========================================================================
%  STEP 1 — Read measured S-parameters
%% ========================================================================
 
data_S21_with  = readmatrix('S-Parameters_S2,1_WITH_SAMPLE.txt');
data_S21_empty = readmatrix('S-Parameters_S2,1_EMPTY_WG.txt');
 
%% ========================================================================
%  STEP 2 — Reconstruct and normalize measured S21
%% ========================================================================
 
f   = data_S21_with(:,1) * 1e9;   % [Hz]
w   = 2*pi*f;                      % [rad/s]
k0  = w / c0;                      % [rad/m]
 
S21_with  = data_S21_with(:,2)  .* exp(1j .* data_S21_with(:,3)  .* pi/180);
S21_empty = data_S21_empty(:,2) .* exp(1j .* data_S21_empty(:,3) .* pi/180);
 
S21_norm = S21_with ./ S21_empty;
 
%% ========================================================================
%  STEP 3 — Simulate S21 with true material properties
%% ========================================================================
 
[~, S21_sim] = calcul_S(eps_mat, f, w, k0, kc, mu0, mu1, mu2, mu3, eps1, eps3, d);
 
%% ========================================================================
%  STEP 4 — Extract eps' and tan_delta at each frequency
%  Mini grid 25x25 to find best starting point (no assumption on values)
%  then fminsearch to refine
%% ========================================================================
 
fprintf('Extracting eps and tan_delta at each frequency...\n');
 
N_f         = length(f);
eps_real_f  = zeros(N_f, 1);
tan_delta_f = zeros(N_f, 1);
 
%% Grid for starting point
N_grid       = 25;
eps_real_vec = linspace(1, 10, N_grid);
eps_img_vec  = linspace(0, 1,  N_grid);
 
for n = 1:N_f
 
    fn  = f(n);
    wn  = w(n);
    k0n = k0(n);
 
    S21_ref_n = S21_norm(n);
 
    %% Mini grid search — find best starting point
    DELTA_grid = zeros(N_grid, N_grid);
    for i = 1:N_grid
        for j = 1:N_grid
            eps_test = eps_real_vec(i) - 1j*eps_img_vec(j);
            [~, S21_c] = calcul_S(eps_test, fn, wn, k0n, kc, mu0, mu1, mu2, mu3, eps1, eps3, d);
            DELTA_grid(j,i) = abs(S21_c - S21_ref_n)^2;
        end
    end
 
    %% Best grid point → starting point for fminsearch
    [~, idx_min]       = min(DELTA_grid(:));
    [row_min, col_min] = ind2sub(size(DELTA_grid), idx_min);
    x0 = [eps_real_vec(col_min), eps_img_vec(row_min)];
 
    %% fminsearch from best grid point
    x_opt = fminsearch(@(x) delta_func_single(x, S21_ref_n, fn, wn, k0n, ...
        kc, mu0, mu1, mu2, mu3, eps1, eps3, d), x0);
 
    eps_real_f(n)  = x_opt(1);
    tan_delta_f(n) = x_opt(2) / x_opt(1);
 
end
 
%% Console — resultats
fprintf('=============================================================\n');
fprintf('  Extraction Results - frequency by frequency\n');
fprintf('=============================================================\n');
fprintf('  eps_real  mean = %.6f\n', mean(eps_real_f));
fprintf('  eps_real  min  = %.6f\n', min(eps_real_f));
fprintf('  eps_real  max  = %.6f\n', max(eps_real_f));
fprintf('  -----------------------------------------------------------\n');
fprintf('  tan_delta mean = %.6f\n', mean(tan_delta_f));
fprintf('  tan_delta min  = %.6f\n', min(tan_delta_f));
fprintf('  tan_delta max  = %.6f\n', max(tan_delta_f));
fprintf('=============================================================\n\n');
 
 
%% ========================================================================
%  STEP 5 — Plot 1 : Measured vs Simulated S21
%% ========================================================================
 
f_GHz = f / 1e9;
 
figure('Name', 'Measured vs Simulated S21', 'NumberTitle', 'off');
 
subplot(2,1,1);
plot(f_GHz, 20*log10(abs(S21_norm)), 'b-',  'LineWidth', 2, 'DisplayName', 'Measured');
hold on;
plot(f_GHz, 20*log10(abs(S21_sim)),  'r--', 'LineWidth', 2, 'DisplayName', ...
    sprintf('Simulated (\\epsilon''=%.1f, tan\\delta=%.2f)', ...
    eps_real_ref, eps_img_ref/eps_real_ref));
xlabel('Frequency [GHz]'); ylabel('|S21| [dB]');
title('S21 magnitude'); legend; grid on;
 
subplot(2,1,2);
plot(f_GHz, angle(S21_norm)*180/pi, 'b-',  'LineWidth', 2, 'DisplayName', 'Measured');
hold on;
plot(f_GHz, angle(S21_sim)*180/pi,  'r--', 'LineWidth', 2, 'DisplayName', ...
    sprintf('Simulated (\\epsilon''=%.1f, tan\\delta=%.2f)', ...
    eps_real_ref, eps_img_ref/eps_real_ref));
xlabel('Frequency [GHz]'); ylabel('Phase [deg]');
title('S21 phase'); legend; grid on;
 
sgtitle(sprintf('Measured vs Simulated — \\epsilon''=%.1f  tan\\delta=%.2f', ...
    eps_real_ref, eps_img_ref/eps_real_ref));
 
%% ========================================================================
%  STEP 6 — Plot 2 : eps' vs frequency
%% ========================================================================
 
figure('Name', 'epsilon vs frequency', 'NumberTitle', 'off');
plot(f_GHz, eps_real_f, 'b-', 'LineWidth', 2, 'DisplayName', 'Extracted \epsilon''');
hold on;
yline(eps_real_ref, 'r--', 'LineWidth', 2, 'DisplayName', ...
    sprintf('True value \\epsilon''= %.1f', eps_real_ref));
xlabel('Frequency [GHz]'); ylabel('\epsilon''');
title('\epsilon'' extracted at each frequency');
legend; grid on;
 
%% ========================================================================
%  STEP 7 — Plot 3 : tan_delta vs frequency
%% ========================================================================
 
figure('Name', 'tan delta vs frequency', 'NumberTitle', 'off');
plot(f_GHz, tan_delta_f, 'b-', 'LineWidth', 2, 'DisplayName', 'Extracted tan\delta');
hold on;
yline(eps_img_ref/eps_real_ref, 'r--', 'LineWidth', 2, 'DisplayName', ...
    sprintf('True value tan\\delta = %.2f', eps_img_ref/eps_real_ref));
xlabel('Frequency [GHz]'); ylabel('tan\delta');
title('tan\delta extracted at each frequency');
legend; grid on;
 
fprintf('Done!\n');
 
%% ========================================================================
%  FUNCTION : delta function — single frequency
%% ========================================================================
 
function delta = delta_func_single(x, S21_ref, f, w, k0, kc, mu0, mu1, mu2, mu3, eps1, eps3, d)
 
    eps_real = x(1);
    eps_img  = x(2);
 
    if eps_real < 1 || eps_real > 10 || eps_img < 0 || eps_img > 1
        delta = 1e10;
        return;
    end
 
    eps_test = eps_real - 1j*eps_img;
    [~, S21_c] = calcul_S(eps_test, f, w, k0, kc, mu0, mu1, mu2, mu3, eps1, eps3, d);
    delta = abs(S21_c - S21_ref)^2;
end
 
%% ========================================================================
%  FUNCTION : compute S11 and S21
%% ========================================================================
 
function [S11, S21] = calcul_S(eps_r, f, w, k0, kc, mu0, mu1, mu2, mu3, eps1, eps3, d)
 
    f  = f(:);
    w  = w(:);
    k0 = k0(:);
 
    kz1 = sqrt( k0.^2 .* (mu1*eps1) - kc^2 );
    kz2 = sqrt( k0.^2 .* (mu2*eps_r) - kc^2 );
    kz3 = sqrt( k0.^2 .* (mu3*eps3) - kc^2 );
 
    C1  = 1;
    S11 = zeros(length(f), 1);
    S21 = zeros(length(f), 1);
 
    for n = 1:length(f)
 
        Y1 = kz1(n) / (w(n) * mu0 * mu1);
        Y2 = kz2(n) / (w(n) * mu0 * mu2);
        Y3 = kz3(n) / (w(n) * mu0 * mu3);
 
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
 


