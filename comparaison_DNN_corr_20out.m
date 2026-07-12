clear; clc; close all;

%% ========================================================================
%  WR90: Comparison fminsearch vs DNN 250k noise corr 20 outputs
%  — real lab data
%% ========================================================================

%% Physical constants
c0  = 299792458;
mu0 = 4*pi*1e-7;
a   = 22.86e-3;
kc  = pi / a;
d   = 5e-3;
mu1 = 1; mu2 = 1; mu3 = 1;
eps1 = 1; eps3 = 1;

%% True values
eps_real_true  = 3.5;
eps_img_true   = 0.175;
tan_delta_true = 0.05;

%% ========================================================================
%  STEP 1 — Load and normalize S21 lab data
%% ========================================================================

data_S21_with  = readmatrix('S-Parameters_S2,1_WITH_SAMPLE.txt');
data_S21_empty = readmatrix('S-Parameters_S2,1_EMPTY_WG.txt');

f_raw = data_S21_with(:,1) * 1e9;
w_raw = 2*pi*f_raw;
k0_raw = w_raw / c0;

S21_with  = data_S21_with(:,2)  .* exp(1j .* data_S21_with(:,3)  .* pi/180);
S21_empty = data_S21_empty(:,2) .* exp(1j .* data_S21_empty(:,3) .* pi/180);
S21_norm  = S21_with ./ S21_empty;

fprintf('S21 loaded. |S21_norm| mean = %.4f\n\n', mean(abs(S21_norm)));

%% ========================================================================
%  STEP 2 — Interpolate to 100 frequencies
%% ========================================================================

N_freq    = 100;
f_interp  = linspace(8.2e9, 12.4e9, N_freq);
w_interp  = 2*pi*f_interp;
k0_interp = w_interp / c0;
f_GHz     = f_interp / 1e9;

S21_interp = interp1(f_raw, S21_norm, f_interp, 'linear');

fprintf('Interpolation done — %d frequencies.\n\n', N_freq);

%% ========================================================================
%  STEP 3 — DNN 20 outputs prediction
%% ========================================================================

fprintf('Loading DNN 250k noise corr 20 outputs...\n');
load('DNN_trained_250k_noise_corr_20out.mat');

X_input   = [real(S21_interp), imag(S21_interp)];
X_input_n = (X_input - X_mean) ./ X_std;
Y_pred_n  = predict(net, X_input_n);

% Denormalize
eps_real_dnn = Y_pred_n(1:N_verif)      * Y_real_std + Y_real_mean;
eps_img_dnn  = Y_pred_n(N_verif+1:end)  * Y_img_std  + Y_img_mean;
tan_delta_dnn = eps_img_dnn ./ eps_real_dnn;

f_verif_GHz = f_verif / 1e9;

fprintf('DNN done.\n');
fprintf('  Mean eps_real = %.4f\n', mean(eps_real_dnn));
fprintf('  Mean eps_img  = %.4f\n', mean(eps_img_dnn));
fprintf('  Mean tan(d)   = %.4f\n\n', mean(tan_delta_dnn));

%% ========================================================================
%  STEP 4 — fminsearch frequency by frequency
%% ========================================================================

fprintf('Running fminsearch frequency by frequency...\n');

x0        = [3.5, 0.1];
opts_fmin = optimset('MaxFunEvals', 10000, 'MaxIter', 5000, ...
                     'TolFun', 1e-10, 'TolX', 1e-10);

eps_real_fmin = zeros(1, N_freq);
eps_img_fmin  = zeros(1, N_freq);

hh = waitbar(0, 'fminsearch frequency by frequency...');
for ff = 1:N_freq
    x_opt = fminsearch(@(x) delta_func_single(x, S21_interp(ff), ...
        f_interp(ff), w_interp(ff), k0_interp(ff), ...
        kc, mu0, mu1, mu2, mu3, eps1, eps3, d), x0, opts_fmin);
    eps_real_fmin(ff) = x_opt(1);
    eps_img_fmin(ff)  = x_opt(2);
    waitbar(ff/N_freq, hh);
end
close(hh);

tan_delta_fmin = eps_img_fmin ./ eps_real_fmin;

fprintf('fminsearch done.\n');
fprintf('  Mean eps_real = %.4f\n', mean(eps_real_fmin));
fprintf('  Mean eps_img  = %.4f\n', mean(eps_img_fmin));
fprintf('  Mean tan(d)   = %.4f\n\n', mean(tan_delta_fmin));

%% ========================================================================
%  STEP 5 — Console comparison
%% ========================================================================

fprintf('=============================================================\n');
fprintf('  Comparison — Real lab data\n');
fprintf('=============================================================\n');
fprintf('  %-20s   eps_real   eps_img   tan(delta)\n', 'Method');
fprintf('  %-20s   %.4f     %.4f    %.4f\n', 'True value',   eps_real_true,        eps_img_true,         tan_delta_true);
fprintf('  %-20s   %.4f     %.4f    %.4f\n', 'fminsearch',   mean(eps_real_fmin),  mean(eps_img_fmin),   mean(tan_delta_fmin));
fprintf('  %-20s   %.4f     %.4f    %.4f\n', 'DNN 20 out',   mean(eps_real_dnn),   mean(eps_img_dnn),    mean(tan_delta_dnn));
fprintf('=============================================================\n\n');

% Errors
err_fmin_real = abs(mean(eps_real_fmin) - eps_real_true) / eps_real_true * 100;
err_fmin_tan  = abs(mean(tan_delta_fmin) - tan_delta_true) / tan_delta_true * 100;
err_dnn_real  = abs(mean(eps_real_dnn) - eps_real_true) / eps_real_true * 100;
err_dnn_tan   = abs(mean(tan_delta_dnn) - tan_delta_true) / tan_delta_true * 100;

fprintf('Errors :\n');
fprintf('  fminsearch : eps_real=%.2f%%  tan(d)=%.2f%%\n', err_fmin_real, err_fmin_tan);
fprintf('  DNN 20 out : eps_real=%.2f%%  tan(d)=%.2f%%\n\n', err_dnn_real, err_dnn_tan);

%% ========================================================================
%  STEP 6 — Plot eps_real vs frequency
%% ========================================================================

figure('Name', 'eps_real vs frequency', 'NumberTitle', 'off');
hold on;

plot(f_GHz, eps_real_fmin, 'b-', 'LineWidth', 1.5, ...
    'DisplayName', sprintf('fminsearch — mean=%.4f', mean(eps_real_fmin)));
plot(f_verif_GHz, eps_real_dnn, 'r-o', 'LineWidth', 2, 'MarkerSize', 6, ...
    'DisplayName', sprintf('DNN 20out — mean=%.4f', mean(eps_real_dnn)));

xlabel('Frequency [GHz]'); ylabel('\epsilon''');
title('\epsilon'' vs frequency — fminsearch vs DNN 20 outputs');
legend('Location', 'best'); grid on;

%% ========================================================================
%  STEP 7 — Plot eps_img vs frequency
%% ========================================================================

figure('Name', 'eps_img vs frequency', 'NumberTitle', 'off');
hold on;

plot(f_GHz, eps_img_fmin, 'b-', 'LineWidth', 1.5, ...
    'DisplayName', sprintf('fminsearch — mean=%.4f', mean(eps_img_fmin)));
plot(f_verif_GHz, eps_img_dnn, 'r-o', 'LineWidth', 2, 'MarkerSize', 6, ...
    'DisplayName', sprintf('DNN 20out — mean=%.4f', mean(eps_img_dnn)));

xlabel('Frequency [GHz]'); ylabel('\epsilon''''');
title('\epsilon'''' vs frequency — fminsearch vs DNN 20 outputs');
legend('Location', 'best'); grid on;

%% ========================================================================
%  STEP 8 — Plot tan_delta vs frequency
%% ========================================================================

figure('Name', 'tan_delta vs frequency', 'NumberTitle', 'off');
hold on;

plot(f_GHz, tan_delta_fmin, 'b-', 'LineWidth', 1.5, ...
    'DisplayName', sprintf('fminsearch — mean=%.4f', mean(tan_delta_fmin)));
plot(f_verif_GHz, tan_delta_dnn, 'r-o', 'LineWidth', 2, 'MarkerSize', 6, ...
    'DisplayName', sprintf('DNN 20out — mean=%.4f', mean(tan_delta_dnn)));

xlabel('Frequency [GHz]'); ylabel('tan(\delta)');
title('tan(\delta) vs frequency — fminsearch vs DNN 20 outputs');
legend('Location', 'best'); grid on;

fprintf('Done!\n');


%% ========================================================================
%  STEP 9 — Plot S21 measured vs computed with DNN properties
%% ========================================================================

% Compute S21 using mean DNN properties over full 100 frequencies
eps_real_dnn_mean = mean(eps_real_dnn);
eps_img_dnn_mean  = mean(eps_img_dnn);
eps_dnn_mean      = eps_real_dnn_mean - 1j * eps_img_dnn_mean;

S21_computed_dnn = zeros(1, N_freq);
for ff = 1:N_freq
    [~, S21_computed_dnn(ff)] = calcul_S_single(eps_dnn_mean, ...
        f_interp(ff), w_interp(ff), k0_interp(ff), ...
        kc, mu0, mu1, mu2, mu3, eps1, eps3, d);
end

% Magnitude
figure('Name', 'S21 magnitude — Measured vs DNN', 'NumberTitle', 'off');
hold on;
plot(f_GHz, 20*log10(abs(S21_interp)), 'b-', 'LineWidth', 1.5, ...
    'DisplayName', 'Measured');
plot(f_GHz, 20*log10(abs(S21_computed_dnn)), 'r--', 'LineWidth', 2, ...
    'DisplayName', sprintf('DNN (\epsilon''=%.4f, tan\delta=%.4f)', ...
    eps_real_dnn_mean, mean(tan_delta_dnn)));
xlabel('Frequency [GHz]'); ylabel('|S_{21}| [dB]');
title('S_{21} Magnitude — Measured vs Computed (DNN properties)');
legend('Location', 'best'); grid on;

% Phase
figure('Name', 'S21 phase — Measured vs DNN', 'NumberTitle', 'off');
hold on;
plot(f_GHz, angle(S21_interp)*180/pi, 'b-', 'LineWidth', 1.5, ...
    'DisplayName', 'Measured');
plot(f_GHz, angle(S21_computed_dnn)*180/pi, 'r--', 'LineWidth', 2, ...
    'DisplayName', sprintf('DNN (\epsilon''=%.4f, tan\delta=%.4f)', ...
    eps_real_dnn_mean, mean(tan_delta_dnn)));
xlabel('Frequency [GHz]'); ylabel('Phase S_{21} [deg]');
title('S_{21} Phase — Measured vs Computed (DNN properties)');
legend('Location', 'best'); grid on;

fprintf('S21 DNN comparison plots done.\n');

%% ========================================================================
%  FUNCTION : delta — single frequency
%% ========================================================================

function delta = delta_func_single(x, S21_ref, f, w, k0, kc, mu0, mu1, mu2, mu3, eps1, eps3, d)
    eps_test   = x(1) - 1j*x(2);
    [~, S21_c] = calcul_S_single(eps_test, f, w, k0, kc, mu0, mu1, mu2, mu3, eps1, eps3, d);
    delta      = abs(S21_c - S21_ref);
end

%% ========================================================================
%  FUNCTION : compute S21 — single frequency
%% ========================================================================

function [S11, S21] = calcul_S_single(eps_r, f, w, k0, kc, mu0, mu1, mu2, mu3, eps1, eps3, d)
    kz1 = sqrt(k0^2 * (mu1*eps1) - kc^2);
    kz2 = sqrt(k0^2 * (mu2*eps_r) - kc^2);
    kz3 = sqrt(k0^2 * (mu3*eps3)  - kc^2);
    C1  = 1;
    Y1  = kz1 / (w*mu0*mu1);
    Y2  = kz2 / (w*mu0*mu2);
    Y3  = kz3 / (w*mu0*mu3);
    e2p = exp(-1j*kz2*d);
    e2m = exp(+1j*kz2*d);
    e3  = exp(-1j*kz3*d);
    A = [ -1,   +1,    +1,    0    ; ...
          +Y1,  +Y2,   -Y2,   0    ; ...
           0,   e2p,   e2m,  -e3   ; ...
           0, Y2*e2p,-Y2*e2m,-Y3*e3];
    b   = [C1; Y1*C1; 0; 0];
    C   = A \ b;
    S11 = C(1);
    S21 = C(4);
end