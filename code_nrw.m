clear; clc; close all;

%% ========================================================================
%  WR90: NRW method — extract epsilon and mu from S11 and S21
%% ========================================================================

%% Physical constants
c0  = 299792458;    % [m/s]
mu0 = 4*pi*1e-7;    % [H/m]

%% WR90 waveguide
a  = 22.86e-3;      % [m]
kc = pi / a;        % [rad/m]

%% Sample thickness
d = 5e-3;           % [m]

%% Surrounding medium (air)
mu1  = 1;
eps1 = 1;

%% True values (reference)
eps_real_ref = 4.5;
eps_img_ref  = 4.5 * 0.025;          % = 0.1125
eps2_ref     = eps_real_ref - 1j*eps_img_ref;
mu_real_ref  = 1.0;
mu_img_ref   = 0.0;
mu2_ref      = mu_real_ref - 1j*mu_img_ref;

%% Frequencies — keep everything as row vectors (1×N)
f  = linspace(8.2e9, 12.4e9, 100);  % row vector
w  = 2*pi*f;
k0 = w / c0;

%% ========================================================================
%  STEP 1 — Analytical S11 and S21 for a slab in waveguide
%% ========================================================================

% kz in air and in material  (choose branch: real part >= 0)
kz1 = sqrt(k0.^2 * (mu1*eps1)         - kc^2);
kz2 = sqrt(k0.^2 * (mu2_ref*eps2_ref) - kc^2);

% Enforce correct branch: Re(kz) >= 0
kz1 = conj(kz1) .* (real(kz1) < 0) + kz1 .* (real(kz1) >= 0);
kz2 = conj(kz2) .* (real(kz2) < 0) + kz2 .* (real(kz2) >= 0);

% Wave impedances in waveguide: eta = omega*mu0*mu / kz
eta1 = w * mu0 * mu1     ./ kz1;
eta2 = w * mu0 * mu2_ref ./ kz2;

% Interface reflection coefficient
Gamma_true = (eta2 - eta1) ./ (eta2 + eta1);

% Round-trip propagation factor
P_true = exp(-1j .* kz2 .* d);

% Analytical S-parameters
denom = 1 - Gamma_true.^2 .* P_true.^2;
S11   = Gamma_true .* (1 - P_true.^2) ./ denom;
S21   = (1 - Gamma_true.^2) .* P_true ./ denom;

fprintf('Reference S-parameters computed.\n');
fprintf('  eps_real = %.4f\n', eps_real_ref);
fprintf('  eps_img  = %.4f\n', eps_img_ref);
fprintf('  mu_real  = %.4f\n', mu_real_ref);
fprintf('  mu_img   = %.4f\n\n', mu_img_ref);

%% ========================================================================
%  STEP 2 — NRW inversion
%% ========================================================================

%% x = (1 - S21^2 + S11^2) / (2*S11)
x_nrw = (1 - S21.^2 + S11.^2) ./ (2 .* S11);

%% Gamma: choose root with |Gamma| <= 1
Gamma1 = x_nrw + sqrt(x_nrw.^2 - 1);
Gamma2 = x_nrw - sqrt(x_nrw.^2 - 1);

Gamma = zeros(1, length(f));
for n = 1:length(f)
    if abs(Gamma2(n)) <= abs(Gamma1(n))
        Gamma(n) = Gamma2(n);
    else
        Gamma(n) = Gamma1(n);
    end
end

%% P
P = (S11 + S21 - Gamma) ./ (1 - Gamma .* (S11 + S21));

%% kz2 from P = exp(-j*kz2*d)
% log(P) = -j*kz2*d  =>  kz2 = j*log(P)/d
% Enforce Re(kz2) >= 0 for physical solution
kz2_nrw = 1j .* log(P) ./ d;

% If real part is negative, take conjugate (flip branch)
neg = real(kz2_nrw) < 0;
kz2_nrw(neg) = -kz2_nrw(neg);

%% eta2 from Gamma definition: Gamma = (eta2-eta1)/(eta2+eta1)
eta1_nrw = w * mu0 * mu1 ./ kz1;
eta2_nrw = eta1_nrw .* (1 + Gamma) ./ (1 - Gamma);

%% mu_r = eta2 * kz2 / (omega * mu0)
mu_r_nrw = eta2_nrw .* kz2_nrw ./ (w * mu0);

%% eps_r from dispersion: kz2^2 = k0^2*eps*mu - kc^2
eps_r_nrw = (kz2_nrw.^2 + kc^2) ./ (k0.^2 .* mu_r_nrw);

%% ========================================================================
%  STEP 3 — Extract real and imaginary parts
%% ========================================================================

eps_real_nrw =  real(eps_r_nrw);
eps_img_nrw  = -imag(eps_r_nrw);
mu_real_nrw  =  real(mu_r_nrw);
mu_img_nrw   = -imag(mu_r_nrw);

%% ========================================================================
%  STEP 4 — Console display
%% ========================================================================

fprintf('=============================================================\n');
fprintf('  NRW Results (mean over frequency)\n');
fprintf('=============================================================\n');
fprintf('  %-12s | %-12s | %-12s\n', 'Parameter', 'NRW', 'Reference');
fprintf('  ------------+-------------+-------------\n');
fprintf('  %-12s | %-12.6f | %-12.6f\n', 'eps_real', mean(eps_real_nrw), eps_real_ref);
fprintf('  %-12s | %-12.6f | %-12.6f\n', 'eps_img',  mean(eps_img_nrw),  eps_img_ref);
fprintf('  %-12s | %-12.6f | %-12.6f\n', 'mu_real',  mean(mu_real_nrw),  mu_real_ref);
fprintf('  %-12s | %-12.6f | %-12.6f\n', 'mu_img',   mean(mu_img_nrw),   mu_img_ref);
fprintf('  ------------+-------------+-------------\n');
fprintf('\nDone!\n');

%% ========================================================================
%  STEP 5 — Plots
%% ========================================================================

f_GHz = f / 1e9;   % still a row vector

figure('Name','NRW Results','NumberTitle','off');

subplot(2,2,1);
plot(f_GHz, eps_real_nrw, 'b-', 'LineWidth', 2); hold on;
yline(eps_real_ref, 'r--', 'LineWidth', 1.5);
xlabel('Frequency [GHz]'); ylabel('\epsilon''');
title('Real part of \epsilon');
legend('NRW','Reference','Location','best'); grid on;

subplot(2,2,2);
plot(f_GHz, eps_img_nrw, 'b-', 'LineWidth', 2); hold on;
yline(eps_img_ref, 'r--', 'LineWidth', 1.5);
xlabel('Frequency [GHz]'); ylabel('\epsilon''''');
title('Imaginary part of \epsilon');
legend('NRW','Reference','Location','best'); grid on;

subplot(2,2,3);
plot(f_GHz, mu_real_nrw, 'b-', 'LineWidth', 2); hold on;
yline(mu_real_ref, 'r--', 'LineWidth', 1.5);
xlabel('Frequency [GHz]'); ylabel('\mu''');
title('Real part of \mu');
legend('NRW','Reference','Location','best'); grid on;

subplot(2,2,4);
plot(f_GHz, mu_img_nrw, 'b-', 'LineWidth', 2); hold on;
yline(mu_img_ref, 'r--', 'LineWidth', 1.5);
xlabel('Frequency [GHz]'); ylabel('\mu''''');
title('Imaginary part of \mu');
legend('NRW','Reference','Location','best'); grid on;

sgtitle('NRW Method — Extracted \epsilon and \mu vs Reference');