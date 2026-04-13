%% helper_Generate_Initial_Gains.m
% Print two stable initial gains for Version 1
% 1) Kbar0_beh : stabilizing behavior gain for augmented true model
% 2) Ko0_rl    : stabilizing initial output-feedback gain for exact lifted homogeneous closed loop
%
% This version uses the correct stability criterion:
%   discrete-time asymptotic stability <=> spectral radius < 1
%
% It will:
%   - build Kbar0_beh and verify rho(A_aug-Bbar*Kbar0_beh) < 1
%   - build Ko_seed_bridge = Kbar0_beh * Mbar_ls
%   - verify rho(Acl(Ko_seed_bridge)) < 1
%   - if yes, directly use Ko0_rl = Ko_seed_bridge
%   - if no, stop and report failure
%
% Output:
%   prints Kbar0_beh and Ko0_rl
%   saves version1_exported_seed_simple_stable.mat

clear; clc; close all;
rng(7);

%% ===================== hidden plant and internal model =====================
dt = 0.05;
A  = [1, dt; 0, 1];
B  = [0.5*dt^2; dt];
C  = [1, 0];

n = size(A,1);
m = size(B,2);
p = size(C,1);

w_ref = 0.8;
th    = w_ref * dt;
S     = [cos(th),  sin(th); ...
        -sin(th),  cos(th)];
F     = S;
G     = [0; 1];
Tff   = [1, 0];

q     = size(S,1);
nz    = n + p*q;
rzeta = nz*m + nz*p + nz*p;

A_aug = [A,      -B*Tff; ...
        -G*C,     F];
Bbar  = [B; zeros(p*q, m)];
Cbar  = [C, zeros(p, p*q)];

Qy   = 10;
Rbar = 1;
Qx   = Cbar' * Qy * Cbar;

%% ===================== Step 1: build Kbar0_beh =====================
[Kbar_star,~,~] = dlqr(A_aug, Bbar, Qx, Rbar);

Delta = [0.25, -0.15, 0.10, -0.10];
alpha = 1.0;

while true
    Kbar0_beh = Kbar_star + alpha * Delta;
    rho_beh   = max(abs(eig(A_aug - Bbar * Kbar0_beh)));
    if rho_beh < 1
        break;
    end
    alpha = 0.5 * alpha;
    if alpha < 1e-10
        error('Could not find a stabilizing Kbar0_beh.');
    end
end

fprintf('\n================ behavior gain check ================\n');
fprintf('rho(A_aug - Bbar*Kbar0_beh) = %.12f\n', rho_beh);

if rho_beh < 1
    fprintf('PASS: Kbar0_beh is Schur-stable.\n');
else
    error('FAIL: Kbar0_beh is not Schur-stable.');
end

%% ===================== Step 2: collect data =====================
Azeta_poles = [0.15, 0.25, 0.35, 0.45];
A_zeta      = companion_from_poles(Azeta_poles);
b           = [zeros(nz-1,1); 1];

k0 = 250;
kf = 2600;

x      = zeros(n,      kf+2);
zbar   = zeros(p*q,    kf+2);
y      = zeros(p,      kf+2);
r      = zeros(nz,     kf+2);
ubar   = zeros(m,      kf+1);
theta  = zeros(p,      kf+1);

zeta_u     = zeros(nz*m, kf+2);
zeta_y     = zeros(nz*p, kf+2);
zeta_theta = zeros(nz*p, kf+2);

x(:,1)    = [0.6; -0.3];
zbar(:,1) = [0.2; -0.1];
y(:,1)    = C * x(:,1);
r(:,1)    = [x(:,1); zbar(:,1)];

for k = 1:(kf+1)
    theta(:,k) = 0.9*sin(0.11*k) + 0.5*sin(0.27*k) + 0.3*sin(0.49*k) + 0.15*randn;
end

for k = 1:(kf+1)
    xi_k      = 0.8 * (0.8*sin(0.13*k) + 0.6*sin(0.71*k) + 0.3*randn);
    ubar(:,k) = -Kbar0_beh * r(:,k) + xi_k;

    zeta_u(:,k+1)     = kron(eye(m), A_zeta) * zeta_u(:,k)     + kron(ubar(:,k),  b);
    zeta_y(:,k+1)     = kron(eye(p), A_zeta) * zeta_y(:,k)     + kron(y(:,k),     b);
    zeta_theta(:,k+1) = kron(eye(p), A_zeta) * zeta_theta(:,k) + kron(theta(:,k), b);

    zbar(:,k+1) = F * zbar(:,k) - G * y(:,k) + G * theta(:,k);
    x(:,k+1)    = A * x(:,k) + B * (ubar(:,k) - Tff * zbar(:,k));
    y(:,k+1)    = C * x(:,k+1);
    r(:,k+1)    = [x(:,k+1); zbar(:,k+1)];
end

zeta_bar = [zeta_u; zeta_y; zeta_theta];

%% ===================== Step 3: bridge seed =====================
Mbar_ls        = r(:, k0:(kf+1)) / zeta_bar(:, k0:(kf+1));
Ko_seed_bridge = Kbar0_beh * Mbar_ls;

Acl_bridge = build_Acl(Ko_seed_bridge, A_aug, Bbar, Cbar, A_zeta, b, m, p, nz);
rho_bridge = max(abs(eig(Acl_bridge)));

fprintf('\n================ bridge seed check ================\n');
fprintf('rho(Acl(Ko_seed_bridge)) = %.12f\n', rho_bridge);

if rho_bridge < 1
    fprintf('PASS: Ko_seed_bridge is Schur-stable for the exact lifted homogeneous closed loop.\n');
    Ko0_rl = Ko_seed_bridge;
else
    error('FAIL: Ko_seed_bridge is not Schur-stable. Need another search method.');
end

%% ===================== optional Lyapunov info =====================
Qcert = eye(size(Acl_bridge,1));
Pcert = dlyap(Acl_bridge', Qcert);
Pcert = 0.5 * (Pcert + Pcert');

lyap_res = norm(Acl_bridge' * Pcert * Acl_bridge - Pcert + Qcert, 'fro');
mineigP  = min(eig(Pcert));

fprintf('\nSupplementary info (not used as pass/fail):\n');
fprintf('min eig(Pcert)    = %.12e\n', mineigP);
fprintf('Lyapunov residual = %.12e\n', lyap_res);

%% ===================== final print =====================
fprintf('\n================ FINAL EXPORTED SEEDS ================\n');

fprintf('Kbar0_beh = [');
fprintf(' %.16g', Kbar0_beh);
fprintf(' ];\n');

fprintf('Ko0_rl    = [');
fprintf(' %.16g', Ko0_rl);
fprintf(' ];\n');

save('version1_exported_seed_simple_stable.mat', ...
    'Kbar0_beh', 'Ko0_rl', 'rho_beh', 'rho_bridge', 'mineigP', 'lyap_res');

fprintf('\nSaved version1_exported_seed_simple_stable.mat\n');

%% ===================== local functions =====================

function Acl = build_Acl(Ko, A_aug, Bbar, Cbar, A_zeta, b, m, p, nz)
    rzeta = nz*m + nz*p + nz*p;

    Azu  = kron(eye(m), A_zeta);
    Azy  = kron(eye(p), A_zeta);
    Azth = kron(eye(p), A_zeta);

    Eu = kron(eye(m), b);
    Ey = kron(eye(p), b);

    nxcl = nz + rzeta;

    ir  = 1:nz;
    izu = nz + (1:nz*m);
    izy = nz + nz*m + (1:nz*p);
    izt = nz + nz*m + nz*p + (1:nz*p);

    Acl = zeros(nxcl, nxcl);

    % r_{k+1} = A_aug r_k + Bbar ubar_k, ubar_k = -Ko * zeta_bar_k
    Acl(ir, ir) = A_aug;
    Acl(ir, nz+1:end) = -Bbar * Ko;

    % zeta_u^+ = Azu zeta_u + Eu ubar
    Acl(izu, izu) = Azu;
    Acl(izu, nz+1:end) = Acl(izu, nz+1:end) - Eu * Ko;

    % zeta_y^+ = Azy zeta_y + Ey*y
    Acl(izy, ir) = Ey * Cbar;
    Acl(izy, izy) = Azy;

    % zeta_theta^+ = Azth zeta_theta (homogeneous test: theta = 0)
    Acl(izt, izt) = Azth;
end

function Acomp = companion_from_poles(poles)
    coeff = poly(poles);
    nloc = numel(poles);
    Acomp = zeros(nloc,nloc);
    Acomp(1:nloc-1, 2:nloc) = eye(nloc-1);
    Acomp(nloc,:) = -fliplr(coeff(2:end));
end