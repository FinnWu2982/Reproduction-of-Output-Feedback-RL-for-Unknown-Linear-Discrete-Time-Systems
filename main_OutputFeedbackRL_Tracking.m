%% main_OutputFeedbackRL_Tracking.m
% Strict paper-aligned Version 1 for the double-integrator example.
% Main principles:
%   1) Keep the paper's behavior policy form:  ubar(k) = -Kbar0_beh*r(k) + xi(k)
%   2) Keep Algorithm 1 main loop: pre-collection, data collection,
%      rank check, solve (29), update by (47), fresh test by (48)
%   3) Do NOT generate Kbar0_beh or Ko0_rl inside this script using any
%      hidden-model helper or oracle bridge.
%   4) Treat Kbar0_beh and Ko0_rl purely as ASSUMED GIVEN stabilizing gains,
%      consistent with the paper's assumptions.
%
% Notes:
% - The two gains below are fixed assumed constants for this example.
% - This script does NOT use Mbar_ls = r / zeta_bar.
% - This script does NOT auto-compute Kbar0_beh via dlqr(A_aug,Bbar,...).
% - This script does NOT call any oracle export helper.

clear; clc; close all;
rng(7);

%% ========================= 1) Hidden true system =========================
dt = 0.05;
A  = [1, dt;
      0,  1];
B  = [0.5*dt^2;
      dt];
C  = [1, 0];

n = size(A,1);
m = size(B,2);
p = size(C,1);

w_ref = 0.8;
th    = w_ref * dt;
S     = [ cos(th),  sin(th);
         -sin(th),  cos(th)];
Rref  = [1, 0];
q     = size(S,1);

F   = S;
G   = [0; 1];
Tff = [1, 0];

%% ===================== 2) Paper augmented dimensions =====================
nz    = n + p*q;
rzeta = nz*m + nz*p + nz*p;

A_aug = [A,      -B*Tff;
        -G*C,     F   ];
Bbar  = [B; zeros(p*q, m)];
Cbar  = [C, zeros(p, p*q)];
Qy    = 10;
Rbar  = 1;
Qx    = Cbar' * Qy * Cbar; %#ok<NASGU>

%% ============== 3) ASSUMED GIVEN stabilizing gains (paper) ==============
% These are treated as externally supplied stabilizing gains.
% They are NOT generated in this script.
Kbar0_beh = [ 10.05163281182561 4.118302443414524 -1.393399395251945 -0.8446666490365405 ];
Ko0_rl    = [ 3.213966771747198 -0.210886568319449 -3.496086601504385 0.2184796631851589 -2571.963130221277 7898.878091870995 -8106.312547733252 2782.860467366305 -0.1238686722536858 -0.3174038799216355 0.113887858959932 -0.8446666490363473 ];

assert(isequal(size(Kbar0_beh), [m, nz]), 'Kbar0_beh must have size [%d x %d].', m, nz);
assert(isequal(size(Ko0_rl),    [m, rzeta]), 'Ko0_rl must have size [%d x %d].', m, rzeta);
assert(all(isfinite(Kbar0_beh), 'all'), 'Kbar0_beh contains non-finite entries.');
assert(all(isfinite(Ko0_rl),    'all'), 'Ko0_rl contains non-finite entries.');

fprintf('Assumed Kbar0_beh = ['); fprintf(' %.16g', Kbar0_beh); fprintf(' ]\n');
fprintf('Assumed Ko0_rl    = ['); fprintf(' %.16g', Ko0_rl);    fprintf(' ]\n');

%% =================== 4) Build A_zeta from chosen poles ===================
Azeta_poles = [0.15, 0.25, 0.35, 0.45];
A_zeta      = companion_from_poles(Azeta_poles);
b           = [zeros(nz-1,1); 1];

%% ====================== 5) Training / data collection ====================
k0    = 250;
kf    = 2600;
Ndata = kf - k0;

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
y(:,1)    = C*x(:,1);
r(:,1)    = [x(:,1); zbar(:,1)];

for k = 1:(kf+1)
    theta(:,k) = 0.9*sin(0.11*k) + 0.5*sin(0.27*k) + 0.3*sin(0.49*k) + 0.15*randn;
end

for k = 1:(kf+1)
    xi_k      = 0.8*(0.8*sin(0.13*k) + 0.6*sin(0.71*k) + 0.3*randn);
    ubar(:,k) = -Kbar0_beh*r(:,k) + xi_k;

    zeta_u(:,k+1)     = kron(eye(m), A_zeta)*zeta_u(:,k)     + kron(ubar(:,k),  b);
    zeta_y(:,k+1)     = kron(eye(p), A_zeta)*zeta_y(:,k)     + kron(y(:,k),     b);
    zeta_theta(:,k+1) = kron(eye(p), A_zeta)*zeta_theta(:,k) + kron(theta(:,k), b);

    zbar(:,k+1) = F*zbar(:,k) - G*y(:,k) + G*theta(:,k);
    x(:,k+1)    = A*x(:,k) + B*(ubar(:,k) - Tff*zbar(:,k));
    y(:,k+1)    = C*x(:,k+1);
    r(:,k+1)    = [x(:,k+1); zbar(:,k+1)];
end

zeta_bar = [zeta_u; zeta_y; zeta_theta];
Ko = Ko0_rl;

%% ================= 6) Build paper data matrices (25)-(29) ================
ns_rzeta = nsym(rzeta);
ns_m     = nsym(m);
ns_p     = nsym(p);

Cz          = zeros(Ndata, ns_rzeta);
DzetaZeta   = zeros(Ndata, rzeta*rzeta);
DuZeta      = zeros(Ndata, m*rzeta);
DthetaZeta  = zeros(Ndata, p*rzeta);
DuTheta     = zeros(Ndata, m*p);
Du          = zeros(Ndata, ns_m);
Dtheta      = zeros(Ndata, ns_p);
Dyy         = zeros(Ndata, p*p);

sample_idx = 0;
for k = k0:(kf-1)
    sample_idx = sample_idx + 1;
    zk  = zeta_bar(:,k);
    zk1 = zeta_bar(:,k+1);
    uk  = ubar(:,k);
    thk = theta(:,k);
    yk  = y(:,k);

    Cz(sample_idx,:)         = (vecv(zk1) - vecv(zk)).';
    DzetaZeta(sample_idx,:)  = kron(zk,  zk).';
    DuZeta(sample_idx,:)     = kron(uk,  zk).';
    DthetaZeta(sample_idx,:) = kron(thk, zk).';
    DuTheta(sample_idx,:)    = kron(uk,  thk).';
    Du(sample_idx,:)         = vecv(uk).';
    Dtheta(sample_idx,:)     = vecv(thk).';
    Dyy(sample_idx,:)        = kron(yk,  yk).';
end

rank_left   = rank([DzetaZeta, DuZeta, Du, DthetaZeta, DuTheta, Dtheta], 1e-8);
rank_needed = ns_rzeta + rzeta*m + ns_m + rzeta*p + p*m + ns_p;

fprintf('rank(data matrix) = %d\n', rank_left);
fprintf('rank required by (46) = %d\n', rank_needed);

%% ====================== 7) Paper RL iteration (29),(47) ==================
max_iter = 1000;
tol_gain = 1e-4;
Ko_hist   = zeros(max_iter+1, numel(Ko));
Ko_hist(1,:) = Ko(:).';
diff_hist = nan(max_iter,1);

for j = 1:max_iter
    DKozeta = zeros(Ndata, ns_m);
    sample_idx = 0;
    for k = k0:(kf-1)
        sample_idx = sample_idx + 1;
        zk = zeta_bar(:,k);
        DKozeta(sample_idx,:) = vecv(Ko*zk).';
    end

    Omega_j = [Cz, ...
        -2*DzetaZeta*kron(eye(rzeta), Ko.') - 2*DuZeta, ...
        -Du + DKozeta, ...
        -2*DthetaZeta, ...
        -2*DuTheta, ...
        -Dtheta];

    nu_j = -DzetaZeta*reshape(Ko.'*Rbar*Ko, [], 1) - Dyy*reshape(Qy, [], 1);
    Lhat_vec = pinv(Omega_j) * nu_j;

    idx = 1;
    idx = idx + ns_rzeta;                 % LhatP, not used explicitly in gain update
    Lhat1_v = Lhat_vec(idx : idx+rzeta*m-1);  idx = idx + rzeta*m;
    Lhat2_s = Lhat_vec(idx : idx+ns_m-1);     idx = idx + ns_m;
    idx = idx + rzeta*p;                  %#ok<NASGU>
    idx = idx + p*m;                      %#ok<NASGU>
    idx = idx + ns_p;                     %#ok<NASGU>

    Lhat1 = reshape(Lhat1_v, [rzeta, m]);
    Lhat2 = svec_inv(Lhat2_s, m);
    Ko_new = (Rbar + Lhat2) \ (Lhat1.');

    diff_hist(j) = norm(Ko_new - Ko, 'fro');
    fprintf('iter %03d: ||Ko_{new}-Ko||_F = %.3e\n', j, diff_hist(j));

    Ko = Ko_new;
    Ko_hist(j+1,:) = Ko(:).';

    if diff_hist(j) < tol_gain
        Ko_hist   = Ko_hist(1:(j+1), :);
        diff_hist = diff_hist(1:j);
        break;
    end
end

Ko_learned = Ko;
fprintf('\nLearned gain size = [%d x %d]\n', size(Ko_learned,1), size(Ko_learned,2));
fprintf('Ko_learned = ['); fprintf(' %.16g', Ko_learned); fprintf(' ]\n');

%% ================= 8) Fresh test phase for controller (48) ===============
Nctrl = 500;
x_test   = zeros(n,   Nctrl+1);
z_test   = zeros(p*q, Nctrl+1);
y_test   = zeros(p,   Nctrl+1);
u_test   = zeros(m,   Nctrl);
yd_test  = zeros(p,   Nctrl+1);
err_test = zeros(p,   Nctrl+1);

zu_test  = zeros(nz*m, Nctrl+1);
zy_test  = zeros(nz*p, Nctrl+1);
zth_test = zeros(nz*p, Nctrl+1);
xd_test  = zeros(q,    Nctrl+1);

x_test(:,1)  = [0.5; -0.2];
z_test(:,1)  = [0.1;  0.0];
y_test(:,1)  = C*x_test(:,1);
xd_test(:,1) = [1; 0];
yd_test(:,1) = Rref*xd_test(:,1);
err_test(:,1)= y_test(:,1) - yd_test(:,1);

for k = 1:Nctrl
    zeta_k = [zu_test(:,k); zy_test(:,k); zth_test(:,k)];
    ubar_test_k = -Ko_learned * zeta_k;
    u_test(:,k) = ubar_test_k - Tff*z_test(:,k);

    zu_test(:,k+1)  = kron(eye(m), A_zeta)*zu_test(:,k)  + kron(ubar_test_k,  b);
    zy_test(:,k+1)  = kron(eye(p), A_zeta)*zy_test(:,k)  + kron(y_test(:,k),  b);
    zth_test(:,k+1) = kron(eye(p), A_zeta)*zth_test(:,k) + kron(yd_test(:,k), b);

    z_test(:,k+1)   = F*z_test(:,k) - G*y_test(:,k) + G*yd_test(:,k);
    x_test(:,k+1)   = A*x_test(:,k) + B*u_test(:,k);
    y_test(:,k+1)   = C*x_test(:,k+1);
    xd_test(:,k+1)  = S*xd_test(:,k);
    yd_test(:,k+1)  = Rref*xd_test(:,k+1);
    err_test(:,k+1) = y_test(:,k+1) - yd_test(:,k+1);
end

rms_tail_learned = sqrt(mean(err_test(:, end-199:end).^2, 'all'));
fprintf('Fresh-test tail RMS tracking error (learned Ko, last 200 steps) = %.6e\n', rms_tail_learned);

%% ===================== 9) Extra analysis and plots =======================
outdir = 'version1_results';
if ~exist(outdir, 'dir')
    mkdir(outdir);
end

t_iter = 1:length(diff_hist);
t_y    = 0:Nctrl;
t_u    = 0:(Nctrl-1);

rmse_all      = sqrt(mean(err_test(:).^2));
rmse_tail     = sqrt(mean(err_test(:, end-199:end).^2, 'all'));
mae_all       = mean(abs(err_test(:)));
mae_tail      = mean(abs(err_test(:, end-199:end)), 'all');
max_abs_err   = max(abs(err_test(:)));
u_rms         = sqrt(mean(u_test(:).^2));
u_max         = max(abs(u_test(:)));

fprintf('\n================ EXTRA SUMMARY METRICS ================\n');
fprintf('Full-horizon RMSE              = %.6e\n', rmse_all);
fprintf('Tail RMSE (last 200 steps)     = %.6e\n', rmse_tail);
fprintf('Full-horizon MAE               = %.6e\n', mae_all);
fprintf('Tail MAE  (last 200 steps)     = %.6e\n', mae_tail);
fprintf('Maximum absolute tracking err  = %.6e\n', max_abs_err);
fprintf('RMS control input              = %.6e\n', u_rms);
fprintf('Maximum absolute control input = %.6e\n', u_max);

summary_fid = fopen(fullfile(outdir, 'summary_metrics.txt'), 'w');
fprintf(summary_fid, 'rank(data matrix) = %d\n', rank_left);
fprintf(summary_fid, 'rank required by (46) = %d\n', rank_needed);
fprintf(summary_fid, 'Number of RL iterations = %d\n', length(diff_hist));
fprintf(summary_fid, 'Final gain update norm = %.6e\n', diff_hist(end));
fprintf(summary_fid, 'Full-horizon RMSE = %.6e\n', rmse_all);
fprintf(summary_fid, 'Tail RMSE (last 200 steps) = %.6e\n', rmse_tail);
fprintf(summary_fid, 'Full-horizon MAE = %.6e\n', mae_all);
fprintf(summary_fid, 'Tail MAE (last 200 steps) = %.6e\n', mae_tail);
fprintf(summary_fid, 'Maximum absolute tracking error = %.6e\n', max_abs_err);
fprintf(summary_fid, 'RMS control input = %.6e\n', u_rms);
fprintf(summary_fid, 'Maximum absolute control input = %.6e\n', u_max);
fclose(summary_fid);

% ---------- Figure 1: RL gain-update convergence ----------
fig1 = figure('Color','w');
semilogy(t_iter, diff_hist, 'o-', 'LineWidth', 1.5, 'MarkerSize', 5);
grid on;
xlabel('Iteration');
ylabel('||K_{o,new} - K_o||_F');
title('RL Gain Update Convergence');
saveas(fig1, fullfile(outdir, 'fig1_gain_convergence.png'));

% ---------- Figure 2: output tracking ----------
fig2 = figure('Color','w');
plot(t_y, yd_test(1,:), '--', 'LineWidth', 1.8); hold on;
plot(t_y, y_test(1,:), '-', 'LineWidth', 1.8);
grid on;
xlabel('Time step');
ylabel('Output');
legend('Reference y_d', 'Plant output y', 'Location', 'best');
title('Output Tracking Performance');
saveas(fig2, fullfile(outdir, 'fig2_output_tracking.png'));

% ---------- Figure 3: tracking error ----------
fig3 = figure('Color','w');
plot(t_y, err_test(1,:), 'LineWidth', 1.8);
grid on;
xlabel('Time step');
ylabel('Tracking error');
title('Tracking Error y - y_d');
saveas(fig3, fullfile(outdir, 'fig3_tracking_error.png'));

% ---------- Figure 4: control input ----------
fig4 = figure('Color','w');
plot(t_u, u_test(1,:), 'LineWidth', 1.8);
grid on;
xlabel('Time step');
ylabel('Control input');
title('Control Input During Fresh Test');
saveas(fig4, fullfile(outdir, 'fig4_control_input.png'));

% ---------- optional: combined compact figure ----------
fig5 = figure('Color','w', 'Position', [100 100 900 700]);

subplot(2,2,1);
semilogy(t_iter, diff_hist, 'o-', 'LineWidth', 1.2, 'MarkerSize', 4);
grid on;
xlabel('Iteration');
ylabel('||K_{o,new} - K_o||_F');
title('Gain Convergence');

subplot(2,2,2);
plot(t_y, yd_test(1,:), '--', 'LineWidth', 1.5); hold on;
plot(t_y, y_test(1,:), '-', 'LineWidth', 1.5);
grid on;
xlabel('Time step');
ylabel('Output');
legend('y_d', 'y', 'Location', 'best');
title('Output Tracking');

subplot(2,2,3);
plot(t_y, err_test(1,:), 'LineWidth', 1.5);
grid on;
xlabel('Time step');
ylabel('Error');
title('Tracking Error');

subplot(2,2,4);
plot(t_u, u_test(1,:), 'LineWidth', 1.5);
grid on;
xlabel('Time step');
ylabel('Input');
title('Control Input');

saveas(fig5, fullfile(outdir, 'fig5_combined_summary.png'));

%% ============================== helpers ==================================
function Acomp = companion_from_poles(poles)
    coeff = poly(poles);
    nloc = numel(poles);
    Acomp = zeros(nloc,nloc);
    Acomp(1:nloc-1, 2:nloc) = eye(nloc-1);
    Acomp(nloc,:) = -fliplr(coeff(2:end));
end

function out = vecv(x)
    x = x(:);
    nx = length(x);
    out = zeros(nx*(nx+1)/2,1);
    idx = 1;
    for i = 1:nx
        for j = i:nx
            out(idx) = x(i)*x(j);
            idx = idx + 1;
        end
    end
end

function nout = nsym(dim)
    nout = dim*(dim+1)/2;
end

function M = svec_inv(v, n)
    M = zeros(n,n);
    idx = 1;
    for i = 1:n
        for j = i:n
            if i == j
                M(i,j) = v(idx);
            else
                M(i,j) = 0.5*v(idx);
                M(j,i) = 0.5*v(idx);
            end
            idx = idx + 1;
        end
    end
end
