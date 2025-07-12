%--------------------------------------------------------------------------
%----------- D^2 x(t)+c D x(t)+(delta+epsi*cos(omega t))x(T) + k x(t-2*pi)=0
%  Stability Analysis Using direct RK4 FTM
%----------- Code by Balaji adireddi
%----------- Indian Institute of Technology Hyderabad
%----------- Date:13-07-2025 ------------------------------------------

clc
clear all %#ok
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaulttextInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
format long
% Parameters
alpha=0.5;
omega=1;               % Angular frequency
c=0.1;                   % Damping coefficient
kd=-0.04;                  % Delay coeffieient
T=2*pi;                % Fundamental period
N_Disc=500;              % Number of points for parameter grid
Delta=linspace(-0.2,1.5,N_Disc); % Range for delta
Epsilon=linspace(0,1.5,N_Disc);   % Range for epsilon
N = 30;                 % Number of basis functions
% Precompute Mass Matrix Identity (Avoid redundant computations)
Mass_mat = M_Mat(N);
M_Zero = zeros(N, N);
M_Identity=[Mass_mat M_Zero;M_Zero Mass_mat];
tspan= linspace(0,T,1001); 
% Precompute Basis Functions
s_span = linspace(-1, 0,1001); 
Basis_fun = phi(s_span, N); % Orthonormal basis functions

% Initialize result storage
Lambda_Stability = [];              % stability results
Lambda_Unstability = [];              % Unstability results
Lambda=[];
for i = 1:N_Disc
    disp(N_Disc-(i-1)); % Display progress
    delta = Delta(i);
    Lambda_Stability_i = [];
    Lambda_Unstability_i = [];
    Lambda_temp=[];
    % Parallel Execution for Inner Loop
    parfor j = 1:N_Disc
        [Lambda_all,Unstability,Stability]= Galerkin(alpha,Basis_fun, s_span, M_Identity,delta,Epsilon(j),kd,c,omega,N,tspan,T);
        Lambda_temp=[Lambda_temp; Lambda_all];
        Lambda_Unstability_i = [Lambda_Unstability_i; Unstability];
        Lambda_Stability_i = [Lambda_Stability_i; Stability];
    end
    Lambda=[Lambda; Lambda_temp];
    Lambda_Unstability=[Lambda_Unstability;Lambda_Unstability_i];
    Lambda_Stability=[Lambda_Stability;Lambda_Stability_i];
    % save('FODDE_FTM_RK4_Ex_1_N100_alpha0pt5.mat','Lambda','alpha','i','N_Disc','Lambda_Unstability','Lambda_Stability','N','Delta','Epsilon','omega','c','kd','T');

end


%% Plotting
[X, Y] = meshgrid(Delta, Epsilon);
lambdaMatrix = Lambda(:,3);  % Convert cell array to matrix
indices = lambdaMatrix < 01;  % Logical indices of points meeting the condition

% Extract the coordinates of points that satisfy the condition
X_valid = X(indices);
Y_valid = Y(indices);

%% Stability Surface Plot
[X,Y] = meshgrid(Delta,Epsilon);
Z = nan(N_Disc, N_Disc);
for ij = 1:numel(Lambda_Stability(:,1))
    Indx_1 = find(abs(X-Lambda_Stability(ij,1))<1e-5);
    Indx_2 = find(abs(Y-Lambda_Stability(ij,2))<1e-5);
    Indx_C = intersect(Indx_1,Indx_2);
    Z(Indx_C) = Lambda_Stability(ij,3);
end

f = figure(2);
set(groot, 'DefaultAxesTickLabelInterpreter', 'latex');
ax = axes('Parent', f);
h = surf(X, Y, Z, 'Parent', ax);
set(h, 'EdgeColor', 'none'); % Remove grid lines on surface
view(ax, [0, 90]); % Top-down view
colormap(jet(10)); % Use jet colormap
colorbar
colorbar_handle = colorbar;
hold on
xlabel('$\delta$', 'Interpreter', 'latex');
ylabel('$\epsilon$', 'Interpreter', 'latex');
axis([Delta(1) Delta(end) Epsilon(1) Epsilon(end)]);
set(get(gca,'YLabel'),'Rotation',0)
set(gca,'FontSize',20);
set(colorbar_handle, 'TickLabelInterpreter', 'latex');


%% Galerkin Arnoldi Method for Stability Analysis
function [Lambda,Lambda_Unstability,Lambda_Stability] = Galerkin(alpha,Basis_fun, s_span, M_Identity,delta,epsi,kd,c,omega,N,tspan,T)
Lambda_Stability = [];
Lambda_Unstability=[];
Lambda=[];
tau=T; % Periodic delay
N_dim = 2*N; % Total system dimension
Q = zeros(N_dim, N_dim); % Arnoldi orthonormal basis matrix
q_init = zeros(N_dim, 1);
q_init(1) = 1;
Q(:, 1) = q_init;
H = zeros(N_dim, N_dim); % Arnoldi Hessenberg matrix
Eig_M = zeros(N_dim, 1); % Eigenvalue container
Eig_Large = NaN;    %  Initialize Eig_Large before use

for k_iter = 1:N_dim
    [x, x_dot] = RK4_solver(alpha,c, delta, epsi, omega, kd, tspan, tau,N, Q(:,k_iter));
    x_new = interp1(tspan, x, T+tau.*s_span);
    x_dot_new = interp1(tspan, x_dot, T+tau.*s_span);
    v = compute_init(Basis_fun,x_new,x_dot_new,M_Identity,N);
    % Arnoldi Process
    for i=1:k_iter
        H(i,k_iter)=Q(:,i)'*v;
        v=v-H(i,k_iter)*Q(:, i);
    end

    if k_iter<N_dim
        H(k_iter+1,k_iter)=norm(v);
        if H(k_iter+1,k_iter)==0
            return
        end
        Q(:,k_iter+1)=v/H(k_iter+1,k_iter);
        for ij=1:k_iter
            Q(:,k_iter+1)=Q(:,k_iter+1)-(Q(:,ij)'*Q(:,k_iter+1))*Q(:,ij);
        end
    end

    % Compute Eigenvalues and Check Convergence
    Eig_M(k_iter) = max(abs(eig(H)));
    if k_iter > 2 && abs(Eig_M(k_iter) - Eig_M(k_iter-1)) < 1e-10
        Eig_Large = Eig_M(k_iter);
        break;
    end
end

if Eig_Large < 1+1e-8
    Lambda_Stability = [Lambda_Stability; delta epsi Eig_Large];

else
    Lambda_Unstability=[Lambda_Unstability;delta epsi Eig_Large];
end
Lambda=[Lambda;delta epsi Eig_Large];
end

% Basis Function
function g = phi(s,N)
Num_elm=numel(s);
g=zeros(N,Num_elm);
for i=1:Num_elm
    g(1,i)=1;
    g(2,i)=1+2*s(i);
    for j=3:N
        g(j,i)=((2*j-3)*g(2,i)*g(j-1,i)-(j-2)*g(j-2,i))/(j-1);
    end
end
end

function [x_out, x_dot_out] = RK4_solver(alpha,c,delta,epsi,omega,kd,tspan,tau,N,Q_temp)
N_steps = length(tspan);
dt = tspan(2)-tspan(1);
x_out = zeros(1, N_steps);
x_dot_out = zeros(1, N_steps);
D_alpha = zeros(1, N_steps);
history_fun = @(t) dde_hist(t,tau,N,Q_temp);
init_first=history_fun(tspan(1));
% Initializing delay condition
for i = 1:N
    x_out(i) = init_first(1);      % Initial displacement (changeable based on requirements)
    x_dot_out(i) = init_first(2);  % Initial velocity
end

for ii = 2:N_steps-1
    t = tspan(ii);
    t_delayed = t - tau;
    
    if t_delayed < tspan(1)
        temp=history_fun(t_delayed);
        x_tau = temp(1);
    else
        x_tau = interp1(tspan, x_out, t_delayed, 'linear', 'extrap'); % Interpolate for accuracy
    end
        
    k1_x = x_dot_out(ii);
    k1_x_dot = -c * D_alpha(ii) - (delta + epsi * cos(omega*t)) * x_out(ii) - kd * x_tau;
    
    k2_x = x_dot_out(ii) + 0.5*dt*k1_x_dot;
    k2_x_dot = -c * D_alpha(ii) - (delta + epsi * cos(omega*(t + 0.5*dt))) * (x_out(ii) + 0.5*dt*k1_x) - kd * x_tau;
    
    k3_x = x_dot_out(ii) + 0.5*dt*k2_x_dot;
    k3_x_dot = -c * D_alpha(ii) - (delta + epsi * cos(omega*(t + 0.5*dt))) * (x_out(ii) + 0.5*dt*k2_x) - kd * x_tau;
    
    k4_x = x_dot_out(ii) + dt*k3_x_dot;
    k4_x_dot = -c * D_alpha(ii) - (delta + epsi * cos(omega*(t + dt))) * (x_out(ii) + dt*k3_x) - kd * x_tau;
    
    x_out(ii+1) = x_out(ii) +(dt/6)*(k1_x+2*k2_x+2*k3_x+k4_x);
    x_dot_out(ii+1) = x_dot_out(ii)+(dt/6)*(k1_x_dot+2*k2_x_dot+2*k3_x_dot+k4_x_dot);
    % Compute fractional derivative D^alpha
    sum_D_alpha = 0;
    for jj = 1:ii
        sum_D_alpha = sum_D_alpha+(x_dot_out(ii+1-jj)+x_dot_out(ii+1-(jj-1)))/(2*((tspan(jj)/2+tspan(jj+1)/2)^alpha))*dt;
    end
    D_alpha(ii+1) = 1/gamma(1-alpha)*sum_D_alpha;
end
end

% History Function
function dhist = dde_hist(t,tau,N,r)
Basis=phi(t/tau,N); % Orthonormal basis
Sol=zeros(2,2*N);
for i=1:2
    Sol(i,(i-1)*N+1:i*N)=Basis;
end
dhist=Sol*r;
end

% Mass Matrix for Orthonormal Basis
function M=M_Mat(N)
M=zeros(N, N);
for i=1:N
    for j=1:N
        if i==j
            M(i, j)=1/(2*i-1);
        end
    end
end
end
% Function to compute r
function r = compute_init(Basis,x,x_dot,M_Identity,N)
SimpsonResults =zeros(2*N,1); % To store Simpson1_3rd results
SimpsonResults(1:N) = Simpson1_3rd(Basis*diag(x),-1,0)'; % Compute Simpson1_3rd(Proj_s,-1,0)
SimpsonResults(N+1:2*N) = Simpson1_3rd(Basis*diag(x_dot),-1,0)'; % Compute Simpson1_3rd(Proj_s,-1,0)
r = M_Identity\SimpsonResults;
end