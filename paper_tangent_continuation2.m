% William Lippincott
% paper_tangent_continuation2.m
% this code attempts to use newton's method to solve the 4 odes rather than
% the color identification from graphing that the paper outlined.

% this code is the tangent continuation method applied to the set of ODEs
% in the paper.

% this iteration of the paper_tangent_continuation method will search for
% different ways to find bifurcation curves in the bifurcation space
clc, clear, close all

%% Definitions of ODES
% the full coupled ODES: (with Dirichlet B.C.)
% -u'' = (lambda * abs(v)^r-1 * v) + (abs(v)^p-1 * v)   in (0,1)
% -v'' = (abs(u)^q-1 * u)                               in (0,1)
% parameters for the ODE
% 0 < r < 1/q and p > max{1, 1/q}
% for now, we want u'' + v.^3 = 0 and v'' + u.^3 = 0. 
% let lambda = 0 and p = 3. Also let q = 3
% this can be changed by user in the following input code 
lambda = 0;

%% --- USER INPUT: PARAMETERS FOR THE COUPLED ODES ---
fprintf('-------------------------------------------\n');
fprintf('Enter parameters for the coupled ODE system:\n');
fprintf('  -u'' = lambda*|v|^(r-1)v + |v|^(p-1)v\n');
fprintf('  -v'' = |u|^(q-1)u\n');
fprintf('with Dirichlet BCs on (0,1)\n');
fprintf('-------------------------------------------\n\n');

% user types in values for what the paramters should be
%lambda = input('Enter lambda: ');
q = input('Enter q: ');
r = input('Enter r (0 < r < 1/q): ');
p = input('Enter p (must be > max(1,1/q)): ');

% set defaults if user does not enter anything or the input breaks the
% conditions
%if isempty(lambda)
 %   lambda = 0; fprintf('lambda set to default value: 0\n');
%end
if isempty(q) || q <= 0
    fprintf('q must be > 0');
end
if isempty(r) || r <= 0 || r < 1/q
    fprintf('r must be > 0 and < 1/q. Set to default value: 0.25\n');
end
if isempty(p) || p > max(1, 1/q)
    fprintf('p must be > max(1,1/q). Set to default value: 2\n');
end

fprintf('\nParameters accepted:\n');
fprintf(' lambda = %g\n r = %g\n p = %g\n q = %g\n', lambda, r, p, q);
fprintf('-------------------------------------------\n\n');

%% Initial Solution
% use Newton's method to first solve the ode. use spdiags and D2 version
n = 100;
N = n-1;
a = 0; b = 1; dx = (b-a) / n;
x = linspace(a, b, n+1)'; 
z = x(2: end-1);
% with these parameters for p, q, we know that our solution probably looks
% like sin(pi*z) because that is what is looked like for u'' + u = 0
% fix initial guess
u1 = sin(2*pi*z); v1 = sin(2*pi*z);
%u1 = sin(2*pi*z); v1 = sin(2*pi*z);

% build difference matrices
D = kron([1 -2 1], ones(N, 1));
D2 = spdiags(D, [-1, 0, 1], N, N) / dx^2;

% create object function, G
G11 = @(u1,v1) D2*u1 + (lambda*(abs(v1)).^(r-1).* v1) + (abs(v1).^(p-1).* v1);
G22 = @(u1,v1) D2*v1 + (abs(u1)).^(q-1).* u1;
G   = @(u1,v1) [G11(u1,v1); G22(u1,v1)];

% create jacobian
% construct the jacobian matrix which should be a 2x2 matrix.
% J11 = top left:     partial G11 / partial u
% J12 = top right:    partial G11 / partial v
% J21 = bottom left:  partial G22 / partial u
% J22 = bottom right: partial G22 / partial v
J11 = @(u1,v1) D2;
J12 = @(u1,v1) spdiags(lambda*r*abs(v1).^(r-1) + p*abs(v1).^(p-1), 0, N, N);
J21 = @(u1,v1) spdiags(q*abs(u1).^(q-1), 0, N, N);
J22 = @(u1,v1) D2;
J   = @(u1,v1) [J11(u1,v1), J12(u1,v1); J21(u1,v1), J22(u1,v1) ];

% define some variables
max_num_iterations = 10;
tol = 1e-10;
% create an array of values spaced by 3.7 apart so that we can search for
% the bifurcation curves of different modes of this system of ODES
%constant_span = 3.708:(3.708):(3*(3.708) + 3.708);
constant_span = 1;
total_iters = length(constant_span);
% preallocate some matrices
u1_values = zeros(length(u1), total_iters);
v1_values = zeros(length(v1), total_iters);
step1 = 0;

for idx = constant_span
    step1 = step1 + 1;
    w1 = [idx*sin(2*pi*z);idx*sin(2*pi*z)];
    for i = 1:max_num_iterations
        u1 = w1(1:N);
        v1 = w1(N+1:end);
        chi = J(u1, v1) \ G(u1, v1);
        w1 = w1 - chi;
    
        error1 = norm(chi) * sqrt(dx);
        error2 = norm(G(u1,v1)) * sqrt(dx);
        if error1 < tol
            break;
        end
    end
    u1_values(:,step1) = w1(1:N);
    v1_values(:,step1) = w1(N+1:end);
end

fprintf('Newton''s Method \lambda =0:\n')
fprintf('max value for u1 is %g\n', max(abs(u1)))
fprintf('max value for v1 is %g\n\n\n', max(abs(v1)))
figure;

plot(z, u1_values(:,1), 'k-o', z, v1_values(:,1), 'k-*')
     %z, u1_values(:,2), 'b-o', z, v1_values(:,2), 'b-*', ...
     %z, u1_values(:,3), 'r-o', z, v1_values(:,3), 'r-*')
     %z, u1_values(:,4), 'y-o', z, v1_values(:,4), 'y-*', ...
     %z, u1_values(:,5), 'm-o', z, v1_values(:,5), 'm-*', ...
     %z, u1_values(:,6), 'c-o', z, v1_values(:,6), 'c-*')
grid on
legend('u - first mode','v - first mode','u - second mode','v - second mode', ...
       'u - third mode','v - third mode',...
       'Location','best')

%% Tangent Continuation
% this section will implement the s–continaution part with Newton's method
% and difference matrices
n = 100;
N = n-1;
a = 0; b = 1; dx = (b-a) / n;
x = linspace(a, b, n+1)'; 
z = x(2: end-1);
% if delta is too large, we may miss important artefacts about the
% bifurcation curve. if too small, it will require an unnecssary amount of
% iterations. delta is the small change to multiply V by
delta = 0.1; 
max_steps = 600; % number of times to do the continuation

% preallocate storage arrays
lambda_branch = zeros(max_steps,1);
u_max_values  = zeros(max_steps,1);
v_max_values  = zeros(max_steps,1);

% build difference matrices
D = kron([1 -2 1], ones(N, 1));
D2 = spdiags(D, [-1, 0, 1], N, N) / dx^2;

% preallocate storage arrays for lambda, u_max, and v_max values across all
% modes
lambda_branch_mode = zeros(max_steps,total_iters);
u_max_values_mode  = zeros(max_steps,total_iters);
v_max_values_mode  = zeros(max_steps,total_iters);

tol2 = 1e-10;
for jj = 1:total_iters
    step = 0; % initialize and reset counter
    lambda = 0;  % begin with lambda = 0. also reset here 
    % set u and v equal to the respective u1 and v1 array for when lambda =
    % 0 and for each mode
    u = u1_values(:, jj); v = v1_values(:, jj);
    % initialize and reset prev values
    u_prev = u; v_prev = v; lambda_prev = 0;
    u_prev2 = u1; v_prev2 = v1; lambda_prev2 = 0; 
    while lambda >= 0 && step < max_steps
        step = step + 1;
        tang = 0;
        if step == 1
            sec = [zeros(N,1); zeros(N,1); 1];
        else
            sec = [u_prev - u_prev2; v_prev - v_prev2; lambda_prev - lambda_prev2];
        end
        V = sec / norm(sec);
        P = [u_prev; v_prev; lambda_prev];     % u=(n-1)x1   s=1x1
        P = P + (delta * V);                     % predictor along tangent
        u = P(1:N);
        v = P(N+1:2*N);
        lambda = P(end);
    
        err2 = inf;
        max_while_iter = 20; while_iter = 0; i = 0;
        % Newton corrector on augmented system:
        while err2 > tol2 && while_iter < max_while_iter
            while_iter = while_iter + 1;
            % Residuals, create object function, G
            % -u'' = (lambda * abs(v)^r-1 * v) + (abs(v)^p-1 * v)   in (0,1)
            % -v'' = (abs(u)^q-1 * u)                               in (0,1)
            G11_cont = @(u,v,lambda) D2*u + (lambda*(abs(v)).^(r-1).* v) + (abs(v).^(p-1).* v);
            G22_cont = @(u,v,lambda) D2*v + (abs(u)).^(q-1).* u;
            % the function for G33 is 
            % ([u;v;lambda] - [u_prev;v_prev;lambda_prev]) * V' (dotted with V
            % transpose). This is the kappa value - our hyperplane. this should
            % always equal 0
            %kappa = V.' * ([u; v; lambda] - [u_prev; v_prev; lambda_prev]);
            %if step == 1, kappa = 0; end
            G33_cont = 0;
            G_cont = [G11_cont(u, v, lambda); G22_cont(u, v, lambda); G33_cont];
    
            % create jacobian
            % construct the jacobian matrix which should be a 2x2 matrix.
            % J11  =  partial G11 / partial u
            % J12  =  partial G11 / partial v
            % J13  =  partial G11 / partial lambda
            % J21  =  partial G22 / partial u
            % J22  =  partial G22 / partial v
            % J23  =  partial G22 / partial lambda 
            % J31  =  partial G33 / partial u      
            % J32  =  partial G33 / partial v      
            % J33  =  partial G33 / partial lambda 
            J11_cont = @(u,v,lambda) D2;
            %J12_cont = @(u,v,lambda) spdiags((lambda*r*abs(v).^(r-2).* v) + (p*abs(v).^(p-2).* v), 0, N, N);
            J12_cont = @(u,v,lambda) spdiags(lambda*(r*abs(v).^(r-1)) + (p*abs(v).^(p-1)), 0, N, N);
            J13_cont = @(u,v,lambda) ((abs(v)).^(r-1).* v);
            %J21_cont = @(u,v,lambda) spdiags((q*abs(u).^(q-2).* u), 0, N, N);
            J21_cont = @(u,v,lambda) spdiags(q*abs(u).^(q-1), 0, N, N);
            J22_cont = @(u,v,lambda) D2;
            J23_cont = @(u,v,lambda) zeros(N,1);                           
            J31_cont = @(u,v,lambda) V(1:N).';                              
            J32_cont = @(u,v,lambda) V(N+1:2*N).';               
            J33_cont = @(u,v,lambda) V(2*N+1);  
            J_cont = [J11_cont(u, v, lambda), J12_cont(u, v, lambda), J13_cont(u, v, lambda);
                      J21_cont(u, v, lambda), J22_cont(u, v, lambda), J23_cont(u, v, lambda);
                      J31_cont(u, v, lambda), J32_cont(u, v, lambda), J33_cont(u, v, lambda)];
    
            % Newton step: (w;lambda)^(k+1) = (w;lambda)^k - J \ G
            chi = J_cont \ G_cont;
            P = P - chi;
            lambda = P(end, 1);
            u = P(1:N, 1); 
            v = P(N+1:2*N, 1); 
            err2 = norm(chi,2); % error computation
            i_tracker(while_iter) = i;
            err2_tracker(while_iter) = err2;
            i = i+1; % loop counter
        end
        u_prev2 = u_prev;       
        v_prev2 = v_prev;   
        lambda_prev2 = lambda_prev;
        u_prev  = u;       
        v_prev  = v;      
        lambda_prev  = lambda;
        lambda_branch(step) = lambda;
        u_max_values(step) = max(abs(u));
        v_max_values(step) = max(abs(v));
        if lambda_branch(step) <= 0 
            lambda = 0;
            break;
        end
        fprintf('When Lambda is: %g:\nu_max: %g\n v_max: %g\n\n', ...
            lambda_branch(step), u_max_values(step), v_max_values(step))
    end
    lambda_branch_mode(:,jj) = lambda_branch;
    u_max_values_mode(:,jj) = u_max_values;
    v_max_values_mode(:,jj) = v_max_values;
end

%% Plotting
%figure;
%plot(i_tracker(1:end), err2_tracker(1:end), 'b-*')
%grid on; title('Error per Iteration')
%xlabel('Iteration'); ylabel('Error (norm of \chi)');
%legend('Iteration','Location','best')
%hold off; 

figure;
plot(lambda_branch_mode(:,1), v_max_values_mode(:,1), 'k-o', ...
     lambda_branch_mode(:,2), v_max_values_mode(:,2), 'b-o', ...
     lambda_branch_mode(:,3), v_max_values_mode(:,3), 'r-o')
     %lambda_branch_mode(:,4), v_max_values_mode(:,4), 'y-o', ...
     %lambda_branch_mode(:,5), v_max_values_mode(:,5), 'm-o', ...
     %lambda_branch_mode(:,6), v_max_values_mode(:,6), 'c-o')
xlabel('lambda'); ylabel('max(v(x)) for x \in [0,1]'); grid on
hold off

figure;
plot(lambda_branch_mode(:,1), u_max_values_mode(:,1), 'k-o', ...
     lambda_branch_mode(:,2), u_max_values_mode(:,2), 'b-o', ...
     lambda_branch_mode(:,3), u_max_values_mode(:,3), 'r-o')
     %lambda_branch_mode(:,4), u_max_values_mode(:,4), 'y-o', ...
     %lambda_branch_mode(:,5), u_max_values_mode(:,5), 'm-o', ...
     %lambda_branch_mode(:,6), u_max_values_mode(:,6), 'c-o')
xlabel('lambda'); ylabel('max(u(x)) for x \in [0,1]'); grid on
hold off

figure;
scatter3(lambda_branch, u_max_values, v_max_values);
xlabel('lambda'); ylabel('max u'); zlabel('max v'); grid on

