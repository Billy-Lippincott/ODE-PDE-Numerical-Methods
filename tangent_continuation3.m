%% William Lippincott

% this code is the tangent_continuation3.m script in which our methodology
% is different. The novelty of this code is the following:
%
%     - iterates over as many modes for the ODE as user wants
%     - includes all modes on the same curve
%
% this code will use the arclength update to approximate the next point on
% the plot of lambda values against the norm of u. it will do this for the
% differential equation of u'' + lambda*u + u^3 = 0
clear, clc

%% use Newton's method to first solve the ode. use spdiags and D2 version
% this is not the trivial solution one. this is for s=0
% input parameters from Dr N's book
n = 10;
N = n-1;
a = 0; b = 1; dx = (b-a) / n;
x = linspace(a, b, n+1)'; 
z = x(2: end-1);
% the problem statement gives us this definition
ymax = 2 * sqrt(2) * ellipticF(pi/2, -1);
% the book says to copy listings 3.1, 3.2, 3.3 which has s=8, but part a, b
% have s=0 for this BVP. leave as s=0 for now
s = 0;
% functions
f = @(u1) s*u1 + u1.^3;
f_prime = @(u1) s + 3*u1.^2;
% the function that Dr. N uses
u1 = 10*sin(pi * z);

D = kron([1 -2 1], ones(N, 1));
D2 = spdiags(D, [-1, 0, 1], N, N) / dx^2;
G = @(u1) D2*u1 + f(u1);
J = @(u1) D2 + spdiags(f_prime(u1), 0, N, N);

max_num_iterations = 10;
tol = 1e-10;
for i = 1:max_num_iterations
    chi = J(u1) \ G(u1);
    u1   = u1 - chi;

    error1 = norm(chi) * sqrt(dx);
    error2 = norm(G(u1)) * sqrt(dx);
    if error1 < tol
        break;
    end
end

sup_Newt_err = abs(ymax - max(abs(u1)));
fprintf('Newton''s Method s=0:\n')
fprintf('value is %d\n', max(abs(u1)))
fprintf('sup_Newt_err = %g\n\n', sup_Newt_err)

%% use Newton's method to solve for the trivial solution
% input parameters from Dr N's book
n = 10;
N = n-1;
a = 0; b = 1; dx = (b-a) / n;
x = linspace(a, b, n+1)'; 
z = x(2: end-1);
% the problem statement gives us this definition
ymax = 2 * sqrt(2) * ellipticF(pi/2, -1);
s = pi^2; % this is the first point for s where there bifurcation curve intersects on the s axis
% functions
f = @(u2) s*u2 + u2.^3;
f_prime = @(u2) s + 3*u2.^2;
% just enter 0 for the entire u vector so that we get the trivial solution
u2 = zeros(N, 1);

D = kron([1 -2 1], ones(N, 1));
D2 = spdiags(D, [-1, 0, 1], N, N) / dx^2;
G = @(u2) D2*u2 + f(u2);
J = @(u2) D2 + spdiags(f_prime(u2), 0, N, N);

max_num_iterations = 10;
tol = 1e-10;
for i = 1:max_num_iterations
    chi = J(u2) \ G(u2);
    u2   = u2 - chi;

    error1 = norm(chi) * sqrt(dx);
    error2 = norm(G(u2)) * sqrt(dx);
    if error1 < tol
        break;
    end
end
u2_prev = u2;
fprintf('Newton''s Method trivial solution:\n')
fprintf('value is %d\n\n', max(abs(u2)))

%% this section will implement the s–continaution part
n = 70; N = n-1; a = 0; b = 1; 
dx = (b-a) / n;
x = linspace(a, b, n+1)'; 
z = x(2: end-1);

% if delta is too large, we may miss important artefacts about the
% bifurcation curve. if too small, it will require an unnecssary amount of
% iterations. delta is the small change to multiply V by
delta = 0.01; 

max_steps = 10000; % number of times to do the continuation
u = zeros(n-1, 1);

% Difference matrices to be used later in the loop
D = kron([1 -2 1], ones(N, 1));
D2 = spdiags(D, [-1, 0, 1], N, N) / dx^2;
tol2 = 1e-10;

% change the k-value to  include however many modes we want for the
% eigenvalues of the ODE
% only use even natural number values of k
k = 1:5; k = length(k);

% preallocate cell arrays to store each modes arrays
s_cell = cell(2*k, 1);
y_max_all_values_cell = cell(2*k,1);

% iterate first for positive curves. second for negative curves
for i_tangent = 1:2
    % use the for loop to iterate over integer values of n that are solutions
    % to the ODE
    if i_tangent == 1
        for n_eig = 1:k
            % use eigs if the eigenfunction is not known analytically
            v1 = sin(n_eig*pi*z);          % eigenvector for the ODE.
            v1 = v1 / norm(v1);            % normalize
            % start at this mode's eigenvalue. where s=0 on the bifurcation
            % curve
            s = (n_eig^2) * pi^2;
        
            % reset continuation state for this mode
            step = 0;
            u_prev = zeros(N,1);
            u_pprev = u_prev;
            s_pprev = s;
        
            % recreate storage for this mode for each iteration
            s_branch = nan(max_steps+1,1);
            y_max_values = nan(max_steps+1,1);
            s_branch(1) = s;
            y_max_values(1) = 0;
        
            while s > 0 && step < max_steps
                step = step + 1;
                if step == 1
                    V = [v1; 0];            % V has n-1 values of u and 1 value of s
                    V = V / norm(V);        % kind of overkill, v1 already norm
                    P = [u; s];             % u=(n-1)x1   s=1x1
                    P = P + delta * V;      % predictor along tangent 
                    u = P(1:end-1);         % use predicted values for Newton's method
                    s = P(end);             % use predicted values for Newton's method
                else
                    sec = [(u_prev - u_pprev); s - s_pprev];   
                    if norm(sec)==0
                        sec = [v1; 0]; 
                    end
                    V = sec / norm(sec);
                    P = [u; s];             % u=(n-1)x1   s=1x1
                    P = P + delta * V;      % predictor along tangent 
                    u = P(1:end-1);
                    s = P(end);
                end
            
                err2 = inf; max_while_iter = 20; while_iter = 0; % initialize!
                % Newton corrector on augmented system:
                while err2 > tol2 && while_iter < max_while_iter
                    % safeguard with max iterations
                    while_iter = while_iter + 1;
        
                    % move these to outide the loops to save time - object
                    % function should not change value per loop
                    % Residuals
                    G1 = @(u,s) D2*u + s*u + u.^3;
                    % G2 should always be 0, since the dot product of t' with -t should
                    % be 0   
                    G2 = 0; % should always be 0
                    G = [G1(u, s); G2];
            
                    % compute the jacobian now
                    J1 = @(u,s) D2 + spdiags(s + 3*u.^2, 0, N, N);
                    J = [J1(u,s), u; V'];
            
                    % Newton step: (u,lambda)^(k+1) = (u,lambda)^k - J \ G
                    chi = J \ G;
                    P = P - chi;
                    s = P(end, 1);
                    u = P(1:end-1, 1); 
                    u_new = u;
                    err2 = norm(chi,2); % error computation
                end
                    % store values in branch array
            s_branch(step+1) = s;
            y_max_values(step+1) = max(u);
            u_pprev = u_prev;    
            s_pprev = s;
            u_prev = u;  
            if s_branch(step) <= 0 
                s = 0;
                break;
            end
            end
            % storage
            s_cell{n_eig} = s_branch;
            y_max_all_values_cell{n_eig} = y_max_values;
        end
    else
        % negative values branch
        for n_eig = 1:k
            u = zeros(N,1);          % reset the state to predict from
            u_prev = zeros(N,1);
            u_pprev = u_prev;
            v1 = sin(n_eig*pi*z);          % eigenvector for the ODE.
            % make this one negative
            v1 = v1 / norm(v1);           % normalize
            % start at this mode's eigenvalue. where s=0 on the bifurcation
            % curve
            s = (n_eig^2) * pi^2;
        
            % reset continuation state for this mode
            step = 0;
            u_prev = zeros(N,1);
            u_pprev = u_prev;
            s_pprev = s;
        
            % recreate storage for this mode for each iteration
            s_branch = nan(max_steps+1,1);
            y_max_values = nan(max_steps+1,1);
            s_branch(1) = s;
            y_max_values(1) = 0;
        
            while s > 0 && step < max_steps
                step = step + 1;
                if step == 1
                    V = [-v1; 0];            % V has n-1 values of u and 1 value of s
                    V = V / norm(V);        % kind of overkill, v1 already norm
                    P = [u; s];             % u=(n-1)x1   s=1x1
                    P = P + delta * V;      % predictor along tangent 
                    u = P(1:end-1);         % use predicted values for Newton's method
                    s = P(end);             % use predicted values for Newton's method
                else
                    sec = [(u_prev - u_pprev); s - s_pprev];   
                    if norm(sec)==0
                        sec = [v1; 0]; 
                    end
                    V = sec / norm(sec);
                    P = [u; s];             % u=(n-1)x1   s=1x1
                    P = P + delta * V;      % predictor along tangent 
                    u = P(1:end-1);
                    s = P(end);
                end
            
                err2 = inf; max_while_iter = 20; while_iter = 0; % initialize!
                % Newton corrector on augmented system:
                while err2 > tol2 && while_iter < max_while_iter
                    % safeguard with max iterations
                    while_iter = while_iter + 1;
        
                    % Residuals
                    G1 = @(u,s) D2*u + s*u + u.^3;
                    % G2 should always be 0, since the dot product of t' with -t should
                    % be 0   
                    G2 = 0; % should always be 0
                    G  = [G1(u, s); G2];
            
                    % compute the jacobian now
                    J1 = @(u,s) D2 + spdiags(s + 3*u.^2, 0, N, N);
                    J = [J1(u,s), u; V'];
            
                    % Newton step: (u,lambda)^(k+1) = (u,lambda)^k - J \ G
                    chi = J \ G;
                    P = P - chi;
                    s = P(end, 1);
                    u = P(1:end-1, 1); 
                    u_new = u;
                    err2 = norm(chi,2); % error computation
                end
                    % store values in branch array
            s_branch(step+1) = s;
            y_max_values(step+1) = min(u);
            u_pprev = u_prev;    
            s_pprev = s;
            u_prev = u;  
            if s_branch(step) <= 0 
                s = 0;
                break;
            end
            end
            % storage for the negative group of curves
            s_cell{n_eig + k} = s_branch;
            y_max_all_values_cell{n_eig + k} = y_max_values;    
        end
    end
end
   
figure;
for j = 1:(2*k)
    % now plot via the for loop to include each curve
    plot(s_cell{j}, y_max_all_values_cell{j}, 'k-o');
    grid on
    xlabel('s')
    ylabel('max v'); 
    hold on;

    fprintf(['The value of y_max for the tangent continuation' ...
    ' algorithm when s=0 is: %.8g\n'], max(y_max_all_values_cell{j}))
    % calculate the max value of y when s=0 to compare it to known value
    tan_err = abs(ymax - max(y_max_all_values_cell{j}));
    fprintf(['the error of y_max for the tangent continuation' ...
        ' algorithm when s=0 is: %g\n\n'], tan_err)
end
