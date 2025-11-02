% William Lippincott
% this code attempts to use newton's method to solve the 4 odes rather than
% the color identification from graphing that the paper outlined.

% the difference in this code is that we will attempt to solve the pde
% system rather than just the ode system

% this code is the tangent continuation method applied to the set of ODEs
% in the paper.
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

%% Initial Solution
% use Newton's method to first solve the ode. use spdiags and D2 version
n = 50;
N = n-1;
a = 0; b = 1;
dx = (b-a) / n;
x = linspace(a, b, n+1)'; 
z = x(2: end-1);
% with these parameters for p, q, we know that our solution probably looks
% like sin(pi*z) because that is what is looked like for u'' + u = 0
% fix initial guess.
% create the initial guess
[X, Y] = meshgrid(z,z);
U0 = 6*(sin(2*pi*X).*sin(pi*Y) + sin(1*pi*X).*sin(2*pi*Y));       

u1 = U0(:);                      

% build difference matrices
D = kron([1 -2 1], ones(N, 1));
D2 = spdiags(D, [-1, 0, 1], N, N) / dx^2;
% this next lines are novel for the pde version
I = speye(N);
L = kron(I, D2) + kron(D2, I);       

% create object function, G
G = @(u1) L*u1 + lambda*u1 + u1.^3;

% create jacobian
J = @(u1) L + spdiags(lambda + 3*u1.^2, 0, N^2, N^2) ;

max_num_iterations = 10;
tol = 1e-10;
for i = 1:max_num_iterations
    chi = J(u1) \ G(u1);
    u1 = u1 - chi;

    error1 = norm(chi) * sqrt(dx);
    error2 = norm(G(u1)) * sqrt(dx);
    if error1 < tol
        break;
    end
end

fprintf('Newton''s Method \lambda =0:\n')
fprintf('max value for u1 is %g\n', max(abs(u1)))
U = reshape(u1, N, N);              
figure; 
imagesc(z, z, U);
colorbar
title('u(x,y)'); xlabel('x'); ylabel('y');
