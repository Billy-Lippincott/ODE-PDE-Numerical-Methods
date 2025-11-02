% William Lippincott
% this code attempts to use newton's method to solve the 4 odes rather than
% the color identification from graphing that the paper outlined
clc, clear, close all

% preallocate an array for the maximum v value for each lambda value
lambda = 0:49;
max_v = zeros(size(lambda));
max_u = zeros(size(lambda));

target_dlam = +1;  % use -1 to go down in lambda
num_steps = 40;  % how many arc steps we want to solve for
lam_arc = []; max_u_arc = []; max_v_arc = []; % preallocate

% initial slopes shooting method, these are just first. imagine slopes
% as the direction that football takes from the kick of the kicker in
% the analogy that the paper uses
duo_c = 10;
dvo_c = 10;
C = [duo_c; dvo_c]; 
j = 1; % initialize lambda counter
C_store = nan(2, numel(lambda));

for sidx = 1:num_steps
    % initialize and define important vlaues
    h = 1e-7; 
    k = 0; % k is the counter              
    error = 1e8;            
    tol = 1e-8;            
    dx = 1e-4;  
    % independent variable x spans 0 to 1. this is what the paper uses
    x = 0:0.001:1;
    % initial guess matrices
    Y0_first = [0; 0; C];
    Y0_dx1 = [0; 0; C + [dx;0]];
    Y0_dx2 = [0; 0; C + [0;dx]];
    % ode solver for the first set of consants and then with one small dx
    % change. need two ode45 solvers for the dx change
    [~, Y_first] = ode45(@(x,Y) derivative(x,Y,lambda(j)), x, Y0_first);
    [~, Y1_dx] = ode45(@(x,Y) derivative(x,Y,lambda(j)), x, Y0_dx1);
    [~, Y2_dx] = ode45(@(x,Y) derivative(x,Y,lambda(j)), x, Y0_dx2);
    % the following is [u_residual; v_residual]
    Res_first = [Y_first(end,1); Y_first(end,2)];    
    Res1_dx = [Y1_dx(end,1); Y1_dx(end,2)];
    Res2_dx = [Y2_dx(end,1); Y2_dx(end,2)];
    % begin while loop
    while error > tol
        if k > 150
            warning(['\n\nThis value of lambda did not converge in under' ...
                ' 150 iterations of Newton''s method. lambda: %.2g\n\n'], j)
            break;
        end
        if k==0
            J = [(Res1_dx - Res_first)/dx, (Res2_dx - Res_first)/dx];
            C_prev = C;
            % prepare C_prev to be used in solver
            %C_prev = [C_prev(1), 0; 0, C_prev(2)];
            alpha = 0.5;
            C_updated = C - (alpha * (J \ Res_first));
            
            % store solved lambda, and C values 
            C_a = C_updated;
            lam_a = lambda(j);
        elseif k == 1
            Y0_prev = [0; 0; C_prev];
            Y01_new = [0; 0; C(1); C_prev(2)];
            Y02_new = [0; 0; C_prev(1); C(2)];
            %[~, Y_prev] = ode45(@(x,Y) derivative(x,Y,j), x, C_prev(:,1));
            [~, Y_prev] = ode45(@(x,Y) derivative(x,Y,lambda(j)), x, Y0_prev);
            [~, Y1_new] = ode45(@(x,Y) derivative(x,Y,lambda(j)), x, Y01_new);
            [~, Y2_new] = ode45(@(x,Y) derivative(x,Y,lambda(j)), x, Y02_new);
            Res_prev = [Y_prev(end,1); Y_prev(end, 2)];
            Res1_new  = [Y1_new(end,1); Y1_new(end,2)];
            Res2_new  = [Y2_new(end,1); Y2_new(end,2)];
            d1 = C(1) - C_prev(1); if d1 == 0, d1 = dx; end
            d2 = C(2) - C_prev(2); if d2 == 0, d2 = dx; end
            J = [(Res1_new - Res_prev)/d1, (Res2_new - Res_prev)/d2];
            C_prev = C;
            [~, Y_current] = ode45(@(x,Y) derivative(x,Y,lambda(j)), x, [0; 0; C]);
            Res_curr = [Y_current(end,1); Y_current(end,2)];
            alpha = 0.5; % damping constant
            C_updated = C - alpha * (J \ Res_curr);

            % store solved lambda and C values
            C_b = C_updated;
            lam_b = lambda(j);
        else
            Y0_prev = [0; 0; C_prev];
            Y01_new = [0; 0; C(1); C_prev(2)];
            Y02_new = [0; 0; C_prev(1); C(2)];
            %[~, Y_prev] = ode45(@(x,Y) derivative(x,Y,j), x, C_prev(:,1));
            [~, Y_prev] = ode45(@(x,Y) derivative(x,Y,lambda(j)), x, Y0_prev);
            [~, Y1_new] = ode45(@(x,Y) derivative(x,Y,lambda(j)), x, Y01_new);
            [~, Y2_new] = ode45(@(x,Y) derivative(x,Y,lambda(j)), x, Y02_new);
            Res_prev = [Y_prev(end,1); Y_prev(end, 2)];
            Res1_new  = [Y1_new(end,1); Y1_new(end,2)];
            Res2_new  = [Y2_new(end,1); Y2_new(end,2)];
            d1 = C(1) - C_prev(1); if d1 == 0, d1 = dx; end
            d2 = C(2) - C_prev(2); if d2 == 0, d2 = dx; end
            J = [(Res1_new - Res_prev)/d1, (Res2_new - Res_prev)/d2];
            C_prev = C;
            [~, Y_current] = ode45(@(x,Y) derivative(x,Y,lambda(j)), x, [0; 0; C]);
            Res_curr = [Y_current(end,1); Y_current(end,2)];
            alpha = 0.5; % damping constant
            C_updated = C - alpha * (J \ Res_curr);

            [C_next, lam_next] = pseudo_arc_step(C_a, lam_a, C_b, lam_b, target_dlam);
            [~, Yn] = ode45(@(x,Y) derivative(x,Y,lam_next), [0 1], [0;0;C_next(1);C_next(2)]);
            lam_arc(end+1) = lam_next;
            max_u_arc(end+1) = max(Yn(:,1));
            max_v_arc(end+1) = max(Yn(:,2));

            % slide window for the next secant
            C_a = C_b;  lam_a = lam_b;
            C_b = C_next; lam_b = lam_next;
        
        end
        % compute error
        error = norm(C_updated - C) / max(1, norm(C_updated));
        C = C_updated; % update for next iteration
        % put C in digestable form for ode solver
        %C = [C(1), C_prev(1); C_prev(2), C(2)];
        k = k+1; % update again
    end
    % create the solution matrix defined how the paper has for consistency
    Sol = C_updated;
    C_store(:, j) = Sol;
    % print to the command window if a solution is not found for duo or dvo
    if isnan(Sol(1))
        fprintf('\nsolution not found for duo while on lambda = %.2g\n\n', lambda(j))
    elseif isnan(Sol(2))
        fprintf('\nsolution not found for dvo while on lambda = %.2g\n\n', lambda(j))
    else
        fprintf(['Newton''s Method took %d iterations to converge\n', ...
         'to [%.5e  %.5e]\n', ...
         'within the desired tolerance, %.1e\n while ' ...
         'on lambda = %.2g\n\n'], k, Sol(1), Sol(2), tol, lambda(j));
    end

    % next compute the actual solutions to the problem, u and v, using the
    % values just found for dvo and duo as the BC
    Y0_sol_c = [0; 0; Sol(1); Sol(2)];
    [~, Y_sol_c] = ode45(@(x,Y) derivative(x,Y,lambda(j)), x, Y0_sol_c);

    % compute the max value of v for each lambda value
    max_v(j) = max(Y_sol_c(:, 2));
    max_u(j) = max(Y_sol_c(:, 1));
    % update j-counter (lambda counter)
    j = j + 1; 
end


figure;
plot(lambda, max_v, 'o-')
xlabel('lambda')
ylabel('max(v(x)) for x \in [0,1]')
grid on

figure;
scatter3(lambda, max_u, max_v);
xlabel('lambda')
ylabel('max u')
zlabel('max v')
grid on

figure;
plot(lambda, max_u, 'o-')
xlabel('lambda')
ylabel('max(u(x)) for x \in [0,1]')
grid on

figure; hold on
plot(lambda, max_v, 'o-');                
plot(lam_arc, max_v_arc, '.-');    
xlabel('lambda');
ylabel('max v'); 
legend('grid','arc')
grid on
hold off

%% Subfunctions
% subfunction for the deriavtive of our 4 first order odes that we switched
% from the 2 second order odes
function d_dt = derivative(~, Y, lambda)
    %  0 < r < 1/q and p > max{1, 1/q}
    % choose any values that satisfy that
    % choose the values from the paper which are below
    r = 1/3; 
    q = 3/2;
    % p = max(1, q) + 1;
    p = 3;
    % define the differential equation by turning the system of two second
    % order odes into 4 first order odes
    % 1. u'(x) = w(x)
    % 2. v'(x) = z(x)
    % 3. w'(x) = -(lambda)(v(x))^r - (v(x))^p
    % 4. z'(x) = -|u(x)|^(q-1) * (u(x))
    % thus Y = [u; v; w; z] so dYdx = [u'; v'; w'; z']
    vplus = max(Y(2),0) + 1e-12; % make sure that v is always positive, also
    % add a small constant to avoid NaN derivatives
    d_dt  = [ Y(3);                           
              Y(4); 
             -lambda*vplus.^r - vplus.^p ; 
             -abs(Y(1)).^(q-1) .* Y(1) ];
end

function [C_out, lambda_out] = pseudo_arc_step(C_a, lambda_a, C_b, lambda_b, target_dlam)

    t = [C_b(:) - C_a(:); lambda_b - lambda_a];
    nrm = max(norm(t), 1e-12);
    t = t / nrm; % normalize
    s = target_dlam / t(3);
    C_pred = C_b + s*(t(1:2));
    lam_pred= lambda_b + s*(t(3));

    C = C_pred; lam = lam_pred;
    tol = 1e-8;
    for it = 1:25
        % residual R(C,lam) = [u(1); v(1)] via shooting
        [~, Y] = ode45(@(x,Y) derivative(x,Y,lam), [0 1], [0;0;C(1);C(2)]);
        R = [Y(end,1); Y(end,2)];
        if any(~isfinite(R)), break; end

        % FD Jacobian wrt C
        dC = 1e-5*(1+abs(C));
        C1 = C; C1(1)=C1(1)+dC(1);
        [~, Y1] = ode45(@(x,Y) derivative(x,Y,lam), [0 1], [0;0;C1(1);C1(2)]);
        R1 = [Y1(end,1); Y1(end,2)];
        C2 = C; C2(2)=C2(2)+dC(2);
        [~, Y2] = ode45(@(x,Y) derivative(x,Y,lam), [0 1], [0;0;C2(1);C2(2)]);
        R2 = [Y2(end,1); Y2(end,2)];
        J = [(R1-R)/dC(1), (R2-R)/dC(2)];    % 2x2

        % FD derivative wrt lambda
        dlam = 1e-5*(1+abs(lam));
        [~, Yl] = ode45(@(x,Y) derivative(x,Y,lam+dlam), [0 1], [0;0;C(1);C(2)]);
        Rl = [Yl(end,1); Yl(end,2)];
        dR_dlam = (Rl - R) / dlam;           % 2x1

        % arclength plane constraint g=0, orthogonal to t through predictor
        g = t(1:2).'* (C - C_pred) + t(3) * (lam - lam_pred);

        % augmented 3x3 Newton system
        J_aug = [J, dR_dlam; t(1:2).', t(3)];
        F_aug = [R; g];

        % regularize if ill-conditioned
        if rcond(J_aug) < 1e-12, J_aug = J_aug + 1e-8*eye(3); end

        % damped update
        delta = J_aug \ F_aug;
        alpha = 1.0;
        C_try = C   - alpha*delta(1:2);
        lam_try = lam - alpha*delta(3);

        % simple backtracking if needed
        for bt = 1:4
            [~, Yt] = ode45(@(x,Y) derivative(x,Y,lam_try), [0 1], [0;0;C_try(1);C_try(2)]);
            Rt = [Yt(end,1); Yt(end,2)];
            gt = t(1:2).'* (C_try - C_pred) + t(3) * (lam_try - lam_pred);
            if norm([Rt; gt]) <= 0.9*norm(F_aug), break; end
            alpha   = 0.5*alpha;
            C_try   = C   - alpha*delta(1:2);
            lam_try = lam - alpha*delta(3);
        end

        C = C_try; lam = lam_try;
    end

    C_out = C; lambda_out = lam;
end