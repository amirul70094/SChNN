function init_NN(n, example_id)

t = sym('t');
alpha = sym('alpha');
mu = sym('mu');

phi = sym('phi', [1, n]);

T = [1, ...
     2*t - 1, ...
     8*t^2 - 8*t + 1, ...
     32*t^3 - 48*t^2 + 18*t - 3, ...
     128*t^4 - 256*t^3 + 160*t^2 - 32*t + 1, ...
     512*t^5 - 1280*t^4 + 1120*t^3 - 400*t^2 + 50*t - 5, ...
     2048*t^6 - 6144*t^5 + 6912*t^4 - 3584*t^3 + 840*t^2 - 72*t + 1, ...
     8192*t^7 - 28672*t^6 + 39424*t^5 - 26880*t^4 + 9408*t^3 - 1568*t^2 + 98*t - 7];

mu_substituted = sum(phi .* T(1:n));

A = 1; %initial condition ex 1,2,4

z_tr_mu = A + t * (- (2*mu^3)./125 + (8*mu^2)./25 + (3*mu)./5);



z_tr_phi = subs(z_tr_mu, mu, mu_substituted);
z_tr_phi = collect(expand(z_tr_phi), t);

D_alpha_sum = compute_fractional_derivative(z_tr_phi, t, alpha);


switch example_id
    case 1
        f_target = exp(t) .* (2 - igamma(1 - alpha, t) ./ gamma(1 - alpha)) - z_tr_phi;
    case 2
        a_param = 1.2;
        f_target = t.^a_param * (1 + t.^(.5 + a_param) + ...
                   gamma(1 + a_param)./gamma(1 + a_param - alpha) * t .^(- alpha)) ...
                   - z_tr_phi - sqrt(t) * z_tr_phi.^2;
end

residual = D_alpha_sum - f_target;
loss_sym = mean(residual^2);
phi_cell = num2cell(phi);

loss_fun = matlabFunction(loss_sym, 'Vars', [phi_cell, {t, alpha}]);
grad_sym = gradient(loss_sym, phi);
loss_grad = matlabFunction(grad_sym, 'Vars', [phi_cell, {t, alpha}]);
z_tr_phi_fun = matlabFunction(z_tr_phi, 'Vars', [phi_cell, {t}]);

save('function_handles.mat','z_tr_phi_fun','loss_grad','loss_fun','example_id','n');
    function D = compute_fractional_derivative(expr, t_var, a_var)
        [c, p] = coeffs(expr, t_var);
        D = sym(0);
        for i = 1:length(p)
            deg = feval(symengine, 'degree', p(i), t_var);
            if deg > 0
                term = (gamma(deg + 1) / gamma(deg + 1 - a_var)) * t_var^(deg - a_var);
                D = D + c(i) * term;
            end
        end
    end
end