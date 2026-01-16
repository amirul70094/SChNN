function [w_optim, final_loss, elapsed, weight_history, loss_history] = train_BFGS(alpha_val, f_loss, f_grad, x_train, num_params)
    % rng(0);
    
    weights_init = randn(num_params, 1);
    
    weight_history = [];
    loss_history = [];
    
    % Define output function to capture iterations
    function stop = outputfcn(x, optimValues, state)
        stop = false;
        switch state
            case 'init'
                % Initialize history
                weight_history = x;
                loss_history = optimValues.fval;
            case 'iter'
                % Store current iteration
                weight_history = [weight_history, x];
                loss_history = [loss_history; optimValues.fval];
            case 'done'
                % Final processing if needed
        end
    end
    
    options = optimoptions('fminunc', ...
        'Algorithm', 'quasi-newton', ...
        'HessUpdate', 'bfgs', ...
        'SpecifyObjectiveGradient', true, ...
        'MaxIterations', 1000, ...
        'OptimalityTolerance', 1e-16, ...
        'FunctionTolerance', 1e-16, ...
        'StepTolerance', 1e-16, ...
        'OutputFcn', @outputfcn);  % Add output function


        % 'Display', 'iter-detailed', ...
    
    tic;
    [w_optim, final_loss] = fminunc(@objective_with_grad, weights_init, options);
    elapsed = toc;
    
    % Save data for plotting
    save('optimization_history.mat', 'weight_history', 'loss_history', 'w_optim', 'final_loss');
    
    % --- Nested Objective Function ---
    function [L, G] = objective_with_grad(w)
        w_cell = num2cell(w);
        loss_vals = f_loss(w_cell{:}, x_train, alpha_val);
        L = mean(loss_vals);
        
        if nargout > 1
            grad_raw = f_grad(w_cell{:}, x_train, alpha_val);
            G_matrix = reshape(grad_raw, [length(x_train), num_params]);
            G = mean(G_matrix, 1)';
        end
    end
end