function main_NN()
    load('function_handles.mat','z_tr_phi_fun','loss_grad','loss_fun','example_id','n');
    COLLOCATION_POINTS = (0:0.01:1)';
    % rng(0);
    % Define variable-order alpha(t)
    alpha_t = 1 - 0.5*cos(COLLOCATION_POINTS);
    % alpha_t = COLLOCATION_POINTS .* cos(COLLOCATION_POINTS); 
    % alpha_t = (COLLOCATION_POINTS + 2 .* exp(COLLOCATION_POINTS))./7;
    % alpha_t = .9 + 0.1 .*sin(COLLOCATION_POINTS);
    % alpha_t = 1;
    
    fprintf('Starting BFGS...\n');
    [weights, final_loss, time_taken, weight_history, loss_history] = train_BFGS(alpha_t, loss_fun, loss_grad, COLLOCATION_POINTS, n);
    
    fprintf('\nOptimization Summary for Ex:%d n=%d:\n',example_id, n);
    fprintf('Final Mean Squared Loss: %.6e\n', final_loss);
    fprintf('Total Computation Time: %.2f seconds\n', time_taken);
    
    %% --- Results Visualization ---
    analyze_and_plot(weights, alpha_t, z_tr_phi_fun, n);
    weights_plot(weight_history)
    
    %%
    function analyze_and_plot(weights, alpha_val, sol_handle, n)
        
        t_test = (0:0.1:1)'; 
        % t_test = sort(rand(100, 1));
    
        
        % Analytical solution for Example 3
        y_exact = exp(t_test);
        % y_exact = t_test .^ 1.2;
    
    switch n
        case 8
            y_pred = sol_handle(weights(1), weights(2), weights(3), weights(4), ...
                                weights(5), weights(6), weights(7), weights(8), t_test);
        case 7
            y_pred = sol_handle(weights(1), weights(2), weights(3), weights(4), ...
                                weights(5), weights(6), weights(7), t_test);
        case 6
            y_pred = sol_handle(weights(1), weights(2), weights(3), weights(4), ...
                                weights(5), weights(6), t_test);
    end
        
    
        abs_error = abs(y_exact - y_pred);
    
        fprintf('\n%s\n', repmat('-', 1, 45));
        fprintf('%-10s | %-15s | %-15s\n', 't', 'Exact y(t)', 'Abs. Error');
        fprintf('%s\n', repmat('-', 1, 45));
        for i = 1:length(t_test)
            fprintf('%-10.6f | %-15.6e | %-15.6e\n', ...
                t_test(i), y_exact(i), abs_error(i));
        end
        fprintf('%s\n', repmat('-', 2, 45));
    
        fprintf("\nMax error: %.4e", max(abs_error));

    
    
        figure;
        plot(t_test, y_pred, 'r-', 'LineWidth', 2, 'DisplayName', 'SChNN'); hold on;
        plot(t_test, y_exact, 'b--', 'LineWidth', 2, 'DisplayName', 'Exact');
        legend;
        xlabel('t');
        ylabel('\chi(t)');
        grid off;
        % 
        t_test = linspace(0, 1, length(abs_error))';    
        figure;
        plot(t_test, abs_error, 'r-', 'LineWidth', 2); 
        xlabel('t');
        ylabel('abs error');
        grid on;
        legend('SChNN abs error');
    end

function weights_plot(weights_history)
    figure;
    hold on;
    
    % Plot each weight with a different color
    for i = 1:size(weights_history, 1)
        plot(weights_history(i,:), 'DisplayName', sprintf('\\phi_{%d}', i));
    end
    
    xlabel('Epoch');
    ylabel('\phi Values');
    title('\phi Values');
    legend('show', 'Location', 'best');
    grid off;
    hold off;
end
end
