%% Aircraft Route Optimization Using Simulated Annealing

clear; close all; clc;

% Core setup
A = [0, 0]; B = [150, 0]; C = 20; lambda = 1.2; N = 5;
initial_temp = 100; cooling_rate = 0.95; min_temp = 0.1; iterations_per_temp = 50;

% Analyze for different grid sizes
grid_sizes = [25, 50, 75, 100, 150, 500];
times = zeros(size(grid_sizes));
costs = zeros(size(grid_sizes));

for i = 1:length(grid_sizes)
    current_grid_size = grid_sizes(i);
    fprintf('Running optimization with grid size = %d\n', current_grid_size);
    
    [x_grid, y_grid, h] = generate_terrain(A, B, C, current_grid_size);
    L = norm(B - A);
    omega = 2*pi*(1:N)/L;
    a_max = C ./ (lambda.^(0:N-1));
    
    tic;
    [best_solution, cost_history, ~] = run_simulated_annealing(...
        A, B, x_grid, y_grid, h, omega, a_max, N, ...
        initial_temp, cooling_rate, min_temp, iterations_per_temp);
    times(i) = toc;
    costs(i) = cost_history(end);
    
    fprintf('Grid Size = %d: Time = %.2f seconds, Final Cost = %.4f\n', ...
        current_grid_size, times(i), cost_history(end));
end

% Visualize with standard grid
[x_grid, y_grid, h] = generate_terrain(A, B, C, 100);
L = norm(B - A);
omega = 2*pi*(1:N)/L;
a_max = C ./ (lambda.^(0:N-1));

[best_solution, cost_history, time_per_iter] = run_simulated_annealing(...
    A, B, x_grid, y_grid, h, omega, a_max, N, ...
    initial_temp, cooling_rate, min_temp, iterations_per_temp);

plot_optimized_route(A, B, x_grid, y_grid, h, best_solution, omega);
plot_convergence(cost_history);
plot_performance_analysis(grid_sizes, times, costs, time_per_iter);

%% Terrain Generation
function [x_grid, y_grid, h] = generate_terrain(A, B, C, grid_size, features)
    x = linspace(A(1), B(1), grid_size);
    y = linspace(-C, C, grid_size);
    [x_grid, y_grid] = meshgrid(x, y);
    h = 50 * ones(grid_size, grid_size);
    
    [xx, yy] = meshgrid(1:grid_size, 1:grid_size);
    
    if nargin < 5 || isempty(features)
        features = [
        % [x_pos, y_pos, width, height, is_threat]
        0.4, 0.6, 0.15, 200, 0;     % Hill in upper middle
        0.7, 0.1, 0.1, 100, 0;      % Small hill on right side
        0.5, -0.5, 0.12, -150, 1;   % Threat zone/valley in bottom middle
        0.2, -0.2, 0.08, 80, 0;     % Small hill on left side
        0.85, 0.4, 0.07, 120, 0;    % Hill in right middle area
        0.25, -0.3, 0.09, 90, 0;    % Hill in bottom left
        0.65, -0.2, 0.06, -100, 1;  % Small threat zone in bottom right
        0.15, 0.7, 0.13, 180, 0;    % Large hill in upper left
        0.55, 0.3, 0.05, 100, 0;    % Small bump in middle
        0.8, -0.6, 0.11, -130, 1;   % Threat zone in far bottom right
        0.3, -0.0, 0.08, -60, 1;    % Small threat zone in middle left
        0.9, -0.7, 0.10, 150, 0;    % Hill in upper right corner
        0.1, -0.7, 0.14, -120, 1;   % Threat zone in bottom left corner
        0.45, -0.1, 0.04, 50, 0;    % Tiny hill in middle
        0.75, 0.5, 0.12, 160, 0;    % Hill in upper right
        0.05, 0.4, 0.06, 85, 0;     % Small hill in left middle
        0.6, 0.7, 0.08, -110, 1;    % Threat zone in upper middle
        0.35, -0.65, 0.10, 140, 0;  % Hill in bottom
        0.95, 0.0, 0.07, -90, 1;    % Threat zone in far right
        0.2, 0.5, 0.05, 65, 0;      % Small hill in upper left middle
        ];
    end
    
    for i = 1:size(features, 1)
        feat_x = round(grid_size * features(i, 1));
        feat_y = round(grid_size * (features(i, 2) + 0.5));
        feat_width = grid_size * features(i, 3);
        feat_height = features(i, 4);
        is_threat = features(i, 5);
        
        dist = sqrt((xx - feat_x).^2 + (yy - feat_y).^2);
        factor = is_threat * 0.5 + ~is_threat;
        h = h + feat_height * exp(-(dist.^2) / (feat_width^2 * factor));
    end
    
    h = max(h, 10);
    h = imgaussfilt(h, 0.5);
end

%% Cost Function
function cost = calculate_cost(solution, omega, A, B, x_grid, y_grid, h)
    L = norm(B - A);
    x = linspace(0, L, 100);
    y = zeros(size(x));
    
    for i = 1:length(solution)
        y = y + solution(i) * sin(omega(i) * x);
    end
    
    angle = atan2(B(2) - A(2), B(1) - A(1));
    route_x = A(1) + x * cos(angle) - y * sin(angle);
    route_y = A(2) + x * sin(angle) + y * cos(angle);
    
    h_route = interp2(x_grid, y_grid, h, route_x, route_y, 'linear', max(h(:)));
    
    w1 = 1;  % Cross-track deviation weight
    w2 = 2;  % Altitude weight
    
    cross_track_cost = w1 * mean(y.^2);
    altitude_cost = w2 * mean(h_route);
    
    cost = cross_track_cost + altitude_cost;
end

%% Generate Neighbor
function neighbor = generate_neighbor(current_solution, a_max, perturbation_strength)
    neighbor = current_solution;
    idx = randi(length(current_solution));
    max_change = perturbation_strength * a_max(idx);
    change = (2 * rand() - 1) * max_change;
    neighbor(idx) = neighbor(idx) + change;
    neighbor(idx) = max(min(neighbor(idx), a_max(idx)), -a_max(idx));
end

%% Simulated Annealing
function [best_solution, cost_history, time_per_iter] = run_simulated_annealing(...
    A, B, x_grid, y_grid, h, omega, a_max, N, initial_temp, cooling_rate, min_temp, iterations_per_temp)
    
    current_solution = (2*rand(1, N) - 1) .* a_max;
    current_cost = calculate_cost(current_solution, omega, A, B, x_grid, y_grid, h);
    
    best_solution = current_solution;
    best_cost = current_cost;
    
    temp = initial_temp;
    
    iter_count = 0;
    total_iters = ceil(log(min_temp/initial_temp)/log(cooling_rate)) * iterations_per_temp;
    cost_history = zeros(total_iters, 1);
    time_per_iter = zeros(total_iters, 1);
    
    while temp > min_temp
        for i = 1:iterations_per_temp
            tic;
            iter_count = iter_count + 1;
            
            perturbation_strength = temp / initial_temp;
            neighbor_solution = generate_neighbor(current_solution, a_max, perturbation_strength);
            neighbor_cost = calculate_cost(neighbor_solution, omega, A, B, x_grid, y_grid, h);
            
            delta_cost = neighbor_cost - current_cost;
            
            if delta_cost < 0 || rand() < exp(-delta_cost / temp)
                current_solution = neighbor_solution;
                current_cost = neighbor_cost;
                
                if current_cost < best_cost
                    best_solution = current_solution;
                    best_cost = current_cost;
                end
            end
            
            cost_history(iter_count) = best_cost;
            time_per_iter(iter_count) = toc;
            
            if mod(iter_count, 50) == 0
                fprintf('Iteration %d: Temp = %.4f, Best Cost = %.4f\n', ...
                    iter_count, temp, best_cost);
            end
        end
        
        temp = temp * cooling_rate;
    end
    
    cost_history = cost_history(1:iter_count);
    time_per_iter = time_per_iter(1:iter_count);
end

%% Plot Route
function plot_optimized_route(A, B, x_grid, y_grid, h, best_solution, omega)
    figure;
    contourf(x_grid, y_grid, h, 20, 'LineColor', 'none');
    colorbar;
    hold on;
    
    L = norm(B - A);
    x = linspace(0, L, 100);
    y = zeros(size(x));
    
    for i = 1:length(best_solution)
        y = y + best_solution(i) * sin(omega(i) * x);
    end
    
    angle = atan2(B(2) - A(2), B(1) - A(1));
    route_x = A(1) + x * cos(angle) - y * sin(angle);
    route_y = A(2) + x * sin(angle) + y * cos(angle);
    
    plot(route_x, route_y, 'r-', 'LineWidth', 2);
    plot(A(1), A(2), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
    plot(B(1), B(2), 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
    
    title('Optimized Aircraft Route (Simulated Annealing)');
    xlabel('X Position'); ylabel('Y Position');
    legend('Terrain/Threats', 'Optimized Route', 'Start Point', 'End Point');
    grid on;
    
    figure;
    surf(x_grid, y_grid, h, 'EdgeColor', 'none', 'FaceAlpha', 0.7);
    hold on;
    
    route_h = interp2(x_grid, y_grid, h, route_x, route_y, 'linear');
    safety_margin = 10;
    
    plot3(route_x, route_y, route_h + safety_margin, 'r-', 'LineWidth', 2);
    
    h_start = interp2(x_grid, y_grid, h, A(1), A(2), 'linear');
    h_end = interp2(x_grid, y_grid, h, B(1), B(2), 'linear');
    plot3(A(1), A(2), h_start + safety_margin, 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
    plot3(B(1), B(2), h_end + safety_margin, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
    
    title('3D View of Optimized Aircraft Route');
    xlabel('X Position'); ylabel('Y Position'); zlabel('Height');
    legend('Terrain/Threats', 'Optimized Route', 'Start', 'End');
    view(45, 30); grid on;
end

%% Plot Convergence
function plot_convergence(cost_history)
    figure;
    plot(1:length(cost_history), cost_history, 'LineWidth', 2);
    title('Simulated Annealing Convergence');
    xlabel('Iteration'); ylabel('Best Cost');
    grid on;
end

%% Plot Performance
function plot_performance_analysis(grid_sizes, times, costs, time_per_iter)
    figure;
    
    subplot(2, 2, 1);
    plot(grid_sizes, times, 'o-', 'LineWidth', 2);
    title('Grid Size vs. Execution Time');
    xlabel('Grid Size'); ylabel('Time (seconds)');
    grid on;
    
    subplot(2, 2, 2);
    plot(grid_sizes, costs, 'o-', 'LineWidth', 2);
    title('Grid Size vs. Final Cost');
    xlabel('Grid Size'); ylabel('Best Cost');
    grid on;
    
    subplot(2, 2, 3);
    plot(1:length(time_per_iter), time_per_iter * 1000, 'LineWidth', 2);
    title('Time per Iteration');
    xlabel('Iteration'); ylabel('Time (ms)');
    grid on;
    
    subplot(2, 2, 4);
    efficiency = costs ./ times;
    plot(grid_sizes, efficiency, 'o-', 'LineWidth', 2);
    title('Cost-Time Efficiency');
    xlabel('Grid Size'); ylabel('Cost/Time Ratio');
    grid on;
    
    sgtitle('Performance Analysis of Simulated Annealing');
    set(gcf, 'Position', [100, 100, 900, 700]);
end