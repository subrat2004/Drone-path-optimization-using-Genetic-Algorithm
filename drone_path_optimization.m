%% Aircraft Route Optimization Using Genetic Algorithms (Single-File Implementation)
% This implementation follows the approach described in the paper
% "Aircraft Route Optimization Using Genetic Algorithms" by Li Qing et al.

clear; close all; clc;

% ============================= MAIN PARAMETERS =============================
A = [0, 0];          % Start point (x, y)
B = [150, 0];        % Target point (x, y)
C = 20;              % Max cross-track deviation
lambda = 1.2;        % Amplitude attenuation factor
N = 5;               % Chromosome length (sine wave components)
M = 100;             % Population size
max_generations = 170; % Number of generations

% ====================== PERFORMANCE ANALYSIS SETUP ======================
% Using grid sizes for analysis instead of chromosome length
grid_sizes = [25, 50, 75, 100, 150,500];  % Different grid sizes for analysis
times = zeros(size(grid_sizes));      % To store execution times
costs = zeros(size(grid_sizes));      % To store best costs

% ====================== PERFORMANCE ANALYSIS LOOP ======================
% Analysis based on different grid sizes
for i = 1:length(grid_sizes)
    current_grid_size = grid_sizes(i);
    fprintf('Running optimization with grid size = %d\n', current_grid_size);
    
    % Generate terrain with current grid size
    [x_grid, y_grid, h] = generate_terrain(A, B, C, current_grid_size);
    
    % Initialize population
    [population, omega, a_max] = initialize_population(M, N, lambda, C, A, B);
    
    % Measure execution time
    tic;
    [best_chromosome, best_cost_history, ~] = run_ga(...
        population, omega, a_max, A, B, x_grid, y_grid, h, max_generations, N);
    times(i) = toc;
    
    % Store best cost
    costs(i) = best_cost_history(end);
    
    fprintf('Grid Size = %d: Time = %.2f seconds, Final Cost = %.4f\n', ...
        current_grid_size, times(i), best_cost_history(end));
end

% ======================== VISUALIZATION ========================
% Run the standard case for visualization
[x_grid, y_grid, h] = generate_terrain(A, B, C, 100);  % Use default 100x100 grid for visualization
[population, omega, a_max] = initialize_population(M, N, lambda, C, A, B);
[best_chromosome, best_cost_history, time_per_gen] = run_ga(...
    population, omega, a_max, A, B, x_grid, y_grid, h, max_generations, N);

% Plot optimized route
plot_optimized_route(A, B, x_grid, y_grid, h, best_chromosome, omega);

% Plot convergence
plot_convergence(best_cost_history);

% Plot performance analysis
plot_performance_analysis(grid_sizes, times, costs, time_per_gen);

% ======================== FUNCTIONS ========================
function [x_grid, y_grid, h] = generate_terrain(A, B, C, grid_size)
    % Generate terrain with simple isolated features for 2D contour visualization
    % A: start point, B: end point, C: corridor width, grid_size: resolution
    
    % Create the grid
    x = linspace(A(1), B(1), grid_size);
    y = linspace(-C, C, grid_size);
    [x_grid, y_grid] = meshgrid(x, y);
    
    % Start with a flat base
    h = 50 * ones(grid_size, grid_size);
    
    % Create coordinates for feature placement
    [xx, yy] = meshgrid(1:grid_size, 1:grid_size);
    
    % Add features as simple Gaussian bumps (hills) and dips (valleys)
    % Format: [x_center, y_center, width, height, is_threat]
    % x,y are in normalized coordinates (0-1)
    % width is relative to grid_size
    % height is the actual height value (positive for hills, negative for valleys)
    % is_threat: 1 if it's a threat zone (will be colored differently)
    
    features = [
    % [x_pos, y_pos, width, height, is_threat]
    0.4, 0.6, 0.15, 200, 0;     % Hill in upper middle
    0.7, 0.1, 0.1, 100, 0;      % Small hill on right side
    0.5, -0.5, 0.12, -150, 1;   % Threat zone/valley in bottom middle
    0.2, -0.2, 0.08, 80, 0;      % Small hill on left side
    0.85, 0.4, 0.07, 120, 0;    % Hill in right middle area
    0.25, -0.3, 0.09, 90, 0;    % Hill in bottom left
    0.65, -0.2, 0.06, -100, 1;  % Small threat zone in bottom right
    0.15, 0.7, 0.13, 180, 0;    % Large hill in upper left
    0.55, 0.3, 0.05, 100, 0;     % Small bump in middle
    0.8, -0.6, 0.11, -130, 1;   % Threat zone in far bottom right
    0.3, -0.0, 0.08, -60, 1;     % Small threat zone in middle left
    0.9, -0.7, 0.10, 150, 0;     % Hill in upper right corner
    0.1, -0.7, 0.14, -120, 1;   % Threat zone in bottom left corner
    0.45, -0.1, 0.04, 50, 0;    % Tiny hill in middle
    0.75, 0.5, 0.12, 160, 0;    % Hill in upper right
    0.05, 0.4, 0.06, 85, 0;     % Small hill in left middle
    0.6, 0.7, 0.08, -110, 1;    % Threat zone in upper middle
    0.35, -0.65, 0.10, 140, 0;  % Hill in bottom
    0.95, 0.0, 0.07, -90, 1;    % Threat zone in far right
    0.2, 0.5, 0.05, 65, 0;      % Small hill in upper left middle
];
    
    % Apply each feature
    for i = 1:size(features, 1)
        % Convert normalized positions to grid coordinates
        feat_x = round(grid_size * features(i, 1));
        feat_y = round(grid_size * (features(i, 2) + 0.5)); % Shift y to match -C to C range
        feat_width = grid_size * features(i, 3);
        feat_height = features(i, 4);
        is_threat = features(i, 5);
        
        % Calculate distance from feature center
        dist = sqrt((xx - feat_x).^2 + (yy - feat_y).^2);
        
        % Create a Gaussian bump or dip
        if is_threat
            % For threats, make a steeper feature
            feat_effect = feat_height * exp(-(dist.^2) / (feat_width^2 * 0.5));
        else
            % For regular terrain, make a smoother feature
            feat_effect = feat_height * exp(-(dist.^2) / (feat_width^2));
        end
        
        % Add to height map
        h = h + feat_effect;
    end
    
    % Ensure minimum height
    h = max(h, 10);
    
    % Apply very minimal smoothing
    h = imgaussfilt(h, 0.5);
end
function [population, omega, a_max] = initialize_population(M, N, lambda, C, A, B)
    % Initialize a population of chromosomes
    % M: population size
    % N: chromosome length (number of sine components)
    
    L = norm(B - A);                  % Distance between start and target
    omega = 2*pi*(1:N)/L;             % Frequencies as per paper
    a_max = C ./ (lambda.^(0:N-1));   % Maximum amplitude constraints
    
    % Initialize random population within amplitude constraints
    population = zeros(M, N);
    for i = 1:M
        for j = 1:N
            population(i, j) = (2*rand() - 1) * a_max(j);
        end
    end
end

function cost = calculate_cost(chromosome, omega, A, B, x_grid, y_grid, h)
    % Calculate cost for a given chromosome
    
    % Generate route from chromosome
    L = norm(B - A);
    x = linspace(0, L, 100);
    y = zeros(size(x));
    
    % Sum up sine components
    for i = 1:length(chromosome)
        y = y + chromosome(i) * sin(omega(i) * x);
    end
    
    % Map route to actual coordinates
    angle = atan2(B(2) - A(2), B(1) - A(1));
    route_x = A(1) + x * cos(angle) - y * sin(angle);
    route_y = A(2) + x * sin(angle) + y * cos(angle);
    
    % Calculate terrain height along route using interpolation
    h_route = interp2(x_grid, y_grid, h, route_x, route_y, 'linear', max(h(:)));
    
    % Cost components from the paper
    w1 = 1;  % Cross-track deviation weight
    w2 = 2;  % Altitude weight
    w3 = 0;  % Threat weight (handled in terrain map)
    
    % Calculate cost based on paper's cost function
    cross_track_cost = w1 * sum(y.^2) / length(y);
    altitude_cost = w2 * sum(h_route) / length(h_route);
    
    cost = cross_track_cost + altitude_cost;
end

function [best_chromosome, best_cost_history, time_per_gen] = run_ga(...
    population, omega, a_max, A, B, x_grid, y_grid, h, max_generations, N)
    
    M = size(population, 1);
    best_cost_history = zeros(max_generations, 1);
    time_per_gen = zeros(max_generations, 1);
    
    for gen = 1:max_generations
        tic;
        
        % --- Evaluate Population ---
        costs = zeros(M, 1);
        for i = 1:M
            costs(i) = calculate_cost(population(i,:), omega, A, B, x_grid, y_grid, h);
        end
        
        % --- Sort Population by Cost ---
        [sorted_costs, idx] = sort(costs);
        population = population(idx, :);
        
        % --- Record Best Cost ---
        best_cost_history(gen) = sorted_costs(1);
        
        % If last generation, exit after recording best cost
        if gen == max_generations
            time_per_gen(gen) = toc;
            break;
        end
        
        % --- Elitism: Keep best, replace worst ---
        elite_count = 5;
        population(M-elite_count+1:M, :) = population(1:elite_count, :);
        
        % --- Crossover ---
        crossover_pairs = 20;  % Number of pairs to crossover
        for i = 1:crossover_pairs
            % Select two random chromosomes
            idx1 = randi(M);
            idx2 = randi(M);
            
            % Select random crossover point
            crossover_point = randi(N);
            
            % Swap genes
            temp = population(idx1, crossover_point);
            population(idx1, crossover_point) = population(idx2, crossover_point);
            population(idx2, crossover_point) = temp;
        end
        
        % --- Mutation ---
        mutation_count = 5;  % Number of genes to mutate
        for i = 1:mutation_count
            % Select random chromosome and gene
            chrom_idx = randi(M);
            gene_idx = randi(N);
            
            % Generate new random value within constraints
            population(chrom_idx, gene_idx) = (2*rand() - 1) * a_max(gene_idx);
        end
        
        time_per_gen(gen) = toc;
        
        % Display progress
        if mod(gen, 10) == 0
            fprintf('Generation %d: Best Cost = %.4f\n', gen, sorted_costs(1));
        end
    end
    
    best_chromosome = population(1, :);
end

function plot_optimized_route(A, B, x_grid, y_grid, h, best_chromosome, omega)
    % Plot terrain as contour map
    figure;
    contourf(x_grid, y_grid, h, 20, 'LineColor', 'none');
    colorbar;
    hold on;
    
    % Generate optimized route
    L = norm(B - A);
    x = linspace(0, L, 100);
    y = zeros(size(x));
    
    % Sum sine components
    for i = 1:length(best_chromosome)
        y = y + best_chromosome(i) * sin(omega(i) * x);
    end
    
    % Map to actual coordinates
    angle = atan2(B(2) - A(2), B(1) - A(1));
    route_x = A(1) + x * cos(angle) - y * sin(angle);
    route_y = A(2) + x * sin(angle) + y * cos(angle);
    
    % Plot route
    plot(route_x, route_y, 'r-', 'LineWidth', 2);
    
    % Plot start and end points
    plot(A(1), A(2), 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
    plot(B(1), B(2), 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
    
    % Formatting
    title('Optimized Aircraft Route');
    xlabel('X Position');
    ylabel('Y Position');
    legend('Terrain/Threats', 'Optimized Route', 'Start Point', 'End Point');
    grid on;
    
    % 3D Visualization
    figure;
    surf(x_grid, y_grid, h, 'EdgeColor', 'none', 'FaceAlpha', 0.7);
    hold on;
    
    % Get heights along route
    route_h = interp2(x_grid, y_grid, h, route_x, route_y, 'linear');
    safety_margin = 10;  % Add a small safety margin above terrain
    
    % Plot 3D route
    plot3(route_x, route_y, route_h + safety_margin, 'r-', 'LineWidth', 2);
    
    % Plot start and end points
    h_start = interp2(x_grid, y_grid, h, A(1), A(2), 'linear');
    h_end = interp2(x_grid, y_grid, h, B(1), B(2), 'linear');
    plot3(A(1), A(2), h_start + safety_margin, 'go', 'MarkerSize', 10, 'MarkerFaceColor', 'g');
    plot3(B(1), B(2), h_end + safety_margin, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
    
    % Formatting
    title('3D View of Optimized Aircraft Route');
    xlabel('X Position');
    ylabel('Y Position');
    zlabel('Height');
    legend('Terrain/Threats', 'Optimized Route', 'Start Point', 'End Point');
    view(45, 30);
    grid on;
end

function plot_convergence(best_cost_history)
    figure;
    plot(1:length(best_cost_history), best_cost_history, 'LineWidth', 2);
    title('Genetic Algorithm Convergence');
    xlabel('Generation');
    ylabel('Best Cost');
    grid on;
end

function plot_performance_analysis(grid_sizes, times, costs, time_per_gen)
    % Plot grid size vs execution time
    figure;
    
    % Plot total time vs grid size
    subplot(2, 2, 1);
    plot(grid_sizes, times, 'o-', 'LineWidth', 2);
    title('Grid Size vs. Execution Time');
    xlabel('Grid Size');
    ylabel('Time (seconds)');
    grid on;
    
    % Plot grid size vs best cost
    subplot(2, 2, 2);
    plot(grid_sizes, costs, 'o-', 'LineWidth', 2);
    title('Grid Size vs. Final Cost');
    xlabel('Grid Size');
    ylabel('Best Cost');
    grid on;
    
    % Plot time per generation
    subplot(2, 2, 3);
    plot(1:length(time_per_gen), time_per_gen * 1000, 'LineWidth', 2);
    title('Time per Generation');
    xlabel('Generation');
    ylabel('Time (ms)');
    grid on;
    
    % Plot time efficiency (cost reduction per time unit)
    subplot(2, 2, 4);
    efficiency = costs ./ times;
    plot(grid_sizes, efficiency, 'o-', 'LineWidth', 2);
    title('Cost-Time Efficiency');
    xlabel('Grid Size');
    ylabel('Cost/Time Ratio');
    grid on;
    
    % Adjust figure layout
    sgtitle('Performance Analysis of Genetic Algorithm');
    set(gcf, 'Position', [100, 100, 900, 700]);
end