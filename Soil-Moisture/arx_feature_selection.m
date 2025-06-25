clear all;
clc;
%% Load and prepare data
data = readtable("C:\Users\HP\Downloads\Soil_Moisture_Data_Sets.xlsx");
data = sortrows(data, {'latitude', 'longitude', 'time'});
numeric_date = datenum(data{:,1});
inputs = [numeric_date, data.latitude, data.longitude, data.clay_content, data.sand_content, data.silt_content];
output = data.sm_aux;
%inputs = table2array(inputs);
output = output(:);

%% Step 1: Determine ARX orders from ACF/PACF/CCF (example values - replace with your analysis)
n_a = 2;    % Output lags (from PACF cutoff)
n_b = 1;    % Input lags (from CCF peaks)
nk = 1;     % Input delay (from first significant CCF lag)

%% Step 2: Split data FIRST (70% train, 30% test)
num_samples = size(inputs, 1);
num_train = round(0.7 * num_samples);
train_inputs_raw = inputs(1:num_train, :);
train_output_raw = output(1:num_train);
test_inputs_raw = inputs(num_train+1:end, :);
test_output_raw = output(num_train+1:end);

%% Step 3: Create ARX features WITH DELAY (nk)
function [X_arx, y_target] = create_arx_features(inputs, output, n_a, n_b, nk)
    max_lag = max([n_a, n_b + nk - 1]);
    X_arx = [];
    for t = max_lag+1:length(output)
        t
        % Past outputs (y(t-1), ..., y(t-n_a))
        past_output = output(t-n_a:t-1);
        
        % Past inputs with delay (u(t-nk), ..., u(t-nk-n_b+1))
        past_inputs = inputs(t-nk-n_b+1:t-nk, :);
        
        X_arx = [X_arx; past_output(:)', past_inputs(:)'];
    end
    y_target = output(max_lag+1:end);
end

% Create features for train/test
[X_train, y_train] = create_arx_features(train_inputs_raw, train_output_raw, n_a, n_b, nk);
[X_test, y_test] = create_arx_features(test_inputs_raw, test_output_raw, n_a, n_b, nk);

%% Step 4: Feature selection on TRAINING data only
%corr_train = corr(X_train, y_train);
%threshold = 0.5;
%selected_features = abs(corr_train) >= threshold;

%if all(selected_features == 0)
    %warning('No features meet threshold. Using top 3 features.');
    %[~, top_idx] = sort(abs(corr_train), 'descend');
    %selected_features(top_idx(1:3)) = true;
%end

X_train_selected = X_train;
%(:, selected_features);
X_test_selected = X_test;
%(:, selected_features);

%% Step 5: KNN-GPR modeling
K = 30;

% Find K nearest neighbors for each test point
Samples_Index = knnsearch(X_train_selected, X_test_selected, 'K', K);

% Number of test samples
num_test_samples = size(y_test, 1);

% Preallocate prediction array
y_pred = zeros(num_test_samples, 1);

for i = 1:num_test_samples
    fprintf('Processing test sample %d of %d\n', i, num_test_samples);
    
    % Extract the local neighborhood
    local_inputs = X_train_selected(Samples_Index(i, :), :);
    local_outputs = y_train(Samples_Index(i, :));
    
    % Fit local GPR model
    model = fitrgp(local_inputs, local_outputs, ...
                   'KernelFunction', 'squaredexponential', ...
                   'Standardize', true);
    
    % Predict for current test point
    y_pred(i) = predict(model, X_test_selected(i, :));
end



%% Step 6: Evaluation
R2 = corr(y_test, y_pred)^2;
rmse = sqrt(mean((y_test - y_pred).^2));

fprintf('ARX Model Performance (nk=%d):\n', nk);
fprintf('R²: %.4f\n', R2);
fprintf('RMSE: %.4f\n', rmse);

% Plot results
figure;
subplot(2,1,1);
plot(y_test, 'b', 'LineWidth', 1.5); hold on;
plot(y_pred, 'r--', 'LineWidth', 1);
legend('Actual', 'Predicted');
title(sprintf('ARX-KNN-GPR Predictions (nk=%d)', nk));
xlabel('Sample Index');
ylabel('Soil Moisture');

subplot(2,1,2);
scatter(y_test, y_pred, 20, 'filled');
hold on;
plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--');
xlabel('Actual Values');
ylabel('Predicted Values');
title('Prediction vs Actual');
grid on;