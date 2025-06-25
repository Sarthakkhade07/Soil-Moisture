clear all;
clc;

%% Load and prepare data
data = readtable("C:\Users\HP\Downloads\Soil_Moisture_Data_Sets.xlsx");
data = sortrows(data, {'latitude', 'longitude', 'time'});
numeric_date = datenum(data{:,1});
inputs = [numeric_date, data.latitude, data.longitude, data.clay_content, data.sand_content, data.silt_content];
output = data.sm_aux;
output = output(:);

%% Step 1: Determine ARX orders
n_a = 2;    % Output lags
n_b = 2;    % Input lags
nk = 1;     % Input delay

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

%% Create features for train/test
[X_train, y_train] = create_arx_features(train_inputs_raw, train_output_raw, n_a, n_b, nk);
[X_test, y_test] = create_arx_features(test_inputs_raw, test_output_raw, n_a, n_b, nk);

%% Standardize training data
mu = mean(X_train);
sigma = std(X_train);
X_train_std = (X_train - mu) ./ sigma;

% PCA on training data
[coeff, score, ~, ~, explained] = pca(X_train_std); 

% Keep components explaining % variance
cum_var = cumsum(explained);
num_pcs = find(cum_var >= 98, 1); 
X_train_pca = score(:, 1:num_pcs); 

% Transform test data using PC1 loadings
X_test_std = (X_test - mu) ./ sigma; 
X_test_pca = X_test_std * coeff(:, 1:num_pcs); 

%% KNN-GPR Modeling
K = 30;
num_test_samples = size(X_test_pca, 1);
y_pred = zeros(num_test_samples, 1);

% Find K nearest neighbors for each test point
Samples_Index = knnsearch(X_train_pca, X_test_pca, 'K', K);

for i = 1:num_test_samples
    fprintf('Processing test sample %d of %d\n', i, num_test_samples);
    
    % Extract the local neighborhood
    local_inputs = X_train_pca(Samples_Index(i, :),:);
    local_outputs = y_train(Samples_Index(i, :));
    
    % Fit local GPR model
    model = fitrgp(local_inputs, local_outputs, ...
                   'KernelFunction', 'squaredexponential', ...
                   'Standardize', true);
    
    % Predict for current test point
    y_pred(i) = predict(model, X_test_pca(i,:));
end

R2 = corr(y_test, y_pred)^2;
disp(['R² (PCA-ARX): ', num2str(R2)]);

%% Visualization
% Plot explained variance
figure;
subplot(2,2,1);
plot(explained(1:10), 'bo-', 'LineWidth', 2);
xlabel('Principal Component');
ylabel('Variance Explained (%)');
title('PCA: Explained Variance');
grid on;

% Scree plot
subplot(2,2,2);
bar(explained(1:10));
xlabel('Principal Component');
ylabel('Variance Explained (%)');
title('Scree Plot');

% PC1 Loadings plot
subplot(2,2,[3 4]);
stem(coeff(:,1), 'filled', 'LineWidth', 1.5);
xlabel('ARX Features');
ylabel('Loading Weight');
title('PC1 Component Loadings');
grid on;

%% Projection Plot
figure;
scatter(X_train_pca, y_train, 50, 'filled');
hold on;
scatter(X_test_pca, y_test, 50, 'x', 'LineWidth', 1.5);
xlabel('Projection on PC1');
ylabel('Soil Moisture');
title('PC1 Projection: Train (circles) vs Test (crosses)');
legend('Training Data', 'Test Data');
grid on;