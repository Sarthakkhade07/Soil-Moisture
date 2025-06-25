% Read data
clear all;
clc;
data = readtable("C:\Users\HP\Downloads\Soil_Moisture_Data_Sets.xlsx");
data = sortrows(data, {'latitude', 'longitude', 'time'});

% Convert date column to numeric format
numeric_date = datenum(data{:,1});
numerical_features = data{:,2:7};  % Columns 2 to 7 (input features + output)
numerical_data = [numeric_date, numerical_features];

% Split dataset into train (70%) and test (30%)
num_samples = size(numerical_data, 1);
num_train = round(0.7 * num_samples);
%rand_indices = randperm(num_samples);

train_data = numerical_data((1:num_train), :);
test_data = numerical_data((num_train+1:end), :);

% Extract input and output
input_training_data = train_data(:, 1:6);
output_training_data = train_data(:, 7);
input_testing_data = test_data(:, 1:6);
output_testing_data = test_data(:, 7);
% Run KNN search
K = 30;  % Number of nearest neighbors
Samples_Index = knnsearch(input_training_data, input_testing_data, 'K', K);

num_test_samples = size(input_testing_data, 1);
ypred = zeros(num_test_samples, 1);

for i = 1:num_test_samples
    i
    % Extract nearest neighbors (local dataset)
    local_data_input = input_training_data(Samples_Index(i, :), :);
    local_data_output = output_training_data(Samples_Index(i, :));
    
    gprMdl = fitrgp(local_data_input, local_data_output, 'KernelFunction', 'squaredexponential', 'Standardize', true);

    % Predict using the GPR model
    ypred(i) = predict(gprMdl, input_testing_data(i, :));

end

% Compute R² Value
R2_model = (corr(output_testing_data, ypred))^2;
disp("R² Value: ");
disp(R2_model);

% Plot Measured vs Predicted Values
figure;
plot(1:num_test_samples, output_testing_data, 'b', 'LineWidth', 1.5);
hold on;
plot(1:num_test_samples, ypred, 'r', 'LineWidth', 1.5);
legend('Measured', 'Predicted');
xlabel('Index');
ylabel('Soil Moisture');
title('Measured vs Predicted Soil Moisture');
grid on;
