%% Bipartite Neural Network: Training & Evaluation (Data Leakage Fixed)
clear; clc; close all;

% =========================================================================
% 1. DATA LOADING & STRICT FEATURE SEPARATION
% =========================================================================
filename = 'Final_Scored_Transactions.csv';
fprintf('Loading data and prepping features...\n');
T = readtable(filename, 'VariableNamingRule', 'preserve');

% Define Inputs (X): ONLY Time and Ratio. 
% (Transposed because patternnet expects features as rows and observations as columns)
X = [T.Hour_Of_Day'; T.Magnitude_Ratio']; 

% Define Target (Y): STRICTLY the Z-Score threshold.
z_threshold = 2.0;
Y = double(T.ZScore_Magnitude > z_threshold)';

% =========================================================================
% 2. TRAIN THE NEURAL NETWORK
% =========================================================================
% Create a pattern recognition network with 10 hidden neurons (matching your LaTeX)
hiddenLayerSize = 10;
net = patternnet(hiddenLayerSize);

% Train the network on the clean, separated data
fprintf('Training the Neural Network (this might take a few seconds)...\n');
[net, tr] = train(net, X, Y);

% Generate the new, mathematically valid continuous probabilities
predicted_scores = net(X)'; 
actual_labels = Y';

% =========================================================================
% 3. APPLY TOP 1% THRESHOLD & CALCULATE METRICS
% =========================================================================
percent_to_flag = 0.01; 
num_to_flag = round(percent_to_flag * height(T));
sorted_probs = sort(predicted_scores, 'descend');
prob_threshold = sorted_probs(num_to_flag);

% Flag transactions that meet the threshold
predicted_labels = predicted_scores >= prob_threshold;

% Calculate Confusion Matrix
TP = sum(actual_labels == 1 & predicted_labels == 1);
FP = sum(actual_labels == 0 & predicted_labels == 1);
TN = sum(actual_labels == 0 & predicted_labels == 0);
FN = sum(actual_labels == 1 & predicted_labels == 0);

% Calculate Precision, Recall, F1
precision = TP / (TP + FP);
recall = TP / (TP + FN);
f1_score = 2 * (precision * recall) / (precision + recall);

% Calculate AUCs
[~, ~, ~, roc_auc] = perfcurve(actual_labels, predicted_scores, 1);
[~, ~, ~, pr_auc] = perfcurve(actual_labels, predicted_scores, 1, 'XCrit', 'reca', 'YCrit', 'prec');

% Print Table Output (Formatted to match your existing LaTeX table)
fprintf('\n--- NEW BIPARTITE NETWORK STATS (Top 1%%) ---\n');
fprintf('ROC-AUC:   %s\n', strrep(sprintf('%.4f', roc_auc), '0.', '.'));
fprintf('PR-AUC:    %s\n', strrep(sprintf('%.4f', pr_auc), '0.', '.'));
fprintf('Precision: %s\n', strrep(sprintf('%.4f', precision), '0.', '.'));
fprintf('Recall:    %s\n', strrep(sprintf('%.4f', recall), '0.', '.'));
fprintf('F1 Score:  %s\n\n', strrep(sprintf('%.4f', f1_score), '0.', '.'));