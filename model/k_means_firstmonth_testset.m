%% K-Means Fraud Clustering (k = 2) using ONLY numeric columns - FIRST 87645 ROWS ONLY
clear; clc;

%% 1) Select CSV file
[fileName, filePath] = uigetfile('*.csv', 'Select the credit card transaction CSV file');
if isequal(fileName, 0)
    error('No file selected. Script terminated.');
end

csvFile = fullfile(filePath, fileName);
fprintf('Reading file: %s\n', csvFile);

%% 2) Read table
T = readtable(csvFile, 'TextType', 'string', 'VariableNamingRule', 'preserve');
fprintf('Loaded dataset with %d rows and %d columns.\n', height(T), width(T));

%% 3) Keep ONLY the first 87645 rows
maxRows = 87645;
if height(T) > maxRows
    T = T(1:maxRows, :);
    fprintf('Truncated to first %d rows.\n', maxRows);
else
    fprintf('Dataset has only %d rows, using all available rows.\n', height(T));
end

%% 4) Check for is_fraud column
varNames = string(T.Properties.VariableNames);
fraudIdx = find(strcmpi(varNames, "is_fraud"), 1);

if isempty(fraudIdx)
    error('The dataset must contain a column named "is_fraud".');
end

fraudColName = T.Properties.VariableNames{fraudIdx};

%% 5) Extract true labels from is_fraud
yTrueRaw = T.(fraudColName);

if isnumeric(yTrueRaw) || islogical(yTrueRaw)
    yTrue = double(yTrueRaw);
elseif isstring(yTrueRaw) || iscategorical(yTrueRaw) || iscellstr(yTrueRaw)
    yStr = lower(strtrim(string(yTrueRaw)));
    yTrue = nan(size(yStr));
    yTrue(yStr == "1" | yStr == "true" | yStr == "fraud" | yStr == "yes") = 1;
    yTrue(yStr == "0" | yStr == "false" | yStr == "legit" | yStr == "legitimate" | yStr == "no") = 0;
    
    if any(isnan(yTrue))
        error('Could not convert is_fraud into binary 0/1 values.');
    end
else
    error('Unsupported data type for is_fraud.');
end

if ~all(ismember(unique(yTrue(~isnan(yTrue))), [0 1]))
    error('is_fraud must contain only 0 and 1 values.');
end

%% 6) Keep ONLY numeric/logical columns for clustering, excluding is_fraud
numericMask = false(1, width(T));

for j = 1:width(T)
    col = T.(T.Properties.VariableNames{j});
    if isnumeric(col) || islogical(col)
        numericMask(j) = true;
    end
end

numericMask(fraudIdx) = false;  % exclude is_fraud from clustering

Xtable = T(:, numericMask);
featureNames = string(Xtable.Properties.VariableNames);

fprintf('Using %d numeric feature columns for clustering.\n', width(Xtable));

if width(Xtable) == 0
    error('No numeric feature columns found besides is_fraud.');
end

%% 7) Convert numeric table to matrix
X = table2array(Xtable);
X = double(X);

%% 8) Remove rows where is_fraud is missing
validLabelRows = ~isnan(yTrue);
X = X(validLabelRows, :);
yTrue = yTrue(validLabelRows);

fprintf('After removing rows with missing fraud labels: %d rows remain.\n', length(yTrue));

%% 9) Remove columns that are all missing or constant
allMissingCols = all(isnan(X), 1);
constantCols = false(1, size(X,2));

for j = 1:size(X,2)
    col = X(:,j);
    col = col(~isnan(col));
    if isempty(col)
        constantCols(j) = true;
    else
        constantCols(j) = numel(unique(col)) <= 1;
    end
end

removeCols = allMissingCols | constantCols;
X(:, removeCols) = [];
featureNames(removeCols) = [];

fprintf('Remaining usable numeric columns: %d\n', size(X,2));

if isempty(X)
    error('No usable numeric columns remain after cleaning.');
end

%% 10) Remove rows with too many missing values if desired
% Here we keep rows and fill missing values with column medians

for j = 1:size(X,2)
    col = X(:,j);
    missingMask = isnan(col);
    if any(missingMask)
        medVal = median(col(~missingMask), 'omitnan');
        if isnan(medVal)
            medVal = 0;
        end
        col(missingMask) = medVal;
        X(:,j) = col;
    end
end

%% 11) Standardize features
Xz = zscore(X);

badCols = any(isnan(Xz), 1) | any(isinf(Xz), 1);
Xz(:, badCols) = [];
featureNames(badCols) = [];

if isempty(Xz)
    error('No usable standardized columns remain.');
end

fprintf('Final clustering matrix size: %d rows x %d columns.\n', size(Xz,1), size(Xz,2));

%% 12) Run k-means with k = 2
rng(1);

opts = statset('UseParallel', false, 'MaxIter', 300);

[idx, C, sumd] = kmeans(Xz, 2, ...
    'Distance', 'sqeuclidean', ...
    'Replicates', 5, ...
    'Options', opts, ...
    'Display', 'final');

fprintf('\nK-means complete.\n');
fprintf('Cluster 1 size: %d\n', sum(idx == 1));
fprintf('Cluster 2 size: %d\n', sum(idx == 2));

%% 13) Map clusters to fraud labels for best accuracy
yPredA = double(idx == 2); % cluster 2 = fraud
yPredB = double(idx == 1); % cluster 1 = fraud

accA = mean(yPredA == yTrue);
accB = mean(yPredB == yTrue);

if accA >= accB
    yPred = yPredA;
    chosenMapping = 'Cluster 1 -> Legit (0), Cluster 2 -> Fraud (1)';
else
    yPred = yPredB;
    chosenMapping = 'Cluster 1 -> Fraud (1), Cluster 2 -> Legit (0)';
end

fprintf('\nChosen mapping: %s\n', chosenMapping);

%% 14) Evaluation metrics
TP = sum((yPred == 1) & (yTrue == 1));
TN = sum((yPred == 0) & (yTrue == 0));
FP = sum((yPred == 1) & (yTrue == 0));
FN = sum((yPred == 0) & (yTrue == 1));

accuracy  = (TP + TN) / numel(yTrue);
precision = TP / max(TP + FP, 1);
recall    = TP / max(TP + FN, 1);
f1        = 2 * precision * recall / max(precision + recall, eps);

fprintf('\n===== Evaluation Against is_fraud =====\n');
fprintf('Accuracy : %.4f\n', accuracy);
fprintf('Precision: %.4f\n', precision);
fprintf('Recall   : %.4f\n', recall);
fprintf('F1 Score : %.4f\n', f1);

fprintf('\nConfusion Matrix (rows = true, cols = predicted)\n');
fprintf('            Pred 0     Pred 1\n');
fprintf('True 0   %10d %10d\n', TN, FP);
fprintf('True 1   %10d %10d\n', FN, TP);
%% 18) Calculate PR-AUC and ROC-AUC
% For clustering, we need scoring values to compute these curves.
% Since k-means doesn't naturally output probabilities, we use:
% Option 1: Distance to cluster centroids (negative distance as score)

% Calculate distances to both centroids for each point
distToCentroid1 = zeros(size(Xz,1), 1);
distToCentroid2 = zeros(size(Xz,1), 1);

for i = 1:size(Xz,1)
    distToCentroid1(i) = norm(Xz(i,:) - C(1,:));
    distToCentroid2(i) = norm(Xz(i,:) - C(2,:));
end

% Score based on relative distance (closer to fraud cluster = higher score)
if chosenMapping == 'Cluster 1 -> Fraud (1), Cluster 2 -> Legit (0)'
    % Cluster 1 is fraud, so closer to centroid 1 = higher fraud probability
    fraudScores = exp(-distToCentroid1) ./ (exp(-distToCentroid1) + exp(-distToCentroid2));
else
    % Cluster 2 is fraud, so closer to centroid 2 = higher fraud probability
    fraudScores = exp(-distToCentroid2) ./ (exp(-distToCentroid1) + exp(-distToCentroid2));
end

% Calculate ROC curve and AUC
[Xroc, Yroc, Troc, AUCroc] = perfcurve(yTrue, fraudScores, 1);
fprintf('\n===== ROC-AUC =====\n');
fprintf('ROC-AUC: %.4f (%.2f%%)\n', AUCroc, AUCroc * 100);

% Calculate Precision-Recall curve and AUC
% For PR curve, we need to compute precision and recall at various thresholds
[precisionPR, recallPR, Tpr, AUCpr] = perfcurve(yTrue, fraudScores, 1, 'xCrit', 'reca', 'yCrit', 'prec');
fprintf('PR-AUC : %.4f (%.2f%%)\n', AUCpr, AUCpr * 100);

%% 19) Plot ROC and PR curves
figure('Position', [100, 100, 1200, 500]);

% ROC Curve subplot
subplot(1,2,1);
plot(Xroc, Yroc, 'b-', 'LineWidth', 2);
hold on;
plot([0 1], [0 1], 'k--', 'LineWidth', 1);
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title(sprintf('ROC Curve (AUC = %.4f)', AUCroc));
grid on;
legend('K-Means (k=2)', 'Random Classifier', 'Location', 'southeast');

% PR Curve subplot
subplot(1,2,2);
plot(recallPR, precisionPR, 'r-', 'LineWidth', 2);
xlabel('Recall');
ylabel('Precision');
title(sprintf('Precision-Recall Curve (AUC = %.4f)', AUCpr));
grid on;

% Add baseline (fraud prevalence)
fraudPrevalence = mean(yTrue);
hold on;
yline(fraudPrevalence, 'k--', 'LineWidth', 1);
legend('K-Means (k=2)', sprintf('Baseline (Prevalence = %.3f)', fraudPrevalence), 'Location', 'best');

sgtitle('K-Means Clustering Performance (k=2)');

%% 20) Save figure
[figFile, figPath] = uiputfile('kmeans_curves_first_87645.png', 'Save ROC/PR curves as');
if ~isequal(figFile, 0)
    saveas(gcf, fullfile(figPath, figFile));
    fprintf('Curves saved to: %s\n', fullfile(figPath, figFile));
end

%% 21) Display summary with new metrics
fprintf('\n===== COMPLETE PERFORMANCE SUMMARY =====\n');
fprintf('Accuracy : %.4f (%.2f%%)\n', accuracy, accuracy*100);
fprintf('Precision: %.4f (%.2f%%)\n', precision, precision*100);
fprintf('Recall   : %.4f (%.2f%%)\n', recall, recall*100);
fprintf('F1 Score : %.4f (%.2f%%)\n', f1, f1*100);
fprintf('ROC-AUC  : %.4f (%.2f%%)\n', AUCroc, AUCroc*100);
fprintf('PR-AUC   : %.4f (%.2f%%)\n', AUCpr, AUCpr*100);
fprintf('Fraud prevalence in dataset: %.4f (%.2f%%)\n', fraudPrevalence, fraudPrevalence*100);
%% 15) Fraud rate inside each cluster
cluster1FraudRate = mean(yTrue(idx == 1));
cluster2FraudRate = mean(yTrue(idx == 2));

fprintf('\nFraud rate in Cluster 1: %.4f\n', cluster1FraudRate);
fprintf('Fraud rate in Cluster 2: %.4f\n', cluster2FraudRate);

%% 16) Display which numeric columns were used
fprintf('\nNumeric columns used for clustering:\n');
disp(featureNames');

%% 17) Save results
resultTable = table((1:length(yTrue))', idx, yTrue, yPred, ...
    'VariableNames', {'RowID', 'Cluster', 'TrueLabel_is_fraud', 'PredictedFraudLabel'});

[outFile, outPath] = uiputfile('kmeans_results_first_87645.csv', 'Save clustering results as');
if ~isequal(outFile, 0)
    writetable(resultTable, fullfile(outPath, outFile));
    fprintf('Results saved to: %s\n', fullfile(outPath, outFile));
else
    fprintf('Results not saved.\n');
end
