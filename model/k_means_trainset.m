%% K-Means Fraud Clustering (k = 2) using ONLY numeric columns
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

%% 3) Check for is_fraud column
varNames = string(T.Properties.VariableNames);
fraudIdx = find(strcmpi(varNames, "is_fraud"), 1);

if isempty(fraudIdx)
    error('The dataset must contain a column named "is_fraud".');
end

fraudColName = T.Properties.VariableNames{fraudIdx};

%% 4) Extract true labels from is_fraud
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

%% 5) Keep ONLY numeric/logical columns for clustering, excluding is_fraud
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

%% 6) Convert numeric table to matrix
X = table2array(Xtable);
X = double(X);

%% 7) Remove rows where is_fraud is missing
validLabelRows = ~isnan(yTrue);
X = X(validLabelRows, :);
yTrue = yTrue(validLabelRows);

%% 8) Remove columns that are all missing or constant
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

%% 9) Remove rows with too many missing values if desired
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

%% 10) Standardize features
Xz = zscore(X);

badCols = any(isnan(Xz), 1) | any(isinf(Xz), 1);
Xz(:, badCols) = [];
featureNames(badCols) = [];

if isempty(Xz)
    error('No usable standardized columns remain.');
end

fprintf('Final clustering matrix size: %d rows x %d columns.\n', size(Xz,1), size(Xz,2));

%% 11) Run k-means with k = 2
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

%% 12) Map clusters to fraud labels for best accuracy
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

%% 13) Evaluation metrics
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

%% 14) Fraud rate inside each cluster
cluster1FraudRate = mean(yTrue(idx == 1));
cluster2FraudRate = mean(yTrue(idx == 2));

fprintf('\nFraud rate in Cluster 1: %.4f\n', cluster1FraudRate);
fprintf('Fraud rate in Cluster 2: %.4f\n', cluster2FraudRate);

%% 15) Display which numeric columns were used
fprintf('\nNumeric columns used for clustering:\n');
disp(featureNames');

%% 16) Save results
resultTable = table((1:length(yTrue))', idx, yTrue, yPred, ...
    'VariableNames', {'RowID', 'Cluster', 'TrueLabel_is_fraud', 'PredictedFraudLabel'});

[outFile, outPath] = uiputfile('kmeans_results.csv', 'Save clustering results as');
if ~isequal(outFile, 0)
    writetable(resultTable, fullfile(outPath, outFile));
    fprintf('Results saved to: %s\n', fullfile(outPath, outFile));
else
    fprintf('Results not saved.\n');
end
