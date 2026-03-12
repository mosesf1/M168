# Fraud Detection using Network Features for Synthetic Data: A Comparative Study
## M168 Final Project 

This repository contains the implementation & analysis for credit card fraud detection using machine learning & network-based transaction features. Our project evaluates both supervised & unsupervised models on a large synthetic credit card transaction dataset & compares their effectiveness in identifying fraudulent transactions.

Fraud detection presents a challenging machine learning problem due to extreme class imbalance, evolving fraud patterns & complex behavioral relationships between cardholders & merchants. This study explores how behavioral, geographic & network-based features can improve fraud detection performance.

## Models Implemented
- Random Forest
- XGBoost
- Isolation Forest
- K-Means Clustering
- Neural Networks
Both supervised classification models and unsupervised anomaly detection models are evaluated.

## Dataset
Synthetic credit card transaction dataset with network-based features derived from the APATE framework.

## Structure
- `model/` – machine learning implementations
- `Data/` – processed feature datasets
  - `intrinsic/` - generate intrinsic feature
  - `network/` - generate network feature
  - `merge/` - sample code to merge the extracted features
- `fraud_detection/` - original `fraudTest1m.csv` and `fraudTrain.csv`
- `report/` – LaTeX project report

## 0. How to Generate Features
For all files in either `Data/intrinsic` or `Data/network`, use the `fraudTest1m.csv` and `fraudTrain.csv`. Update the paths accordingly.

In `Data/intrinsic`,
- `Feature_Extraction.ipynb` generates `feat_train.csv` and `feat_test1m.csv`
- `FraudRate_and_Distance.ipynb` generates `distance_train.csv` and `distance_train_1m.csv`.

In `Data/network`,
- `APATE network optimized.ipynb` generates `apate_features_train.csv` and `apate_features_test1m.csv`
- `Bipartite Centrality (Birank) Analysis.ipynb` generates `tx_birank_priors_train.csv`, `tx_birank_priors_test1m.csv`, `card_birank.csv` and `merchant_birank.csv`
- Reading `card_birank.csv` and `merchant_birank.csv` in `APATE network optimized.ipynb` gives 'apate_birank_features_train.csv' and 'apate_birank_features_test1m.csv'.

Using `merge/merge.ipynb` can generate a consolidated features dataframe csv.

## 1. How to Replicate the Bipartite Neural Network

This script trains a bipartite neural network to detect anomalous transactions. To prevent data leakage, the model is trained strictly on temporal and magnitude-ratio features, while the Z-score is isolated as the ground-truth target label.

### Prerequisites
To run this code, you will need the following installed on your machine:
* **MATLAB** (R2021a or newer recommended)
* **Deep Learning Toolbox** (Required for the `patternnet` and `train` functions)

### Data Requirements
You must have the processed dataset file named `Merged_Test_is_fraud.csv` located in your current MATLAB working directory. 

The script expects this CSV to contain the following specific columns:
* `Hour_Of_Day`: The extracted hour of the transaction (0-23).
* `Magnitude_Ratio`: The ratio of the transaction amount to the historical median.
* `ZScore_Magnitude`: The calculated Z-score for the transaction (used strictly to define the `> 2.0` anomaly threshold for the target variable).

### Running the Model
1. Open MATLAB and ensure your "Current Folder" is set to the directory containing both the script (`Bipartite_Network.m`) and your `Merged_Test_is_fraud.csv` data file.
2. Open the script in the MATLAB editor.
3. Click **Run** (or type the script name in the Command Window and hit Enter).

### Expected Output
The script will silently train the 10-neuron hidden layer using scaled conjugate gradient backpropagation. Once finished, it applies a top 1% risk threshold and prints the evaluation metrics directly to the MATLAB Command Window in a format ready for academic tables:

```text
Loading data and prepping features...
Training the Neural Network (this might take a few seconds)...

--- NEW BIPARTITE NETWORK STATS (Top 1%) ---
ROC-AUC:   .xxxx
PR-AUC:    .xxxx
Precision: .xxxx
Recall:    .xxxx
F1 Score:  .xxxx
```


## 2. How to Replicate the XGBoost Model
This script trains an XGBoost machine learning model to identify fraudulent transactions.

### Prerequisites
To run this code, you will need the following installed in your environment:
**xgboost, sklearn, numpy, matplotlib, pandas**

### Data Requirements
 You must have the processed dataset files named 'Merged_train_is_fraud.csv' and 'Merged_Test_is_fraud.csv' located in your current working directory.

### Running the Model
1. Open your IDE and ensure the folder you have open contains both the script (`XGBoost.ipynb`) and your 'Merged_train_is_fraud.csv' and 'Merged_Test_is_fraud.csv' data files.
2. For the most optimized hyperparameters, ensure the parameters in the cell labelled # p3.2 are the only parameters being used.
3. Run all cells

### Expected Output
The script will train the model using an ensemble of decision trees. Once finished, it will output the results based on the metrics: ROC_AUC, PR_AUC, then show the Confusion Matrix produced and accuracy of the model, and our classification of precision, recall, and f-1 score rounded to 4 decimal places:

```text
ROC_AUC x.xxxx
PR_AUC x.xxxx

Confusion Matrix 
 [[xxxxx   xx]
 [   xx   xxx]] 

Accuracy of the model is: xx.xxxx 

Classification 
               precision    recall  f1-score   support

           0     .xxxx    .xxxx    .xxxx     xxxxx
           1     .xxxx    .xxxx    .xxxx       xxx

    accuracy                         .xxxx     xxxxx
   macro avg     .xxxx    .xxxx    .xxxx     xxxxx
weighted avg     .xxxx    .xxxx    .xxxx     xxxxx
```


# 4. How to Replicate the K-Means Clustering Model
This script runs the provided data through K-Means Clustering to detect fraudulent transactions.

### Prerequisites
To run this code, you will need the following installed on your machine:
* **MATLAB** (R2021a or newer recommended)
* **Statistics and Machine Learning ToolBox**

### Data Requirements
 You must have the processed dataset files named 'Merged_train_is_fraud.csv' and 'Merged_Test_is_fraud.csv' located in any drive on your machine. 
 
 ### Running the Model
1. Open the Matlab IDE, where you will put either k means code of the two into the compiler
2. The next part depends on which you have selected. Follow accordingly
-If running k_means_trainset simply run the code and select the "Merged_train_is_fraud.csv" wherever it is located on your machine, it will prompt you to look for it in one of your drives.
-If running k_means_firstmonth_testset simply run the code and select the "Merged_test_is_fraud.csv" wherever it is located on your machine, it will prompt you to look for it in one of your drives.
4. The results will now pop up in the command window.

## Expected Outcome 
```
Cluster 1 size: xxxxxxxx
Cluster 2 size: xxxxx

Chosen mapping: Cluster 1 -> Legit (0), Cluster 2 -> Fraud (1)

===== Evaluation Against is_fraud =====
Accuracy : 0.xxxx
Precision: 0.xxxx
Recall   : 0.xxxx
F1 Score : 0.xxxx

Confusion Matrix (rows = true, cols = predicted)
           Pred 0     Pred 1
True 0        xxxxx      xxxxx
True 1          xxx         xx

Fraud rate in Cluster 1: 0.xxxx
Fraud rate in Cluster 2: 0.xxxx
```
# 4. How to Replicate the Random Forest Model

This notebook trains a Random Forest classifier to identify fraudulent credit card transactions using a combination of transaction-level, behavioral, geographic, and network-based features. It includes model training, baseline evaluation, ROC and PR analysis, feature importance extraction, and validation-based threshold selection.

## Prerequisites
To run this notebook, install the following Python packages in your environment:
**xgboost, scikit-learn, numpy, matplotlib, pandas**
If you are using Google Colab, the notebook also mounts Google Drive to access the dataset files.

### Data Requirements
 You must have the processed dataset files named 'Merged_train_is_fraud.csv' and 'Merged_Test_is_fraud.csv' located in your current working directory.

If you are running the notebook locally, update the file path in the notebook accordingly.

The notebook expects these CSV files to contain a target column:

- `is_fraud`: binary fraud label

It also expects a set of model features including transaction amount, geographic variables, temporal variables, behavioral variables, and network-derived variables. Examples from the notebook include:

- `amt`
- `lat`
- `long`
- `city_pop`
- `category`
- `unix_time`
- `merch_lat`
- `merch_long`
- `log_amt`
- `merchant_txn_count_past`
- `merchant_unique_cards_past`
- `card_txn_count_1h`
- `amt_to_card_median`
- `amt_to_category_median`
- `hour_of_day`
- `day_of_week`
- `is_weekend`
- `dist_home_merchant_km`
- `recency_sec`
- `dist_prev_merchant_km`
- `implied_speed_kmh`
- `State Fraud Rate`
- `Avg Card Dist (km)`
- `Dist Ratio`

The notebook automatically separates numeric and categorical columns and applies preprocessing before model training.

## Running the Model
1. Open the notebook `RandomForest.ipynb` in Jupyter Notebook, JupyterLab, or Google Colab.
2. Make sure the dataset files `features_train.csv` and `features_test.csv` are accessible from the path used in the notebook.
3. Run all cells in order.

## Model Details
The notebook builds a preprocessing-and-model pipeline using:

- median imputation for numeric features
- most-frequent imputation for categorical features
- one-hot encoding for categorical variables
- a `RandomForestClassifier`

The baseline Random Forest model is initialized with settings including:

- `n_estimators=100`
- `random_state=42`
- `n_jobs=-1`
- `class_weight='balanced_subsample'`
- `min_samples_split=20`
- `min_samples_leaf=5`

## Expected Output
The notebook will:

1. Load training and test data
2. Train the Random Forest pipeline
3. Report baseline evaluation metrics on the test set
4. Plot ROC curves, including a zoomed low-FPR ROC view
5. Display top transformed feature importances
6. Perform validation-based threshold tuning using three strategies:
   - best F1 on validation
   - best precision subject to recall ≥ 0.80
   - best recall subject to FPR ≤ 0.1%
7. Refit on the full training data and evaluate all selected thresholds on the test set

Typical printed output includes metrics in a format similar to:

```text
=== Random Forest Baseline Results ===
ROC-AUC: x.xxxxxx
PR-AUC : x.xxxxxx

Confusion Matrix @ threshold=0.5
[[xxxxx   xx]
 [   xx  xxx]]

Classification Report
              precision    recall  f1-score   support

           0     .xxxx     .xxxx     .xxxx    xxxxx
           1     .xxxx     .xxxx     .xxxx      xxx

    accuracy                         .xxxx    xxxxx
   macro avg     .xxxx     .xxxx     .xxxx    xxxxx
weighted avg     .xxxx     .xxxx     .xxxx    xxxxx
```

The threshold-tuning section also produces a summary table comparing the different threshold-selection strategies on the test set.

# 4. How to Replicate the Isolation Forest Model

This notebook trains an Isolation Forest model to detect anomalous credit card transactions using a combination of transaction-level, behavioral, geographic, and network-based features. Because Isolation Forest is an unsupervised anomaly detection method, it is trained without fraud labels and instead identifies transactions that deviate from normal behavior. The notebook includes preprocessing, missing-value imputation, feature scaling, anomaly scoring, quantile-based thresholding, and evaluation using classification metrics and ROC-AUC.

## Prerequisites

To run this notebook, install the following Python packages in your environment: scikit-learn, numpy, matplotlib, and pandas.
If you are using Google Colab, the notebook may also mount Google Drive to access the dataset files.

### Data Requirements

You must have the processed dataset files named features_train.csv and features_test.csv located in your current working directory.
If you are running the notebook locally, update the file paths in the notebook accordingly.
The notebook expects these CSV files to contain a target column:
is_fraud: binary fraud label used only for evaluation
It also expects a set of model features including transaction amount, geographic variables, temporal variables, behavioral variables, and network-derived variables. (Examples listed above)
The notebook drops identifier columns such as trans_num, one-hot encodes categorical variables such as category, imputes missing values using median imputation, and standard-scales the features before fitting the model.

### Model Notes

The Isolation Forest model is trained on the feature matrix only, typically using legitimate transactions only from the training set so that the model learns normal transaction behavior. After fitting, the notebook computes anomaly scores on the test set and classifies the most anomalous transactions as fraud using a quantile-based threshold. Fraud labels are never used during training and are reserved only for post-hoc evaluation.


## Notes
- The notebook is currently configured for Google Colab through `drive.mount()`.
- If running outside Colab, remove or modify the Google Drive mounting cells.

