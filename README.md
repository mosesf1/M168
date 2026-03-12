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
- `models/` – machine learning implementations
- `Data/` – processed feature datasets
  - `intrinsic/` - generate intrinsic feature
  - `network/` - generate network feature
  - `merge/` - sample code to merge the extracted features
- `report/` – LaTeX project report

## How to Replicate the Bipartite Neural Network

This script trains a bipartite neural network to detect anomalous transactions. To prevent data leakage, the model is trained strictly on temporal and magnitude-ratio features, while the Z-score is isolated as the ground-truth target label.

### Prerequisites
To run this code, you will need the following installed on your machine:
* **MATLAB** (R2021a or newer recommended)
* **Deep Learning Toolbox** (Required for the `patternnet` and `train` functions)

### Data Requirements
You must have the processed dataset file named `Final_Scored_Transactions.csv` located in your current MATLAB working directory. 

The script expects this CSV to contain the following specific columns:
* `Hour_Of_Day`: The extracted hour of the transaction (0-23).
* `Magnitude_Ratio`: The ratio of the transaction amount to the historical median.
* `ZScore_Magnitude`: The calculated Z-score for the transaction (used strictly to define the `> 2.0` anomaly threshold for the target variable).

### Running the Model
1. Open MATLAB and ensure your "Current Folder" is set to the directory containing both the script (`Bipartite_Network.m`) and your `Final_Scored_Transactions.csv` data file.
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


## How to Replicate the XGBoost Model
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

