import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

#i apply Isolation Forest to detect transactions that deviate from normal spatial, temporal, and graph-based transaction behavior

train = pd.read_csv("/Users/preeti.kar/Downloads/features_train.csv")
test = pd.read_csv("/Users/preeti.kar/Downloads/features_test.csv")

drop_cols = ["trans_num", "is_fraud"] #ids have no relation, and is_fraud is the target variable we want to predict (unsupervised) 

X_train = train.drop(columns=drop_cols)
X_test = test.drop(columns=drop_cols)

y_test = test["is_fraud"] #extract the target variable for evaluation

X_train = pd.get_dummies(X_train, columns=["category"], drop_first=True) #convert categories to binary columns, ex. "category_groceruy" = 1 if the transaction is in the grocery category, 0 otherwise
X_test = pd.get_dummies(X_test, columns=["category"], drop_first=True)

X_test = X_test.reindex(columns=X_train.columns, fill_value=0) #ensure that the test set has the same columns as the training set, filling any missing columns with 0 (ex if a category is present in the training set but not in the test set)
imputer = SimpleImputer(strategy="median") #fill missing objects with the median of each col

X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = StandardScaler() #scaling object, calculates mean and sd from training data

X_train_scaled = scaler.fit_transform(X_train) #fit_transform calculates the mean and sd from X_train and scales X_train, ensuring model is not exposed to information from the test set during training
X_test_scaled = scaler.transform(X_test) 


iso = IsolationForest(
    n_estimators=300, #number of trees in the forest, more trees -> improve performance -> increase computation time
    contamination=0.02, #expected proportion of anomolies 
    random_state=42, 
    n_jobs=-1
)

X_train_legit = X_train_scaled[train["is_fraud"] == 0]
iso.fit(X_train_legit)

scores = iso.decision_function(X_test_scaled) #anomaly score for the test set, lower score -> more likely to be anomaly
threshold = pd.Series(scores).quantile(0.035) #set threshold at the 1st percentile of the scores, we flag the top 1% most anomalous transactions as fraud
pred_fraud = (scores <= threshold).astype(int)


print(classification_report(y_test, pred_fraud))

auc = roc_auc_score(y_test, -scores)
print("AUC:", auc)

test["anomaly_score"] = -scores
test["pred_fraud"] = pred_fraud
