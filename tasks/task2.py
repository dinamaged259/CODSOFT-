
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import pandas as pd
df = pd.read_csv('task2.csv')

df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['dob'] = pd.to_datetime(df['dob'], errors='coerce')  # Handle potential date format issues

df['cc_num'] = df['cc_num'].astype(str)

numeric_columns = ['amt', 'zip', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')
  

X = df.drop(columns=['is_fraud'])
y = df['is_fraud']

categorical_columns = ['gender', 'merchant', 'category']
X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

logistic_reg_model = LogisticRegression()
decision_tree_model = DecisionTreeClassifier()
random_forest_model = RandomForestClassifier()

logistic_reg_model.fit(X_train, y_train)
decision_tree_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)

def evaluate_model(model, X, y):
    y_pred = model.predict(X)

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc_auc = roc_auc_score(y, y_pred)

    return accuracy, precision, recall, f1, roc_auc

logistic_reg_accuracy, logistic_reg_precision, logistic_reg_recall, logistic_reg_f1, logistic_reg_roc_auc = evaluate_model(logistic_reg_model, X_test, y_test)
decision_tree_accuracy, decision_tree_precision, decision_tree_recall, decision_tree_f1, decision_tree_roc_auc = evaluate_model(decision_tree_model, X_test, y_test)
random_forest_accuracy, random_forest_precision, random_forest_recall, random_forest_f1, random_forest_roc_auc = evaluate_model(random_forest_model, X_test, y_test)

print("Logistic Regression:")
print(f"Accuracy: {logistic_reg_accuracy:.2f}")
print(f"Precision: {logistic_reg_precision:.2f}")
print(f"Recall: {logistic_reg_recall:.2f}")
print(f"F1 Score: {logistic_reg_f1:.2f}")
print(f"ROC AUC: {logistic_reg_roc_auc:.2f}")
print("\n")

print("Decision Tree:")
print(f"Accuracy: {decision_tree_accuracy:.2f}")
print(f"Precision: {decision_tree_precision:.2f}")
print(f"Recall: {decision_tree_recall:.2f}")
print(f"F1 Score: {decision_tree_f1:.2f}")
print(f"ROC AUC: {decision_tree_roc_auc:.2f}")
print("\n")

print("Random Forest:")
print(f"Accuracy: {random_forest_accuracy:.2f}")
print(f"Precision: {random_forest_precision:.2f}")
print(f"Recall: {random_forest_recall:.2f}")
print(f"F1 Score: {random_forest_f1:.2f}")
print(f"ROC AUC: {random_forest_roc_auc:.2f}")
print("\n")

