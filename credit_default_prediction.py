# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 11:30:57 2024

@author: DIVYANSHU
"""

import pandas as pd
# Importing the dataset
dataset = pd.read_csv('/Users/DIVYANSHU/Desktop/ML course/credit_default_pred/GiveMeSomeCredit-training.csv')



# Impute missing values in numerical columns with the mean
dataset.fillna(dataset.mean(), inplace=True)
missing_values = dataset.isnull().sum()
print(missing_values)


# Separate target variable and features
target = dataset.iloc[:, 1]  # Assuming the target is the first column
features = dataset.iloc[:, 2:]  # All other columns are features  


# Split data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

# Check if there are still non-numeric columns
print(features.dtypes)





import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Model evaluation and preprocessing tools

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV



# Set style for plots
sns.set(style='whitegrid')


############  DECISION TREE CLASSIFIER  ################
# Initialize and train the model
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Predict and evaluate
y_pred = decision_tree.predict(X_val)
print(f"Decision Tree Accuracy: {accuracy_score(y_val, y_pred)}")


# Define parameter grid
param_grid_dt = {
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 4, 8],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Grid Search
grid_search_dt = GridSearchCV(estimator=DecisionTreeClassifier(random_state=42), param_grid=param_grid_dt, cv=5, scoring='roc_auc')
grid_search_dt.fit(X_train, y_train)

print(f"Best Parameters for Decision Tree: {grid_search_dt.best_params_}")

# Train best model
best_dt = grid_search_dt.best_estimator_
y_pred = best_dt.predict(X_val)
print(f"Optimized Decision Tree Accuracy: {accuracy_score(y_val, y_pred)}")



#####################   RANDOM FOREST  #########################33
# Initialize and train the model
random_forest = RandomForestClassifier(random_state=42)
random_forest.fit(X_train, y_train)

# Predict and evaluate
y_pred = random_forest.predict(X_val)
print(f"Random Forest Accuracy: {accuracy_score(y_val, y_pred)}")


param_grid_rf = {
    'n_estimators': [50, 100, 200],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
}


random_search_rf = RandomizedSearchCV(estimator=random_forest,
                                       param_distributions=param_grid_rf,
                                       n_iter=10,  # Number of parameter settings to try
                                       cv=5,
                                       scoring='roc_auc',
                                       n_jobs=-1,  # Use all available cores
                                       random_state=42)

# Fit the model
random_search_rf.fit(X_train, y_train)

# Get the best estimator
best_rf = random_search_rf.best_estimator_

# Print best parameters
print(f"Best Parameters for Random Forest: {random_search_rf.best_params_}")



######################## XG BOOST  ###############################3
# Initialize and train the model
xgboost = XGBClassifier(random_state=42, eval_metric='logloss')
xgboost.fit(X_train, y_train)

# Predict and evaluate
y_pred = xgboost.predict(X_val)
print(f"XGBoost Accuracy: {accuracy_score(y_val, y_pred)}")


# Define parameter grid for XGBoost
param_dist_xgb = {
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2],    
    'lambda': [0, 0.1, 1],
    'alpha': [0, 0.1, 1]
}

# Initialize RandomizedSearchCV
random_search_xgb = RandomizedSearchCV(estimator=xgboost,
                                        param_distributions=param_dist_xgb,
                                        n_iter=10,  # Number of parameter settings to try
                                        cv=5,
                                        scoring='roc_auc',
                                        n_jobs=-1,  # Use all available cores
                                        random_state=42)

# Fit the model
random_search_xgb.fit(X_train, y_train)

# Get the best estimator
best_xgb = random_search_xgb.best_estimator_

# Print best parameters
print(f"Best Parameters for XGBoost: {random_search_xgb.best_params_}")

# Evaluate on validation data
y_pred_xgb = best_xgb.predict(X_val)
y_prob_xgb = best_xgb.predict_proba(X_val)[:, 1]
print(f"Optimized XGBoost Accuracy: {accuracy_score(y_val, y_pred_xgb)}")








# Initialize models
models = {
    'Decision Tree': best_dt,
    'Random Forest': best_rf,
    'XGBoost': best_xgb
}




import matplotlib.pyplot as plt
# Dictionary to store results
model_performance = {}

plt.figure(figsize=(10, 8))

for model_name, model in models.items():
    print(f"Training {model_name}...")
    model.fit(X_train, y_train)
    
    # Predict on validation data
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]  # For ROC and AUC
    
    # Compute metrics
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)
    
    # Store results
    model_performance[model_name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc
    }
    
    # Plot ROC curve
    fpr, tpr, _ = roc_curve(y_val, y_prob)
    plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.2f})")

# Add diagonal line for no discrimination
plt.plot([0, 1], [0, 1], 'k--')

# Add labels and title
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')

# Add legend to identify each model
plt.legend(loc='best')
# Show plot
plt.show()



# Print model performance
for model_name, metrics in model_performance.items():
    print(f"{model_name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.2f}")
        
        
        
        
# Print model performance comparison
performance_df = pd.DataFrame(model_performance).T
print("\nModel Performance Comparison:\n", performance_df)

# Plot confusion matrices for each model
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for i, (model_name, model) in enumerate(models.items()):
    sns.heatmap(confusion_matrix(y_val, model.predict(X_val)), annot=True, fmt='d', ax=axes[i])
    axes[i].set_title(f'Confusion Matrix: {model_name}')
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')

plt.tight_layout()
plt.show()

import joblib

joblib.dump(best_rf, 'best_rf.pkl')







