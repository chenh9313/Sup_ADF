#!/usr/bin/env python
# coding: utf-8

# ###### RandomForest

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification

# Read date
data = pd.read_table('balanced_ImmuneGene.txt', sep=",")
type(data)
data.Label.unique()

# Get all the features 
X = data[data.columns.tolist()[3:]]
y = data['Label']
X

# ###### feature correlation
# Compute the correlation matrix
correlation_matrix = X.corr()

# Set up the matplotlib figure
plt.figure(figsize=(15, 8))

# Create a heatmap using seaborn
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)

#save png
plt.savefig("Res-FeatureCorrelation.png")

# Show the plot
plt.show()


# ###### Drop highly related features
# Create correlation matrix
corr_matrix = X.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.5)]

# Drop features 
X.drop(to_drop, axis=1, inplace=True)
to_drop

# ###### Droped feature cor
# Compute the correlation matrix
correlation_matrix = X.corr()

# Set up the matplotlib figure
plt.figure(figsize=(15, 8))

# Create a heatmap using seaborn
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=.5)

#save png
plt.savefig("Res-Dropped_featureCorrelation.png")

# Show the plot
plt.show()

#result from the above code
Best Hyperparameters: {'bootstrap': False, 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 500}
Accuracy: 0.9917202223711706

# Split data as 80% for training, 20% for test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standard Scaling
scaler = StandardScaler()
X_standard = scaler.fit_transform(X_train)

# Must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(X_test)

# Instantiate the Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=500, random_state=0, max_features = "sqrt",
                                      min_samples_leaf = 2, min_samples_split = 2, max_depth = 12,
                                      bootstrap = 'TRUE')

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")


# ###### ROC
# Compute ROC curve and area under the curve (AUC)
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve - random forest')
plt.legend(loc='lower right')

#save ROC.png
plt.savefig("Res-ROC_RandomForest.pdf")

plt.show()

#### cross_val
# Generate some example data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize the Random Forest classifier
#rf_classifier = RandomForestClassifier(n_estimators=10, random_state=0)
rf_classifier = RandomForestClassifier(n_estimators=500, random_state=0, max_features = "sqrt",
                                      min_samples_leaf = 2, min_samples_split = 2, max_depth = 12,
                                      bootstrap = 'TRUE')


# Set up K-fold cross-validation
num_folds = 500  # You can adjust the number of folds as needed
kf = KFold(n_splits=num_folds, shuffle=True, random_state=0)

# Perform cross-validation and get accuracy scores
cv_scores = cross_val_score(rf_classifier, X, y, cv=kf, scoring='accuracy')

# Display the cross-validation scores
print("Cross-validation Scores:", cv_scores)

# Calculate and display the mean and standard deviation of the scores
print("Mean Accuracy:", np.mean(cv_scores))
print("Standard Deviation:", np.std(cv_scores))


# Generate some example data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize the Random Forest classifier
#rf_classifier = RandomForestClassifier(n_estimators=10, random_state=0)
rf_classifier = RandomForestClassifier(n_estimators=500, random_state=0, max_features = "sqrt",
                                      min_samples_leaf = 2, min_samples_split = 2, max_depth = 12,
                                      bootstrap = 'TRUE')

# Perform cross-validation
cv_scores = cross_val_score(rf_classifier, X, y, cv=500)  # 5-fold cross-validation

# Print the cross-validation scores
print("Cross-validation scores:", cv_scores)

# Calculate and print the mean and standard deviation of the cross-validation scores
print("Mean CV accuracy:", cv_scores.mean())
print("Standard deviation of CV accuracy:", cv_scores.std())

# Display the cross-validation scores
print("Cross-validation Scores:", cv_scores)

# Calculate and display the mean and standard deviation of the scores
print("Mean Accuracy:", np.mean(cv_scores))
print("Standard Deviation:", np.std(cv_scores))

#A test gene with expresison value
real_data = pd.read_table('Unpredicted_real_data.txt', sep=",")

X_new = real_data[real_data.columns.tolist()[2:]]#get all the feature name
X_new = X_new.drop(columns=to_drop)

# Standard Scaling
scaler = StandardScaler()
X_new_scale = scaler.fit_transform(X_new)

knn_prediction = rf_classifier.predict(X_new_scale)
X_new

new_dataframe

# Add a new column 'Predicted' to the original DataFrame
real_data['Predicted_Label'] = knn_prediction
# Create a new DataFrame with selected columns
new_dataframe = real_data[['Predicted_Label', 'Name', 'Isoform']]
new_dataframe

# Save DataFrame as a CSV file
new_dataframe.to_csv('Res-RandomForest-predicted_output.csv', index=False)

