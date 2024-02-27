#!/usr/bin/env python
# coding: utf-8

#K-Nearest Neighbors (KNN) 
#each feature has the same weight

get_ipython().run_line_magic('matplotlib', 'notebook')
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

data = pd.read_table('balanced_ImmuneGene.txt', sep=",")
type(data)
data.head()
data.Label.unique()

# Get all the features
X = data[data.columns.tolist()[3:]]
y = data['Label']

# Create correlation matrix
corr_matrix = X.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.5)]

# Drop features 
X.drop(to_drop, axis=1, inplace=True)
to_drop

# Split data as 80%/20% train/test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(X_test)


# Find best n_neighbors and num_folds
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a range of n_neighbors values to try
neighbors_range = list(range(1, 5))

# Perform grid search
accuracy_scores = []
for n_neighbors in neighbors_range:
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    scores = cross_val_score(knn_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    accuracy_scores.append(np.mean(scores))
    
# Find the best n_neighbors value
best_n_neighbors = neighbors_range[np.argmax(accuracy_scores)]
print("Best n_neighbors:", best_n_neighbors)

# Plot results
plt.plot(neighbors_range, accuracy_scores, marker='o')
plt.xlabel('n_neighbors')
plt.ylabel('Cross-validated Accuracy')
plt.title('Grid Search for Best n_neighbors')
plt.show()


#Cross-validation for KNN
# Instantiate the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=1)

# Choose the number of folds for cross-validation
num_folds = 500

# Use KFold for custom control over the cross-validation process
kf = KFold(n_splits=num_folds, shuffle=True, random_state=0)

# Perform cross-validation and print the results
accuracy_scores = cross_val_score(knn_classifier, X, y, cv=kf, scoring='accuracy')
precision_scores = cross_val_score(knn_classifier, X, y, cv=kf, scoring='precision')
recall_scores = cross_val_score(knn_classifier, X, y, cv=kf, scoring='recall')
f1_scores = cross_val_score(knn_classifier, X, y, cv=kf, scoring='f1')

# Print the mean and standard deviation of the cross-validated scores
print(f"Accuracy: {np.mean(accuracy_scores):.4f} (+/- {np.std(accuracy_scores):.4f})")
print(f"Precision: {np.mean(precision_scores):.4f} (+/- {np.std(precision_scores):.4f})")
print(f"Recall: {np.mean(recall_scores):.4f} (+/- {np.std(recall_scores):.4f})")
print(f"F1 Score: {np.mean(f1_scores):.4f} (+/- {np.std(f1_scores):.4f})")

# KNN model
# Instantiate the KNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=1)

# Train the model
knn_classifier.fit(X_train_scaled, y_train)

# Make predictions on the test set
y_pred = knn_classifier.predict(X_test_scaled)

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
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('Receiver Operating Characteristic (ROC) Curve - k-nearest neighbors')
plt.legend(loc='lower right')

#save ROC.png
plt.savefig("Res-ROC_KNN.pdf")

plt.show()

# Predict real data
#read real data
real_data = pd.read_table('Unpredicted_real_data.txt', sep=",")
real_data

X_new = real_data[real_data.columns.tolist()[2:]]#get all the feature name
X_new = X_new.drop(columns=to_drop)
X_new

# Standard Scaling
scaler = StandardScaler()
X_new_standard = scaler.fit_transform(X_new)

knn_prediction = knn_classifier.predict(X_new_standard)
real_data['Predicted_Label'] = knn_prediction

# Add a new column 'Predicted' to the original DataFrame
real_data['Predicted_Label'] = knn_prediction
# Create a new DataFrame with selected columns
new_dataframe = real_data[['Predicted_Label', 'Name', 'Isoform']]
new_dataframe

# Print or save the new DataFrame
#print(new_dataframe)

# Save the new DataFrame to a CSV file
new_dataframe.to_csv('Res-KNN-predicted_output.csv', index=False)




