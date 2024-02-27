#!/usr/bin/env python
# coding: utf-8

# ###### Decision Tree

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from adspy_shared_utilities import plot_feature_importances
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

# Read data
data = pd.read_table('balanced_ImmuneGene.txt', sep=",")
type(data)

# Check data label
data.Label.unique()

# Get all the features name
X = data[data.columns.tolist()[3:]]
y = data['Label']

# ###### drop highly cor feature
# Create correlation matrix
corr_matrix = X.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.5)]

# Drop features 
X.drop(to_drop, axis=1, inplace=True)

# ###### find the best value for hyperparameters
# spleit data as 0.8/0.2 fro train and test data
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the parameter grid to search
param_grid = {
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# Create Decision Tree classifier
dt_classifier = DecisionTreeClassifier(random_state=0)

# Create GridSearchCV object
grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, 
                           cv=5, scoring='accuracy', n_jobs=-1)

# Perform grid search on the training data
grid_search.fit(X_train, y_train)

# Print the best hyperparameters found
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Get the best model from the grid search
best_dt_model = grid_search.best_estimator_

# Make predictions on the test set using the best model
y_pred = best_dt_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

feature_names = list(X.columns)
feature_names

# Assume dt_model is your trained DecisionTreeClassifier
plt.figure(figsize=(15, 10))
plot_tree(best_dt_model, filled=True, feature_names = feature_names)
plt.savefig("Result-DecisionTree.pdf")
plt.show()


# ###### add best best value for hyperparameters fro Deciosin Tress
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Instantiate the Decision Tree classifier
#dt_classifier = DecisionTreeClassifier(max_depth = 4, min_samples_leaf = 8,random_state = 0).fit(X_train, y_train)
dt_classifier = DecisionTreeClassifier(random_state=0,
                                       max_depth=12,
                                       max_features = 'sqrt',
                                       min_samples_leaf = 2,
                                       min_samples_split = 2)

# Train the model
dt_classifier.fit(X_train, y_train)

# Make predictions
y_pred = dt_classifier.predict(X_test)

# Evaluate using scikit-learn metrics
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
plt.title('Receiver Operating Characteristic (ROC) Curve - decision tree')
plt.legend(loc='lower right')

#save ROC.png
plt.savefig("Res-ROC_DecisionTree.pdf")

plt.show()


# ###### Cross-validation is used to assess how well a Decision Tree model

# In[17]:


from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold

#dt_classifier = DecisionTreeClassifier(max_depth=None)  # You can set other hyperparameters as needed
dt_classifier = DecisionTreeClassifier(random_state=0,
                                       max_depth=12,
                                       max_features = 'sqrt',
                                       min_samples_leaf = 2,
                                       min_samples_split = 2)
num_folds = 500

# Create a KFold cross-validation object
kf = KFold(n_splits=num_folds, shuffle=True, random_state=0)

# Use cross_val_score to evaluate the Decision Tree model
cross_val_scores = cross_val_score(dt_classifier, X_train, y_train, cv=kf, scoring='accuracy')

# Print the cross-validation scores
print("Cross-Validation Scores:", cross_val_scores)


# In[18]:


average_accuracy = cross_val_scores.mean()
print("Average Accuracy:", average_accuracy)
std_accuracy = cross_val_scores.std()
print("std Accuracy:", std_accuracy)


# In[ ]:





# In[115]:


# Split the data into training and testing sets. 80% is training, 20% is test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Instantiate the Decision Tree classifier
#dt_classifier = DecisionTreeClassifier(max_depth = 4, min_samples_leaf = 8,random_state = 0).fit(X_train, y_train)
dt_classifier = DecisionTreeClassifier(random_state=0,
                                       max_depth=12,
                                       max_features = 'sqrt',
                                       min_samples_leaf = 2,
                                       min_samples_split = 2)

# Train the model
dt_classifier.fit(X_train, y_train)

# Make predictions
y_pred = dt_classifier.predict(X_test)

# Evaluate using scikit-learn metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

feature_name = X.columns.tolist()[0:]

plt.figure(figsize=(10,4), dpi=80)
plot_feature_importances(dt_classifier, feature_name)
plt.show()
plt.savefig('Decision_tree_feature_importances.png')


# ###### Predict real data
# Read real data
real_data = pd.read_table('Unpredicted_real_data.txt', sep=",")

X_new = real_data[real_data.columns.tolist()[2:]]#get all the feature name
X_new = X_new.drop(columns=to_drop)

# Standard Scaling
scaler = StandardScaler()
X_new_standard = scaler.fit_transform(X_new)
dt_prediction = dt_classifier.predict(X_new_standard)

# Add a new column 'Predicted' to the original DataFrame
real_data['Predicted_Label'] = dt_prediction
# Create a new DataFrame with selected columns
new_dataframe = real_data[['Predicted_Label', 'Name', 'Isoform']]
new_dataframe

# Save the DataFrame as a CSV file
new_dataframe.to_csv('Res-DecisionTree-predicted_output.csv', index=False)

