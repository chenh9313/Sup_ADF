#!/usr/bin/env python
# coding: utf-8

# ###### Neural Network 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

# Read data
data = pd.read_table('balanced_ImmuneGene.txt', sep=",")
data.shape

# Check label
data.Label.unique()

# Get features
X = data[data.columns.tolist()[3:]]#get all the feature name
y = data['Label']

#drop highly cor feature
# Create correlation matrix
corr_matrix = X.corr().abs()

# Select upper triangle of correlation matrix
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find features with correlation greater than 0.95
to_drop = [column for column in upper.columns if any(upper[column] > 0.5)]

# Drop features 
X.drop(to_drop, axis=1, inplace=True)

#Split and Scale data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standard Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Must apply the scaling to the test set that we computed for the training set
X_test_scaled = scaler.transform(X_test)


# ###### find the best value for NeuralNetwork, the follw resluts comes from the follwing code that I ran on HPCC
# find teh best value
print("find best value for NeuralNetwork")
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

param_grid = {
    'hidden_layer_sizes': [(1,), (2,), (3,)],
    'activation': ['relu', 'tanh'],
    'learning_rate_init': [0.001, 0.01, 0.1],
    # Add other hyperparameters...
}

model = MLPClassifier()
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)


# The best value for NeuralNetwork got from the above code
Best Hyperparameters: {'activation': 'tanh', 'hidden_layer_sizes': (3,), 'learning_rate_init': 0.01}

# Define the model
model = Sequential([
    Dense(64, activation='tanh', input_shape=(X.shape[1],)),  # Input layer with 64 neurons and ReLU activation
    Dense(32, activation='tanh'),  # Hidden layer with 32 neurons and ReLU activation
    Dense(1, activation='sigmoid')  # Output layer with 1 neuron (binary classification) and sigmoid activation
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model #what shoudl be my batch size
model.fit(X_train_scaled, y_train, epochs=496, batch_size=256, validation_data=(X_test_scaled, y_test))

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
print(f'Test Accuracy: {test_accuracy}')

# Make predictions
predictions = model.predict(X_test_scaled)
predictions

# Evaluate the model
y_pred_proba = model.predict(X_test_scaled)
y_pred = (y_pred_proba > 0.5).astype(int)  # Threshold predictions at 0.5

# Convert probabilities to binary predictions

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
plt.title('Receiver Operating Characteristic (ROC) Curve - Neural Networks')
plt.legend(loc='lower right')

# Save ROC.png
plt.savefig("Res-ROC_NeuralNetworks.pdf")
plt.show()

# ###### corss_val
# Define a function to create your neural network model
def create_nn_model():
    model = Sequential()
    model.add(Dense(64, activation='tanh', input_shape=(X.shape[1],)))
    model.add(Dense(32, activation='tanh'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create a KerasClassifier wrapper for your model
nn_classifier = KerasClassifier(build_fn=create_nn_model, epochs=496, batch_size=256, verbose=0)

# Set up Stratified K-fold cross-validation (for classification tasks)
num_folds = 500  # You can adjust the number of folds as needed
skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=0)

# Perform cross-validation and get accuracy scores
cv_scores = cross_val_score(nn_classifier, X_train_scaled, y_train, cv=skf, scoring='accuracy')

# Display the cross-validation scores
print("Cross-validation Scores:", cv_scores)

# Calculate and display the mean and standard deviation of the scores
print("Mean Accuracy:", np.mean(cv_scores))
print("Standard Deviation:", np.std(cv_scores))


# ###### Predict real data
# Real Date
real_data = pd.read_table('Unpredicted_real_data.txt', sep=",")

X_new = real_data[real_data.columns.tolist()[2:]]#get all the feature name
X_new = X_new.drop(columns=to_drop)

# Standard Scaling
scaler = StandardScaler()
X_new_standard = scaler.fit_transform(X_new)

NN_prediction = model.predict(X_new_standard)
NN_y_pred = (NN_prediction > 0.5).astype(int)

# Add a new column 'Predicted' to the original DataFrame
real_data['Predicted_Label'] = NN_y_pred
# Create a new DataFrame with selected columns
new_dataframe = real_data[['Predicted_Label', 'Name', 'Isoform']]
new_dataframe
# Print or save the new DataFrame
#print(new_dataframe)

# Save the DataFrame as a CSV file
new_dataframe.to_csv('Res-NeuralNetwork-predicted_output.csv', index=False)
