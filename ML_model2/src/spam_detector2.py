# Spam Detection using Autoencoder-Based Learning (Using Dataset 2)
# Created by: Leo Martinez III in Spring 2024

# Using the provided dataset, the machine learning program is capable of detecting whether or not an email should be considered spam.

# Dataset:https://www.kaggle.com/datasets/balaka18/email-spam-classification-dataset-csv
# Citation: Biswas, Balaka Kaggle Email Spam Classification Dataset CSV (2020)

"""Imports"""

# Handling Data Imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Evaluation Metrics Imports
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Preprocessing Imports
from sklearn.compose import ColumnTransformer
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder

# Import Neural Network Libraries
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import plot_model, to_categorical
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as pyplot
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

#%%

"""Load Dataset"""

# --- Loading Dataset ---
# Kaggle Email Spam Classification Dataset CSV, 3002 features, 5172 samples, Binary Classification (non-spam = 0, spam = 1)
dataset = pd.read_csv('emails.csv')

# Index is not needed
dataset.drop('Email No.', axis=1, inplace=True)

#%%
# Distribution of classes in dataset
dataset['Prediction'].value_counts()

#%%

"""Preprocessing Data"""

# Only one categorical feature 'class', which is already label encoded, so no categorical preprocessing needed.

# Define the columns to scale
columns_to_scale = dataset.columns[:-1]  # Exclude the last column

# Scale numerical columns using MinMax
scaler = MinMaxScaler()
for column in columns_to_scale:
    dataset[column] = scaler.fit_transform(dataset[[column]])

# Assign all features except the last feature ('class') to X and make the Target Variable Y equal to 'class'
X = dataset.iloc[:, :-1].values
y = dataset['Prediction'].values

#%%

"""Train-Test Data Split"""

# Split the large dataset into by 60% Training and 40% Testing (can be adjusted by changing the value for test_size)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

#%%

"""Deep Autoencoder Architecture + Training"""

# Autoencoder consisting of 5 hidden layers and 1 input/output layer
# Used for feature learning of the dataset (can be repurposed for other datasets)

# Number of features in the input data (3002 - 1 ('Present') - 1 ('Email No.') = 3000 total features)
n_inputs = 3000

# Define the input layer
visible = Input(shape=(n_inputs,))

# Hidden layers with increased capacity and regularization
e = Dense(1500, activation='relu')(visible) # Hidden Layer 1
#e = Dropout(0.1)(e)
#e = BatchNormalization()(e)

e = Dense(1000, activation='selu')(e) # Hidden Layer 2

bottleneck = Dense(500, activation='relu')(e) # Hidden Layer 3 (Latent Space)
#bottleneck = Dropout(0.1)(e)
#bottlneck = BatchNormalization()(e)

d = Dense(1000, activation='selu')(bottleneck) # Hidden Layer 4

d = Dense(1500, activation='relu')(d) # Hidden Layer 5
#d = Dropout(0.1)(e)
#d = BatchNormalization()(e)

# Output Layer
output = Dense(3000, activation='linear')(d)

# Define the model
model = Model(inputs=visible, outputs=output)

# Compile the model with ReduceLROnPlateau callback
model.compile(optimizer='adam', loss='mse')

# Set up early stopping and learning rate reduction callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=0.0001)

# Fit the model with augmented data
history = model.fit(X_train, X_train, epochs=4, batch_size=32, verbose=2,
                    validation_split=0.1, callbacks=[early_stopping, reduce_lr])
# Define a deep network model
neural_network = Model(inputs=visible, outputs=output)
plot_model(neural_network, 'autoencoder.png', show_shapes=True)

# Save the neural_network model in Keras format
neural_network.save('autoencoder_model.keras')

#%%

# Preprocessing & Autoencoder have been applied prior to training
# Random Forest ML Ensemble Algorithm for Binary Classification

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, matthews_corrcoef
from sklearn.model_selection import cross_val_score
from keras.models import load_model

# Load the model from file
encoder = load_model('autoencoder_model.keras')

# Encode the training and testing data
X_train_encoded = encoder.predict(X_train)
X_test_encoded = encoder.predict(X_test)

# Create a RandomForestClassifier
clf = RandomForestClassifier(n_estimators=50, random_state=42)

# Perform k-fold cross-validation (k = 10)
cv_scores = cross_val_score(clf, X_train_encoded, y_train, cv=2)

# Train the model on the entire training set
clf.fit(X_train_encoded, y_train)

# Evaluate the model on the test set
y_pred = clf.predict(X_test_encoded)

# Print the cross-validation scores
print("Cross-Validation Scores:")
print(cv_scores)
print("\nMean CV Score:", cv_scores.mean(),"\n")

# Print classification report and confusion matrix on the test set
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred), "\n")

# Calculate AUC
y_prob = clf.predict_proba(X_test)[:, 1]  # Get the probability of the malware class
auc_score = roc_auc_score(y_test, y_prob)
print("AUC Score:", auc_score)

# Calculate MCC
mcc_score = matthews_corrcoef(y_test, y_pred)
print("MCC Score:", mcc_score)