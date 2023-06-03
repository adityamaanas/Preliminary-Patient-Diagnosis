# Import the necessary libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import wordembeddingsgensim as we
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(we.data['Embeddings'].values, we.data['Disease'].values, test_size=0.25)

#print(X_train.shape)
#print(y_train.shape)

# Pad the sequences to the same length
max_length = max(len(seq) for seq in X_train)
X_train = pad_sequences(X_train, maxlen=max_length, padding='post')
X_test = pad_sequences(X_test, maxlen=max_length, padding='post')

# Flatten the sequences
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Train the model
model = RandomForestClassifier(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)

# Evaluate the model
score = model.score(X_test, y_test)
print('Score:', score)

# Tune the hyperparameters
parameters = {'n_estimators': [100, 200, 300], 'max_depth': [5, 10, 15]}
clf = GridSearchCV(model, parameters, scoring='accuracy')
clf.fit(X_train, y_train)

# Print the best parameters
best_params = clf.best_params_
print('Best parameters:', best_params)

# Deploy the model
model = clf.best_estimator_
