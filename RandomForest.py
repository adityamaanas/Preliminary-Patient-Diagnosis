import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv("C:/Users/vishw/OneDrive/Desktop/dataset.csv")
df_padded = data.fillna(0)
print(df_padded)
df_encoded = pd.get_dummies(df_padded)

# Split the data into input features (X) and target variable (y)
X = df_encoded.drop(df_encoded.columns[1], axis=1)  # Remove the target variable column if present
print(X)
y = df_encoded[df_encoded.columns[1]]  # Set the target variable column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=3, max_depth=2, min_samples_leaf=50)
bagging_classifier = BaggingClassifier(base_estimator=model, n_estimators=10, random_state=42)
bagging_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = bagging_classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
#print("Mean Squared Error: ", mean_squared_error(y_test['x'], y_pred))
#print("R2 Score: ", r2_score(y_test['x'], y_pred))