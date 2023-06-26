import pandas as pd
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("D:/Files/repos/openlab/dataset.csv")

# Split the data into input features (X) and target variable (y)
X = data.drop('Disease', axis=1)
y = data['Disease']

# Preprocess the data
X_padded = X.fillna(0)
X_encoded = pd.get_dummies(X_padded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

#print("Training set: ", X_train)

# Create and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=3, max_depth=2, min_samples_leaf=50)
bagging_classifier = BaggingClassifier(base_estimator=model, n_estimators=10, random_state=42)
bagging_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = bagging_classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Take a list of symptoms as input
input_symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions']
input_encoded = pd.get_dummies(pd.DataFrame(input_symptoms)).reindex(columns=X.columns, fill_value=0)

#print("Input:", input_encoded)

# Predict the corresponding disease for the input symptoms
prediction = bagging_classifier.predict(input_encoded)
print("Prediction:", prediction)
