# Import the necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Load the data
data = pd.read_csv('englishtraining.csv')
#print(data.head())

# Clean the data
data = data.dropna()
data = data.apply(lambda x: x.str.lower() if x.dtype == "object" else x)  # Normalize string columns to lowercase

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, data['Disease'], test_size=0.25)

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
