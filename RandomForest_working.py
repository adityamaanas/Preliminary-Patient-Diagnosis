import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings

# Filter out the warning
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Load the dataset
data = pd.read_csv("D:/Files/repos/openlab/dataset.csv")

# Split the data into input features (X) and target variable (y)
X = data.drop('Disease', axis=1)
y = data['Disease']

# Preprocess the data
X_padded = X.fillna(0)
X_padded = X_padded.astype(str).apply(lambda x: x.str.lower())
#print(type(X_padded.iloc[0,0]))
# delete the spaces in the symptom names
X_padded = X_padded.apply(lambda x: x.str.replace(" ", ""))

#label_encoder = LabelEncoder()

# Fit and transform the categorical data to numerical labels
#X_encoded = X.apply(LabelEncoder().fit_transform)

# melt the dataframe to convert it from wide to long format
X_melted = pd.melt(X_padded)

# write a for loop to map every element in X_melted['value'] to a unique integer
symptom_dict = {}
val = 1
for i in X_melted['value'].unique().tolist():
    # delete spaces in the symptom names
    i = i.replace(" ", "")
    symptom_dict[i] = val
    val += 1

# Encode the categorical data to numerical labels using the dictionary
X_encoded = X_padded.replace(symptom_dict)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Create and train the Random Forest Classifier
bagging_classifier = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=110, random_state=42)
bagging_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = bagging_classifier.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Take a list of symptoms as input
input_symptoms = ['fatigue', 'weight_loss', 'lethargy', 'irregular_sugar_level', 'blurred_and_distorted_vision', 'obesity', 'excessive_hunger', 'increased_appetite', 'polyuria']

# convert the input symptoms to strings and convert them to lower case
input_symptoms = [str(item).lower() for item in input_symptoms]

# replace the input symptoms with corresponding integers from the input using the same dictionary
input_encoded = []
for item in input_symptoms:
    input_encoded.append(symptom_dict[item])

# pad the input_encoded list with 0s to match the length of the input features
input_encoded = np.pad(input_encoded, (0, len(X_encoded.columns) - len(input_encoded)), 'constant')

# Predict the corresponding disease for the input symptoms
prediction = bagging_classifier.predict([input_encoded])

print("Prediction:", prediction[0])