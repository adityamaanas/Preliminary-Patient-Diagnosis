#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#region dataset import
df = pd.read_csv('C:/Users/adipi/OneDrive - Amrita vishwa vidyapeetham\
                 /Repos/openlab/englishtraining.csv')
df.head()
#endregion

#region preprocessing
    #dropping the disease code column
df.drop(df.columns[[0]], axis=1, inplace=True)
df.head()

    #creating a second dataframe
df2 = pd.read_csv('C:/Users/adipi/OneDrive - Amrita vishwa vidyapeetham\
                 /Repos/openlab/englishtraining.csv')
df2.drop(df2.columns[[0]], axis=1, inplace=True)
df2.head()

#def flatten(listOfLists):
#    "Flatten one level of nesting"
#    return chain.from_iterable(listOfLists)

#region label encoding
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
X = df2.Symptoms
encoded_X = label_encoder.fit_transform(X)
label_encoder_name_mapping = dict(zip(label_encoder.classes_,
                                         label_encoder.transform(label_encoder.classes_)))
print("Mapping of Label Encoded Classes", label_encoder_name_mapping, sep="\n")
print("Label Encoded Target Variable", encoded_X, sep="\n")
#endregion label encoding
#endregion preprocessing

#region KNN
import math
def euclidean_distance(point1, point2):
    sum_squared_distance = 0
    for i in range(len(point1)):
        sum_squared_distance += math.pow(point1[i] - point2[i], 2)
    return math.sqrt(sum_squared_distance)
#endregion KNN