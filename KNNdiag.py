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
#from sklearn.preprocessing import LabelEncoder
#label_encoder = LabelEncoder()
#X = df2.Symptoms
#encoded_X = label_encoder.fit_transform(X)
#label_encoder_name_mapping = dict(zip(label_encoder.classes_,
#                                         label_encoder.transform(label_encoder.classes_)))
#print("Mapping of Label Encoded Classes", label_encoder_name_mapping, sep="\n")
#print("Label Encoded Target Variable", encoded_X, sep="\n")
#endregion label encoding
#endregion preprocessing

#region KNN
import math
def euclidean_distance(point1, point2):
    sum_squared_distance = 0
    for i in range(len(point1)):
        sum_squared_distance += math.pow(point1[i] - point2[i], 2)
    return math.sqrt(sum_squared_distance)

def knn(data, query, k, distance_fn, choice_fn):
    neighbor_distances_and_indices = []
    
    # 3. For each example in the data
    for index, example in enumerate(data):
        # 3.1 Calculate the distance between the query example and the current
        # example from the data.
        distance = distance_fn(example[:-1], query)
        
        # 3.2 Add the distance and the index of the example to an ordered collection
        neighbor_distances_and_indices.append((distance, index))
    
    # 4. Sort the ordered collection of distances and indices from
    # smallest to largest (in ascending order) by the distances
    sorted_neighbor_distances_and_indices = sorted(neighbor_distances_and_indices)
    
    # 5. Pick the first K entries from the sorted collection
    k_nearest_distances_and_indices = sorted_neighbor_distances_and_indices[:k]
    
    # 6. Get the labels of the selected K entries
    k_nearest_labels = [data[i][1] for distance, i in k_nearest_distances_and_indices]

    # 7. If regression (choice_fn = mean), return the average of the K labels
    # 8. If classification (choice_fn = mode), return the mode of the K labels
    return k_nearest_distances_and_indices , choice_fn(k_nearest_labels)

#region KNN test
temp = df.iloc[3,1:]
print(list(temp))

temp[12] = 1
temp[37] = 1
temp[112] = 0

def diagnose(symptoms, k_recommendations):
    dataset = []
    with open('/content/drive/MyDrive/DiseasesToSymptoms.csv', 'r') as md:
        # Discard the first line (headings)
        next(md)

        # Read the data into memory
        for line in md.readlines():
            data_row = line.strip().split(',')
            dataset.append(data_row)

    # Prepare the data for use in the knn algorithm by picking
    # the relevant columns and converting the numeric columns
    # to numbers since they were read in as strings
    diagnosis = []
    for row in dataset:
        data_row = list(map(float, row[2:]))
        diagnosis.append(data_row)

    # Use the KNN algorithm to get the 5 movies that are most
    # similar to The Post.
    recommendation_indices, _ = knn(
        diagnosis, symptoms, k=k_recommendations,
        distance_fn=euclidean_distance, choice_fn=lambda x: None
    )

    final = []
    for _, index in recommendation_indices:
        final.append(dataset[index])

    return final

if __name__ == '__main__':
    disease_x = temp # feature vector for
    a = diagnose(symptoms=disease_x, k_recommendations=3)

    # Print recommended movie titles
    for recommendation in a:
        print(recommendation[1])
#endregion KNN test

#endregion KNN