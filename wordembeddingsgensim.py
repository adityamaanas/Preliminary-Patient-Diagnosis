import pandas as pd

data = pd.read_csv('englishtraining.csv')

# Clean the data
data = data.dropna()
data = data.apply(lambda x: x.str.lower() if x.dtype == "object" else x)  # Normalize string columns to lowercase

# region creating vocabularytxt
tokenized_data = list(data.iloc[:, 1])
#print(tokenized_data)

vocabulary = set()

# Iterate over tokenized data and add words to the vocabulary set
for sentence in tokenized_data:
    vocabulary.add(sentence)

# Print the vocabulary
#print(vocabulary)
# endregion

# region creating word embeddings
import gensim
from gensim.models import KeyedVectors

# Path to the pre-trained GloVe embeddings file
glove_file = "D:/Files/repos/openlab/glove.6B.100d.txt"

# Load GloVe embeddings
glove_model = KeyedVectors.load_word2vec_format(glove_file, binary=False)

word_to_vector = {}

import numpy as np

def get_random_vector(dim):
    return np.random.randn(dim)

def vocab_index(word):
    return int(glove_model.key_to_index[word])

for word in vocabulary:
    #print(vocab_index(word))
    if word not in glove_file:
        # Handle out-of-vocabulary words
        # Assign a random vector or a special "unknown" vector
        word_to_vector[word] = get_random_vector(100)
    else:
        word_to_vector[word] = glove_model.vectors[vocab_index]

# Convert tokenized sentences to word embeddings
embeddings = []

for sentence in tokenized_data:
    sentence_embeddings = []
    
    for word in sentence:
        if word in word_to_vector:
            sentence_embeddings.append(word_to_vector[word])
        else:
            # Handle out-of-vocabulary words
            sentence_embeddings.append(get_random_vector(100))
    
    embeddings.append(sentence_embeddings)

#print(embeddings)
# endregion creating word embeddings

# add the embeddings to the dataframe
data['Embeddings'] = embeddings

# Pad the word embeddings to the same length

max_length = max(len(embedding) for embedding in embeddings)
padded_embeddings = []

for embedding in embeddings:
    if len(embedding) < max_length:
        padding = [np.zeros(100)] * (max_length - len(embedding))
        padded_embedding = embedding + padding
    else:
        padded_embedding = embedding[:max_length]
    
    padded_embeddings.append(padded_embedding)

# Convert the padded embeddings to numpy array
padded_embeddings = np.array(padded_embeddings)

# find the length of each embedding
for embedding in padded_embeddings:
    print(len(embedding))
