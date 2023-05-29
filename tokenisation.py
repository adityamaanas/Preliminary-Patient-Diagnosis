# import english-train.json
import pandas as pd
from tabulate import tabulate
df = pd.read_json(r'english-train.json')
#print(df.head())
#print(df.info())
#print(df.iloc[2,0])
#print(df.iloc[2,1])

# Tokenisation
import spacy
from spacy import displacy

med7 = spacy.load("en_ner_bc5cdr_md")
# print(med7.get_pipe("ner").labels)

# create distinct colours for labels
col_dict = {}
seven_colours = ['#e6194B', '#3cb44b', '#ffe119', '#ffd8b1', '#f58231', '#f032e6', '#42d4f4']
for label, colour in zip(med7.pipe_labels['ner'], seven_colours):
    col_dict[label] = colour

options = {'ents': med7.pipe_labels['ner'], 'colors':col_dict}

def tokenisation(txt):
    #text = str(df.iloc[2,1])
    text = txt
    doc = med7(text)

    filtered_entities = [ent.text for ent in doc.ents if ent.label_ in ["DISEASE", "SYMPTOM"]]
    print("filtered:", filtered_entities)
    #print(displacy.render(doc, style='ent', jupyter=True, options=options))

    print([(ent.text, ent.label_) for ent in doc.ents])
    return filtered_entities

if __name__ == "__main__":
    tokenisation(txt)