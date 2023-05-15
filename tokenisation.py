# import english-train.json
import pandas as pd
from tabulate import tabulate
df = pd.read_json(r'english-train.json')
#print(df.head())
#print(df.info())
print(df.iloc[0,0])
print(df.iloc[0,1])

# 