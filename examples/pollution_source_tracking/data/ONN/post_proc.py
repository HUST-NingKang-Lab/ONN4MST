import pandas as pd
from collections import OrderedDict


with open('batch_0.txt', 'r') as f:
    con = pd.Series(f.read().splitlines())

con = con.str.split()  # 0 -> id|pred, 1 -> id|true 2 -> probas

df = pd.DataFrame([OrderedDict(zip(line[0].split('|')[1].split(','), line[2].split(','))) for line in con])
print(df)

df[sorted(df.columns.tolist(), key=lambda x: list(x).count('-'))].to_csv('probas_for_each_sample.csv')





