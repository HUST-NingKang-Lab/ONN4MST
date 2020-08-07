import pandas as pd


df = pd.read_csv('overall_analysis_source_contributions_matrix.txt', sep='\t')
df = df.T
df['GroupID'] = df.index.to_series().apply(lambda x: '_'.join(x.split('_')[1:]) if x != 'Unknown' else x)
df = df.groupby(by='GroupID').sum().T
df['Env'] = df.index.to_series().apply(lambda x: '_'.join(x.split('_')[1:]))
df['SampleID'] = df.index.tolist()
df.to_csv('HumanSourceTracking.by-FEAST.csv')


