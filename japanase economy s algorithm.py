"""Implement a Python program for  the most specific hypothesis using Find-S algorithm for the following given dataset and show the output:
Origin	Manufacturer	Color	Decade	Type	Example Type
Japan	Honda	Blue	1980	Economy	Positive
Japan	Toyota	Green	1970	Sports	Negative
Japan	Toyota	Blue	1990	Economy	Positive
USA	Chrysler	Red	1980	Economy	Negative
Japan	Honda	White	1980	Economy	Positive"""
import pandas as pd
data = {
    'Origin': ['Japan', 'Japan', 'Japan', 'USA', 'Japan'],
    'Manufacturer': ['Honda', 'Toyota', 'Toyota', 'Chrysler', 'Honda'],
    'Color': ['Blue', 'Green', 'Blue', 'Red', 'White'],
    'Decade': ['1980', '1970', '1990', '1980', '1980'],
    'Type': ['Economy', 'Sports', 'Economy', 'Economy', 'Economy'],
    'Example Type': ['Positive', 'Negative', 'Positive', 'Negative', 'Positive']
}
df=pd.DataFrame(data)
def s_algorithm(df):
    hypothesis=['?' for _ in range(len(df.columns)-1)]
    for i,row in df.iterrows():
        if row['Example Type']=='Positive':
            if hypothesis==['?' for _ in range(len(df.columns)-1)]:
                hypothesis=row.iloc[:-1].tolist()
                    
            else:
                for j in range(len(hypothesis)):
                    if hypothesis[j]!=row.iloc[j]:
                        hypothesis[j]='?'
    return hypothesis
hypothesis=s_algorithm(df)
print("most specific hypothesis using s algorithm:",hypothesis)

