"""Implement a Python program for  the most specific hypothesis using Find-S algorithm for the following given dataset and show the output:
Example	Shape	Size	Color	Surface	Thickness	Target Concept
1	Circular	Large	Light	Smooth	Thick	Malignant (+)
2	Circular	Large	Light	Irregular	Thick	Malignant (+)
3	Oval	Large	Dark	Smooth	Thin	Benign (-)
4	Oval	Large	Light	Irregular	Thick	Malignant (+)
"""
import pandas as pd
data = {
    'Example': [1, 2, 3, 4],
    'Shape': ['Circular', 'Circular', 'Oval', 'Oval'],
    'Size': ['Large', 'Large', 'Large', 'Large'],
    'Color': ['Light', 'Light', 'Dark', 'Light'],
    'Surface': ['Smooth', 'Irregular', 'Smooth', 'Irregular'],
    'Thickness': ['Thick', 'Thick', 'Thin', 'Thick'],
    'Target Concept': ['Malignant (+)', 'Malignant (+)', 'Benign (-)', 'Malignant (+)']
}
df=pd.DataFrame(data)
def s_algorithm(df):
    hypothesis=['?' for _ in range(len(columns)-1)]
    for i,row in df,iterrows():
        if row['Target Concept']=='Malignant(+)':
            if hypothesis==['?' for _ in range(len(df.columns)-1)]:
                hypothesis=row.iloc[:-1].tolist()
            else:
                for j in range(len(hypothesis)):
                    if hypothesis[j]!=row.iloc[j]:
                        hypothesis[j]='?'
    return hypothesis
result=s_algorithm(df)
print("mossst specific hypothesis using s algorithm:",result)
