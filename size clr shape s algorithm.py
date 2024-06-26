"""2.	Implement a Python program for  the most specific hypothesis using Find-S
algorithm for the following given dataset and show the output:
Size	Color	Shape	Class
Big	Red	Circle	No
Small	Red	Triangle	No
Small	Red	Circle	Yes
Big	Blue	Circle	No
Small	Blue	Circle	Yes
"""
import pandas as pd
data={
    'Size': ['Big', 'Small', 'Small', 'Big', 'Small'],
    'Color': ['Red', 'Red', 'Red', 'Blue', 'Blue'],
    'Shape': ['Circle', 'Triangle', 'Circle', 'Circle', 'Circle'],
    'Class': ['No', 'No', 'Yes', 'No', 'Yes']
    }
df=pd.DataFrame(data)
def s_algorithm(df):
    columns=df.columns
    hypothesis=['?' for _ in range(len(columns)-1)]
    for i,row in df.iterrows():
        if row['Class']=='Yes':
            if hypothesis==['?' for _ in range(len(columns)-1)]:
                hypothesis=row.iloc[:-1].tolist()
            else:
                for j in range(len(hypothesis)):
                    if hypothesis[j]!=row.iloc[j]:
                        hypothesis[j]='?'
    return hypothesis
result=s_algorithm(df)
print("most specific hypothesis using s algorithm:",result)
