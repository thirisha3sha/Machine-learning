"""2.	Implement a Python program for  the most specific hypothesis using Find-S algorithm for the following given dataset and show the output:
Example	Sky	Air Temp	Humidity	Wind	Water	Forecast	Enjoy Sport
1	Sunny	Warm	Normal	Strong	Warm	Same	Yes
2	Sunny	Warm	High	Strong	Warm	Same	Yes
3	Rainy	Cold	High	Strong	Warm	Change	No
4	Sunny	Warm	High	Strong	Cool	Change	Yes"""
import pandas as pd
data = {
    'Sky': ['Sunny', 'Sunny', 'Rainy', 'Sunny'],
    'Air Temp': ['Warm', 'Warm', 'Cold', 'Warm'],
    'Humidity': ['Normal', 'High', 'High', 'High'],
    'Wind': ['Strong', 'Strong', 'Strong', 'Strong'],
    'Water': ['Warm', 'Warm', 'Warm', 'Cool'],
    'Forecast': ['Same', 'Same', 'Change', 'Change'],
    'Enjoy Sport': ['Yes', 'Yes', 'No', 'Yes']
}
df=pd.DataFrame(data)
def s_algorithm(df):
    hypothesis=['0']*(len(df.columns)-1)
    for index,row in df.iterrows():
        if row['Enjoy Sport']=='Yes':
            if hypothesis==['0']*(len(df.columns)-1):
                hypothesis=row.iloc[:-1].tolist()
            else:
                for i in range(len(hypothesis)):
                    if hypothesis[i]!=row.iloc[i]:
                        hypothesis[i]='?'

    return hypothesis
hypothesis=s_algorithm(df)
print("most specific hypothesis found by s-algorithm is:",hypothesis)
