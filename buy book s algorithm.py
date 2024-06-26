"""2.	Implement a Python program for  the most specific hypothesis using Find-S algorithm for the following given dataset and show the output:Example	Citations	Size	In Library	Price	Editions	Buy
1	Some	Small	No	Affordable	Few	No
2	Many	Big	No	Expensive	Many	Yes
3	Many	Medium	No	Expensive	Few	Yes
4	Many	Small	No	Affordable	Many	Yes
"""
import pandas as pd

# Define the dataset
data = {
    'Example': [1, 2, 3, 4],
    'Citations': ['Some', 'Many', 'Many', 'Many'],
    'Size': ['Small', 'Big', 'Medium', 'Small'],
    'In Library': ['No', 'No', 'No', 'No'],
    'Price': ['Affordable', 'Expensive', 'Expensive', 'Affordable'],
    'Editions': ['Few', 'Many', 'Few', 'Many'],
    'Buy': ['No', 'Yes', 'Yes', 'Yes']
}

# Create a DataFrame
df = pd.DataFrame(data)

def s_algorithm(df):
    hypothesis = ['0'] * (len(df.columns) - 1)
    
    for index, row in df.iterrows():
        if row['Buy'] == 'Yes':
            if hypothesis == ['0'] * (len(df.columns) - 1):
                hypothesis = row.iloc[1:-1].tolist()  # Exclude 'Example' and 'Buy'
            else:
                for i in range(len(hypothesis)):
                    if hypothesis[i] != row.iloc[i + 1]:  # Adjust for 'Example'
                        hypothesis[i] = '?'
    
    return hypothesis

# Run the Find-S algorithm
hypothesis = s_algorithm(df)
print("Most specific hypothesis found by Find-S algorithm:", hypothesis)
