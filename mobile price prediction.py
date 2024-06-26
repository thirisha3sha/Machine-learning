"""1.	Mr Arun wants to start his own mobile phone company and he wants to wage an uphill battle with big smartphone brands like Samsung and Apple. But he doesn’t know how to estimate the price of a mobile that can cover both marketing and manufacturing costs. So in this task, you don’t have to predict the actual prices of the mobiles but you have to predict the price range of the mobiles. ”
a)	Read the Mobile price dataset  using the Pandas module 
b)	print the 1st five rows. 
c)	Basic statistical computations on the data set or distribution of data
d)	the columns and their data types
e)	Detects null values in the dataset. If there is any null values replaced it with mode value
f)	Explore the data set using   heatmap
g)	Split the data in to test and train 
h)	Fit in to the model Naive Bayes Classifier
i)	Predict the model
j)	Find the accuracy of the model
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
# Step 1: Create a sample dataset
data={
    'battery_power':np.random.randint(500,5000,1000),#500 to 4999 mAh.
    'bluetooth':np.random.randint(0,2,1000),#(1 for yes, 0 for no)
    'clock_Speed':np.round(np.random.uniform(0.5,3.0,1000),1),#0.5 to 2.9 GHz
    'dual_Sim':np.random.randint(0,2,1000),#(1 for yes, 0 for no).
    'front_Camera':np.random.randint(0,20,1000),#0 to 19 MP
    'four_g':np.random.randint(0,2,1000),# (1 for yes, 0 for no).
    'three_g':np.random.randint(0,2,1000),# (1 for yes, 0 for no).
    'int_memory':np.random.randint(2,256,1000),#2 to 255 GB.
    'mobile_depth':np.round(np.random.uniform(0.1,1.0,1000)),#0.1 to 0.9 cm
    'mobile_weight':np.random.randint(80,250,1000),#80 to 249 g.
    'num_cores':np.random.randint(1,8,1000),# 1 to 7 cores.
    'pc':np.random.randint(0,20,1000),#0 to 19 MP
    'screen_height_in_pixels':np.random.randint(0,1960,1000),# 0 to 1959 pixels.
    'screen_width':np.random.randint(0,1920,1000),#0 to 1919 pixels.
    'ram':np.random.randint(256,8000,1000),#256 to 7999 MB.
    'screen_height(cm)':np.random.randint(5,20,1000),#5 to 19 cm.
    'screen_width':np.random.randint(0,20,1000),#0 to 19 cm.
    'talk_time':np.random.randint(2,20,1000),#2to 19 hours
    'touch_screen':np.random.randint(0,2,1000),#(1 for yes, 0 for no).
    'wifi':np.random.randint(0,2,1000),#(1 for yes, 0 for no).
    'price_range':np.random.randint(0,4,1000)
    #(0, 1, 2, 3, budget, mid-range, high-end, premium)"""
    }
# Step 2: Read the Mobile price dataframe using the Pandas module
df=pd.DataFrame(data)
print("mobile dataset:\n",df)
# Step 3: Print the 1st five rows
head=df.head()
print("\nFirst 5 rows:",head)
# Step 4: Basic statistical computations on the dataset
print("\nstatistical description:",df.describe())
# Step 5: The columns and their data types
print("\ncolumns and their data types:",df.dtypes)
# Step 6: Detect null values in the dataframe and replace them with mode value if any
print("\n null values in the dataframe:",df.isnull().sum())
#Step 7: Explore the dataset using a heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(),annot=True,fmt='.2f')
plt.title('Feature correlation heatmap')
plt.show()
# Step 8: Split the data into test and train sets
x=df.drop('price_range',axis=1)
y=df['price_range']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
# Step 9: Fit the data into a Naive Bayes Classifier
model=GaussianNB()
model.fit(x_train,y_train)
# Step 10: Predict using the model
y_pred=model.predict(x_test)
# Step 11: Find the accuracy of the model
accuracy=accuracy_score(y_test,y_pred)
print("\n accuracy of the naive bayes classifier model:",accuracy)
model.fit(x_train,y_train)






                            
