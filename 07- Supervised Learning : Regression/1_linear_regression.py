
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

df = pd.read_csv('homeprices.csv')
df

get_ipython().run_line_magic('matplotlib', 'inline')
plt.xlabel('area')
plt.ylabel('price')
plt.scatter(df.area,df.price,color='red',marker='+')

new_df = df.drop('price',axis='columns')
new_df

area = df.area
print(area)


price = df.price
price

# Create linear regression object
reg = linear_model.LinearRegression()
reg.fit(new_df,price)
#reg.fit(area,price)

reg.predict([[3300]])

reg.predict([[3300],[3200]])


# **Y = m * X + b (m is coefficient and b is intercept)**

reg.coef_

reg.intercept_


#3300*135.78767123 + 180616.43835616432


# **(1) Predict price of a home with area = 5000 sqr ft**

reg.predict([[5000]])

area_df = pd.read_csv("areas.csv")
area_df.head(3)

p = reg.predict(area_df)
p

area_df['prices']=p
area_df

area_df.to_csv("prediction.csv")


predicted_data=pd.read_csv('prediction.csv');
print(predicted_data)