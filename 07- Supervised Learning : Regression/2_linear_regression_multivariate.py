
import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv('homeprices.csv')
df

df.bedrooms.median()

df.bedrooms = df.bedrooms.fillna(df.bedrooms.median())
df

feature_matrix=df.drop('price',axis='columns')
feature_vector=df.price

reg = linear_model.LinearRegression()
#reg.fit(df.drop('price',axis='columns'),df.price)
reg.fit(feature_matrix,feature_vector)


# **Y = m1*X1 + m2*X2 + m3*X3 + b (m is coefficient and b is intercept)**
reg.coef_

reg.intercept_


# **Find price of home with 3000 sqr ft area, 3 bedrooms, 40 year old**

reg.predict([[3000, 3, 40]])


#112.06244194*3000 + 23388.88007794*3 + -3231.71790863*40 + 221323.00186540384


# **Find price of home with 2500 sqr ft area, 4 bedrooms,  5 year old**

reg.predict([[2500, 4, 5]])
