#!/usr/bin/env python
# coding: utf-8

# # Google Store Ecommerce Data + Fake Retail Data

# In this notebook, let's see one of the easiest ways of forecasting quantities with different regression models.
# 
# # Importing required libraries 

# In[18]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from datetime import date 
import holidays 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# Reading CSV files 

# In[19]:


marketing_spent = pd.read_csv('../Kaggle_datasets/Google_Store_Ecommerce_Data/Marketing_Spend.csv')
online = pd.read_csv('../Kaggle_datasets/Google_Store_Ecommerce_Data/Online.csv')


# In[20]:


marketing_spent.head(2)


# In[21]:


online.head(2)


# 
# 
# 1. Renaming columns as the Date column in names as "Unnamed" in marketing_spent csv file.
# 2. Changing Date format in online csv file from YYYYMMDD to YYYY-MM-DD.

# In[22]:


marketing_spent.rename(columns = {'Unnamed: 0':'Date'}, inplace = True) 
marketing_spent['Date']= pd.to_datetime(marketing_spent['Date'])
online['Date'] = pd.to_datetime(online['Date'], format='%Y%m%d')


# understanding Online dataset

# In[23]:


# understanding marketing spent dataset
print("Total number of rows:",online.shape[0])
print("Total number of colums:",online.shape[1])
print("\n\nList of columns:", online.columns.tolist())
print("\n\nMin Date:", online['Date'].min())
print("Max Date:", online['Date'].max())
print("\n\nDatatypes:\n",online.dtypes)
print("\n\nUnique values:\n",online.nunique())
print("\n\nMissing values:\n",online.isnull().sum())
print("\n\nQuantitative analysis\n", online.describe())


# We have 5 missing values for Quantity. Lets fill the same later. 
# 
# Understanding marketing spent dataset

# In[24]:


# understanding marketing spent dataset
print("Total number of rows:",marketing_spent.shape[0])
print("Total number of colums:",marketing_spent.shape[1])
print("\n\nList of columns:", marketing_spent.columns.tolist())
print("\n\nMin Date:", marketing_spent['Date'].min())
print("Max Date:", marketing_spent['Date'].max())
print("\n\nDatatypes:\n",marketing_spent.dtypes)
print("\n\nUnique values:\n",marketing_spent.nunique())
print("\n\nMissing values:\n",marketing_spent.isnull().sum())
print("\n\nQuantitative analysis\n", marketing_spent.describe())


# # Combining 2 datasets

# In[25]:


df = pd.merge(
    marketing_spent,
    online,
    left_on=['Date'],
    right_on=['Date'])
df.shape


# Lets rename columns and impute missing values

# In[26]:


df.columns = df.columns.str.replace(' ', '_')
df = df.fillna(axis=0, method='ffill')
df.head(2)


# # Correlation matrix

# In[27]:


import seaborn as sns
df_correlation = df[['Date','Offline_Spend','Online_Spend','Transaction_ID','Product_SKU','Product','Product_Category_(Enhanced_E-commerce)',
 'Avg._Price','Revenue','Tax', 'Quantity','Delivery']]

upper_triangle = np.zeros_like(df_correlation.corr(), dtype = np.bool)
upper_triangle[np.triu_indices_from(upper_triangle)] = True #make sure we don't show half of the other triangle
f, ax = plt.subplots(figsize = (15, 10))
sns.heatmap(df_correlation.corr(),ax=ax,mask=upper_triangle,annot=True, fmt='.2f',linewidths=0.5,cmap=sns.diverging_palette(10, 133, as_cmap=True))


# In[28]:


df1 = df.copy()
df1 = df1[['Date', 'Quantity']]
print("Shape:",df1.shape,"\n")
print(df1.info(),"\n")
print("Missing values:\n",df1.isnull().sum())
print("\nDescription:\n", df1.describe())


# We group by "Date" and sum up the "Quantities"

# In[29]:


df1 = df1.groupby(['Date'])['Quantity'].sum().reset_index()
df1.head(5)


# Let us integrate national holidays from UK in out dataset and will have a bad impact on our forcasted values. Let us first filter it out.

# In[30]:


# holidays
uk_holidays = []
for date in holidays.UnitedKingdom(years = 2017).items():
    uk_holidays.append(str(date[0]))

holidays = pd.DataFrame(uk_holidays, columns=['Holiday Date']) 
holidays.head()


# In[31]:


df1['Holidays'] = df1['Date'].isin(uk_holidays)
df1.head()


# In[32]:


# removing the holidays
df1 = df1[df1["Holidays"]==False]
df1.head()


# In[33]:


df2 = df1[["Date","Quantity"]]
df2.head()


# In[34]:


# creating testing data for 2018 Jan month
date_2018 = "2018-01-01"
# index = pd.date_range(date_2018, periods=30, freq='D')
#creating Quantity column
# columns = ['Quantity']
test = pd.DataFrame()
test['Date'] = pd.date_range(start=date_2018, periods=30, freq='D')
# # extracting more features from the train dataset
test['Year'] = pd.to_datetime(test['Date']).dt.year
test['Week'] = pd.to_datetime(test['Date']).dt.week
test['Day'] = pd.to_datetime(test['Date']).dt.day
test['Weekday'] = pd.to_datetime(test['Date']).dt.dayofweek

test["Quantity"] = ""
test.head(8)


# In[35]:


# extracting more features from the train dataset
df2['Year'] = pd.to_datetime(df2['Date']).dt.year
df2['Week'] = pd.to_datetime(df2['Date']).dt.week
df2['Day'] = pd.to_datetime(df2['Date']).dt.day
df2['Weekday'] = pd.to_datetime(df2['Date']).dt.dayofweek
df2.head(8)


# Before running the model lets do EDA

# In[36]:


df2.describe()


# In[37]:


sns.boxplot(x=df2['Quantity'])


# Lets plot some weekly

# In[38]:


# weekly trend
sns.lineplot(df2['Week'], df2['Quantity'])


# # Removing outliers

# In[39]:


# removing outliers
df2 = df2[df2['Quantity']<3500]
df2.describe()


# In[40]:


sns.boxplot(x=df2['Quantity'])


# # Running the Random-Forest-Regressor
# Before running the Random forest regressor let us evaluate wheather there are better algorithms for the dataset

# In[41]:


df2


# # Splitting the dataset

# In[42]:


#Breaking the data and selecting features , predictors
from sklearn.model_selection import train_test_split
predictors=df2.drop(['Quantity','Date'],axis=1)
target=df2['Quantity']
x_train,x_cv,y_train,y_cv=train_test_split(predictors,target,test_size=0.1,random_state=7)

# x_train -> Year, Week, Day, Weekday ...... shape(314*4)
# x_cv Year, Week, Day, Weekday ...... shape(35*4)
# y_train -> quantity ......... shape(314,)
# y_cv -> qunatity ........... shape(35,)


# In[43]:


#Comparing Algorithms
def scores(i):
    lin = i()
    lin.fit(x_train, y_train)
    y_pred=lin.predict(x_cv)
    lin_r= r2_score(y_cv, y_pred)
    s.append(lin_r)
#Checking the scores by using our function
algos=[LinearRegression,KNeighborsRegressor,RandomForestRegressor,Lasso,ElasticNet,DecisionTreeRegressor]
s=[]
for i in algos:
    scores(i)


# In[44]:


#Checking the score
models = pd.DataFrame({
    'Models': ['LinearRegression', 'KNeighborsRegressor', 
              'RandomForestRegressor', 'Lasso','DecisionTreeRegressor'],
    'R2-Score': [s[0],s[1],s[2],s[3],s[4]]})
models.sort_values(by='R2-Score', ascending=False)


# As we can see Linear regression will give the best R squared score for our dataset.
# As we have evaluated, let's run for Random Forest Regressor and see how it performs.

# In[45]:


#Hypertuned Model
model = RandomForestRegressor()
# bootstrap=True, criterion='mse', max_depth=None,
#                       max_features='auto', max_leaf_nodes=None,
#                       min_impurity_decrease=0.0, min_impurity_split=None,
#                       min_samples_leaf=4, min_samples_split=2,
#                       min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=3,
#                       oob_score=True, random_state=7, verbose=0,
#                       warm_start=False



model.fit(x_train,y_train)


# In[46]:


pred = model.predict(x_cv)
# R2 Score
r2_score(pred, y_cv)


# In[47]:


def mean_percentage_error(y_cv, pred): 
    y_cv, pred = np.array(y_cv), np.array(pred)
    return np.mean(np.array((y_cv - pred) / y_cv)) * 100
mean_percentage_error(y_cv, pred)


# let's predict for the 2018 Jan dataset, we have created earlier

# In[48]:


test1=test.drop(['Quantity', 'Date'],axis=1)
pred2=model.predict(test1)
test['Quantity']=pred2.round(0)
test.head()


# # Predicting Quantities for 2018 Jan 

# In[50]:


result=test[['Date','Quantity']]
result

