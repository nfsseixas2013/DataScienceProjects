#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Bibliotecas
import pandas as pd
import numpy as np
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from pandas.plotting import scatter_matrix
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression, Ridge, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel


# Business problem:
# Our goal is to build a machine learning model that is able to make predictions about the average occupancy rate of homes in the Boston region, USA, by homeowners. The variable to be predicted is a numerical value that represents the median occupancy rate for homes in Boston. For each house, we have several explanatory variables.

# We will use the Boston Housing Dataset, which is a dataset that has the average occupancy rate of homes, along with 13 other variables that may be related to home prices. These are factors like socioeconomic conditions, environmental conditions, educational facilities, and some other similar factors. There are 506 observations in the data for 14 variables. There are 12 numeric variables in our dataset and 1 categorical variable. The objective of this project is to build a linear regression model to estimate the average occupancy rate of homes by homeowners in Boston.

# Links:
# [1]: https://www.portalsaofrancisco.com.br/quimica/oxido-nitrico

# In[3]:


# Let's load the dataset
from sklearn.datasets import load_boston


# In[4]:


boston = load_boston()


# In[5]:


dataset = pd.DataFrame(boston.data, columns = boston.feature_names)


# In[6]:


dataset['target'] = boston.target


# In[7]:


dataset.head()


# Dataset Description
# 
# 1. CRIM: per capita crime rate by town 
# 2. ZN: proportion of residential land zoned for lots over 25,000 sq.ft. 
# 3. INDUS: proportion of non-residential acres per town 
# 4. CHAS: Charles River dummy variable (= 1 if tract bounds river; 0 otherwise) 
# 5. NOX: nitric oxides concentration (parts per 10 million) 
# 6. RM: average number of rooms per dwelling 
# 7. AGE: proportion of owner-occupied units built prior to 1940 
# 8. DIS: weighted distances to five Boston employment centres 
# 9. RAD: index of accessibility to radial highways 
# 10. TAX: full-value property-tax rate per 10,000 
# 11. PTRATIO: pupil-teacher ratio by town 
# 12. B: 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town 
# 13. LSTAT: % lower status of the population 
# 14. TARGET: Median value of owner-occupied homes in $1000's

# # Cleaning data

# In[8]:


# Verifying if there is null values by column
dataset.isnull().sum()


# In[41]:


# ok. There isn't. Now we can go further.


# # Exploratory Analysis

# In[9]:


# Verifying correlation with the target.
dataset.corr()


# As we can see, all variables has considerable correlation with the target. Let's see between the predictors

# In[10]:


Correlation = dataset.iloc[:,:-1].corr() # Except the target

plt.figure(figsize=(12,8))
corr_map = sns.heatmap(
Correlation,
vmin = -1,
vmax = 1,
cmap = sns.diverging_palette(20,220, n = 400),
center = 0,
square = True)

corr_map.set_xticklabels(
corr_map.get_xticklabels(),
rotation = 45, horizontalalignment = 'right')
plt.savefig('Correlation.png')


# In[44]:


# Let's take a look at those high correlated!


# In[11]:


high_correlated = []
for i in Correlation.columns:
    for j in Correlation.columns:
        if i != j:
            if (Correlation[i][j] >= 0.7 or Correlation[i][j] <= -0.7) and (i,j) not in high_correlated and (j,i) not in high_correlated:
                high_correlated.append((i,j))


# In[12]:


high_correlated


# In[13]:


# This the first set of correlated data to be seen.
df1 = dataset[['INDUS', 'NOX', 'TAX', 'DIS']]


# In[14]:


df1.head()


# In[15]:


sns.pairplot(df1)


# In[16]:


df1.corr()


# We can see high correlation between industrial zone (INDUS), area of employment (DIS), polluent gas NOX, and the full-value property-tax rate per 10,000 (TAX). When the house is near of city centre, it tends to be expensive. Industrial area is highly correlated with nitric Oxide concentration. Nitric oxide is a colorless gas. It is also known as nitrogen monoxide and has the chemical formula NO. It is considered an air pollutant responsible for the depletion of the ozone layer. Nitric oxide reacts with oxygen (O2) and ozone (O3) to form nitrogen dioxide (NO2), a brown smoke and an environmental pollutant. Nitric oxide generated from car engines, industries, and power plants is the cause of acid rain and air pollution[1]. The more the distance from city centre, the less is NO, which means, negative correlation. I can also see asymmetric distributions.

# In[17]:


df1.skew()


# By the results above, we get asymmetric positive distributions. So the data is biased.

# In[18]:


df2 = dataset[['RAD', 'TAX']]


# In[19]:


sns.pairplot(df2)


# In[20]:


df2.corr()


# As we can see, the accessibility to highways and the taxes are highly correlated. So far, we have found that the predictor variables are strongly correlated, which can cause colinearity. If we put all the variables in the machine learning model, the model struggles to get the impact of multicolinear variables on the target variable.

# In[21]:


# Looking for outliers 
plt.figure(figsize=(12,8))
dataset.iloc[:,:-1].boxplot()
plt.savefig('boxold.png')


# The boxplot indicates the necessity of replacing or remove outliers.

# # Preprocessing

# In[22]:


# Replacing outliers:
'''
Formula:
Q1 - 1.5*IQR
q3 + 1.5*IQR
'''
def replace_outliers(dataset):
    data = dataset.copy()
    q1 = np.quantile(data, 0.25)
    q3 = np.quantile(data, 0.75)
    IQR = q3 - q1
    for i in range(0, len(data)):
        if data[i] < q1 - 1.5*IQR:
            data.iat[i] = np.mean(data)
        elif data[i] > q3 + 1.5*IQR:
            data.iat[i] = np.mean(data)
    return data
            


# In[23]:


dataset2 = dataset.copy()
for i in dataset2.columns:
    if i != 'target':
        dataset2[i] = replace_outliers(dataset2[i])


# In[24]:


# Checking the outliers 
plt.figure(figsize=(12,8))
dataset2.iloc[:,:-1].boxplot()
plt.savefig('boxafter.png')


# In[25]:


dataset3 = dataset2.copy()


# In[26]:


## Now, it's time to split the dataset between train and test
previsores = dataset3.iloc[:,:-1]


# In[27]:


previsores


# In[28]:


target = dataset3.iloc[:,-1:]


# In[29]:


target


# In[ ]:


# Now we have to normalize them in order to get a fair analysis of importance by the model


# In[30]:


n_previsores = MinMaxScaler()


# In[31]:


n_target = MinMaxScaler()


# In[32]:


previsores_normalized = n_previsores.fit_transform(previsores)


# In[33]:


target_normalized = n_target.fit_transform(target)


# In[34]:


target_normalized.shape


# In[35]:


previsores_normalized.shape


# In[36]:


previsores_normalized


# In[ ]:


# Time for splitting the data


# In[48]:


X_train,X_test, Y_train, Y_test = train_test_split(previsores_normalized, target_normalized, test_size = 0.3, random_state=42)


# In[49]:


len(Y_test)


# In[50]:


len(X_test)


# In[51]:


len(X_train)


# In[52]:


len(Y_train)


# # Modelling our Machine Learning!

# In[53]:


modelo = LinearRegression(normalize = False)


# In[54]:


modelo.fit(X_train, Y_train)


# In[55]:


r2_score(Y_test, modelo.predict(X_test))


# Using all variables we have 0.63 of R_squared. Let's see how can we improve using Feature Selection with random_forest
# 
# 

# # Improving the model

# In[56]:


sel = SelectFromModel(RandomForestRegressor(n_estimators = 100))
sel.fit(X_train, Y_train)


# In[57]:


## Let's see the variables it has chosen.
sel.get_support()


# It picked up only 2 variables. RM and LSTAT

# In[58]:


sel.get_feature_names_out()


# In[61]:


X_train = X_train[:,[5,12]]


# In[62]:


X_test = X_test[:,[5,12]]


# In[63]:


modelo_v2 = LinearRegression(normalize = False)


# In[64]:


modelo_v2.fit(X_train, Y_train)


# # Evaluation

# In[65]:


r2_score(Y_test, modelo_v2.predict(X_test))


# Worse than the previous one. We must put more variables into our model. Let's try other methods to feature selection

# # Keep Improving!

# In[66]:


X_train,X_test, Y_train, Y_test = train_test_split(previsores_normalized, target_normalized, test_size = 0.3, random_state=42)


# In[67]:


forest = RandomForestRegressor(n_estimators = 100)


# In[68]:


forest.fit(X_train, Y_train)


# In[69]:


forest.feature_importances_


# In[70]:


np.sort(forest.feature_importances_)


# In[72]:


# let's get the last 4:
# 12,5,7,4 indexes
X_train = X_train[:,[4,5,7,12]]
X_test = X_test[:,[4,5,7,12]]


# In[73]:


modelo_v3 = LinearRegression(normalize = False)


# In[74]:


modelo_v3.fit(X_train, Y_train)


# In[75]:


r2_score(Y_test, modelo_v3.predict(X_test))


# ## Still not good! Testing with the random forest...

# In[76]:


forest2 = RandomForestRegressor(n_estimators = 100)


# In[77]:


forest2.fit(X_train, Y_train)


# In[78]:


r2_score(Y_test, forest2.predict(X_test))


# In[90]:


predictions = forest2.predict(X_test)


# # Visualizing the data

# In[116]:


# reverting normalization


# In[91]:


predictions = predictions.reshape(-1,1)


# In[92]:


predictions.shape


# In[93]:


predictions = n_target.inverse_transform(predictions)


# In[94]:


predictions


# In[95]:


p = pd.DataFrame(predictions)


# In[96]:


p.columns = ['data']


# In[97]:


p.index


# In[98]:


Y = n_target.inverse_transform(Y_test)


# In[100]:


Y = pd.DataFrame(Y)


# In[103]:


Y.index


# In[104]:


plt.figure(figsize=(12, 8), dpi = 200)
plt.scatter(Y.index, Y, color = 'blue',label = 'Real values')
plt.plot(p.index,p.data, color = 'red',label = 'Predictions')
plt.xlabel('Rounds', fontsize = 15)
plt.ylabel('Target values', fontsize = 15)
plt.legend()
plt.savefig('Predictions.png')


# In[ ]:




