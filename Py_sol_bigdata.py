#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import os
import statistics
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import svm, feature_selection, linear_model


# In[2]:


pathName = '/Users/aesthetic/Desktop/CO3093Labs' 
fileName = 'Manhattan12.csv' 
filePath = os.path.join(pathName, fileName)


#loading data from csv file
data = pd.read_csv(filePath, header=4)


# In[3]:


#this method computes the summary statistics on each column in the dataset
def describe(df, col):
    ## Compute the summary stats
    desc = df[col].describe()


     ## Change the name of the 25% to lower Quartile
    idl=desc.index.tolist()
    idl[4]= 'Lower Quartile'
    desc.index = idl
        
        
    ## Change the name of the 50% index to median
    idx = desc.index.tolist() 
    idx[5] = 'median' 
    desc.index = idx
    
    ## Change the name of the 75% index to upper Quartile
    
    idu= desc.index.tolist()
    idu[6]= 'Upper Quartile'
    desc.index = idu
    
    return desc

describe(data,'BLOCK' )


# In[4]:


data.dtypes


# In[5]:


# renaming a few columns so that they are easier to work with

data=data.rename(columns={'SALE\nPRICE':'SALE_PRICE'})
data=data.rename(columns={'COMMERCIAL UNITS':'COMMERCIAL_UNITS'})
data=data.rename(columns={'RESIDENTIAL UNITS':'RESIDENTIAL_UNITS'})
data=data.rename(columns={'LAND SQUARE FEET':'LAND_SQUARE_FEET'})
data=data.rename(columns={'GROSS SQUARE FEET':'GROSS_SQUARE_FEET'})
data=data.rename(columns={'TOTAL UNITS':'TOTAL_UNITS'})


# In[6]:


# this method replaces strings and commas
def replace_char(df,col,char,replacement):
    replaced= data[col] = data[col].replace({char:replacement}, regex = True)
    return replaced




replace_char(data,'SALE_PRICE', '\$', '')
replace_char(data,'SALE_PRICE', ',', '')
replace_char(data,'LAND_SQUARE_FEET', ',', '')
replace_char(data,'GROSS_SQUARE_FEET', ',', '')


# In[7]:


#chaning a few columns to nummerical values  

cols_numeric= ['SALE_PRICE', 'GROSS_SQUARE_FEET', 'LAND_SQUARE_FEET', 'TOTAL_UNITS', 'RESIDENTIAL_UNITS'] 
    
for col in cols_numeric:
    data[col] = pd.to_numeric(data[col], errors="coerce")
    
#that dates are formatted correctly.

data['SALE DATE'] = pd.to_datetime(data['SALE DATE'])


# In[8]:



cols_categorical= ['NEIGHBORHOOD','BUILDING CLASS CATEGORY', 'TAX CLASS AT PRESENT', 'BUILDING CLASS AT PRESENT', 'BUILDING CLASS AT TIME OF SALE']

# chaning all empty spaces to nans
data=data.replace(r'^\s*$', np.nan, regex=True)


# In[9]:


data.shape


# In[10]:


# dropping a few columns
data.drop(['BOROUGH'], axis=1, inplace=True)
data.drop(['EASE-MENT'], axis=1, inplace=True)
data.drop(['APART\nMENT\nNUMBER'], axis=1, inplace=True)


# In[11]:


data.shape


# In[12]:


data.corr()


# In[13]:



cols_for_prediction=['TOTAL_UNITS','LAND_SQUARE_FEET,GROSS_SQUARE_FEET', 'YEAR BUILT, TAX CLASS AT TIME OF SALE']


# In[14]:


data.shape


# In[15]:


col_notNull= ['LAND_SQUARE_FEET','GROSS_SQUARE_FEET','SALE_PRICE','AGE_OF_HOUSE_AT_SALE','TOTAL_UNITS']

for cols in col_notNull:
    data = data[data[col].notna()]


# In[16]:


data.head()

# data['RESIDENTIAL_UNITS']


# In[17]:


# data_nuls = data.replace(0, np.NaN) 

data['LAND_SQUARE_FEET'] = data['LAND_SQUARE_FEET'].replace(0, np.NaN) 
data['GROSS_SQUARE_FEET'] = data['GROSS_SQUARE_FEET'].replace(0, np.NaN) 
data['TOTAL_UNITS'] = data['TOTAL_UNITS'].replace(0, np.NaN) 
data['SALE_PRICE'] = data['SALE_PRICE'].replace(0, np.NaN)


# cols_numeric= ['SALE_PRICE', 'GROSS_SQUARE_FEET', 'LAND_SQUARE_FEET', 'BLOCK','LOT', 'EASE-MENT', 'ZIP CODE','YEAR BUILT','TAX CLASS AT TIME OF SALE'] 

 



# for cols in cols_numeric


# In[18]:


pivot=data.pivot_table(index='NEIGHBORHOOD', values='SALE_PRICE', aggfunc=np.median)
pivot


# In[19]:


pivot.plot(kind='bar', color='Green', figsize=(20,10))


# In[20]:


data.shape


# In[21]:


miss=data.isnull().sum()/len(data)
miss=miss[miss>0]
miss.sort_values(inplace=True)
miss


# In[22]:


miss=miss.to_frame()
miss.columns=['count']
miss.index.names=['Name']
miss['Name']=miss.index
miss


# In[23]:


#plot the missing values
sns.set(style='whitegrid',color_codes=True)
sns.barplot(x='Name', y='count',data=miss)
plt.xticks(rotation=90)
sns


# In[24]:


data.head()


# In[25]:


plt.scatter(x=data['YEAR BUILT'],y=data['SALE_PRICE'])
ax =plt.gca()
ax.get_yaxis().get_major_formatter().set_scientific(False)
# plt.figsize(30,6)
plt.draw()


# In[26]:


data['AGE_OF_HOUSE_AT_SALE'] = data['SALE DATE'].dt.year - data['YEAR BUILT']
data['AGE_OF_HOUSE_AT_SALE'] = data['AGE_OF_HOUSE_AT_SALE'].replace(0, np.NaN) 
data = data.drop(['SALE DATE','YEAR BUILT'],axis=1)


# In[27]:


data.shape


# In[28]:


data= data.drop_duplicates(subset=None, keep='first', inplace=False)


# In[29]:


data.shape


# In[30]:


data_initialModel=data.dropna(how='any')


# In[31]:


# plt.figure(figsize=(30,15))
# plt.scatter(x=data['YEAR BUILT'],y=data['SALE_PRICE'])
# ax =plt.gca()
# ax.get_yaxis().get_major_formatter().set_scientific(False)
# plt.draw()


# In[32]:


data.sort_values('SALE_PRICE').tail(1)


# In[33]:


data.shape


# In[34]:


corr = data.corr()
sns.heatmap(corr)


# In[35]:


corr['SALE_PRICE']


# In[36]:


from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(data_initialModel, test_size=0.3)
print("Training size: {}, Testing size: {}".format(len(df_train), len(df_test))) 
print("Samples: {} Features: {}".format(*df_train.shape))


# In[37]:


from sklearn import svm, feature_selection, linear_model 
df = data_initialModel.select_dtypes(include=[np.number]).copy() 


feature_cols = df.columns.values.tolist() 

feature_cols.remove('SALE_PRICE')






XO = df[feature_cols]
YO = df['SALE_PRICE']
estimator = svm.SVR(kernel="linear")
selector = feature_selection.RFE(estimator, 5, step=1) 

selector = selector.fit(XO, YO)
select_features = np.array(feature_cols)[selector.ranking_ == 1].tolist() 
print(select_features)


# In[38]:


X = df[select_features]
Y = df['SALE_PRICE']
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2) 
lm = linear_model.LinearRegression()
lm.fit(trainX, trainY)
# Inspect the calculated model equations
print("Y-axis intercept {:6.4f}".format(lm.intercept_)) 
print("Weight coefficients:")
for feat, coef in zip(select_features, lm.coef_):
    print(" {:>20}: {:6.4f}".format(feat, coef))
# The value of R^2
print("R squared for the training data is {:4.3f}".format(lm.score(trainX, trainY))) 
print("Score against test data: {:4.3f}".format(lm.score(testX, testY)))


# In[39]:


# data = data[data.SALE_PRICE > 0]


# In[40]:


# data = data[(data.RESIDENTIAL_UNITS + data.COMMERCIAL_UNITS) == data.TOTAL_UNITS]


# In[41]:


data.shape


# In[42]:


data.isnull().any()


# In[43]:


data['AGE_OF_HOUSE_AT_SALE'] = data['AGE_OF_HOUSE_AT_SALE'].fillna(statistics.mode(data['AGE_OF_HOUSE_AT_SALE']))


# In[44]:


data['NEIGHBORHOOD']= data['NEIGHBORHOOD'].fillna(method='bfill')
data['BUILDING CLASS CATEGORY']= data['BUILDING CLASS CATEGORY'].fillna(method='bfill')
data['TAX CLASS AT PRESENT']= data['TAX CLASS AT PRESENT'].fillna(method='bfill')
data['BUILDING CLASS AT PRESENT']= data['BUILDING CLASS AT PRESENT'].fillna(method='bfill')


# In[45]:


data = data[(data.RESIDENTIAL_UNITS + data.COMMERCIAL_UNITS) == data.TOTAL_UNITS]


# In[46]:


data=data.dropna(how='any')


# In[47]:


data.shape


# In[48]:


data.head()


# In[49]:


def replace_mean(df,col):
    df[col]= df[col].fillna(df[col].mean())
    
# replace_mean(data, 'COMMERCIAL_UNITS')
# replace_mean(data, 'RESIDENTIAL_UNITS')
# replace_mean(data, 'GROSS_SQUARE_FEET')
# replace_mean(data, 'LAND_SQUARE_FEET')


# In[50]:


plt.figure(figsize=(15,6))

sns.boxplot(x='SALE_PRICE', data=data)
plt.ticklabel_format(style='plain', axis='x')
plt.title('Boxplot of SALE PRICE in USD')
plt.show()


# In[51]:


# # # Remove observations that fall outside those caps
data = data[(data['SALE_PRICE'] < 10000000) ]
# data.corr()


# In[52]:


data.shape


# In[53]:


def remove_outlier(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower_bound  = q1 - (1.5  * iqr)
    upper_bound = q3 + (1.5 * iqr)
    out_df = df.loc[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return out_df

# data= remove_outlier(data,"SALE_PRICE")


# In[54]:


# plt.figure(figsize=(10,6))
# sns.boxplot(x='YEAR BUILT', data=data,showfliers=True)


# In[55]:


data.corr()


# In[56]:


data['SALE_PRICE'].skew()


# In[57]:


plt.figure(figsize=(15,6))

sns.boxplot(x='SALE_PRICE', data=data)
plt.ticklabel_format(style='plain', axis='x')
plt.title('Boxplot of SALE PRICE in USD')
plt.show()


# In[58]:


sns.distplot(data['SALE_PRICE'])


# In[59]:


# SALE PRICE is highly right skewed. So, we will log transform it so that it give better results.
data["ln_price"] = np.log(data["SALE_PRICE"])
data['ln_price'].skew()


# In[60]:


plt.figure(figsize=(15,6))

sns.boxplot(x='ln_price', data=data)
plt.ticklabel_format(style='plain', axis='x')
plt.title('Boxplot of SALE PRICE in USD')
plt.show()


# In[61]:


# corr['ln_price']
data.corr()


# In[62]:


# data= remove_outlier(data,"ln_price")
data = data[ (data['ln_price'] > 10)]
sns.distplot(data['ln_price']) 

# Well now we can see the symmetry and thus it is normalised.


# In[63]:


data.corr()


# In[64]:


data['ln_price'].skew()


# In[65]:


# data['YEAR BUILT'] = data['YEAR BUILT'].fillna(statistics.mode(data['YEAR BUILT']))
# data['TAX CLASS AT PRESENT'] = data['TAX CLASS AT PRESENT'].fillna(statistics.mode(data['TAX CLASS AT PRESENT']))


# In[66]:


# data['NEIGHBORHOOD']= data['NEIGHBORHOOD'].fillna(method='bfill')
# data['BUILDING CLASS CATEGORY']= data['BUILDING CLASS CATEGORY'].fillna(method='bfill')


# In[67]:


# data=data.dropna(how='any')


# In[68]:


data.shape


# In[69]:


corr = data.corr()
sns.heatmap(corr)


# In[70]:


plt.figure(figsize=(10,6))
sns.boxplot(x='AGE_OF_HOUSE_AT_SALE', data=data,showfliers=True)


# In[71]:


plt.figure(figsize=(30,15))
plt.scatter(x=data['AGE_OF_HOUSE_AT_SALE'],y=data['SALE_PRICE'])
ax =plt.gca()
ax.get_yaxis().get_major_formatter().set_scientific(False)
plt.draw()


# In[72]:


plt.figure(figsize=(15,6))

sns.boxplot(x='AGE_OF_HOUSE_AT_SALE', data=data)
plt.ticklabel_format(style='plain', axis='x')
plt.title('Boxplot of SALE PRICE in USD')
plt.show()


# In[73]:


# removing outliers in AGE_OF_HOUSE_AT_SALE
data = data[(data['AGE_OF_HOUSE_AT_SALE'] > 0) & (data['AGE_OF_HOUSE_AT_SALE'] < 2000)]


# In[74]:


# a boxplot showng age distribution
plt.figure(figsize=(15,6))

sns.boxplot(x='AGE_OF_HOUSE_AT_SALE', data=data)
plt.ticklabel_format(style='plain', axis='x')
plt.title('Boxplot of SALE PRICE in USD')
plt.show()


# In[75]:


corr['ln_price']


# In[76]:


data.shape


# In[77]:


data.isnull().any()


# In[78]:


# box plot showing the distribution of land square feet
plt.figure(figsize=(10,6))
sns.boxplot(x='LAND_SQUARE_FEET', data=data,showfliers=True)


# In[79]:


# scatter plot showing landsquare feet
plt.figure(figsize=(30,15))
plt.scatter(x=data['LAND_SQUARE_FEET'],y=data['SALE_PRICE'])
ax =plt.gca()
ax.get_yaxis().get_major_formatter().set_scientific(False)
plt.draw()


# In[80]:


print(data['LAND_SQUARE_FEET'].skew())


# In[81]:


data = data[(data['LAND_SQUARE_FEET'] < 20000) ]
# data= remove_outlier(data,"LAND_SQUARE_FEET")


# In[82]:


data.shape


# In[83]:


plt.figure(figsize=(30,15))
plt.scatter(x=data['LAND_SQUARE_FEET'],y=data['SALE_PRICE'])
ax =plt.gca()
ax.get_yaxis().get_major_formatter().set_scientific(False)
plt.draw()


# In[84]:


# data.drop(['ZIP CODE'], axis=1, inplace=True)
data.drop(['ADDRESS'], axis=1, inplace=True)
#Dropping few columns
data.drop(['BLOCK'], axis=1, inplace=True)
data.drop(['LOT'], axis=1, inplace=True)
data.drop(['ZIP CODE'], axis=1, inplace=True)


# In[85]:


print(data['LAND_SQUARE_FEET'].skew())


# In[86]:


data.shape


# In[87]:


plt.figure(figsize=(10,6))
sns.boxplot(x='GROSS_SQUARE_FEET', data=data,showfliers=True)


# In[88]:


plt.figure(figsize=(30,15))
plt.scatter(x=data['GROSS_SQUARE_FEET'],y=data['SALE_PRICE'])
ax =plt.gca()
ax.get_yaxis().get_major_formatter().set_scientific(False)
plt.draw()


# In[89]:


# data = data[(data['GROSS_SQUARE_FEET'] < 100000) ]


# In[90]:


plt.figure(figsize=(10,6))
sns.boxplot(x='TOTAL_UNITS', data=data,showfliers=True)


# In[91]:


data.shape


# In[92]:


data = data[(data['TOTAL_UNITS'] < 75) ]


# In[93]:


plt.figure(figsize=(10,6))
sns.boxplot(x='TOTAL_UNITS', data=data,showfliers=True)


# In[94]:


data.shape


# In[95]:


# data=data.dropna(how='any')
data.corr()


# In[96]:


one_hot_features = ['NEIGHBORHOOD','BUILDING CLASS CATEGORY','TAX CLASS AT PRESENT','TAX CLASS AT TIME OF SALE']

one_hot_encoded = pd.get_dummies(data[one_hot_features])
one_hot_encoded.info(verbose=True, memory_usage=True, null_counts=True)
data = data.drop(one_hot_features,axis=1)
data = pd.concat([data, one_hot_encoded] ,axis=1)


# In[97]:


# this method normalises the dataset
def normalize(df):
    num_cols = df.select_dtypes(include=[np.number]).copy() 
    num_cols.drop('SALE_PRICE', axis='columns', inplace=True) # drop the price column 
    df_norm = ((num_cols-num_cols.min())/(num_cols.max()-num_cols.min()))
    return df_norm

data_norm = normalize(data) 
data_norm.describe()


# In[98]:


data_norm.head()


# In[99]:


c= data.corr()
print(c['SALE_PRICE'].sort_values(ascending=False).head(30))


# In[101]:


corr = data.corr()


# In[102]:


corr['ln_price']


# In[103]:



# from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(data_norm, test_size=0.3)
print("Training size: {}, Testing size: {}".format(len(df_train), len(df_test))) 
print("Samples: {} Features: {}".format(*df_train.shape))


# In[104]:


# from sklearn import svm, feature_selection, linear_model 
df = data_norm.select_dtypes(include=[np.number]).copy() 
feature_cols = df.columns.values.tolist() 
feature_cols.remove('ln_price')
XO = df[feature_cols]
YO = df['ln_price']
estimator = svm.SVR(kernel="linear")
selector = feature_selection.RFE(estimator, 30, step=1) 
selector = selector.fit(XO, YO)
select_features = np.array(feature_cols)[selector.ranking_ == 1].tolist() 
print(select_features)



# In[105]:


X = df[select_features]
Y = df['ln_price']
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2) 
lm = linear_model.LinearRegression()
lm.fit(trainX, trainY)
# Inspect the calculated model equations
print("Y-axis intercept {:6.4f}".format(lm.intercept_)) 
print("Weight coefficients:")
for feat, coef in zip(select_features, lm.coef_):
    print(" {:>20}: {:6.4f}".format(feat, coef))
# The value of R^2
print("R squared for the training data is {:4.3f}".format(lm.score(trainX, trainY))) 
print("Score against test data: {:4.3f}".format(lm.score(testX, testY)))


# In[106]:


data.shape


# In[107]:


# A method to calculate mean squared error
def mse(df, pred, obs):
    n = df.shape[0]
    return sum((df[pred]-df[obs])**2)/n
data_norm['pred'] = lm.predict(X)
print("Mean Squared error: {}".format(mse(data_norm,'pred', 'ln_price')))


# In[108]:


# a graph showing actual vs predicted values
import matplotlib.pylab as plb
pred_trainY = lm.predict(trainX)
plt.figure(figsize=(14, 8))
plt.plot(trainY, pred_trainY, 'o')  
plb.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title="Plot of predicted vs actual prices" 
plt.show()


# In[ ]:


 

