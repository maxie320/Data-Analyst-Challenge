#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing libraries
import pandas as pd
import numpy as np
import swifter
import operator, functools
import ast
import matplotlib.pyplot as plt
######TASK 1########
#read data
TakeHomeData = pd.read_csv('takeHome.csv')
#check missing values - sanity check
print(TakeHomeData.isna().sum())
print(TakeHomeData.dtypes)
#clean values - sanity check
TakeHomeData= TakeHomeData.fillna('{"n":"0"}')
TakeHomeData['ltv'] = TakeHomeData['ltv'].swifter.apply(lambda x: ast.literal_eval(x))
TakeHomeData['ttime'] = TakeHomeData['ttime'].swifter.apply(lambda x: ast.literal_eval(x))
TakeHomeData['abt'] = TakeHomeData['abt'].swifter.apply(lambda x: ast.literal_eval(x))
TakeHomeData['tue'] = TakeHomeData['tue'].swifter.apply(lambda x: ast.literal_eval(x))
#extract data
get_values = functools.partial(map, operator.itemgetter('n'))
a = pd.DataFrame(get_values(TakeHomeData['ltv'])).rename(columns = {0:'ltv'})
b = pd.DataFrame(get_values(TakeHomeData['ttime'])).rename(columns = {0:'ttime'})
c = pd.DataFrame(get_values(TakeHomeData['abt'])).rename(columns = {0:'abt'})
d = pd.DataFrame(get_values(TakeHomeData['tue'])).rename(columns = {0:'tue'})
#sum up the frames and assign a singular data type
TakeHomeData_clean = pd.concat([a,b,c,d],axis = 1)
TakeHomeData_clean.iloc[:,0:4] = TakeHomeData_clean.iloc[:,0:4].astype(float)
#make pivot to aggregate across ab groups
q_pivot = pd.pivot_table(TakeHomeData_clean,values=['ltv','ttime','tue'],columns = ['abt'],aggfunc=sum)
#make transpose pivot
q_pivot = q_pivot.transpose().reset_index()
#compute difference
d = {}
diff = {}
map_dict = {}
j = 0
for n in range(0,max(q_pivot.index)+1):
    d[n] = pd.DataFrame(q_pivot.iloc[:, :].values - q_pivot.iloc[n:n+1, :].values)
    diff[n] = d[n][d[n][0] > 0]
    if n == 0:
        for x in range(min(list(diff[n][0])),max(list(diff[n][0]))+1):
            map_dict[x] = f'1-{x+1}'
        j += 1
    if j == 59:
        break
    if n ==j:
        for x in range(min(list(diff[n][0])),max(list(diff[n][0]))+1):
            map_dict[x] = f'{j+1}-{x+j+1}'
        j += 1
    diff[n][0] = diff[n][0].map(map_dict)
    map_dict = {}
#concat difference dataframes
df = pd.DataFrame()
for y in range(0,max(q_pivot.index)):
    df = pd.concat([df,diff[y]],axis = 0)
#renaming dataframes
df = df.rename(columns = {0:'abt_diff',1:'ltv_diff',2:'ttime_diff',3:'tue_diff'})
#taking absolute values
df[['ltv_diff','ttime_diff','tue_diff']] = df[['ltv_diff','ttime_diff','tue_diff']].abs()
#tue_diff is in positive correlation with ttime_diff
print(df.plot(kind='scatter',x='ttime_diff',y='tue_diff',color='green'))
#ltv_diff is in positive correlation with ttime_diff
print(df.plot(kind='scatter',x='ltv_diff',y='ttime_diff',color='blue'))
#ltv_diff and tue_diff are positively correlated but not strongly correlated
print(df.plot(kind='scatter',x='ltv_diff',y='tue_diff',color='red'))
#data points are evenly distributed across all groups
print(df.plot(kind='bar',x='abt_diff',y=['ltv_diff']))
#data points are skewed for tue_diff in first 20 categories
print(df.plot(kind='bar',x='abt_diff',y=['tue_diff']))
#data points are stale for ttime_diff in first 20 categories
print(df.plot(kind='bar',x='abt_diff',y=['ttime_diff']))
######TASK 2########
econs = pd.read_csv('econs.txt',delimiter='\t')
sts = pd.read_csv('sts.csv')
print(sts.isna().sum())
sts['state'] = sts['state'].swifter.apply(lambda x: ast.literal_eval(x))
state = pd.DataFrame(get_values(sts['state'])).rename(columns = {0:'state_clean'})
sts_new = pd.concat([sts,state],axis = 1).drop('state',axis = 1)
sts_new['sts'] = sts_new['sts'].swifter.apply(lambda x: ast.literal_eval(x))

