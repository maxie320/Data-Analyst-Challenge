#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing libraries
import pandas as pd
import numpy as np
import swifter
import operator, functools
import ast
import matplotlib.pyplot as plt
import sklearn.cluster as cluster
import sklearn.metrics as metrics
import sklearn.datasets
import sklearn.tree
import collections
import sklearn.tree as tree
import sys
from IPython.display import Image
import seaborn as sns
######TASK 1########
#read data
TakeHomeData = pd.read_csv('takeHome.csv')
#check missing values - sanity check
print(TakeHomeData.isna().sum())
#check dtypes values - sanity check
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
print(df)
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
data_num = df[['ltv_diff','ttime_diff','tue_diff']]
## Scale the data, using pandas
def scale(x):
    return (x-np.mean(x))/np.std(x)
data_scaled=data_num.apply(scale,axis=0)
print(data_scaled.head())
print (data_scaled)
print ("Type of output is "+str(type(data_scaled)))
print ("Shape of the object is "+str(data_scaled.shape))
kmeans=cluster.KMeans(n_clusters=3,init="k-means++")
kmeans=kmeans.fit(data_scaled)
kmeans.labels_
kmeans.cluster_centers_
## Elbow method
from scipy.spatial.distance import cdist
K=range(1,20)
wss = []
for k in K:
    kmeans = cluster.KMeans(n_clusters=k,init="k-means++")
    kmeans.fit(data_scaled)
    wss.append(sum(np.min(cdist(data_scaled, kmeans.cluster_centers_, 'euclidean'), 
                                      axis=1)) / data_scaled.shape[0])
plt.plot(K, wss, 'bx')
plt.xlabel('k')
plt.ylabel('Average distortion')
plt.title('Selecting k with the Elbow Method')
print(plt.show())
for i in range(2,20):
    labels=cluster.KMeans(n_clusters=i,random_state=200).fit(data_scaled).labels_
    print ("Silhoutte score for k= "+str(i)+" is "+str(metrics.silhouette_score(data_scaled,labels,metric="euclidean",
                                 sample_size=1000,random_state=200)))
labels=cluster.KMeans(n_clusters=3,random_state=200).fit(data_scaled).labels_
print(metrics.silhouette_score(data_scaled,labels,metric="euclidean",sample_size=1000,random_state=200))
sys.path.append('/Users/shubhambharadwaj/Desktop/InterView Deck/Python&DLAI/K means/Code -K-Means/')
import cluster_profiles as cluster_profiles
kmeans=cluster.KMeans(n_clusters=3,random_state=200).fit(data_scaled)
print(cluster_profiles.get_zprofiles(data=data_num.copy(),kmeans=kmeans))
print(cluster_profiles.get_profiles(data=data_num.copy(),kmeans=kmeans))
plt.figure(figsize=(12,8))
sns.scatterplot(x='ltv_diff', y='ttime_diff', data=data_num, hue=kmeans.labels_, s=100)
print(plt.show())
plt.figure(figsize=(12,8))
sns.scatterplot(x='ttime_diff', y='tue_diff', data=data_num, hue=kmeans.labels_, s=100)
print(plt.show())
plt.figure(figsize=(12,8))
sns.scatterplot(x='ltv_diff', y='tue_diff', data=data_num, hue=kmeans.labels_, s=100)
print(plt.show())
econs = pd.read_csv('econs.txt',delimiter='\t')
print(econs)
econs_grouped = econs.groupby('econ',as_index = False).agg({'cumGXP':["max","min"],'cumPXP':["max","min"],'cumTime':"mean",'cumUnits':"mean"})
print(econs_grouped.columns)
cols = []
for x in econs_grouped.columns:
    cols.append(f'{x[0]}_{x[1]}')
econs_grouped.columns = cols
print(econs_grouped)
dict_ = {}
for x in range(1,7):
    dict_[x] = 'Control'
for y in range(55,61):
    dict_[y] = 'Control'
for z in range(7,15):
    dict_[z] = 'New Economy 1'
for z in range(19,35):
    dict_[z] = 'New Economy 1'
for e in range(15,17):
    dict_[e] = 'New Economy 2'
for p in range(35,45):
    dict_[p] = 'New Economy 2'
for l in range(17,19):
    dict_[l] = 'New Economy 3'
for o in range(45,55):
    dict_[o] = 'New Economy 3'
TakeHomeData_clean['abt_EcoGroup'] = TakeHomeData_clean['abt'].map(dict_)
X = TakeHomeData_clean.drop(['abt','abt_EcoGroup','tue'],axis = 1)
y = TakeHomeData_clean['tue']
import sklearn.model_selection as model_selection
X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.2,random_state=200)
reg=tree.DecisionTreeRegressor(max_depth=3)
reg.fit(X_train,y_train)
print(reg.score(X_test,y_test))
reg.feature_importances_
pd.Series(reg.feature_importances_,index=X.columns).sort_values(ascending=False).head(5)
import pydotplus
import graphviz
dot_data = tree.export_graphviz(reg, out_file=None, 
                         feature_names=X.columns,  
                         filled=True, rounded=True,  
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data)
######DT FOR MODEL 2 ############
######TASK 2########
dec_paths = reg.decision_path(X_train)
samples = collections.defaultdict(list)
for d, dec in enumerate(dec_paths):
    for i in range(reg.tree_.node_count):
        if dec.toarray()[0][i] == 1:
            samples[i].append(d)
##Estimating two sample decision cases with samples 1272 and 206
print("Decision cases 1",len(set(samples[4])))
print("Decision cases 2",len(set(samples[11])))
print("Common Cases:",set(samples[4]).intersection(set(samples[11])))
tue_min_1 = pd.DataFrame(samples[4]).rename(columns = {0:'indextue_1'})
tue_min_2 = pd.DataFrame(samples[11]).rename(columns = {0:'indextue_2'})
Image(graph.create_png())
TakeHomeData_clean = TakeHomeData_clean.reset_index()
m = pd.merge(TakeHomeData_clean,tue_min_1, left_on = 'index',right_on = 'indextue_1', how = 'left')
v = pd.merge(TakeHomeData_clean,tue_min_2, left_on = 'index',right_on = 'indextue_2', how = 'left')
mvp = pd.merge(m,v,on = ['index','ltv','ttime','abt','tue','abt_EcoGroup'], how = 'inner')
mvp_new = mvp[(mvp['indextue_1'].isna() == False)|(mvp['indextue_2'].isna() == False)]
mvp_new = mvp_new.drop(['index','indextue_1','indextue_2'],axis = 1).reset_index(drop = True)
mvp_group_1 = pd.DataFrame(mvp_new.groupby('abt_EcoGroup').apply(lambda x: ','.join(map(str,x.abt)))).reset_index().rename(columns = {0:'ab_groups'})
mvp_group_2 = mvp_new.groupby('abt_EcoGroup',as_index = False).agg({'ltv':"mean",
                                                      'ttime':"mean",
                                                      'tue':"mean"})
mvp_group = pd.merge(mvp_group_1,mvp_group_2,on = 'abt_EcoGroup',how = 'inner')
for index in mvp_group.index:
    mvp_group['ab_groups'][index] = list(set(mvp_group['ab_groups'][index].split(',')))
print("As observed, the two target decision nodes significantly demonstrate decrease in tue in various ab groups across different economies")
print(mvp_group)
print(econs_grouped)
sts = pd.read_csv('sts.csv')
print("sts null stats",sts.isna().sum())
sts['state'] = sts['state'].swifter.apply(lambda x: ast.literal_eval(x))
state = pd.DataFrame(get_values(sts['state'])).rename(columns = {0:'state_clean'})
sts_new = pd.concat([sts,state],axis = 1).drop('state',axis = 1)
sts_new['sts'] = sts_new['sts'].swifter.apply(lambda x: ast.literal_eval(x))
b = pd.DataFrame()
for index in list(sts_new.index):
    a = pd.DataFrame(sts_new['sts'][index]['m'])
    b = pd.concat([b,a],axis = 0)

