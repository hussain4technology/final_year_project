#!/usr/bin/env python
# coding: utf-8

# In[138]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from matplotlib import pyplot
import numpy as np
import pandas as pd
import math



# data uploaded form csv file
file1=pd.read_csv('/home/hannu/Documents/wti-daily_csv.csv', parse_dates=True)
file2=pd.read_csv('/home/hannu/Documents/brent-daily_csv.csv', parse_dates=True)
wti_set=pd.read_csv('/home/hannu/Music/wti-daily_csv.csv', parse_dates=True)
brent_set=pd.read_csv('/home/hannu/Music/brent-daily_csv.csv', parse_dates=True)


# data modelling according to the algorithm
file1['lagged_returns']=np.log((file1['y'])/(wti_set['y']))
file2['lagged_returns']=np.log((file2['y'])/(brent_set['y']))




# data is divided into traing and testing dataframes
wti_train_set=file1[0:200]
wti_test_set=file1[200:220]
brent_train_set=file2[0:200]
brent_test_set=file2[200:220]

#plotting timeseries data of brent and wti daily prices
file1_lag=file1[['x','lagged_returns']]
file2_lag=file2[['x','lagged_returns']]


wti_set.plot()
pyplot.title("wti crude-oil prices")
pyplot.xlabel('t')
pyplot.ylabel('Pt') #resolve the subscript problem
pyplot.show() #wti crude-oil prices

brent_set.plot()
pyplot.title("brent crude-oil prices")
pyplot.xlabel('t')
pyplot.ylabel('Pt')#resolve the subscript problem
pyplot.show() #brent-crude oil prices


file1_lag.plot(color="y")
pyplot.ylim(-0.1,0.1)
pyplot.title("wti crude-oil prices")
pyplot.xlabel('t')
pyplot.ylabel('lagged_returns') #resolve the subscript problem
pyplot.show() #wti crude-oil prices


file2_lag.plot(color="y")
pyplot.ylim(-0.1,0.1)
pyplot.title("brent crude-oil prices")
pyplot.xlabel('t')
pyplot.ylabel('lagged_returns') #resolve the subscript problem
pyplot.show() #brent crude-oil prices


m=[4,5,6]
h=[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
n=[10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
from scipy.stats import kurtosis
from scipy.stats import skew

wti_obj={"mean":file1['lagged_returns'].mean(),"max":file1['lagged_returns'].max(),"min":file1['lagged_returns'].min(),"stdev":np.std(file1['lagged_returns']),"skewness":format( skew(file1['lagged_returns']) ),"kurtosis":format( kurtosis(file1['lagged_returns']) )}


brent_obj={"mean":file2['lagged_returns'].mean(),"max":file2['lagged_returns'].max(),"min":file2['lagged_returns'].min(),"stdev":file2['lagged_returns'].std(),"skewness":file2['lagged_returns'].skew(),"kurtosis":format( kurtosis(file2['lagged_returns']) )}

print("wti data")
print(wti_obj)
print("brent data")
print(brent_obj)

s=0
def defuzzifier (file):
    a=[]
    for index, row in file.iterrows():
        if (row['lagged_returns']>0.000000):
                a.append(1)
                s=1
        elif(row['lagged_returns']<0.000000):
                a.append(0)
                s=0
        else:
                a.append(0)
                    
    return a    

count=0

for key,value in file1['lagged_returns'].iteritems():
    if(value==0.0):
       file1['lagged_returns'][key]=0.0
       count=count+1


        
file1['bool_lag']=defuzzifier(file1)
      
print(file1['bool_lag'])       
            
                


# In[141]:





for i in range(len(m)):
    z=file1.iloc[:,2].values.tolist()
    
    j=1
    feature_set=pd.DataFrame()
    print('M')
    print(m[i])
    while(m[i]-j+1>=1):
        z.pop()
        z.insert(0,0.000526)
        k="m"+str(m[i]-j+1)
        feature_set[k]=z
        j=j+1
    
    #model training and predictions occur here
    y=file1.iloc[:,3].values.tolist()
    

    x=feature_set[:][feature_set.columns[::-1]].values.tolist()
    from sklearn.model_selection import train_test_split 
    xtrain, xtest, ytrain, ytest = train_test_split( 
            x, y, test_size = 0.09, random_state = 0) 
    
    from sklearn.preprocessing import StandardScaler 
    sc_x = StandardScaler() 
    xtrain = sc_x.fit_transform(xtrain)  
    xtest = sc_x.transform(xtest) 
 
    
    
    
    #print(ytrain)
    
    
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0).fit(xtrain,ytrain)
     #classifier.fit(xtrain, ytrain) 

    y_pred = classifier.predict(xtest)
    Rt=y_pred.tolist()
    r=ytest
  

    
    del feature_set
    

    from sklearn.metrics import confusion_matrix 
    cm = confusion_matrix(r, Rt) 
  
    print ("Confusion Matrix : \n", cm) 

    
    
    from sklearn.metrics import accuracy_score 
    print ("Accuracy : ", accuracy_score(r, Rt)) 


    
    print(Rt)
    del Rt
    print(r)
    del r
    del xtrain
    del xtest
    del ytrain
    #print(ytest)
    del ytest
    #print(y_pred)
    del y_pred
    del z
    
    
    


# In[ ]:





# In[ ]:




