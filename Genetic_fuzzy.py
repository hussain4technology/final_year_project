#!/usr/bin/env python
# coding: utf-8

# In[17]:


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
                a.append(s)
                    
    return a    

count=0

for key,value in file1['lagged_returns'].iteritems():
    if(value==0.0):
       file1['lagged_returns'][key]=0.0
       count=count+1


        
file1['bool_lag']=defuzzifier(file1)
file2['bool_lag']=defuzzifier(file2)


index=0
#for i in range(len(m)):
z=file1.iloc[:,2].values.tolist()
j=1
feature_set=pd.DataFrame()
print('M')
#print(m[i])
print()
print()
print()
print()
while(4-j+1>=1):
    z.pop()
    z.insert(0,0.000526)
    k="m"+str(4-j+1)
    feature_set[k]=z
    j=j+1

#model training and predictions occur here
y=file1.iloc[:,2].values.tolist()
x=feature_set[:][feature_set.columns[::-1]].values.tolist()

xtrain=feature_set[0:201][feature_set.columns[::-1]].values.tolist()
xtest=feature_set[201:221][feature_set.columns[::-1]].values.tolist()
ytrain=file1.iloc[0:201,2].values.tolist()
ytest=file1.iloc[201:221,2].values.tolist()



V=[]

for i1 in range(n[17]):
    V.append(-0.05251106859673565 + (i1)*((0.07334138704823652+0.05251106859673565)/(n[17]-1)))



one_d_op_distribution=[[0 for i_col in range(n[17])] for j_row in range(len(ytrain))] 


delta_m=((0.07334138704823652+0.05251106859673565)/(n[17]-1))
for i1 in range(n[17]):
    for j1 in range(len(ytrain)):
        e=ytrain[j1]-V[i1]
        if(abs(e)<=delta_m):
            one_d_op_distribution[j1][i1]=(1-(abs(e)/delta_m))
        else:
            one_d_op_distribution[j1][i1]=0






#print(one_d_op_distribution)

U=[[0 for i1_col in range(4)] for j1_row in range(h[0])]

for k2 in range(4):
    for j2 in range(h[0]):
        U[j2][3-k2]=(min(feature_set.iloc[0:201,k2]) + (j2)*((max(feature_set.iloc[0:201,k2])-min(feature_set.iloc[0:201,k2]))/(h[0]-1)))
        
        
    

one_d_ip_distribution=np.zeros(len(ytrain)*4*h[0])
one_d_ip_distribution=one_d_ip_distribution.reshape(4,len(ytrain),h[0])




for i3 in range(4):
   delta_i=((max(feature_set.iloc[0:201,i3])-min(feature_set.iloc[0:201,i3]))/(h[0]-1))                
   for k3 in range(h[0]):
        for j3 in range(len(ytrain)):
            e=xtrain[j3][i3]-U[k3][i3]
            if(abs(e)<delta_i):
                g=one_d_ip_distribution[i3][j3][k3]=(1-(abs(e)/delta_i))
            else:
                one_d_ip_distribution[i3][j3][k3]=0



#print(one_d_ip_distribution)

multi_d_coding=pd.DataFrame()

multi_d_if_distribution=[[0 for i4_col in range(n[17])] for j4_row in range(10000)]

#print(multi_d_if_distribution)



indexs=[]
H=h[0]



for K in range(n[17]):
    index_m=0
    for i4 in range(h[0]):
        for j4 in range(h[0]):
            for k4 in range(h[0]):
                for l4 in range(h[0]):
                    indexs.append((l4)*pow(H,0)+(k4)*pow(H,1)+(j4)*pow(H,2)+(i4)*pow(H,3))
                    input_control=[l4,k4,j4,i4]
                    agg=0
                    for t in range(len(ytrain)):
                        prod=1
                        for I4 in range(4):
                            prod=prod*one_d_ip_distribution[I4][t][input_control[I4]]
                        
                        agg=agg+one_d_op_distribution[t][K]*prod
                        
                    
                    multi_d_if_distribution[index_m][K]=agg
                    index_m=index_m+1
                    

                                    
multi_d_coding['indexs']=indexs
print(multi_d_if_distribution)

del one_d_ip_distribution



del one_d_op_distribution

del U

del V

del x
del y


# In[74]:


from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np
import itertools


#mpl.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')

def cc(arg):
    return mcolors.to_rgba(arg, alpha=0.6)

zs=[]

verts=[[0 for l in range(2)] for k in range(135000)]

verts=np.array(verts)

count=0
for i in range(5000):
    for j in range(27):
        verts[count][0]=i
        verts[count][1]=j
        zs.append(multi_d_if_distribution[i][j])
        count=count+1
        


poly = PolyCollection(verts,facecolors=[cc('k')])

poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=zs, zdir='y')

ax.set_xlabel('X')
ax.set_xlim3d(0, 5000)
ax.set_ylabel('Y')
ax.set_ylim3d(0,26)
ax.set_zlabel('Z')
ax.set_zlim3d(0, 0.4)

plt.show()
        
        


# In[ ]:




