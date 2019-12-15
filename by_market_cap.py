# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:51:00 2019

按市值分类

@author: yuxiang
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from time import time
import os
import gc
%matplotlib inline

daily = pd.read_csv('daily/201809StockDailyData.csv')

daily['InstrumentId'] = daily['WindCode'].apply(lambda x:x[:6])
daily_SH = daily[daily['WindCode'].apply(lambda x:True if 'SH' in x else False)]
today = daily_SH[daily_SH['TradingDay']==20180903]
large_cap = today[today['MarketCapAFloat']>1e10]
small_cap = today[today['MarketCapAFloat']<3e9]

large_cap_list = list(large_cap['InstrumentId'])
small_cap_list = list(small_cap['InstrumentId'])
print(len(large_cap_list),len(small_cap_list))

fac_path = 'fac_1m/20180903/'
l = os.listdir(fac_path)
large = []
small = []
for i in range(len(l)): # i=0;
    file = l[i]
    if file[:6] in large_cap_list:
        large.append(pd.read_csv(fac_path+file,index_col=0))
    elif file[:6] in small_cap_list:
        small.append(pd.read_csv(fac_path+file,index_col=0))

print(len(large),len(small))

fac_large = pd.concat(large)
fac_large['datetime']=pd.to_datetime(fac_large['datetime'],format='%Y-%m-%d %H:%M:%S')
fac_large.reset_index(drop=True,inplace=True)
fac_small = pd.concat(small)
fac_small['datetime']=pd.to_datetime(fac_small['datetime'],format='%Y-%m-%d %H:%M:%S')
fac_small.reset_index(drop=True,inplace=True)

print(list(fac_large.columns))

def plot(temp,factor,log=False,col=None):
    if log:
        temp = np.log(temp)
    plt.axhline(y=0,color='r')
    #plt.axhline(y=0.001,color='r')
    #plt.axhline(y=-0.001,color='r')
    plt.axvline(x=0,color='r')
    plt.scatter(temp,factor['ret_vwapmid_fur'],linewidths=0.001)
    if col:
        plt.title(col)
    plt.xlim(min(temp)*0.9,max(temp)*1.1)
    plt.ylim(min(factor['ret_vwapmid_fur'])-0.0005,max(factor['ret_vwapmid_fur'])+0.0005)
    #plt.plot([-10000,10000],[0,0],color='r')
    plt.show()

print((fac_large['resist_buy_lag']>0).sum(),len(fac_large),(fac_large['resist_buy_lag']>0).sum()/len(fac_large))

temp_large = fac_large[fac_large['resist_buy_lag']>0]
plot(temp_large['resist_buy_lag'],temp_large,1)
for i in range(7):
    temp_c = temp_large[np.log(temp_large['resist_buy_lag'])>i]
    print(i,len(temp_c[temp_c['ret_vwapmid_fur']>0])/len(temp_c))


temp_small = fac_small[fac_small['resist_buy_lag']>0]
plot(temp_small['resist_buy_lag'],temp_small,1)
for i in range(7):
    temp_c = temp_small[np.log(temp_small['resist_buy_lag'])>i]
    print(i,len(temp_c[temp_c['ret_vwapmid_fur']>0.00])/len(temp_c))

#--------
temp_large = fac_large[fac_large['resist_sell_lag']>0]
plot(temp_large['resist_sell_lag'],temp_large,1)
temp_c = temp_large[np.log(temp_large['resist_sell_lag'])<-5]
len(temp_c[temp_c['ret_vwapmid_fur']>0.002])/len(temp_c)

temp_small = fac_small[fac_small['resist_sell_lag']>0]
plot(temp_small['resist_sell_lag'],temp_small,1)
for i in range(7):
    temp_c = temp_small[np.log(temp_small['resist_sell_lag'])<-i]
    print(-i,len(temp_c[temp_c['ret_vwapmid_fur']>0])/len(temp_c))

#-----------------------
plot(fac_small['resist_sell_lag'],fac_small)
temp_small2 = fac_small[fac_small['resist_sell_lag']<0]
plot(abs(temp_small2['resist_sell_lag']),temp_small2,1)
for i in range(7):
    temp_c = temp_small2[np.log(abs(temp_small2['resist_sell_lag']))<-i]
    print(-i,len(temp_c[temp_c['ret_vwapmid_fur']>0])/len(temp_c))

print(len(fac_large[fac_large['ret_vwapmid_fur']>0.002])/len(fac_large))

print(len(fac_small[fac_small['ret_vwapmid_fur']>0.002])/len(fac_small))


(fac_large['resist_buy_lag']==np.inf).sum()
gc.collect()

# ------------------------
d={}
############### large, sell
#large_noinf = fac_large[(fac_large['resist_sell_lag']>-np.inf) & (fac_large['resist_sell_lag']<np.inf)]
##large_noinf['resist_sell_lag'].hist(bins=1000)
#large_noinf = fac_large[(fac_large['resist_sell_lag']>0) & (fac_large['resist_sell_lag']<np.inf)]
#np.log(large_noinf['resist_sell_lag']).hist()
#large_noinf['ret_vwapmid_fur'].hist(bins=50)

large_noinf = fac_large[(fac_large['resist_sell_lag']>0) & (fac_large['resist_sell_lag']<1)]
#np.log(large_noinf['resist_sell_lag']).hist()
#large_noinf['ret_vwapmid_fur'].hist(bins=50)

for i in range(7):
    temp_large_noinf = large_noinf[np.log(large_noinf['resist_sell_lag'])<-i]
    temp_large_noinf['ret_vwapmid_fur'].hist(bins=100)
    plt.show()
    print('''%d
          median: %f
          mean:   %f
          std:    %f
          >0 pct: %f''' % (-i,temp_large_noinf['ret_vwapmid_fur'].median(),
          temp_large_noinf['ret_vwapmid_fur'].mean(),
          temp_large_noinf['ret_vwapmid_fur'].std(),
          (temp_large_noinf['ret_vwapmid_fur']>0).sum()/len(temp_large_noinf)))
    
    d[(-i,'large,sell')]=dict(zip(('median','mean','std','>0 pct','num pct'),(temp_large_noinf['ret_vwapmid_fur'].median(),
          temp_large_noinf['ret_vwapmid_fur'].mean(),
          temp_large_noinf['ret_vwapmid_fur'].std(),
          (temp_large_noinf['ret_vwapmid_fur']>0).sum()/len(temp_large_noinf),
          len(temp_large_noinf)/len(large_noinf))))

large_noinf = fac_large[(fac_large['resist_sell_lag']>-np.inf) & (fac_large['resist_sell_lag']<-100)]
large_noinf['ret_vwapmid_fur'].hist(bins=150)
large_noinf['resist_sell_lag'].hist()

############### small,sell
#small_noinf = fac_small[(fac_small['resist_sell_lag']>-np.inf) & (fac_small['resist_sell_lag']<np.inf)]
##small_noinf['resist_sell_lag'].hist(bins=1000)
#small_noinf = fac_small[(fac_small['resist_sell_lag']>0) & (fac_small['resist_sell_lag']<np.inf)]
#np.log(small_noinf['resist_sell_lag']).hist()
#small_noinf['ret_vwapmid_fur'].hist(bins=150)

small_noinf = fac_small[(fac_small['resist_sell_lag']>0) & (fac_small['resist_sell_lag']<1)]
#np.log(small_noinf['resist_sell_lag']).hist()
#small_noinf['ret_vwapmid_fur'].hist(bins=150)

for i in range(7):
    temp_small_noinf = small_noinf[np.log(small_noinf['resist_sell_lag'])<-i]
    temp_small_noinf['ret_vwapmid_fur'].hist(bins=100)
    plt.show()
    print('''%d
          median: %f
          mean:   %f
          std:    %f
          >0 pct: %f''' % (-i,temp_small_noinf['ret_vwapmid_fur'].median(),
          temp_small_noinf['ret_vwapmid_fur'].mean(),
          temp_small_noinf['ret_vwapmid_fur'].std(),
          (temp_small_noinf['ret_vwapmid_fur']>0).sum()/len(temp_small_noinf)))
    
    d[(-i,'small,sell')]=dict(zip(('median','mean','std','>0 pct','num pct'),(temp_small_noinf['ret_vwapmid_fur'].median(),
          temp_small_noinf['ret_vwapmid_fur'].mean(),
          temp_small_noinf['ret_vwapmid_fur'].std(),
          (temp_small_noinf['ret_vwapmid_fur']>0).sum()/len(temp_small_noinf),
          len(temp_small_noinf)/len(small_noinf))))


############### small,buy
#small_noinf = fac_small[(fac_small['resist_buy_lag']>-np.inf) & (fac_small['resist_buy_lag']<np.inf)]
##small_noinf['resist_buy_lag'].hist(bins=1000)
#small_noinf = fac_small[(fac_small['resist_buy_lag']>0) & (fac_small['resist_buy_lag']<np.inf)]
#np.log(small_noinf['resist_buy_lag']).hist()
#small_noinf['ret_vwapmid_fur'].hist(bins=150)

small_noinf = fac_small[(fac_small['resist_buy_lag']>1) & (fac_small['resist_buy_lag']<np.inf)]
#np.log(small_noinf['resist_buy_lag']).hist()
#small_noinf['ret_vwapmid_fur'].hist(bins=150)

for i in range(7):
    temp_small_noinf = small_noinf[np.log(small_noinf['resist_buy_lag'])>i]
    temp_small_noinf['ret_vwapmid_fur'].hist(bins=100)
    plt.show()
    print('''%d
          median: %f
          mean:   %f
          std:    %f
          >0 pct: %f''' % (i,temp_small_noinf['ret_vwapmid_fur'].median(),
          temp_small_noinf['ret_vwapmid_fur'].mean(),
          temp_small_noinf['ret_vwapmid_fur'].std(),
          (temp_small_noinf['ret_vwapmid_fur']>0).sum()/len(temp_small_noinf)))
    
    d[(i,'small,buy')]=dict(zip(('median','mean','std','>0 pct','num pct'),(temp_small_noinf['ret_vwapmid_fur'].median(),
          temp_small_noinf['ret_vwapmid_fur'].mean(),
          temp_small_noinf['ret_vwapmid_fur'].std(),
          (temp_small_noinf['ret_vwapmid_fur']>0).sum()/len(temp_small_noinf),
          len(temp_small_noinf)/len(small_noinf))))


############### large,buy
#large_noinf = fac_large[(fac_large['resist_buy_lag']>-np.inf) & (fac_large['resist_buy_lag']<np.inf)]
##large_noinf['resist_buy_lag'].hist(bins=1000)
#large_noinf = fac_large[(fac_large['resist_buy_lag']>0) & (fac_large['resist_buy_lag']<np.inf)]
#np.log(large_noinf['resist_buy_lag']).hist()
#large_noinf['ret_vwapmid_fur'].hist(bins=150)

large_noinf = fac_large[(fac_large['resist_buy_lag']>1) & (fac_large['resist_buy_lag']<np.inf)]
#np.log(large_noinf['resist_buy_lag']).hist()
#large_noinf['ret_vwapmid_fur'].hist(bins=150)

for i in range(7):
    temp_large_noinf = large_noinf[np.log(large_noinf['resist_buy_lag'])>i]
    temp_large_noinf['ret_vwapmid_fur'].hist(bins=100)
    plt.show()
    print('''%d
          median: %f
          mean:   %f
          std:    %f
          >0 pct: %f''' % (i,temp_large_noinf['ret_vwapmid_fur'].median(),
          temp_large_noinf['ret_vwapmid_fur'].mean(),
          temp_large_noinf['ret_vwapmid_fur'].std(),
          (temp_large_noinf['ret_vwapmid_fur']>0).sum()/len(temp_large_noinf)))
    
    d[(i,'large,buy')]=dict(zip(('median','mean','std','>0 pct','num pct'),(temp_large_noinf['ret_vwapmid_fur'].median(),
          temp_large_noinf['ret_vwapmid_fur'].mean(),
          temp_large_noinf['ret_vwapmid_fur'].std(),
          (temp_large_noinf['ret_vwapmid_fur']>0).sum()/len(temp_large_noinf),
          len(temp_large_noinf)/len(large_noinf))))

##########
summary = pd.DataFrame.from_dict(d,orient='index')

temp = summary.sort_index()
temp.index = pd.MultiIndex.from_tuples(temp.index)
temp.sort_index(level=1,inplace=True)
# -----------
plot(fac_large['incre_sell_lag'],fac_large)
plot(fac_large['incre_buy_lag'],fac_large)
plot(fac_small['incre_sell_lag'],fac_small)
plot(fac_small['incre_buy_lag'],fac_small)
















