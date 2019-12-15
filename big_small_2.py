# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 09:06:26 2019

大小单总体跟一些个股


@author: yuxiang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from time import time
import gc
import os
%matplotlib inline

file_path = 'fac_1m_all/20180903'
csvs = os.listdir(file_path)

df = []
for i in csvs:
    df.append(pd.read_csv(file_path+'/'+i,index_col=0))
df = pd.concat(df)
df['datetime']=pd.to_datetime(df['datetime'],format='%Y-%m-%d %H:%M:%S')
df.shape


daily = pd.read_csv('daily/201809StockDailyData.csv')
daily['InstrumentId'] = daily['WindCode'].apply(lambda x:int(x[:6]))

temp = daily[daily['InstrumentId'].isin(df['InstrumentId'].unique())]
temp = temp[temp['TradingDay']==20180903]

temp[temp['VWAvgPrice']<15]



# ---------------------------------------------
def plot(temp,factor,log=False,col=None,mid=0):
    if log:
        temp = np.log(temp)
    plt.axhline(y=0,color='r')
    #plt.axhline(y=0.001,color='r')
    #plt.axhline(y=-0.001,color='r')
    plt.axvline(x=0,color='r')
    plt.scatter(temp,factor['ret_vwapmid_fur'],linewidths=0.001,alpha=0.5)
    if col:
        plt.title(col)
    plt.xlim(min(temp)*0.9,max(temp)*1.1)
    plt.ylim(min(factor['ret_vwapmid_fur'])-0.005,max(factor['ret_vwapmid_fur'])+0.005)
    #plt.plot([-10000,10000],[0,0],color='r')
    plt.grid(True)
    plt.show()
    print('''pct:
        1: %f  2: %f
        3: %f  4: %f
        left mean: %f  right mean: %f''' % (
        (factor[temp<mid]['ret_vwapmid_fur']>0).sum()/len(factor),
        (factor[temp>mid]['ret_vwapmid_fur']>0).sum()/len(factor),
        (factor[temp<mid]['ret_vwapmid_fur']<0).sum()/len(factor),
        (factor[temp>mid]['ret_vwapmid_fur']<0).sum()/len(factor),
        factor[temp<mid]['ret_vwapmid_fur'].mean(),
        factor[temp>mid]['ret_vwapmid_fur'].mean()))

        
def plot_mean(factor,temp,intervals,log=False,show=True,label=None):
    x=[]
    y=[]
    if log:
        factor = np.log(factor)
    for interval in intervals:
        temp_interval = temp[(factor>interval[0])&(factor<=interval[1])]
        if not temp_interval.empty:
            y.append(temp_interval['ret_vwapmid_fur'].mean())
            x.append(0.5*(interval[0]+interval[1]))
    plt.scatter(x,y,linewidth=0.001)
    #plt.plot(x,y,linewidth=0.8)
    plt.ylim(min(y),max(y))
    #ind = x[list(map(abs,y)).index(min(list(map(abs,y))))]
    plt.axvline(x=0,color='r',linewidth=0.8)
    plt.axhline(y=0,color='r',linewidth=0.8)
    plt.grid(True)
    if label:
        plt.legend(labels=[label])
    if show:
        plt.show()
    #print('zero point',x[list(map(abs,y)).index(min(list(map(abs,y))))])



def all_plot(fac_name,df,mid=0,log=False):
    plot(df[fac_name],df,mid=mid,log=log)
    if log:
        arr = np.linspace(min(np.log(df[fac_name])),max(np.log(df[fac_name])),100)
    else:
        arr = np.linspace(min(df[fac_name]),max(df[fac_name]),100)
    itv = np.vstack([arr[:-1],arr[1:]]).T
    plot_mean(df[fac_name],df,itv,log=log)
# ---------------------------------------------


df['big_total_ratio'] = df['big_volume_lag']/df['trade_volume_lag']
df['small_total_ratio'] = df['small_volume_lag']/df['trade_volume_lag']
df['big_small_ratio'] = df['big_volume_lag']/df['small_volume_lag']

df['buy_total_ratio'] = df['total_buy_vol_lag']/df['trade_volume_lag']
df['sell_total_ratio'] = df['total_sell_vol_lag']/df['trade_volume_lag']



df['big_buy_total_ratio'] = df['big_buy_volume_lag']/df['trade_volume_lag']
df['small_buy_total_ratio'] = df['small_buy_volume_lag']/df['trade_volume_lag']
df['big_buy_buytotal_ratio'] = df['big_buy_volume_lag']/df['total_buy_vol_lag']
df['small_buy_buytotal_ratio'] = df['small_buy_volume_lag']/df['total_buy_vol_lag']
df['big_buy_small_buy_ratio'] = df['big_buy_volume_lag']/df['small_buy_volume_lag']

df['big_sell_total_ratio'] = df['big_sell_volume_lag']/df['trade_volume_lag']
df['small_sell_total_ratio'] = df['small_sell_volume_lag']/df['trade_volume_lag']
df['big_sell_selltotal_ratio'] = df['big_sell_volume_lag']/df['total_sell_vol_lag']
df['small_sell_selltotal_ratio'] = df['small_sell_volume_lag']/df['total_sell_vol_lag']
df['big_sell_small_sell_ratio'] = df['big_sell_volume_lag']/df['small_sell_volume_lag']

df['big_volume_spread'] = df['big_buy_volume_lag'] - df['big_sell_volume_lag']
df['small_volume_spread'] = df['small_buy_volume_lag'] - df['small_sell_volume_lag']

df['big_spread_total_spread'] = df['big_volume_spread']/df['net_volume_lag']
df['small_spread_total_spread'] = df['small_volume_spread']/df['net_volume_lag']


# -----------------------------------------------

plot(df['net_volume_lag'],df)
all_plot('net_volume_lag',df)
(df['ret_vwapmid_fur']>0).sum()/len(df)

plt.scatter(df['net_volume_lag2'],df['net_volume_lag'],alpha=0.5)
plt.grid(True)
plt.show()

# ------------------
norm_df = []
for Id in df['InstrumentId'].unique(): # Id=df['InstrumentId'].unique()[0]
    temp_df = df[df['InstrumentId']==Id].copy()
    #temp_df['norm_net_volume_lag'] = (temp_df['net_volume_lag']-temp_df['net_volume_lag'].mean())/temp_df['net_volume_lag'].std()
    #temp_df['norm_net_volume_lag'] = temp_df['net_volume_lag']/temp_df['net_volume_lag'].mean()
    temp_df['norm_net_volume_lag'] = temp_df['net_volume_lag']/temp[temp['InstrumentId']==Id]['NonRestrictedShares'].values * 1e4
    norm_df.append(temp_df)
norm_df = pd.concat(norm_df)
all_plot('norm_net_volume_lag',norm_df)

#-------------------
for Id in df['InstrumentId'].unique(): # Id=df['InstrumentId'].unique()[0]
    temp_df = df[df['InstrumentId']==Id].copy()
    #temp_df['norm_net_volume_lag'] = (temp_df['net_volume_lag']-temp_df['net_volume_lag'].mean())/temp_df['net_volume_lag'].std()
    #temp_df['norm_net_volume_lag'] = temp_df['net_volume_lag']/temp_df['net_volume_lag'].mean()
    temp_df['norm_net_volume_lag'] = temp_df['net_volume_lag']/temp[temp['InstrumentId']==Id]['NonRestrictedShares'].values * 1e4
    all_plot('norm_net_volume_lag',temp_df)
    print(Id,temp[temp['InstrumentId']==Id]['MarketCapAFloat'],temp[temp['InstrumentId']==Id]['VWAvgPrice']/temp[temp['InstrumentId']==Id]['SplitFactor'])

#----------------
for Id in df['InstrumentId'].unique()[10:]:
    temp_df = df[df['InstrumentId']==Id]
#    all_plot('net_volume_lag',temp_df)
    plt.scatter(temp_df['net_volume_lag2'],temp_df['net_volume_lag'],alpha=0.5)
    plt.show()
    print(Id,temp[temp['InstrumentId']==Id]['MarketCapAFloat'],temp[temp['InstrumentId']==Id]['VWAvgPrice']/temp[temp['InstrumentId']==Id]['SplitFactor'])
    


#---
temp_df = df[~(df['big_spread_total_spread'].isna()) & (df['big_spread_total_spread']>-np.inf) &
             (df['big_spread_total_spread']<np.inf)]
all_plot('big_spread_total_spread',temp_df)
temp_df = df[~(df['big_spread_total_spread'].isna()) & (df['big_spread_total_spread']>0) &
             (df['big_spread_total_spread']<np.inf)]
all_plot('big_spread_total_spread',temp_df,log=True)
#---
temp_df = df[~(df['small_spread_total_spread'].isna()) & (df['small_spread_total_spread']>-np.inf) &
             (df['small_spread_total_spread']<np.inf)]
all_plot('small_spread_total_spread',temp_df)
temp_df = df[~(df['small_spread_total_spread'].isna()) & (df['small_spread_total_spread']>0) &
             (df['small_spread_total_spread']<np.inf)]
all_plot('small_spread_total_spread',temp_df,log=True)
#---

all_plot('big_total_ratio',df)
all_plot('small_total_ratio',df)
temp_df = df[~(df['big_small_ratio'].isna()) & (df['big_small_ratio']>0) &
             (df['big_small_ratio']<np.inf)]
all_plot('big_small_ratio',temp_df,log=True)

all_plot('buy_total_ratio',df)
all_plot('big_buy_total_ratio',df)
all_plot('small_buy_total_ratio',df)
all_plot('big_buy_buytotal_ratio',df)
all_plot('small_buy_buytotal_ratio',df)
temp_df = df[~(df['big_buy_small_buy_ratio'].isna()) & (df['big_buy_small_buy_ratio']>0) &
             (df['big_buy_small_buy_ratio']<np.inf)]
all_plot('big_buy_small_buy_ratio',temp_df)

####
np.corrcoef(list(df['sell_total_ratio']),list(df['big_sell_total_ratio']))
np.corrcoef(list(df['buy_total_ratio']),list(df['big_buy_total_ratio']))
####

all_plot('sell_total_ratio',df)
all_plot('big_sell_total_ratio',df)
all_plot('small_sell_total_ratio',df)
all_plot('big_sell_selltotal_ratio',df)
all_plot('small_sell_selltotal_ratio',df)
temp_df = df[~(df['big_sell_small_sell_ratio'].isna()) & (df['big_sell_small_sell_ratio']>0) &
             (df['big_sell_small_sell_ratio']<np.inf)]
all_plot('big_sell_small_sell_ratio',temp_df)


all_plot('big_volume_spread',df)
all_plot('small_volume_spread',df)
all_plot('net_volume_lag',df)

# ------------------
norm_df = []
for Id in df['InstrumentId'].unique(): # Id=df['InstrumentId'].unique()[0]
    temp_df = df[df['InstrumentId']==Id].copy()
    #temp_df['norm_net_volume_lag'] = (temp_df['net_volume_lag']-temp_df['net_volume_lag'].mean())/temp_df['net_volume_lag'].std()
    #temp_df['norm_net_volume_lag'] = temp_df['net_volume_lag']/temp_df['net_volume_lag'].mean()
    temp_df['norm_net_volume_lag'] = temp_df['net_volume_lag']/temp[temp['InstrumentId']==Id]['NonRestrictedShares'].values * 1e4
    norm_df.append(temp_df)
norm_df = pd.concat(norm_df)
all_plot('norm_net_volume_lag',norm_df)



big buy/buy total



 










