# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 14:24:15 2019

不存在卖单的情况

@author: caoqiliang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from time import time
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
%matplotlib inline

file_path = "E:/Level2/XSHG_Transaction/201809/"
year = 2018
month = 9
day = 3
file = "20180903.csv"
snapshot = pd.read_csv("E:/Level2/XSHG_SnapShot/201809/20180903.csv")
transaction = pd.read_csv(file_path + file)

snapshot = snapshot.drop_duplicates()
snapshot = snapshot.drop_duplicates(['Time','InstrumentId'],keep='last')
snapshot['datetime'] = pd.to_datetime(snapshot['Date'].astype(str)+snapshot['Time'].astype(str).str.zfill(9),format='%Y%m%d%H%M%S%f')
snapshot = snapshot.reset_index(drop=True)


transaction['datetime'] = pd.to_datetime(transaction['Date'].astype(str)+transaction['Time'].astype(str).str.zfill(9),format='%Y%m%d%H%M%S%f')
#transaction = transaction[transaction['datetime']>dt.datetime(year,month,day,9,30,0,0)]
transaction['TradeDirection'] = ((transaction['BuyOrderId']>transaction['SellOrderId']).astype(int)*2-1).astype(int)
transaction['ActOrderId'] = (transaction['BuyOrderId']*(transaction['TradeDirection']+1)*0.5 - transaction['SellOrderId']*(transaction['TradeDirection']-1)*0.5).astype(int)
transaction['NetVolume'] = (transaction['TradeVolume']*transaction['TradeDirection']).astype(int)
transaction = transaction.reset_index()


window = dt.timedelta(minutes=1)
am = dt.datetime(year,month,day,11,30,0,0)-window
pm1 = dt.datetime(year,month,day,13,00,0,0)+2*window
pm = dt.datetime(year,month,day,14,57,0,0)-window


df_all = pd.DataFrame()
ids = list(set(transaction['InstrumentId']))

for j in range(len(ids)): # Id=600000 
    Id = ids[j]
    transaction_id = transaction[transaction['InstrumentId']==Id]
    snapshot_id = snapshot[snapshot['InstrumentId']==Id]
    t = snapshot_id[snapshot_id['datetime']>=dt.datetime(year,month,day,9,30,0,0)]['datetime'].iloc[0]
    d1={}
    while t < pm:  # t = dt.datetime(year,month,day,9,40,0,0)
        if (t>am) & (t<pm1):
            t = pm1
            continue
        temp = transaction_id[(transaction_id['datetime']>t) & (transaction_id['datetime']<t+window)]
        temp_lag = transaction_id[(transaction_id['datetime']>t-window) & (transaction_id['datetime']<=t)]
        temp_lag2 = transaction_id[(transaction_id['datetime']>t-2*window) & (transaction_id['datetime']<=t-window)]

        if temp.empty or temp_lag.empty or temp_lag2.empty:
            t+=window
            #t+=dt.timedelta(seconds=1)
            continue
        d1[t]={}
        d1[t]['vwap_fur'] = sum(temp['TradeAmount'])/sum(temp['TradeVolume'])
        
        temp_actbuy = temp_lag[temp_lag['TradeDirection']==1]
        temp_actsell = temp_lag[temp_lag['TradeDirection']==-1]
        if temp_actbuy.empty:
            d1[t]['nobuy']=1
            d1[t]['sell_volume'] = sum(temp_actsell['TradeVolume'])
            grouped_actsell = temp_actsell.groupby('ActOrderId').sum()
            d1[t]['ave_sell_vol_lag'] = sum(grouped_actsell['TradeVolume'])/len(grouped_actsell)      
            d1[t]['sig_sell_vol_lag'] = grouped_actsell['TradeVolume'].std()
        
        if temp_actsell.empty:
            d1[t]['nosell']=1
            d1[t]['buy_volume'] = sum(temp_actbuy['TradeVolume'])
            grouped_actbuy = temp_actbuy.groupby('ActOrderId').sum()
            d1[t]['ave_buy_vol_lag'] = sum(grouped_actbuy['TradeVolume'])/len(grouped_actbuy)
            d1[t]['sig_buy_vol_lag'] = grouped_actbuy['TradeVolume'].std()
                          
        t+=window

    df = pd.DataFrame.from_dict(d1,orient='index')
    df['InstrumentId'] = Id
    df.rename(columns={'index':'datetime'},inplace=True)
    df = df.reset_index()
    df_all = pd.concat([df_all,df])
    if not j%10:
        print(j,end=',')
        
        

df_dropna = df_all.dropna(subset = ['nobuy','nosell'],how='all')   
snapshot_formerge = snapshot[['InstrumentId','datetime','BuyPrice1','BuyVolume1','SellPrice1','SellVolume1']]
df_merged = df_dropna.merge(snapshot_formerge,on=['InstrumentId','datetime'],how='left')        
        
df_merged.dropna(subset=['BuyPrice1','BuyVolume1'],inplace=True)        
df_merged['ret_vwapmid_fur'] = np.log(df_merged['vwap_fur']*2/(df_merged['BuyPrice1']+df_merged['SellPrice1']))  


df_merged[df_merged['nosell']==1]['ret_vwapmid_fur'].hist(bins=100)
df_merged[df_merged['nobuy']==1]['ret_vwapmid_fur'].hist(bins=100)


df_merged['volume_imbalance'] = df_merged['BuyVolume1']/df_merged['SellVolume1']


nosell = df_merged[df_merged['nosell']==1]
(nosell['ret_vwapmid_fur']>0.0005).sum()/nosell.shape[0]

nosell['ret_vwapmid_fur'].mean()
(nosell['ret_vwapmid_fur']-0.0005).sum()


nobuy = df_merged[df_merged['nobuy']==1]
(nobuy['ret_vwapmid_fur']<-0.0005).sum()/nobuy.shape[0]


df_merged.columns
for col in [ 'ave_buy_vol_lag','buy_volume','BuyVolume1','SellVolume1','volume_imbalance']:
    if col in ['datetime','InstrumentId']:
        continue
    plt.scatter(np.log(nosell[col]),nosell['ret_vwapmid_fur'])
    plt.title(col)
    plt.xlim(min(np.log(nosell[col]))*0.9,max(np.log(nosell[col]))*1.1)
    plt.ylim(min(nosell['ret_vwapmid_fur'])-0.0005,max(nosell['ret_vwapmid_fur'])+0.0005)
    plt.plot([-100000,100000],[0,0],color='r')
    plt.axvline(x=0,color='r')
    plt.show()





for i in np.linspace(6,11,51):
    nosell_volume = nosell[np.log(nosell['buy_volume'])>i]
    print(i,(nosell_volume['ret_vwapmid_fur']>0.0005).sum()/nosell_volume.shape[0])

for i in np.linspace(6,11,51):
    nosell_volume = nosell[np.log(nosell['BuyVolume1'])>i]
    print(i,(nosell_volume['ret_vwapmid_fur']>0.0005).sum()/nosell_volume.shape[0])

for i in np.linspace(0,4,51):
    nosell_volume = nosell[np.log(nosell['volume_imbalance'])>i]
    print(i,(nosell_volume['ret_vwapmid_fur']>0.0005).sum()/nosell_volume.shape[0])



nosell_volume.shape





