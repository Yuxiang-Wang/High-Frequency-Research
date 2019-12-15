# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:51:13 2019

常规因子计算分析

@author: yuxiang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from time import time
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
    # t = dt.datetime(year,month,day,9,30,0,0)
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
        # d1[t]['last_trade_price'] = temp_lag['TradePrice'].iloc[len(temp_lag)-1]
#        d1[t]['buy_aveprice'] = temp_lag[temp_lag['TradeDirection']==1]['TradePrice'].mean()
#        d1[t]['sell_aveprice'] = temp_lag[temp_lag['TradeDirection']==-1]['TradePrice'].mean()
        d1[t]['vwap_lag'] = sum(temp_lag['TradeAmount'])/sum(temp_lag['TradeVolume'])
        d1[t]['vwap_lag2'] = sum(temp_lag2['TradeAmount'])/sum(temp_lag2['TradeVolume'])

            
        if t in snapshot_id['datetime']:
            d1[t]['midprice']=0.5*(snapshot_id[snapshot_id['datetime']==t]['BuyPrice1']+snapshot_id[snapshot_id['datetime']==t]['SellPrice1'])
            d1[t]['buyprice1'] = snapshot_id[snapshot_id['datetime']==t]['BuyPrice1'] 
            d1[t]['sellprice1'] = snapshot_id[snapshot_id['datetime']==t]['SellPrice1'] 
            d1[t]['buyvolume1'] = snapshot_id[snapshot_id['datetime']==t]['BuyVolume1'] 
            d1[t]['sellvolume1'] = snapshot_id[snapshot_id['datetime']==t]['SellVolume1'] 
        else:
            temp_snap = snapshot_id[snapshot_id['datetime']<t]
            d1[t]['midprice']=0.5*(temp_snap['BuyPrice1'].iloc[len(temp_snap)-1]+temp_snap['SellPrice1'].iloc[len(temp_snap)-1])
            d1[t]['buyprice1'] = temp_snap['BuyPrice1'].iloc[len(temp_snap)-1]
            d1[t]['sellprice1'] = temp_snap['SellPrice1'].iloc[len(temp_snap)-1]
            d1[t]['buyvolume1'] = temp_snap['BuyVolume1'].iloc[len(temp_snap)-1]
            d1[t]['sellvolume1'] = temp_snap['SellVolume1'].iloc[len(temp_snap)-1] 

        d1[t]['ret_vwap_fur'] = np.log(d1[t]['vwap_fur']/d1[t]['vwap_lag'])         
        d1[t]['ret_vwapmid_fur'] = np.log(d1[t]['vwap_fur']/d1[t]['midprice'])


        if t-window in snapshot_id['datetime']:
            d1[t]['midprice_lag']=0.5*(snapshot_id[snapshot_id['datetime']==t-window]['BuyPrice1']+snapshot_id[snapshot_id['datetime']==t-window]['SellPrice1'])
            d1[t]['buyprice1'] = snapshot_id[snapshot_id['datetime']==t-window]['BuyPrice1'] 
            d1[t]['sellprice1'] = snapshot_id[snapshot_id['datetime']==t-window]['SellPrice1'] 
            d1[t]['buyvolume1'] = snapshot_id[snapshot_id['datetime']==t-window]['BuyVolume1'] 
            d1[t]['sellvolume1'] = snapshot_id[snapshot_id['datetime']==t-window]['SellVolume1'] 
        else:
            temp_snap = snapshot_id[snapshot_id['datetime']<t-window]
            d1[t]['midprice_lag']=0.5*(temp_snap['BuyPrice1'].iloc[len(temp_snap)-1]+temp_snap['SellPrice1'].iloc[len(temp_snap)-1])      
            d1[t]['buyprice1'] = temp_snap['BuyPrice1'].iloc[len(temp_snap)-1]
            d1[t]['sellprice1'] = temp_snap['SellPrice1'].iloc[len(temp_snap)-1]
            d1[t]['buyvolume1'] = temp_snap['BuyVolume1'].iloc[len(temp_snap)-1]
            d1[t]['sellvolume1'] = temp_snap['SellVolume1'].iloc[len(temp_snap)-1] 

        d1[t]['ret_vwap_lag'] = np.log(d1[t]['vwap_lag']/d1[t]['vwap_lag2'])  
        d1[t]['ret_vwapmid_lag'] = np.log(d1[t]['vwap_lag']/d1[t]['midprice_lag']) 

        # ------------------------
        d1[t]['net_volume'] = sum(temp_lag['NetVolume'])
        d1[t]['trade_volume'] = sum(temp_lag['TradeVolume'])
        
        temp_actbuy = temp_lag2[temp_lag2['TradeDirection']==1]
        temp_actsell = temp_lag2[temp_lag2['TradeDirection']==-1]
        d1[t]['buy_volume'] = sum(temp_actbuy['TradeVolume'])
        d1[t]['sell_volume'] = sum(temp_actsell['TradeVolume'])
        if d1[t]['sell_volume']==0:
            d1[t]['vol_imbalance'] = np.inf
        else:
            d1[t]['vol_imbalance'] = d1[t]['buy_volume']/d1[t]['sell_volume']
        
        grouped_actvol = temp_lag.groupby('ActOrderId').sum()
        thresh = grouped_actvol['TradeVolume'].quantile(0.75)
        temp_big = grouped_actvol[grouped_actvol['TradeVolume']>thresh]
        temp_small = grouped_actvol[grouped_actvol['TradeVolume']<=thresh]
        if temp_small.empty:
            d1[t]['big_samll_ratio_lag'] = np.inf
        else:
            d1[t]['big_samll_ratio_lag'] = np.log(temp_big['TradeVolume'].sum()/temp_small['TradeVolume'].sum())
        
        d1[t]['big_trade_spread'] = sum(temp_big['NetVolume'])
        d1[t]['small_trade_spread'] = sum(temp_small['NetVolume'])
        
        d1[t]['act_vol_skew_lag'] = np.power((grouped_actvol['NetVolume'] - grouped_actvol['NetVolume'].mean())/grouped_actvol['NetVolume'].std(),3).sum()
        t+=window

        #t+=dt.timedelta(seconds=1)
    df = pd.DataFrame.from_dict(d1,orient='index')
    df['InstrumentId'] = Id
    df = df.reset_index()
    df_all = pd.concat([df_all,df])
    if not j%10:
        print(j,end=',')


df_all.isna().sum()
df_all.rename(columns = {'index':'datetime'},inplace=True)

(df_all==np.inf).sum()
(df_all==-np.inf).sum()

df_noinf = df_all.replace([-np.inf,np.inf],np.nan)

df_noinf.isna().sum()
df_noinf.dropna(inplace=True)


df_noinf['vol_imbalance'] = np.log(df_noinf['vol_imbalance'])
df_noinf['trade_volume_imbalance'] = df_noinf['net_volume']/df_noinf['trade_volume']

for col in l: # col = 'vol_imbalance'; df_noinf.columns[11:21]
    if col in ['index','InstrumentId']:
        continue
    #col = 'big_samll_ratio_lag'
    plt.scatter(df_noinf[col],df_noinf['ret_vwapmid_fur'])
    plt.title(col)
    plt.xlim(min(df_noinf[col])*0.9,max(df_noinf[col])*1.1)
    plt.ylim(min(df_noinf['ret_vwapmid_fur'])-0.0005,max(df_noinf['ret_vwapmid_fur'])+0.0005)
    plt.plot([-100000,100000],[0,0],color='r')
    plt.axvline(x=0,color='r')
    plt.show()
    

    
    
[df_noinf['InstrumentId']==600000]


l = ['ret_vwapmid_lag','ret_vwap_lag','net_volume','big_trade_spread','vol_imbalance','trade_volume_imbalance']
col = l[0]


temp = df_noinf[df_noinf['trade_volume_imbalance']>0]
(temp['ret_vwapmid_fur']>0).sum()
(temp['ret_vwapmid_fur']>0).sum()/temp.shape[0]

# big_trade_spread
for i in range(0,610000,10000):
    temp = df_noinf[(df_noinf[l[3]]>i)&(df_noinf[l[3]]<5000000)]
    #(temp['ret_vwapmid_fur']>0).sum()
    print(i,(temp['ret_vwapmid_fur']>0.0005).sum()/temp.shape[0])

# net_volume
for i in range(0,310000,5000):
    temp = df_noinf[(df_noinf[l[2]]>i)&(df_noinf[l[2]]<500000)]
    #(temp['ret_vwapmid_fur']>0).sum()
    print(i,(temp['ret_vwapmid_fur']>0.0005).sum()/temp.shape[0])

# ret_vwapmid_lag
for i in np.linspace(0,0.005,num=50):
    temp = df_noinf[(df_noinf[l[0]]>i)&(df_noinf[l[0]]<0.02)]
    #(temp['ret_vwapmid_fur']>0).sum()
    print(i,(temp['ret_vwapmid_fur']>0.0005).sum()/temp.shape[0])
    

# ret_vwap_lag
for i in np.linspace(0,0.005,num=50):
    temp = df_noinf[(df_noinf[l[1]]>i)]
    #(temp['ret_vwapmid_fur']>0).sum()
    print(i,(temp['ret_vwapmid_fur']>0.0005).sum()/temp.shape[0])


temp = df_noinf[(df_noinf[l[2]]>100000) & (df_noinf[l[0]]>0.001)]
(temp['ret_vwapmid_fur']>0).sum()/temp.shape[0]




