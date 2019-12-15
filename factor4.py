# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 08:49:32 2019

部分常规因子

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
    t = dt.datetime(year,month,day,9,30,0,0)
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
        d1[t]['last_trade_price'] = temp_lag['TradePrice'].iloc[len(temp_lag)-1]
#        d1[t]['buy_aveprice'] = temp_lag[temp_lag['TradeDirection']==1]['TradePrice'].mean()
#        d1[t]['sell_aveprice'] = temp_lag[temp_lag['TradeDirection']==-1]['TradePrice'].mean()
        d1[t]['vwap_lag'] = sum(temp_lag['TradeAmount'])/sum(temp_lag['TradeVolume'])
        d1[t]['vwap_lag2'] = sum(temp_lag2['TradeAmount'])/sum(temp_lag2['TradeVolume'])

            
        if t in snapshot_id['datetime']:
            d1[t]['midprice']=0.5*(snapshot_id[snapshot_id['datetime']==t]['BuyPrice1']+snapshot_id[snapshot_id['datetime']==t]['SellPrice1'])
        else:
            temp_snap = snapshot_id[snapshot_id['datetime']<t]
            d1[t]['midprice']=0.5*(temp_snap['BuyPrice1'].iloc[len(temp_snap)-1]+temp_snap['SellPrice1'].iloc[len(temp_snap)-1])
        d1[t]['ret_vwap_fur'] = np.log(d1[t]['vwap_fur']/d1[t]['vwap_lag'])         
        d1[t]['ret_vwapmid_fur'] = np.log(d1[t]['vwap_fur']/d1[t]['midprice'])


        if t-window in snapshot_id['datetime']:
            d1[t]['midprice_lag']=0.5*(snapshot_id[snapshot_id['datetime']==t-window]['BuyPrice1']+snapshot_id[snapshot_id['datetime']==t-window]['SellPrice1'])
        else:
            temp_snap = snapshot_id[snapshot_id['datetime']<t-window]
            d1[t]['midprice_lag']=0.5*(temp_snap['BuyPrice1'].iloc[len(temp_snap)-1]+temp_snap['SellPrice1'].iloc[len(temp_snap)-1])      
        d1[t]['ret_vwap_lag'] = np.log(d1[t]['vwap_lag']/d1[t]['vwap_lag2'])  
        d1[t]['ret_vwapmid_lag'] = np.log(d1[t]['vwap_lag']/d1[t]['midprice_lag']) 


        temp_actbuy = temp_lag[temp_lag['TradeDirection']==1]
        temp_actsell = temp_lag[temp_lag['TradeDirection']==-1]
        d1[t]['buy_ave_vol_lag'] = np.log(temp_lag.groupby('BuyOrderId').sum()['TradeVolume'].mean())
        d1[t]['sell_ave_vol_lag'] = np.log(temp_lag.groupby('SellOrderId').sum()['TradeVolume'].mean())
        if temp_actbuy.empty:
            d1[t]['actbuy_weighted_price_lag']=np.nan
            d1[t]['max_actbuy_vol_lag'] =np.nan
            d1[t]['ave_actbuy_vol_lag'] = np.nan
            d1[t]['buy_vol_skew_lag'] = np.nan
            
        else:
            d1[t]['actbuy_weighted_price_lag'] = np.log(sum(temp_actbuy['TradeAmount'])/sum(temp_actbuy['TradeVolume']))
            grouped_buyvol = temp_actbuy.groupby('BuyOrderId').sum()['TradeVolume']
            d1[t]['max_actbuy_vol_lag'] = np.log(grouped_buyvol.max())
            d1[t]['ave_actbuy_vol_lag'] = np.log(grouped_buyvol.mean())
            d1[t]['buy_vol_skew_lag'] = np.power((grouped_buyvol - grouped_buyvol.mean())/grouped_buyvol.std(),3).sum()
                
        if temp_actsell.empty:
            d1[t]['actsell_weighted_price_lag']=np.nan
            d1[t]['trade_volume_imbalance_lag'] = np.inf
            d1[t]['ave_vol_imbalance_lag'] = np.inf
            
            d1[t]['max_actsell_vol_lag'] = np.nan
            d1[t]['ave_actsell_vol_lag'] = np.nan
            d1[t]['sell_vol_skew_lag'] = np.nan

        else:
            d1[t]['actsell_weighted_price_lag'] = np.log(sum(temp_actsell['TradeAmount'])/sum(temp_actsell['TradeVolume']))
            d1[t]['trade_volume_imbalance_lag'] = np.log(sum(temp_actbuy['TradeVolume']) / sum(temp_actsell['TradeVolume']))
            d1[t]['ave_vol_imbalance_lag'] = np.log(d1[t]['buy_ave_vol_lag']/d1[t]['sell_ave_vol_lag'])
            grouped_sellvol = temp_actsell.groupby('SellOrderId').sum()['TradeVolume']
            d1[t]['max_actsell_vol_lag'] = np.log(grouped_sellvol.max())
            d1[t]['ave_actsell_vol_lag'] = np.log(grouped_sellvol.mean())
            d1[t]['sell_vol_skew_lag'] = np.power((grouped_sellvol - grouped_sellvol.mean())/grouped_sellvol.std(),3).sum()

        grouped_actvol = temp_lag.groupby('ActOrderId').sum()
        temp_big = grouped_actvol[grouped_actvol['TradeVolume']>2000]
        temp_small = grouped_actvol[grouped_actvol['TradeVolume']<=2000]
        if temp_small.empty:
            d1[t]['big_samll_ratio_lag'] = np.inf
        else:
            d1[t]['big_samll_ratio_lag'] = np.log(temp_big['TradeVolume'].sum()/temp_small['TradeVolume'].sum())
        
        d1[t]['act_vol_skew_lag'] = np.power((grouped_actvol['NetVolume'] - grouped_actvol['NetVolume'].mean())/grouped_actvol['NetVolume'].std(),3).sum()


        # -----------
        temp_actbuy = temp_lag2[temp_lag2['TradeDirection']==1]
        temp_actsell = temp_lag2[temp_lag2['TradeDirection']==-1]
        d1[t]['buy_ave_vol_lag2'] = np.log(temp_lag2.groupby('BuyOrderId').sum()['TradeVolume'].mean())
        d1[t]['sell_ave_vol_lag2'] = np.log(temp_lag2.groupby('SellOrderId').sum()['TradeVolume'].mean())
        if temp_actbuy.empty:
            d1[t]['actbuy_weighted_price_lag2']=np.nan
            d1[t]['max_actbuy_vol_lag2'] =np.nan
            d1[t]['ave_actbuy_vol_lag2'] = np.nan
            d1[t]['buy_vol_skew_lag2'] = np.nan
            
        else:
            d1[t]['actbuy_weighted_price_lag2'] = np.log(sum(temp_actbuy['TradeAmount'])/sum(temp_actbuy['TradeVolume']))
            grouped_buyvol = temp_actbuy.groupby('BuyOrderId').sum()['TradeVolume']
            d1[t]['max_actbuy_vol_lag2'] = np.log(grouped_buyvol.max())
            d1[t]['ave_actbuy_vol_lag2'] = np.log(grouped_buyvol.mean())
            d1[t]['buy_vol_skew_lag2'] = np.power((grouped_buyvol - grouped_buyvol.mean())/grouped_buyvol.std(),3).sum()
        
        if temp_actsell.empty:
            d1[t]['actsell_weighted_price_lag2']=np.nan
            d1[t]['trade_volume_imbalance_lag2'] = np.inf
            d1[t]['ave_vol_imbalance_lag2'] = np.inf
            
            d1[t]['max_actsell_vol_lag2'] = np.nan
            d1[t]['ave_actsell_vol_lag2'] = np.nan
            d1[t]['sell_vol_skew_lag2'] = np.nan

        else:
            d1[t]['actsell_weighted_price_lag2'] = np.log(sum(temp_actsell['TradeAmount'])/sum(temp_actsell['TradeVolume']))
            d1[t]['trade_volume_imbalance_lag2'] = np.log(sum(temp_actbuy['TradeVolume']) / sum(temp_actsell['TradeVolume']))
            d1[t]['ave_vol_imbalance_lag2'] = np.log(d1[t]['buy_ave_vol_lag2']/d1[t]['sell_ave_vol_lag2'])
            grouped_sellvol = temp_actsell.groupby('SellOrderId').sum()['TradeVolume']
            d1[t]['max_actsell_vol_lag2'] = np.log(grouped_sellvol.max())
            d1[t]['ave_actsell_vol_lag2'] = np.log(grouped_sellvol.mean())
            d1[t]['sell_vol_skew_lag2'] = np.power((grouped_sellvol - grouped_sellvol.mean())/grouped_sellvol.std(),3).sum()

        grouped_actvol = temp_lag2.groupby('ActOrderId').sum()
        temp_big = grouped_actvol[grouped_actvol['TradeVolume']>2000]
        temp_small = grouped_actvol[grouped_actvol['TradeVolume']<=2000]
        if temp_small.empty:
            d1[t]['big_samll_ratio_lag2'] = np.inf
        else:
            d1[t]['big_samll_ratio_lag2'] = np.log(temp_big['TradeVolume'].sum()/temp_small['TradeVolume'].sum())
        
        d1[t]['act_vol_skew_lag2'] = np.power((grouped_actvol['NetVolume'] - grouped_actvol['NetVolume'].mean())/grouped_actvol['NetVolume'].std(),3).sum()

        t+=window
        #---------------
#        d1[t]['dri_actbuy'] = np.log(d1[t]['actbuy_weighted_price_lag']/d1[t]['actbuy_weighted_price_lag2'])
#        d1[t]['dri_actsell'] = np.log(d1[t]['actsell_weighted_price_lag']/d1[t]['actsell_weighted_price_lag2'])
#        d1[t]['dri_avebuyvol'] = np.log(d1[t]['ave_actbuy_vol_lag']/d1[t]['ave_actbuy_vol_lag2'])
#        d1[t]['dri_avesellvol'] = np.log(d1[t]['ave_actsell_vol_lag']/d1[t]['ave_actsell_vol_lag2'])

        
        #t+=dt.timedelta(seconds=1)
    df = pd.DataFrame.from_dict(d1,orient='index')
    df['InstrumentId'] = Id
    df = df.reset_index()
    df_all = pd.concat([df_all,df])
    if not j%10:
        print(j)

d1
df
df_all.shape
df.isna().sum()

df_noinf = df_all.copy()

for col in df_noinf.columns:
    if col in ['index','InstrumentId']:
        continue
    maxv = df_noinf[col].replace(np.inf,np.nan).dropna().max()
    minv = df_noinf[col].replace(-np.inf,np.nan).dropna().min()
    df_noinf.loc[:,col][df_noinf[col]>maxv] = maxv
    df_noinf.loc[:,col][df_noinf[col]<minv] = minv


for col in df_noinf.columns[:15]: 
    if col in ['index','InstrumentId']:
        continue
    # col = 'big_samll_ratio_lag'
    plt.scatter(df_noinf[df_noinf['InstrumentId']==600000][col],df_noinf[df_noinf['InstrumentId']==600000]['ret_vwapmid_fur'])
    plt.title(col)
    plt.xlim(min(df_noinf[col])*0.9,max(df_noinf[col])*1.1)
    plt.ylim(min(df_noinf['ret_vwapmid_fur'])-0.0005,max(df_noinf['ret_vwapmid_fur'])+0.0005)
    plt.plot([-10000,10000],[0,0],color='r')
    plt.show()
    
ids
Id=ids[13]

d={}
for i in range(len(ids)): # i=15
    Id = ids[i]
    temp = df_all[df_all['InstrumentId']==Id]
    #high_ret = temp[temp['ret_vwapmid_fur']>0.0015]  
    high_ret = temp[(temp['ret_vwapmid_fur']>0.0015) & (temp['net_volume']<temp['net_volume'].quantile(0.6)) &
                    (temp['net_volume']>temp['net_volume'].quantile(0.4))]  
    
    transaction_id = transaction[transaction['InstrumentId']==Id]
    
    times = list(high_ret['datetime'])
    print(times)
    #t = times[0]
    for t in times:
        temp_fur = transaction_id[(transaction_id['datetime']>t) & (transaction_id['datetime']<=t+window)]
        temp_lag = transaction_id[(transaction_id['datetime']>t-window) & (transaction_id['datetime']<=t)]
        temp_lag2 = transaction_id[(transaction_id['datetime']>t-2*window) & (transaction_id['datetime']<=t-window)]
            
        temp_lag_group = temp_lag.groupby('ActOrderId').last()[['datetime','TradeDirection']].merge(temp_lag.groupby('ActOrderId').sum()[['TradeVolume','NetVolume']],left_index=True,right_index=True)
        temp_lag_group['price']=temp_lag.groupby('ActOrderId').apply(lambda x: sum(x['TradeAmount'])/sum(x['TradeVolume']))
        temp_lag_group['total_volume'] = sum(temp_lag_group['TradeVolume'])
        temp_lag_group['volume_spread'] = sum(temp_lag_group['NetVolume'])
        
        temp_fur_group = temp_fur.groupby('ActOrderId').last()[['datetime','TradeDirection']].merge(temp_fur.groupby('ActOrderId').sum()[['TradeVolume','NetVolume']],left_index=True,right_index=True)
        temp_fur_group['price']=temp_fur.groupby('ActOrderId').apply(lambda x: sum(x['TradeAmount'])/sum(x['TradeVolume']))
        temp_fur_group['total_volume'] = sum(temp_fur_group['TradeVolume'])
        temp_fur_group['volume_spread'] = sum(temp_fur_group['NetVolume'])
        
        
        temp_lag_group_buy = temp_lag_group[temp_lag_group['TradeDirection']==1]
        temp_lag_group_sell = temp_lag_group[temp_lag_group['TradeDirection']==-1]
        
        temp_lag_buy_big = temp_lag_group_buy[temp_lag_group_buy['TradeVolume']>=transaction_id.groupby('ActOrderId').sum()['TradeVolume'].quantile(0.8)]
        temp_lag_sell_big = temp_lag_group_sell[temp_lag_group_sell['TradeVolume']>=transaction_id.groupby('ActOrderId').sum()['TradeVolume'].quantile(0.8)]
#        print(sum(temp_lag_buy_big['TradeVolume']),
#              sum(temp_lag_group_buy['TradeVolume']),
#              sum(temp_lag_buy_big['TradeVolume'])/sum(temp_lag_group_buy['TradeVolume']))
#        
#        print(sum(temp_lag_sell_big['TradeVolume']),
#              sum(temp_lag_group_sell['TradeVolume']),
#              sum(temp_lag_sell_big['TradeVolume'])/sum(temp_lag_group_sell['TradeVolume']))

        d[(Id,t)]={}
        if sum(temp_lag_group_buy['TradeVolume'])!=0:
            d[(Id,t)]['buy_big'] = sum(temp_lag_buy_big['TradeVolume'])/sum(temp_lag_group_buy['TradeVolume'])
        if sum(temp_lag_group_sell['TradeVolume'])!=0:
            d[(Id,t)]['sell_big'] = sum(temp_lag_sell_big['TradeVolume'])/sum(temp_lag_group_sell['TradeVolume'])
    
    if not i%10:
        print(i,end=',')



pct = pd.DataFrame.from_dict(d,orient='index')
pct.dropna(inplace=True)


pct['buy_big'].hist(bins=20)
pct['sell_big'].hist(bins=20)
(pct['buy_big']!=0).sum()/len(pct)
(pct['sell_big']!=0).sum()/len(pct)

temp_lag_group[(temp_lag_group['TradeVolume']>=transaction_id.groupby('ActOrderId').sum()['TradeVolume'].quantile(0.9)) & 
               (temp_lag_group['])]











    
snapshot_id 




volume increment
week sell_1 sell_2
trade volume and new orders/rest volume, on the same price


short term buy order volume increment
active order increment
resistent order increment



    
        