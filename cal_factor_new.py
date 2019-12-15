# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 17:48:30 2019

因子计算，包括挂单净增量因子跟大小单比例因子。这里分大小单用的分位数是当天的

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



transaction_file_path = "E:/Level2/XSHG_Transaction/201809/"
snapshot_file_path = "E:/Level2/XSHG_SnapShot/201809/"

transaction_files = os.listdir(transaction_file_path)
for i in transaction_files:
    if '.csv' not in i:
        transaction_files.remove(i)
snapshot_files = os.listdir(snapshot_file_path)
for i in snapshot_files:
    if '.csv' not in i:
        snapshot_files.remove(i)


def incre_buy(x):
    vol2 = 0
    vol1 = 0
    for i in map(str,range(1,11)):
        if x['BuyPrice'+i]>=x['lowest_price_lag']:
            vol2 += x['BuyVolume'+i]
        else:
            break
    for i in map(str,range(1,11)):
        if x['BuyPrice'+i+'_lag']>=x['lowest_price_lag']:
            vol1 += x['BuyVolume'+i+'_lag']
        else:
            break
    return vol2 + x['total_sell_vol_lag'] - vol1      
       
def incre_sell(x):
    vol2 = 0
    vol1 = 0
    for i in map(str,range(1,11)):
        if x['SellPrice'+i]<=x['highest_price_lag']:
            vol2 += x['SellVolume'+i]
        else:
            break
    for i in map(str,range(1,11)):
        if x['SellPrice'+i+'_lag']<=x['highest_price_lag']:
            vol1 += x['SellVolume'+i+'_lag']
        else:
            break
    return vol2 + x['total_buy_vol_lag'] - vol1

dir_name = 'fac_1m_all'
try:
    os.mkdir(dir_name)
except:
    pass
window = dt.timedelta(minutes=1)
#for i in range(3): #i=0; 2
#for i in range(3,6): #i=0; 3
for i in range(9,12): #i=0; 3
    file = transaction_files[i]
    print('start',file)
    year = int(file[:4])
    month = int(file[4:6])
    day = int(file[6:8])
    try:
        os.mkdir(dir_name+'/'+file[:8])
    except:
        pass
    output_file_path = (dir_name+'/'+file[:8]+'/')

    snapshot = pd.read_csv(snapshot_file_path+file)
    transaction = pd.read_csv(transaction_file_path + file)


    snapshot = snapshot.drop_duplicates()
    snapshot = snapshot.drop_duplicates(['Time','InstrumentId'],keep='last')
    snapshot['datetime'] = pd.to_datetime(snapshot['Date'].astype(str)+snapshot['Time'].astype(str).str.zfill(9),format='%Y%m%d%H%M%S%f')
    snapshot = snapshot.reset_index(drop=True)  
    snapshot_formerge = snapshot[['InstrumentId','datetime','BuyPrice1','BuyVolume1',
                              'BuyPrice2','BuyVolume2','BuyPrice3','BuyVolume3',
                              'BuyPrice4','BuyVolume4','BuyPrice5','BuyVolume5',
                              'BuyPrice6','BuyVolume6','BuyPrice7','BuyVolume7',
                              'BuyPrice8','BuyVolume8','BuyPrice9','BuyVolume9',
                              'BuyPrice10','BuyVolume10','SellPrice1','SellVolume1',
                              'SellPrice2','SellVolume2','SellPrice3','SellVolume3',
                              'SellPrice4','SellVolume4','SellPrice5','SellVolume5',
                              'SellPrice6','SellVolume6','SellPrice7','SellVolume7',
                              'SellPrice8','SellVolume8','SellPrice9','SellVolume9',
                              'SellPrice10','SellVolume10']]
    snapshot_formerge_lag2 = snapshot_formerge.copy()
    snapshot_formerge_lag2['datetime'] = snapshot_formerge_lag2['datetime']+window
    new_name = []
    for col in snapshot_formerge_lag2.columns:
        if col in ['InstrumentId','datetime']:
            new_name.append(col)
        else:
            new_name.append(col+'_lag')
    snapshot_formerge_lag2.columns = new_name    
    
    transaction['datetime'] = pd.to_datetime(transaction['Date'].astype(str)+transaction['Time'].astype(str).str.zfill(9),format='%Y%m%d%H%M%S%f')
    #transaction = transaction[transaction['datetime']>dt.datetime(year,month,day,9,30,0,0)]
    transaction['TradeDirection'] = ((transaction['BuyOrderId']>transaction['SellOrderId']).astype(int)*2-1).astype(int)
    transaction['ActOrderId'] = (transaction['BuyOrderId']*(transaction['TradeDirection']+1)*0.5 - transaction['SellOrderId']*(transaction['TradeDirection']-1)*0.5).astype(int)
    transaction['NetVolume'] = (transaction['TradeVolume']*transaction['TradeDirection']).astype(int)
    transaction = transaction.reset_index()
    
    trans_grouped = transaction[['InstrumentId','datetime','TradeDirection','ActOrderId']].groupby(['InstrumentId','ActOrderId']).last()
    trans_grouped['NetVolume'] =transaction.groupby(['InstrumentId','ActOrderId']).sum()['NetVolume']
    trans_grouped['TradeVolume'] = transaction.groupby(['InstrumentId','ActOrderId']).sum()['TradeVolume']
    
    print('read snapshot, transaction file')
    
    #am = dt.datetime(year,month,day,11,30,0,0)-window
    #pm1 = dt.datetime(year,month,day,13,00,0,0)+2*window
    #pm = dt.datetime(year,month,day,14,57,0,0)-window
    am = dt.datetime(year,month,day,11,30,0,0)
    pm1 = dt.datetime(year,month,day,13,00,0,0)+window
    pm = dt.datetime(year,month,day,14,57,0,0)
    
    
    df_all = pd.DataFrame()
    ids = list(set(transaction['InstrumentId']))
    print('total: ',len(ids))
   
    
#    for j in range(len(ids)): # j=0,len(ids)
    for j in range(len(ids)): # j=0,len(ids)
        #dtime = [];j=0;dtime.append(time())
        Id = ids[j]
        transaction_id = transaction[transaction['InstrumentId']==Id]
        snapshot_id = snapshot[snapshot['InstrumentId']==Id]
        snapshot_formerge_id = snapshot_formerge[snapshot_formerge['InstrumentId']==Id]
        snapshot_formerge_lag2_id = snapshot_formerge_lag2[snapshot_formerge_lag2['InstrumentId']==Id]
        trans_grouped_id = trans_grouped.loc[Id]
        t = snapshot_id[snapshot_id['datetime']>=dt.datetime(year,month,day,9,30,0,0)]['datetime'].iloc[0]
        d1={}
        
        #dtime.append(time())
        while t < pm:  # t = dt.datetime(year,month,day,9,40,0,0)
            if (t>am) & (t<pm1):
                t = snapshot_id[snapshot_id['datetime']>=pm1]['datetime'].iloc[0]
                continue
            
            t0=time()
            temp_lag = transaction_id[(transaction_id['datetime']>t-window) & (transaction_id['datetime']<=t)]
            temp_grouped_lag = trans_grouped_id[(trans_grouped_id['datetime']>t-window) & (trans_grouped_id['datetime']<=t)]
            
            #if temp.empty or temp_lag.empty or temp_lag2.empty:
            if temp_lag.empty:
                #t+=window
                t+=dt.timedelta(seconds=3)
                continue
            
            d1[t]={}

            trade_volume = temp_lag['TradeVolume'].values
            d1[t]['net_volume_lag'] = temp_lag['NetVolume'].values.sum()
            d1[t]['trade_volume_lag'] = trade_volume.sum()
            d1[t]['vwap_lag'] = temp_lag['TradeAmount'].values.sum()/d1[t]['trade_volume_lag']
            d1[t]['lowest_price_lag'] = temp_lag['TradePrice'].values.min()
            d1[t]['highest_price_lag'] = temp_lag['TradePrice'].values.max()
                    
            d1[t]['total_buy_vol_lag'] = ((temp_lag['TradeDirection'].values+1)*0.5*trade_volume).sum()
            d1[t]['total_sell_vol_lag'] = -((temp_lag['TradeDirection'].values-1)*0.5*trade_volume).sum()
            
#            net_volume = temp_grouped_lag['NetVolume']
#            d1[t]['act_vol_skew_lag'] = np.power((net_volume - net_volume.mean()),3).mean()/np.power(net_volume.std(),3)
# 
#            temp_big = temp_grouped_lag[temp_grouped_lag['TradeVolume']>trans_grouped_id['TradeVolume'].quantile(0.8)]
#            temp_small = temp_grouped_lag[temp_grouped_lag['TradeVolume']<trans_grouped_id['TradeVolume'].quantile(0.5)]
#            d1[t]['big_volume_lag'] = temp_big['TradeVolume'].sum()
#            d1[t]['small_volume_lag'] = temp_small['TradeVolume'].sum()
#            
#            d1[t]['big_buy_volume_lag'] = ((temp_big['TradeDirection']+1)*0.5*temp_big['TradeVolume']).sum()
#            d1[t]['big_sell_volume_lag'] = -((temp_big['TradeDirection']-1)*0.5*temp_big['TradeVolume']).sum()
#            d1[t]['small_buy_volume_lag'] = ((temp_small['TradeDirection']+1)*0.5*temp_small['TradeVolume']).sum()
#            d1[t]['small_sell_volume_lag'] = -((temp_small['TradeDirection']-1)*0.5*temp_small['TradeVolume']).sum()

            net_volume = temp_grouped_lag['NetVolume'].values
            d1[t]['act_vol_skew_lag'] = np.power((net_volume - net_volume.mean()),3).mean()/np.power(net_volume.std(),3)
            
            trade_volume = temp_grouped_lag['TradeVolume'].values
            trade_volume_id = trans_grouped_id['TradeVolume'].values
            trade_direction = temp_grouped_lag['TradeDirection'].values
            temp_big = np.where(trade_volume>np.quantile(trade_volume_id,0.8),trade_volume,0)
            temp_small = np.where(trade_volume<np.quantile(trade_volume_id,0.5),trade_volume,0)
            d1[t]['big_volume_lag'] = temp_big.sum()
            d1[t]['small_volume_lag'] = temp_small.sum()            
            
            d1[t]['big_buy_volume_lag'] = ((trade_direction+1)*0.5*temp_big).sum()
            d1[t]['big_sell_volume_lag'] = -((trade_direction-1)*0.5*temp_big).sum()
            d1[t]['small_buy_volume_lag'] = ((trade_direction+1)*0.5*temp_small).sum()
            d1[t]['small_sell_volume_lag'] = -((trade_direction-1)*0.5*temp_small).sum()

            t+=dt.timedelta(seconds=3)
    
        #dtime.append(time())
        
        df = pd.DataFrame.from_dict(d1,orient='index')
        df['InstrumentId'] = Id
        #df_all = pd.concat([df_all,df])
        df = df.reset_index()
        df.rename(columns={'index':'datetime'},inplace=True)
        
        #-----------
        df_lag = df.copy()
        df_lag['datetime'] = df_lag['datetime']+window
        new_name = []
        for col in df_lag.columns:
            if col in ['InstrumentId','datetime']:
                new_name.append(col)
            else:
                new_name.append(col+'2')
        df_lag.columns = new_name
        df = df.merge(df_lag,on=['InstrumentId','datetime'],how='left')
        #----------
        df_fur = df[['InstrumentId','datetime','vwap_lag','trade_volume_lag']].copy()
        df_fur['datetime'] = df_fur['datetime']-window
        df_fur.columns = ['InstrumentId','datetime','vwap_fur','trade_volume_fur']
        df = df.merge(df_fur,on = ['InstrumentId','datetime'],how='left') 
        #------------
        df = df.merge(snapshot_formerge_id,on=['InstrumentId','datetime'],how='left')
        df = df.merge(snapshot_formerge_lag2_id,on=['InstrumentId','datetime'],how='left')
        
        #------------
        #dtime.append(time())
        
        df['midprice'] = 0.5*(df['BuyPrice1'].values + df['SellPrice1'].values)
        df['ret_vwapmid_fur'] = np.log(df['vwap_fur'].values/df['midprice'].values)
        df['trade_volume_imbalance_lag'] = df['total_buy_vol_lag'].values/df['trade_volume_lag'].values
        df['volume_imbalance'] = df['BuyVolume1'].values/(df['SellVolume1'].values+df['BuyVolume1'].values)
        
        df['incre_buy_lag'] = df.apply(incre_buy,axis=1)
        df['incre_sell_lag'] = df.apply(incre_sell,axis=1)
        
        df['resist_buy_lag'] = df['incre_buy_lag'].values/df['total_sell_vol_lag'].values
        df['resist_sell_lag'] = df['incre_sell_lag'].values/df['total_buy_vol_lag'].values
        #dtime.append(time())


        df.dropna(inplace=True)
        df.reset_index(drop=True,inplace=True)
        
        df.to_csv(output_file_path+str(Id)+'.csv')
        if not j%10:
            print(j,end=',')

    print('\n',file,'done')
    gc.collect()
gc.collect()




np.diff(np.array(dtime))

np.diff(np.array(dcopy))


dcopy = dtime.copy()
l = os.listdir(dir_name)

















