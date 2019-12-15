# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 17:44:53 2019

new order data.
calculation.

@author: yuxiang
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from time import time
import gc
import os
import statsmodels.api as sm
%matplotlib inline

date = 20180903
year=2018
month=9
day = 3
transaction_file_path = "E:/Level2/XSHE_Transaction/201809/"+str(date)+'/'
snapshot_file_path = "E:/Level2/XSHE_SnapShot/201809/"+str(date)+'/'

order_file_path = "E:/Level2/XSHE_Order/201809/"+str(date)+'/'

order_files = os.listdir(order_file_path)

dir_name = 'fac_1m_new'
try:
    os.mkdir(dir_name)
except:
    pass
output_file_path = dir_name+'/'+str(date)+'/'
try:
    os.mkdir(output_file_path)
except:
    pass
window = dt.timedelta(seconds=60)

n=972
for file in order_files[972:]: # file=order_files[n]; file='300007.csv' # 971
    
    order = pd.read_csv(order_file_path+file)
    snapshot = pd.read_csv(snapshot_file_path+file)
    transaction = pd.read_csv(transaction_file_path+file)

    order['datetime'] = pd.to_datetime(order['Date'].astype(str)+order['Time'].astype(str).str.zfill(9),format='%Y%m%d%H%M%S%f')
    order['Direction'] = ((order['OrderDirection']=='Buy').astype(int)*2-1).astype(int)
    order['OrderAmount'] = order['OrderPrice']*order['OrderVolume']

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

    
    transaction['datetime'] = pd.to_datetime(transaction['Date'].astype(str)+transaction['Time'].astype(str).str.zfill(9),format='%Y%m%d%H%M%S%f')
    transaction['TradeDirection'] = ((transaction['BuyOrderId']>transaction['SellOrderId']).astype(int)*2-1).astype(int)
    transaction['ActOrderId'] = (transaction['BuyOrderId']*(transaction['TradeDirection']+1)*0.5 - transaction['SellOrderId']*(transaction['TradeDirection']-1)*0.5).astype(int)
    transaction['NetVolume'] = (transaction['TradeVolume']*transaction['TradeDirection']).astype(int)
    transaction = transaction.reset_index()

    trans_traded = transaction[transaction['TradeType']=='Traded']
    trans_traded_grouped = trans_traded[['InstrumentId','datetime','TradeDirection','ActOrderId','TradeType']].groupby(['InstrumentId','ActOrderId']).last()
    trans_traded_grouped[['NetVolume','TradeVolume','TradeAmount']] =trans_traded.groupby(['InstrumentId','ActOrderId']).sum()[['NetVolume','TradeVolume','TradeAmount']]
    trans_traded_grouped['TradePrice'] = trans_traded_grouped['TradeAmount']/trans_traded_grouped['TradeVolume']
    trans_traded_grouped['HighestPrice'] = trans_traded[['InstrumentId','TradePrice','ActOrderId']].groupby(['InstrumentId','ActOrderId']).max()['TradePrice']
    trans_traded_grouped['LowestPrice'] = trans_traded[['InstrumentId','TradePrice','ActOrderId']].groupby(['InstrumentId','ActOrderId']).min()['TradePrice']
    trans_traded_grouped = trans_traded_grouped.reset_index()
    trans_traded_grouped = trans_traded_grouped.merge(order.rename(columns={'datetime':'OrderTime'})[['OrderTime','OrderPrice',
                                                      'OrderId','OrderVolume',
                                                      'OrderPriceType','OrderDirection']],left_on='ActOrderId',right_on='OrderId',how='left')
    trans_traded_grouped['CancelledTradeAmount'] = (trans_traded_grouped['TradeVolume']*trans_traded_grouped['OrderPrice'])
    
    trans_cancelled = transaction[transaction['TradeType']=='Cancelled']
    trans_cancelled = trans_cancelled.merge(order.rename(columns={'datetime':'OrderTime'})[['OrderTime','OrderPrice',
                                                      'OrderId','OrderVolume','OrderAmount',
                                                      'OrderPriceType','OrderDirection']],left_on='ActOrderId',right_on='OrderId',how='left')
    trans_cancelled['CancelledTradeAmount'] = (trans_cancelled['TradeVolume']*trans_cancelled['OrderPrice'])
    

#    temp = transaction[['InstrumentId','datetime','TradeDirection','ActOrderId','TradeType']].groupby(['datetime','ActOrderId']).last()
#    temp[['NetVolume','TradeVolume','TradeAmount']] =transaction.groupby(['datetime','ActOrderId']).sum()[['NetVolume','TradeVolume','TradeAmount']]
#    temp['TradePrice'] = temp['TradeAmount']/temp['TradeVolume']
#    temp['HighestPrice'] = transaction[['datetime','TradePrice','ActOrderId']].groupby(['datetime','ActOrderId']).max()['TradePrice']
#    temp['LowestPrice'] = transaction[['datetime','TradePrice','ActOrderId']].groupby(['datetime','ActOrderId']).min()['TradePrice']
#    temp = temp.reset_index()
#    temp = temp.merge(order.rename(columns={'datetime':'OrderTime'})[['OrderTime','OrderPrice',
#                                                      'OrderId','OrderVolume',
#                                                      'OrderPriceType','OrderDirection']],left_on='ActOrderId',right_on='OrderId',how='left')
#    temp_sort_buy = temp[temp['TradeDirection']==1].sort_values(['TradeVolume','datetime'])
  
    am = dt.datetime(year,month,day,11,30,0,0)
    pm1 = dt.datetime(year,month,day,13,0,0,0)+window
    pm = dt.datetime(year,month,day,14,56,0,0)
    


    t = snapshot_formerge[snapshot_formerge['datetime']>=dt.datetime(year,month,day,9,30,0,0)]['datetime'].iloc[0]

    d1={}
    while t < pm:  
        if (t>am) & (t<pm1):
            t = snapshot_formerge[snapshot_formerge['datetime']>=pm1]['datetime'].iloc[0]
            continue
        d1[t]={}
        traded = trans_traded_grouped[(trans_traded_grouped['datetime']>=t) & (trans_traded_grouped['datetime']<t+window)]
        
        if traded.empty:
            t+=dt.timedelta(seconds=3)
            continue
        t1=time()
        d1[t]['trade_volume'] = traded['TradeVolume'].values.sum()
        d1[t]['buy_volume'] = (traded['TradeVolume'].values*(traded['TradeDirection'].values+1)*0.5).sum()
        d1[t]['sell_volume'] = -(traded['TradeVolume'].values*(traded['TradeDirection'].values-1)*0.5).sum()

        
        cancelled = trans_cancelled[(trans_cancelled['datetime']>=t) & (trans_cancelled['datetime']<t+window)]

        #trans = transaction[(transaction['datetime']>t) & (transaction['datetime']<=t+window)]
        snaps = snapshot_formerge[(snapshot['datetime']>=t) & (snapshot_formerge['datetime']<t+window)]        
        order_t = order[(order['datetime']>=t) & (order['datetime']<t+window)]

        ##########

        min_quote = np.nanmin(snaps[['BuyPrice1','BuyPrice2','BuyPrice3','BuyPrice4','BuyPrice5']].values)
        max_quote = np.nanmax(snaps[['SellPrice1','SellPrice2','SellPrice3','SellPrice4','SellPrice5']].values)
        if min_quote!=np.nan and max_quote!=np.nan:
            order_range = order_t[(order_t['OrderPrice'].values>=min_quote) & (order_t['OrderPrice'].values<=max_quote)]
            d1[t]['new_buy_vol_above_5'] = (0.5*(order_range['Direction'].values+1)*order_range['OrderVolume'].values).sum()
            d1[t]['new_buy_vwap_above_5'] = (0.5*(order_range['Direction'].values+1)*order_range['OrderAmount'].values).sum()/d1[t]['new_buy_vol_above_5']

            d1[t]['new_sell_vol_above_5'] = -(0.5*(order_range['Direction'].values-1)*order_range['OrderVolume'].values).sum()
            d1[t]['new_sell_vwap_above_5'] = -(0.5*(order_range['Direction'].values-1)*order_range['OrderAmount'].values).sum()/d1[t]['new_sell_vol_above_5']
            #-------
            cancelled_range = cancelled[(cancelled['OrderPrice']<=max_quote) & (cancelled['OrderPrice']>=min_quote)]
            d1[t]['cancelled_buy_vol_above_5'] = ((cancelled_range['TradeDirection'].values+1)*0.5*cancelled_range['TradeVolume']).sum()
            d1[t]['cancelled_buy_vwap_above_5'] = ((cancelled_range['TradeDirection'].values+1)*0.5*cancelled_range['CancelledTradeAmount']).sum()/d1[t]['cancelled_buy_vol_above_5'] 
            d1[t]['cancelled_sell_vol_above_5'] = -((cancelled_range['TradeDirection'].values-1)*0.5*cancelled_range['TradeVolume']).sum()
            d1[t]['cancelled_sell_vwap_above_5'] = -((cancelled_range['TradeDirection'].values-1)*0.5*cancelled_range['CancelledTradeAmount']).sum()/d1[t]['cancelled_sell_vol_above_5']     
        ###########        
        #-----
        min_quote = traded['LowestPrice'].values.min()
        max_quote = traded['HighestPrice'].values.max()
        order_range = order_t[(order_t['OrderPrice'].values>=min_quote) & (order_t['OrderPrice'].values<=max_quote)]
        cancelled_range = cancelled[(cancelled['OrderPrice']<=max_quote) & (cancelled['OrderPrice']>=min_quote)]
        d1[t]['new_buy_vol_above_lowest'] = (0.5*(order_range['Direction'].values+1)*order_range['OrderVolume'].values).sum()
        d1[t]['new_buy_vwap_above_lowest'] = (0.5*(order_range['Direction'].values+1)*order_range['OrderAmount'].values).sum()/d1[t]['new_buy_vol_above_lowest']
        d1[t]['new_sell_vol_above_lowest'] = -(0.5*(order_range['Direction'].values-1)*order_range['OrderVolume'].values).sum()
        d1[t]['new_sell_vwap_above_lowest'] = -(0.5*(order_range['Direction'].values-1)*order_range['OrderAmount'].values).sum()/d1[t]['new_sell_vol_above_5']

        d1[t]['cancelled_buy_vol_above_lowest'] = ((cancelled_range['TradeDirection'].values+1)*0.5*cancelled_range['TradeVolume']).sum()
        d1[t]['cancelled_buy_vwap_above_lowest'] = ((cancelled_range['TradeDirection'].values+1)*0.5*cancelled_range['CancelledTradeAmount']).sum()/d1[t]['cancelled_buy_vol_above_lowest'] 
        d1[t]['cancelled_sell_vol_above_lowest'] = -((cancelled_range['TradeDirection'].values-1)*0.5*cancelled_range['TradeVolume']).sum()
        d1[t]['cancelled_sell_vwap_above_lowest'] = -((cancelled_range['TradeDirection'].values-1)*0.5*cancelled_range['CancelledTradeAmount']).sum()/d1[t]['cancelled_sell_vol_above_lowest']  
        #########
        d1[t]['traded_buy_vol'] = (0.5*(traded['TradeDirection'].values+1)*traded['TradeVolume'].values).sum()
        d1[t]['traded_buy_vwap'] = (0.5*(traded['TradeDirection'].values+1)*traded['TradeAmount'].values).sum()/d1[t]['traded_buy_vol']
        d1[t]['traded_sell_vol'] = -(0.5*(traded['TradeDirection'].values-1)*traded['TradeVolume'].values).sum()
        d1[t]['traded_sell_vwap'] = -(0.5*(traded['TradeDirection'].values-1)*traded['TradeAmount'].values).sum()/d1[t]['traded_sell_vol']
        
        #########
        t+=dt.timedelta(seconds=3)


    
    df = pd.DataFrame.from_dict(d1,orient='index')
    df['InstrumentId'] = int(file[:6])
    df['new_all_vol_above_5'] = df['new_buy_vol_above_5'].values+df['new_sell_vol_above_5'].values
    df['new_all_vwap_above_5'] = (df['new_buy_vwap_above_5'].values*df['new_buy_vol_above_5'].values+df['new_sell_vwap_above_5'].values*df['new_sell_vol_above_5'].values)/df['new_all_vol_above_5'].values
    df['traded_all_vol'] = df['traded_buy_vol'].values + df['traded_sell_vol'].values
    df['traded_all_vwap'] = (df['traded_buy_vwap'].values*df['traded_buy_vol'].values + df['traded_sell_vwap'].values*df['traded_sell_vol'].values)/df['traded_all_vol'].values
    
    df['net_new_buy_vol_above_5'] = df['new_buy_vol_above_5']-df['cancelled_buy_vol_above_5']
    df['net_new_sell_vol_above_5'] = df['new_sell_vol_above_5']-df['cancelled_sell_vol_above_5']
    
    df['net_new_buy_vol_above_lowest'] = df['new_buy_vol_above_lowest']-df['cancelled_buy_vol_above_lowest']
    df['net_new_sell_vol_above_lowest'] = df['new_sell_vol_above_lowest']-df['cancelled_sell_vol_above_lowest']

    df.to_csv(output_file_path+file)
    gc.collect()
    print(n)
    n+=1
    
    
    
    
temp_snap = snaps[lp]




all vol
all vwap


temp_snap = snaps[lp].copy()
temp_snap['BuyPrice4']=np.nan
temp_snap.loc[31,'BuyPrice4']=3
temp_snap.min()



        
   ?pd.merge     
        
        
        
        d1[t]['trade_volume_imbalance'] = 
        traded['TradePrice'].max()
        traded['TradePrice'].min()
    

order_files


