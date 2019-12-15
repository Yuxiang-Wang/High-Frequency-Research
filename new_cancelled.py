# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:52:34 2019

new order data.
cancelled order

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
    transaction = transaction.merge(order[['OrderId','datetime']].rename(columns={'OrderId':'BuyId',
                                    'datetime':'BuyTime'}),left_on='BuyOrderId',right_on='BuyId',how='left')
    transaction = transaction.merge(order[['OrderId','datetime']].rename(columns={'OrderId':'SellId',
                                    'datetime':'SellTime'}),left_on='SellOrderId',right_on='SellId',how='left')
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
    
    
    
    temp1 = trans_cancelled[['InstrumentId','datetime','ActOrderId','TradeType','OrderDirection','OrderPrice','TradeVolume','OrderTime']].rename(columns={'ActOrderId':'OrderId',
                   'TradeType':'OrderPriceType','TradeVolume':'OrderVolume'})
    temp2 = order[['InstrumentId','datetime','OrderId','OrderPriceType','OrderDirection','OrderPrice','OrderVolume']]
    
    temp = pd.concat([temp2,temp1])
    temp = temp.sort_values(['InstrumentId','datetime'])
    

temp_duplicate = temp[temp.duplicated(['InstrumentId','OrderDirection','OrderVolume'],keep=False)]
temp_duplicate = temp_duplicate.sort_values(['OrderVolume','datetime'])
    
    
temp_duplicate = temp[temp.duplicated]   
    
    
    