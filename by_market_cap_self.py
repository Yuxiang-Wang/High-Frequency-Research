# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 16:51:10 2019

1、按市值分研究挂单净增量因子
2、研究因子散点图各区域分布

@author: caoqiliang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from time import time
import os
import gc
%matplotlib inline

fac_path = 'fac_1m/20180903/'
l = os.listdir(fac_path)
fac_data = []
for i in range(len(l)): # i=0;
    file = l[i]
    fac_data.append(pd.read_csv(fac_path+file,index_col=0))
fac_data = pd.concat(fac_data)
fac_data['datetime']=pd.to_datetime(fac_data['datetime'],format='%Y-%m-%d %H:%M:%S')
fac_data['ret_vwapsell_fur'] = np.log(fac_data['vwap_fur']/fac_data['SellPrice1'])


daily = pd.read_csv('daily/201809StockDailyData.csv')
daily['InstrumentId'] = daily['WindCode'].apply(lambda x:int(x[:6]))
daily_SH = daily[daily['WindCode'].apply(lambda x:True if 'SH' in x else False)]
today = daily_SH[daily_SH['TradingDay']==20180903]

temp = fac_data.merge(today[['InstrumentId','MarketCapAFloat']],on = ['InstrumentId'],how = 'left')


conditions = [temp['MarketCapAFloat']>1e10,
              (temp['MarketCapAFloat']<1e10) & (temp['MarketCapAFloat']>5e9),
              (temp['MarketCapAFloat']<5e9) & (temp['MarketCapAFloat']>3e9),
              (temp['MarketCapAFloat']<3e9) & (temp['MarketCapAFloat']>1.5e9),
              (temp['MarketCapAFloat']<1.5e9)]
cons = ['>1e10','5e9 - 1e10','3e9 - 5e9','1.5e9 - 3e9','<1.5e9']
#fac_range = temp[temp['MarketCapAFloat']>1e10]
#fac_range = temp[(temp['MarketCapAFloat']<1e10) & (temp['MarketCapAFloat']>5e9)]

# --------------------------------------
def plot(temp,factor,log=False,col=None):
    if log:
        temp = np.log(temp)
    plt.axhline(y=0,color='r')
    #plt.axhline(y=0.001,color='r')
    #plt.axhline(y=-0.001,color='r')
    plt.axvline(x=0,color='r')
    plt.scatter(temp,factor['ret_vwapsell_fur'],linewidths=0.001)
    if col:
        plt.title(col)
    plt.xlim(min(temp)*0.9,max(temp)*1.1)
    plt.ylim(min(factor['ret_vwapsell_fur'])-0.005,max(factor['ret_vwapsell_fur'])+0.005)
    #plt.plot([-10000,10000],[0,0],color='r')
    plt.show()
    print('''pct:
        1: %f  2: %f
        3: %f  4: %f''' % (
        (factor[temp<0]['ret_vwapsell_fur']>0).sum()/len(factor),
        (factor[temp>0]['ret_vwapsell_fur']>0).sum()/len(factor),
        (factor[temp<0]['ret_vwapsell_fur']<0).sum()/len(factor),
        (factor[temp>0]['ret_vwapsell_fur']<0).sum()/len(factor)))

######### sell
for i in range(len(conditions)):
    con = conditions[i]
    fac_range = temp[con]
    range_noinf = fac_range[(fac_range['resist_sell_lag']>0) & (fac_range['resist_sell_lag']<np.inf)]
    plot(range_noinf['resist_sell_lag'],range_noinf,1,'sell, cap '+cons[i])
    
######### buy
for i in range(len(conditions)):
    con = conditions[i]
    fac_range = temp[con]
    range_noinf = fac_range[(fac_range['resist_buy_lag']>0) & (fac_range['resist_buy_lag']<np.inf)]
    plot(range_noinf['resist_buy_lag'],range_noinf,1,'buy, cap '+cons[i])


# -------------------------------------
summary = []
for con in conditions:
    fac_range = temp[con]
    print(len(fac_range),len(fac_range['InstrumentId'].unique()))
    n = len(fac_range['InstrumentId'].unique())
    
    #--------------------------
    d={}
    ############### range,sell
    range_noinf = fac_range[(fac_range['resist_sell_lag']>0) & (fac_range['resist_sell_lag']<1)]
    for i in range(7):
        temp_range_noinf = range_noinf[np.log(range_noinf['resist_sell_lag'])<-i]
#        temp_range_noinf['ret_vwapsell_fur'].hist(bins=100)
#        plt.show()
#        print('''%d
#              median: %f
#              mean:   %f
#              std:    %f
#              >0 pct: %f''' % (-i,temp_range_noinf['ret_vwapsell_fur'].median(),
#              temp_range_noinf['ret_vwapsell_fur'].mean(),
#              temp_range_noinf['ret_vwapsell_fur'].std(),
#              (temp_range_noinf['ret_vwapsell_fur']>0).sum()/len(temp_range_noinf)))
        
        d[(-i,'range,sell')]=dict(zip(('median','mean','std','>0 pct','95% VaR','num pct',
          'ave signals','num stock'),(temp_range_noinf['ret_vwapsell_fur'].median(),
              temp_range_noinf['ret_vwapsell_fur'].mean(),
              temp_range_noinf['ret_vwapsell_fur'].std(),
              (temp_range_noinf['ret_vwapsell_fur']>0).sum()/len(temp_range_noinf),
              temp_range_noinf['ret_vwapsell_fur'].quantile(0.05),
              len(temp_range_noinf)/len(fac_range),
              len(temp_range_noinf)/n,
              n)))
    
    ############### range,buy
    range_noinf = fac_range[(fac_range['resist_buy_lag']>1) & (fac_range['resist_buy_lag']<np.inf)]
    for i in range(7):
        temp_range_noinf = range_noinf[np.log(range_noinf['resist_buy_lag'])>i]
#        temp_range_noinf['ret_vwapsell_fur'].hist(bins=100)
#        plt.show()
#        print('''%d
#              median: %f
#              mean:   %f
#              std:    %f
#              >0 pct: %f''' % (i,temp_range_noinf['ret_vwapsell_fur'].median(),
#              temp_range_noinf['ret_vwapsell_fur'].mean(),
#              temp_range_noinf['ret_vwapsell_fur'].std(),
#              (temp_range_noinf['ret_vwapsell_fur']>0).sum()/len(temp_range_noinf)))
        
        d[(i,'range,buy')]=dict(zip(('median','mean','std','>0 pct','95% VaR','num pct',
          'ave signals','num stock'),(temp_range_noinf['ret_vwapsell_fur'].median(),
              temp_range_noinf['ret_vwapsell_fur'].mean(),
              temp_range_noinf['ret_vwapsell_fur'].std(),
              (temp_range_noinf['ret_vwapsell_fur']>0).sum()/len(temp_range_noinf),
              temp_range_noinf['ret_vwapsell_fur'].quantile(0.05),
              len(temp_range_noinf)/len(fac_range),
              len(temp_range_noinf)/n,
              n)))
    
    ###############
    #summary = pd.DataFrame.from_dict(d,orient='index')
    summary.append(pd.DataFrame.from_dict(d,orient='index'))



# ---------------------------------------------

today['VWAvgPrice'].sort_values().iloc[:-100].hist(bins=100)















    