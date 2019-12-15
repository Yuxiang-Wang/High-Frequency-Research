# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 13:22:19 2019

也是按市值分类研究挂单净增量因子

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

# gc.collect()

#fac_path = 'fac_1m/20180903/'
fac_path = 'fac_5m/20180903/'
l = os.listdir(fac_path)
fac_data = []
for i in range(len(l)): # i=0;
    file = l[i]
    fac_data.append(pd.read_csv(fac_path+file,index_col=0))
fac_data = pd.concat(fac_data)
fac_data['datetime']=pd.to_datetime(fac_data['datetime'],format='%Y-%m-%d %H:%M:%S')

# fac_data.rename(columns = {'new_buy_lag':'resist_buy_lag','new_sell_lag':'resist_sell_lag'},inplace=True)

daily = pd.read_csv('daily/201809StockDailyData.csv')
daily['InstrumentId'] = daily['WindCode'].apply(lambda x:int(x[:6]))
daily_SH = daily[daily['WindCode'].apply(lambda x:True if 'SH' in x else False)]
today = daily_SH[daily_SH['TradingDay']==20180903]
today['free_cap'] = today['VWAvgPrice']*today['NonRestrictedShares']/today['SplitFactor']


#temp = fac_data.merge(today[['InstrumentId','MarketCapAFloat']],on = ['InstrumentId'],how = 'left')
#conditions_cap = [temp['MarketCapAFloat']>1e10,
#              (temp['MarketCapAFloat']<1e10) & (temp['MarketCapAFloat']>5e9),
#              (temp['MarketCapAFloat']<5e9) & (temp['MarketCapAFloat']>3e9),
#              (temp['MarketCapAFloat']<3e9) & (temp['MarketCapAFloat']>1.5e9),
#              (temp['MarketCapAFloat']<1.5e9)]

temp = fac_data.merge(today[['InstrumentId','free_cap']],on = ['InstrumentId'],how = 'left')
conditions_cap = [temp['free_cap']>1e10,
              (temp['free_cap']<1e10) & (temp['free_cap']>5e9),
              (temp['free_cap']<5e9) & (temp['free_cap']>3e9),
              (temp['free_cap']<3e9) & (temp['free_cap']>1.5e9),
              (temp['free_cap']<1.5e9)]
cons_cap = ['>1e10','5e9 - 1e10','3e9 - 5e9','1.5e9 - 3e9','<1.5e9']
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
    plt.scatter(temp,factor['ret_vwapmid_fur'],linewidths=0.001,alpha=0.5)
    if col:
        plt.title(col)
    plt.xlim(min(temp)*0.9,max(temp)*1.1)
    plt.ylim(min(factor['ret_vwapmid_fur'])-0.005,max(factor['ret_vwapmid_fur'])+0.005)
    #plt.plot([-10000,10000],[0,0],color='r')
    plt.show()
    print('''pct:
        1: %f  2: %f
        3: %f  4: %f
        left mean: %f  right mean: %f''' % (
        (factor[temp<0]['ret_vwapmid_fur']>0).sum()/len(factor),
        (factor[temp>0]['ret_vwapmid_fur']>0).sum()/len(factor),
        (factor[temp<0]['ret_vwapmid_fur']<0).sum()/len(factor),
        (factor[temp>0]['ret_vwapmid_fur']<0).sum()/len(factor),
        factor[temp<0]['ret_vwapmid_fur'].mean(),
        factor[temp>0]['ret_vwapmid_fur'].mean()))


def plot_mean(factor,temp,intervals,log=False,show=True,label=None,ylim = None):
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
    if ylim:
        plt.ylim(ylim)
    else:
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

#------------------
arr = np.linspace(-6,6,100)
interval_res = np.vstack([arr[:-1],arr[1:]]).T
lim = [min(temp['ret_vwapmid_fur']),max(temp['ret_vwapmid_fur'])]

######### sell

fac_range = temp[(temp['resist_sell_lag']>0) & (temp['resist_sell_lag']<np.inf)]
plot_mean(fac_range['resist_sell_lag'],fac_range,interval_res,log=True)


for i in range(len(conditions_cap)):
    con = conditions_cap[i]
    fac_range = temp[con]
    range_noinf = fac_range[(fac_range['resist_sell_lag']>0) & (fac_range['resist_sell_lag']<np.inf)]
    plot(range_noinf['resist_sell_lag'],range_noinf,1,'sell, price '+cons_cap[i])
    print('''
          num stocks:   %f
          ave spread:   %f
          ave price:    %f
          spread/price: %f''' % (
          len(range_noinf['InstrumentId'].unique()),
          (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean(),
          range_noinf['vwap_lag'].mean(),
          (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean()/range_noinf['vwap_lag'].mean()))

for i in range(len(conditions_cap)):
    con = conditions_cap[i]
    cons = cons_cap[i]
    fac_range = temp[con]
    range_noinf = fac_range[(fac_range['resist_sell_lag']>0) & (fac_range['resist_sell_lag']<np.inf)]
    plot_mean(range_noinf['resist_sell_lag'],range_noinf,interval_res,log=True,label=cons)


######### buy
for i in range(len(conditions_cap)):
    con = conditions_cap[i]
    fac_range = temp[con]
    range_noinf = fac_range[(fac_range['resist_buy_lag']>0) & (fac_range['resist_buy_lag']<np.inf)]
    plot(range_noinf['resist_buy_lag'],range_noinf,1,'buy, price '+cons_cap[i])
    print('''
          num stocks:   %f
          ave spread:   %f
          ave price:    %f
          spread/price: %f''' % (
          len(range_noinf['InstrumentId'].unique()),
          (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean(),
          range_noinf['vwap_lag'].mean(),
          (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean()/range_noinf['vwap_lag'].mean()))

for i in range(len(conditions_cap)):
    con = conditions_cap[i]
    cons = cons_cap[i]
    fac_range = temp[con]
    range_noinf = fac_range[(fac_range['resist_buy_lag']>0) & (fac_range['resist_buy_lag']<np.inf)]
    plot_mean(range_noinf['resist_buy_lag'],range_noinf,interval_res,log=True,label=cons)


# -------------------------------------
summary = {}
#for con in conditions: # con = conditions[4]; cons[4]
for j in range(len(conditions_cap)): # j=0
    con = conditions_cap[j]
    cons = cons_cap[j]
    fac_range = temp[con]
    print(len(fac_range),len(fac_range['InstrumentId'].unique()))
    n = len(fac_range['InstrumentId'].unique())
    
    #--------------------------
    d={}
    ############### range,sell
    range_noinf = fac_range[(fac_range['resist_sell_lag']>0) & (fac_range['resist_sell_lag']<1)]
    for i in range(7):
        temp_range_noinf = range_noinf[np.log(range_noinf['resist_sell_lag'])<-i]
#        temp_range_noinf['ret_vwapmid_fur'].hist(bins=100)
#        plt.show()
#        
#        print('''%d
#              median:      %f
#              mean:        %f
#              std:         %f
#              >0 pct:      %f
#              spread:      %f
#              >1/2spread:  %f
#              >-1/2spread: %f''' % (-i,temp_range_noinf['ret_vwapmid_fur'].median(),
#              temp_range_noinf['ret_vwapmid_fur'].mean(),
#              temp_range_noinf['ret_vwapmid_fur'].std(),
#              (temp_range_noinf['ret_vwapmid_fur']>0).sum()/len(temp_range_noinf),
#              (temp_range_noinf['SellPrice1']-temp_range_noinf['BuyPrice1']).mean()/temp_range_noinf['vwap_lag'].mean(),
#              (temp_range_noinf['ret_vwapmid_fur']>0.5*(temp_range_noinf['SellPrice1']-temp_range_noinf['BuyPrice1']).mean()/temp_range_noinf['vwap_lag'].mean()).sum()/len(temp_range_noinf),
#              (temp_range_noinf['ret_vwapmid_fur']>-0.5*(temp_range_noinf['SellPrice1']-temp_range_noinf['BuyPrice1']).mean()/temp_range_noinf['vwap_lag'].mean()).sum()/len(temp_range_noinf)))
#        
        d[(-i,cons+',sell')]=dict(zip(('median','mean','mean-0.5spread','std','>0 pct','95% VaR','spread ret','>1/2spread','>-1/2spread',
          'ave signals','num stock'),(temp_range_noinf['ret_vwapmid_fur'].median(),
              temp_range_noinf['ret_vwapmid_fur'].mean(),
              temp_range_noinf['ret_vwapmid_fur'].mean() - 0.5*(temp_range_noinf['SellPrice1']-temp_range_noinf['BuyPrice1']).mean()/temp_range_noinf['vwap_lag'].mean(),
              temp_range_noinf['ret_vwapmid_fur'].std(),
              (temp_range_noinf['ret_vwapmid_fur']>0).sum()/len(temp_range_noinf),
              temp_range_noinf['ret_vwapmid_fur'].quantile(0.05),
              (temp_range_noinf['SellPrice1']-temp_range_noinf['BuyPrice1']).mean()/temp_range_noinf['vwap_lag'].mean(),
              (temp_range_noinf['ret_vwapmid_fur']>0.5*(temp_range_noinf['SellPrice1']-temp_range_noinf['BuyPrice1']).mean()/temp_range_noinf['vwap_lag'].mean()).sum()/len(temp_range_noinf),
              (temp_range_noinf['ret_vwapmid_fur']>-0.5*(temp_range_noinf['SellPrice1']-temp_range_noinf['BuyPrice1']).mean()/temp_range_noinf['vwap_lag'].mean()).sum()/len(temp_range_noinf),
              len(temp_range_noinf)/n,
              n)))
    
    ############### range,buy
    range_noinf = fac_range[(fac_range['resist_buy_lag']>1) & (fac_range['resist_buy_lag']<np.inf)]
    for i in range(7):
        temp_range_noinf = range_noinf[np.log(range_noinf['resist_buy_lag'])>i]
#        temp_range_noinf['ret_vwapmid_fur'].hist(bins=100)
#        plt.show()
#        print('''%d
#              median:      %f
#              mean:        %f
#              std:         %f
#              >0 pct:      %f
#              spread:      %f
#              >1/2spread:  %f
#              >-1/2spread: %f''' % (i,temp_range_noinf['ret_vwapmid_fur'].median(),
#              temp_range_noinf['ret_vwapmid_fur'].mean(),
#              temp_range_noinf['ret_vwapmid_fur'].std(),
#              (temp_range_noinf['ret_vwapmid_fur']>0).sum()/len(temp_range_noinf),
#              (temp_range_noinf['SellPrice1']-temp_range_noinf['BuyPrice1']).mean()/temp_range_noinf['vwap_lag'].mean(),
#              (temp_range_noinf['ret_vwapmid_fur']>0.5*(temp_range_noinf['SellPrice1']-temp_range_noinf['BuyPrice1']).mean()/temp_range_noinf['vwap_lag'].mean()).sum()/len(temp_range_noinf),
#              (temp_range_noinf['ret_vwapmid_fur']>-0.5*(temp_range_noinf['SellPrice1']-temp_range_noinf['BuyPrice1']).mean()/temp_range_noinf['vwap_lag'].mean()).sum()/len(temp_range_noinf)))
#        
        d[(i,cons+',buy')]=dict(zip(('median','mean','mean-0.5spread','std','>0 pct','95% VaR','spread ret','>1/2spread','>-1/2spread',
          'ave signals','num stock'),(temp_range_noinf['ret_vwapmid_fur'].median(),
              temp_range_noinf['ret_vwapmid_fur'].mean(),
              temp_range_noinf['ret_vwapmid_fur'].mean() - 0.5*(temp_range_noinf['SellPrice1']-temp_range_noinf['BuyPrice1']).mean()/temp_range_noinf['vwap_lag'].mean(),
              temp_range_noinf['ret_vwapmid_fur'].std(),
              (temp_range_noinf['ret_vwapmid_fur']>0).sum()/len(temp_range_noinf),
              temp_range_noinf['ret_vwapmid_fur'].quantile(0.05),
              (temp_range_noinf['SellPrice1']-temp_range_noinf['BuyPrice1']).mean()/temp_range_noinf['vwap_lag'].mean(),
              (temp_range_noinf['ret_vwapmid_fur']>0.5*(temp_range_noinf['SellPrice1']-temp_range_noinf['BuyPrice1']).mean()/temp_range_noinf['vwap_lag'].mean()).sum()/len(temp_range_noinf),
              (temp_range_noinf['ret_vwapmid_fur']>-0.5*(temp_range_noinf['SellPrice1']-temp_range_noinf['BuyPrice1']).mean()/temp_range_noinf['vwap_lag'].mean()).sum()/len(temp_range_noinf),
              len(temp_range_noinf)/n,
              n)))
    
    ###############
    #summary = pd.DataFrame.from_dict(d,orient='index')
    summary[cons] = pd.DataFrame.from_dict(d,orient='index')


################################## correlation

# sell
for i in range(len(conditions_cap)):
    con = conditions_cap[i]
    fac_range = temp[con]
    range_noinf = fac_range[(fac_range['resist_sell_lag']>0) & (fac_range['resist_sell_lag']<np.inf)]     
    print(np.corrcoef(np.log(range_noinf['resist_sell_lag']),range_noinf['ret_vwapmid_fur'])[0][1])

# buy
for i in range(len(conditions_cap)):
    con = conditions_cap[i]
    fac_range = temp[con]
    range_noinf = fac_range[(fac_range['resist_buy_lag']>0) & (fac_range['resist_buy_lag']<np.inf)]     
    print(np.corrcoef(np.log(range_noinf['resist_buy_lag']),range_noinf['ret_vwapmid_fur'])[0][1])
  
  
    
    
    
    
    
    
    