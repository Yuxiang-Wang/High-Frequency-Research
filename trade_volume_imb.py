# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 14:22:36 2019

成交量不平衡因子按市值跟价格分类研究

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

fac_path = 'fac_1m/20180903/'
#fac_path = 'fac_5m/20180903/'
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

temp = fac_data.merge(today[['InstrumentId','MarketCapAFloat']],on = ['InstrumentId'],how = 'left')

def plot(temp,factor,log=False,col=None,mid=0.5):
    if log:
        temp = np.log(temp)
    plt.axhline(y=0,color='r',linewidth=0.8)
    #plt.axhline(y=0.001,color='r')
    #plt.axhline(y=-0.001,color='r')
    plt.axvline(x=mid,color='r',linewidth=0.8)
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
    plt.axvline(x=0.5,color='r',linewidth=0.8)
    plt.axhline(y=0,color='r',linewidth=0.8)
    plt.grid(True)
    if label:
        plt.legend(labels=[label])
    if show:
        plt.show()
    #print('zero point',x[list(map(abs,y)).index(min(list(map(abs,y))))])
        
#-------------------
for i in range(5,10):
    plot(temp['trade_volume_imbalance_lag'],temp,mid=0.1*i)
    
#-------------------
plot(temp['trade_volume_imbalance_lag'],temp)

interval_tvi = [(i/100,(i+1)/100) for i in range(0,100)]
plot_mean(temp['trade_volume_imbalance_lag'],temp,interval_tvi)
#x=[]
#y=[]
#for interval in [(i/100,(i+1)/100) for i in range(0,100)]:
#    temp_interval = temp[(temp['trade_volume_imbalance_lag']>interval[0])&(temp['trade_volume_imbalance_lag']<=interval[1])]
#    y.append(temp_interval['ret_vwapmid_fur'].mean())
#    x.append(0.5*(interval[0]+interval[1]))
#plt.scatter(x,y,linewidth=0.001)
#plt.ylim(min(y),max(y))
#ind = x[list(map(abs,y)).index(min(list(map(abs,y))))]
#plt.axvline(x=ind,color='r')
#plt.axhline(y=0,color='r')
#plt.show()
#print('zero point',x[list(map(abs,y)).index(min(list(map(abs,y))))])

#--------------------

conditions_price = [temp['vwap_lag']>20,
                    (temp['vwap_lag']<=20) & (temp['vwap_lag']>10),
                    (temp['vwap_lag']<=10) & (temp['vwap_lag']>5),
                    (temp['vwap_lag']<=5)]              
cons_price = ['>20','10 - 20','5 - 10','<5']

conditions_cap = [temp['MarketCapAFloat']>1e10,
                  (temp['MarketCapAFloat']<1e10) & (temp['MarketCapAFloat']>5e9),
                  (temp['MarketCapAFloat']<5e9) & (temp['MarketCapAFloat']>3e9),
                  (temp['MarketCapAFloat']<3e9) & (temp['MarketCapAFloat']>1.5e9),
                  (temp['MarketCapAFloat']<1.5e9)]
cons_cap = ['>1e10','5e9 - 1e10','3e9 - 5e9','1.5e9 - 3e9','<1.5e9']


######### 
for i in range(len(conditions_price)):
    con = conditions_price[i]
    cons = cons_price[i]
    range_noinf = temp[con]
    plot(range_noinf['trade_volume_imbalance_lag'],range_noinf,0,'price '+cons,mid=0.5)
    print('''
          num stocks:   %f
          ave spread:   %f
          ave price:    %f
          spread/price: %f''' % (
          len(range_noinf['InstrumentId'].unique()),
          (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean(),
          range_noinf['vwap_lag'].mean(),
          (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean()/range_noinf['vwap_lag'].mean()))

for i in range(len(conditions_price)):
    con = conditions_price[i]
    cons = cons_price[i]
    range_noinf = temp[con]
    plot_mean(range_noinf['trade_volume_imbalance_lag'],range_noinf,interval_tvi,label=cons)



######### 
for i in range(len(conditions_cap)):
    con = conditions_cap[i]
    cons = cons_cap[i]
    range_noinf = temp[con]
    plot(range_noinf['trade_volume_imbalance_lag'],range_noinf,0,'cap '+cons,mid=0.5)
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
    range_noinf = temp[con]
    plot_mean(range_noinf['trade_volume_imbalance_lag'],range_noinf,interval_tvi,label=cons)


#########
for i in range(len(conditions_price)):
    for j in range(len(conditions_cap)):
        con = conditions_price[i] & conditions_cap[j]
        cons = cons_price[i] + '+' + cons_cap[j]
        #fac_range = temp[con]
        range_noinf = temp[con]
        plot(range_noinf['trade_volume_imbalance_lag'],range_noinf,0,'price '+cons,mid=0.5)
        print('''
              num stocks:   %f
              ave spread:   %f
              ave price:    %f
              spread/price: %f''' % (
              len(range_noinf['InstrumentId'].unique()),
              (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean(),
              range_noinf['vwap_lag'].mean(),
              (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean()/range_noinf['vwap_lag'].mean()))

for i in range(len(conditions_price)):
    for j in range(len(conditions_cap)):
        con = conditions_price[i] & conditions_cap[j]
        cons = cons_price[i] + '+' + cons_cap[j]
        range_noinf = temp[con]
        plot_mean(range_noinf['trade_volume_imbalance_lag'],range_noinf,interval_tvi,label=cons)






