# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 15:09:13 2019

大小单按市值跟价格分类

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
print(df.shape)

daily = pd.read_csv('daily/201809StockDailyData.csv')
daily['InstrumentId'] = daily['WindCode'].apply(lambda x:int(x[:6]))
daily_SH = daily[daily['WindCode'].apply(lambda x:True if 'SH' in x else False)]
today = daily_SH[daily_SH['TradingDay']==20180903]
today['free_cap'] = today['VWAvgPrice']*today['NonRestrictedShares']/today['SplitFactor']

#temp = fac_data.merge(today[['InstrumentId','MarketCapAFloat']],on = ['InstrumentId'],how = 'left')
temp = df.merge(today[['InstrumentId','free_cap','NonRestrictedShares']],on = ['InstrumentId'],how = 'left')

gc.collect()

plt.plot([0,1e13],[0,1e13],c='r',alpha=0.3)
plt.scatter(today['MarketCapAFloat'],today['free_cap'],linewidth=0.1,alpha=0.5)

conditions_price = [temp['vwap_lag']>20,
                    (temp['vwap_lag']<=20) & (temp['vwap_lag']>10),
                    (temp['vwap_lag']<=10) & (temp['vwap_lag']>5),
                    (temp['vwap_lag']<=5)]              
cons_price = ['>20','10 - 20','5 - 10','<5']

#conditions_cap = [temp['MarketCapAFloat']>1e10,
#                  (temp['MarketCapAFloat']<1e10) & (temp['MarketCapAFloat']>5e9),
#                  (temp['MarketCapAFloat']<5e9) & (temp['MarketCapAFloat']>3e9),
#                  (temp['MarketCapAFloat']<3e9) & (temp['MarketCapAFloat']>1.5e9),
#                  (temp['MarketCapAFloat']<1.5e9)]

conditions_cap = [temp['free_cap']>1e10,
              (temp['free_cap']<1e10) & (temp['free_cap']>5e9),
              (temp['free_cap']<5e9) & (temp['free_cap']>3e9),
              (temp['free_cap']<3e9) & (temp['free_cap']>1.5e9),
              (temp['free_cap']<1.5e9)]
cons_cap = ['>1e10','5e9 - 1e10','3e9 - 5e9','1.5e9 - 3e9','<1.5e9']

# ---------------------------------------------
def plot(temp,factor,log=False,col=None,mid=0,quadratic=False):
    if log:
        temp = np.log(temp)
    elif quadratic:
        temp = temp*temp
    plt.axhline(y=0,color='r')
    #plt.axhline(y=0.001,color='r')
    #plt.axhline(y=-0.001,color='r')
    plt.axvline(x=mid,color='r')
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
        ret>0 in sample: %f
        left mean: %f  right mean: %f''' % (
        (factor[temp<mid]['ret_vwapmid_fur']>0).sum()/len(factor),
        (factor[temp>mid]['ret_vwapmid_fur']>0).sum()/len(factor),
        (factor[temp<mid]['ret_vwapmid_fur']<0).sum()/len(factor),
        (factor[temp>mid]['ret_vwapmid_fur']<0).sum()/len(factor),
        (factor['ret_vwapmid_fur']>0).sum()/len(factor),
        factor[temp<mid]['ret_vwapmid_fur'].mean(),
        factor[temp>mid]['ret_vwapmid_fur'].mean()))

        
def plot_mean(factor,temp,intervals,log=False,show=True,label=None,quadratic=False):
    x=[]
    y=[]
    if log:
        factor = np.log(factor)
    elif quadratic:
        factor = factor*factor
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



def all_plot(fac_name,df,mid=0,log=False,col=None,quadratic=False):
    plot(df[fac_name],df,mid=mid,log=log,col=col,quadratic=quadratic)
    if log:
        arr = np.linspace(min(np.log(df[fac_name])),max(np.log(df[fac_name])),100)
    else:
        arr = np.linspace(min(df[fac_name]),max(df[fac_name]),100)
    itv = np.vstack([arr[:-1],arr[1:]]).T
    plot_mean(df[fac_name],df,itv,log=log,quadratic=quadratic)
# -----------------------------------
temp['big_total_ratio'] = temp['big_volume_lag']/temp['trade_volume_lag']
temp['small_total_ratio'] = temp['small_volume_lag']/temp['trade_volume_lag']

temp['buy_total_ratio'] = temp['total_buy_vol_lag']/temp['trade_volume_lag']
temp['sell_total_ratio'] = temp['total_sell_vol_lag']/temp['trade_volume_lag']


temp['big_buy_total_ratio'] = temp['big_buy_volume_lag']/temp['trade_volume_lag']
temp['small_buy_total_ratio'] = temp['small_buy_volume_lag']/temp['trade_volume_lag']
temp['big_buy_buytotal_ratio'] = temp['big_buy_volume_lag']/temp['total_buy_vol_lag']
temp['small_buy_buytotal_ratio'] = temp['small_buy_volume_lag']/temp['total_buy_vol_lag']

temp['big_sell_total_ratio'] = temp['big_sell_volume_lag']/temp['trade_volume_lag']
temp['small_sell_total_ratio'] = temp['small_sell_volume_lag']/temp['trade_volume_lag']
temp['big_sell_selltotal_ratio'] = temp['big_sell_volume_lag']/temp['total_sell_vol_lag']
temp['small_sell_selltotal_ratio'] = temp['small_sell_volume_lag']/temp['total_sell_vol_lag']

temp['big_volume_spread'] = temp['big_buy_volume_lag'] - temp['big_sell_volume_lag']
temp['small_volume_spread'] = temp['small_buy_volume_lag'] - temp['small_sell_volume_lag']

temp['net_volume_shares_ratio'] = temp['net_volume_lag']/temp['NonRestrictedShares'] *1e4
temp['ret_buybuy_lag'] = np.log(temp['BuyPrice1']/temp['BuyPrice1_lag'])
temp['ret_sellsell_lag'] = np.log(temp['SellPrice1']/temp['SellPrice1_lag'])
temp['ret_vwapmid_lag'] = np.log(temp['vwap_lag']/(0.5*temp['BuyPrice1_lag']+0.5*temp['SellPrice1_lag']))



all_plot('net_volume_shares_ratio',temp,col='20190903')

for i in range(len(conditions_price)):
    con = conditions_price[i]
    cons = cons_price[i]
    range_noinf = temp[con]
    all_plot('net_volume_shares_ratio',range_noinf,col='price '+cons)
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
    all_plot('net_volume_shares_ratio',range_noinf,col='cap '+cons)
    print('''
          num stocks:   %f
          ave spread:   %f
          ave price:    %f
          spread/price: %f''' % (
          len(range_noinf['InstrumentId'].unique()),
          (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean(),
          range_noinf['vwap_lag'].mean(),
          (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean()/range_noinf['vwap_lag'].mean()))



fac_name = 'small_volume_spread'
for i in range(len(conditions_price)):
    con = conditions_price[i]
    cons = cons_price[i]
    range_noinf = temp[con]
    all_plot(fac_name,range_noinf,col='price '+cons)
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
    all_plot(fac_name,range_noinf,col='price '+cons)
    print('''
          num stocks:   %f
          ave spread:   %f
          ave price:    %f
          spread/price: %f''' % (
          len(range_noinf['InstrumentId'].unique()),
          (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean(),
          range_noinf['vwap_lag'].mean(),
          (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean()/range_noinf['vwap_lag'].mean()))

# -------------------------------
for i in range(len(conditions_cap)):
    con = conditions_cap[i]
    cons = cons_cap[i]
    fac_range = temp[con]
    range_noinf = fac_range[(fac_range['resist_sell_lag']>0) & (fac_range['resist_sell_lag']<np.inf)]
    all_plot('resist_sell_lag',range_noinf,col='sell,price '+cons,log=True)
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
    fac_range = temp[con]
    range_noinf = fac_range[(fac_range['resist_sell_lag']>0) & (fac_range['resist_sell_lag']<np.inf)]
    all_plot('resist_sell_lag',range_noinf,col='sell,price '+cons,log=True)
    print('''
          num stocks:   %f
          ave spread:   %f
          ave price:    %f
          spread/price: %f''' % (
          len(range_noinf['InstrumentId'].unique()),
          (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean(),
          range_noinf['vwap_lag'].mean(),
          (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean()/range_noinf['vwap_lag'].mean()))

# ---------------------------
    
temp_bigsmall = temp[(temp['big_buy_volume_lag']!=0) & (temp['small_buy_volume_lag']!=0)].copy()
temp_bigsmall['big_small_buy_ratio'] = temp_bigsmall['big_buy_volume_lag']/temp_bigsmall['small_buy_volume_lag']    
temp_bigsmall = temp[(temp['big_sell_volume_lag']!=0) & (temp['small_sell_volume_lag']!=0)].copy()
temp_bigsmall['big_small_sell_ratio'] = temp_bigsmall['big_sell_volume_lag']/temp_bigsmall['small_sell_volume_lag']    


fac_name = 'big_small_buy_ratio'
fac_name = 'big_small_sell_ratio'
fac_name = 'big_buy_buytotal_ratio'
fac_name = 'big_sell_selltotal_ratio'




#all_plot(fac_name,temp_bigsmall,log=True,mid=2.5,col=fac_name)
all_plot(fac_name,temp_bigsmall,log=False,mid=0.5,col=fac_name+'^2',quadratic=True)

for i in range(len(conditions_price)):
    con = conditions_price[i]
    cons = cons_price[i]
    range_noinf = temp_bigsmall[con]
    all_plot(fac_name,range_noinf,col='price '+cons,log=True,mid=2.5)
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
    range_noinf = temp_bigsmall[con]
    all_plot(fac_name,range_noinf,col='price '+cons,log=True)
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
        fac_range = temp[con]
        range_noinf = fac_range[(fac_range['resist_sell_lag']>0) & (fac_range['resist_sell_lag']<np.inf)]
        plot(range_noinf['resist_sell_lag'],range_noinf,1,'sell, price '+cons)
        print('''
              num stocks:   %f
              ave spread:   %f
              ave price:    %f
              spread/price: %f''' % (
              len(range_noinf['InstrumentId'].unique()),
              (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean(),
              range_noinf['vwap_lag'].mean(),
              (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean()/range_noinf['vwap_lag'].mean()))


#----------------------------------------------

all_plot('ret_vwapmid_lag',temp)

temp['spread'] = temp['SellPrice1'] - temp['BuyPrice1']
all_plot('spread',temp)
temp['vol_imb'] = temp['BuyVolume1']/(temp['BuyVolume1']+temp['SellVolume1'])
all_plot('vol_imb',temp)

range_noinf = temp[(temp['incre_buy_lag']+temp['incre_sell_lag']!=0)].copy()
range_noinf['incre_imb'] = (range_noinf['incre_buy_lag']+range_noinf['total_buy_vol_lag'])/(
        range_noinf['incre_sell_lag']+range_noinf['total_sell_vol_lag'])
all_plot('incre_imb',range_noinf)

((range_noinf['incre_buy_lag']+range_noinf['total_buy_vol_lag'])<0).sum()/len(range_noinf)
((range_noinf['incre_sell_lag']+range_noinf['total_sell_vol_lag'])<0).sum()/len(range_noinf)


(range_noinf[range_noinf['incre_sell_lag']<0]['ret_vwapmid_fur']>0).sum()/(range_noinf['incre_sell_lag']<0).sum()
(range_noinf[range_noinf['incre_sell_lag']>0]['ret_vwapmid_fur']>0).sum()/(range_noinf['incre_sell_lag']>0).sum()

(range_noinf[range_noinf['incre_sell_lag']<0]['incre_buy_lag']>0).sum()/(range_noinf['incre_sell_lag']<0).sum()

(range_noinf[range_noinf['resist_sell_lag']<0]['ret_vwapmid_fur']>0).sum()/(range_noinf['resist_sell_lag']<0).sum()
(range_noinf[range_noinf['resist_sell_lag']>0]['ret_vwapmid_fur']>0).sum()/(range_noinf['resist_sell_lag']>0).sum()


range_noinf[range_noinf['incre_imb'].isna()][['incre_buy_lag','incre_sell_lag','incre_imb']]
range_noinf['incre_buy_lag'].isna().sum()
range_noinf['incre_sell_lag'].isna().sum()

################# when incre sell < 0
# -------------------------------
for i in range(len(conditions_cap)):
    con = conditions_cap[i]
    cons = cons_cap[i]
    fac_range = temp[con]
#    range_noinf = fac_range[(fac_range['resist_sell_lag']<0) & (fac_range['resist_sell_lag']>-np.inf)]
#    range_noinf['resist_sell_lag'] = abs(range_noinf['resist_sell_lag'])
    range_noinf = fac_range[(fac_range['resist_sell_lag']<np.inf) & (fac_range['resist_sell_lag']>-np.inf)].copy()
#    range_noinf['norm_resist_sell_lag'] = (range_noinf['resist_sell_lag']-range_noinf['resist_sell_lag'].mean())/range_noinf['resist_sell_lag'].std()
    range_noinf['norm_resist_sell_lag'] = range_noinf['resist_sell_lag']/range_noinf['resist_sell_lag'].std()
    all_plot('norm_resist_sell_lag',range_noinf,col='sell,price '+cons,log=False)
    print('''
          num stocks:   %f
          ave spread:   %f
          ave price:    %f
          spread/price: %f''' % (
          len(range_noinf['InstrumentId'].unique()),
          (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean(),
          range_noinf['vwap_lag'].mean(),
          (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean()/range_noinf['vwap_lag'].mean()))

range_noinf['norm_resist_sell_lag'].isna().sum()
(range_noinf['norm_resist_sell_lag']==np.inf).sum()
(range_noinf['norm_resist_sell_lag']==-np.inf).sum()
plt.hist(range_noinf['resist_sell_lag'],bins=100)


for i in range(len(conditions_price)):
    con = conditions_price[i]
    cons = cons_price[i]
    fac_range = temp[con].copy()
    range_noinf = fac_range[(fac_range['resist_sell_lag']<0) & (fac_range['resist_sell_lag']>-np.inf)]
    range_noinf['resist_sell_lag'] = abs(range_noinf['resist_sell_lag'])
    all_plot('resist_sell_lag',range_noinf,col='sell,price '+cons,log=True)
    print('''
          num stocks:   %f
          ave spread:   %f
          ave price:    %f
          spread/price: %f''' % (
          len(range_noinf['InstrumentId'].unique()),
          (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean(),
          range_noinf['vwap_lag'].mean(),
          (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean()/range_noinf['vwap_lag'].mean()))



resist,trade imbalance, net volume/ shares, ret_vwapmid_lag, vol imb in snapshot


range_noinf['norm_resist_sell_lag']








