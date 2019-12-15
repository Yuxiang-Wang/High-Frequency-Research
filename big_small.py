# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 10:43:31 2019

大小单比例因子研究

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

df1 = pd.read_csv('df6_merged.csv',index_col=0)
df1['datetime']=pd.to_datetime(df1['datetime'],format='%Y-%m-%d %H:%M:%S')

big_small_1 = pd.read_csv('big_small_1.csv',index_col=0)
big_small_2 = pd.read_csv('big_small_2.csv',index_col=0)

big_small = pd.concat([big_small_1,big_small_2])
big_small['datetime']=pd.to_datetime(big_small['datetime'],format='%Y-%m-%d %H:%M:%S')

fac_data = df1.merge(big_small,on=['InstrumentId','datetime'])

fac_data['big_total_ratio_lag'] = fac_data['big_volume_lag']/fac_data['total_volume_lag']
fac_data['net_volume'] = fac_data['buy_volume_lag'] - fac_data['sell_volume_lag']

fac_data.columns

fac_data['big_buy_sell_ratio'] = fac_data['big_buy_volume_lag']/fac_data['big_sell_volume_lag']
fac_data['supbig_buy_sell_ratio'] = fac_data['supbig_buy_volume_lag']/fac_data['supbig_sell_volume_lag']
fac_data['big_imbalance'] = fac_data['big_buy_volume_lag']/(fac_data['big_buy_volume_lag']+fac_data['big_sell_volume_lag'])
fac_data['supbig_imbalance'] = fac_data['supbig_buy_volume_lag']/(fac_data['supbig_buy_volume_lag']+fac_data['supbig_sell_volume_lag'])

fac_data['big_buy_total_ratio'] = fac_data['big_buy_volume_lag']/fac_data['buy_volume_lag']
fac_data['big_sell_total_ratio'] = fac_data['big_sell_volume_lag']/fac_data['sell_volume_lag']

fac_data['small_net_volume'] = fac_data['net_volume'] - (fac_data['big_buy_volume_lag'] - fac_data['big_sell_volume_lag'])
fac_data['big_net_volume'] = fac_data['big_buy_volume_lag'] - fac_data['big_sell_volume_lag']
fac_data['bigvolume_netvolume_ratio'] = fac_data['big_net_volume']/fac_data['net_volume']
fac_data['bigbuy_smallnet_ratio'] = fac_data['big_buy_volume_lag']/fac_data['small_net_volume']
fac_data['bigsmall_smallnet_ratio'] = fac_data['big_small_volume_lag']/fac_data['small_net_volume']


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
 
fac_data.isna().sum()

range_noinf = fac_data.dropna()
range_noinf = range_noinf[(range_noinf['big_buy_sell_ratio']>0) & (range_noinf['big_buy_sell_ratio']<np.inf)]
plot(range_noinf['big_buy_sell_ratio'],range_noinf,log=1)
arr = np.linspace(-4,4,100)
interval_bbs = np.vstack([arr[:-1],arr[1:]]).T
plot_mean(range_noinf['big_buy_sell_ratio'],range_noinf,interval_bbs,log=1)
# -----
range_noinf = fac_data.dropna()
range_noinf = range_noinf[(range_noinf['supbig_buy_sell_ratio']>0) & (range_noinf['supbig_buy_sell_ratio']<np.inf)]
plot(range_noinf['supbig_buy_sell_ratio'],range_noinf,log=1)
arr = np.linspace(-4,4,100)
interval_bbs = np.vstack([arr[:-1],arr[1:]]).T
plot_mean(range_noinf['supbig_buy_sell_ratio'],range_noinf,interval_bbs,log=1)
#------
range_noinf = fac_data.dropna()
plot(range_noinf['big_imbalance'],range_noinf)
arr = np.linspace(0,1,100)
interval_bbs = np.vstack([arr[:-1],arr[1:]]).T
plot_mean(range_noinf['big_imbalance'],range_noinf,interval_bbs)
#------
range_noinf = fac_data.dropna()
plot(range_noinf['supbig_imbalance'],range_noinf)
arr = np.linspace(0,1,100)
interval_bbs = np.vstack([arr[:-1],arr[1:]]).T
plot_mean(range_noinf['supbig_imbalance'],range_noinf,interval_bbs)

# -----------------------------
range_noinf = fac_data.dropna()
plot(range_noinf['big_buy_total_ratio'],range_noinf)
arr = np.linspace(-0,1,100)
interval_bbs = np.vstack([arr[:-1],arr[1:]]).T
plot_mean(range_noinf['big_buy_total_ratio'],range_noinf,interval_bbs)
#-----
range_noinf = fac_data.dropna()
plot(range_noinf['big_sell_total_ratio']**2,range_noinf)
arr = np.linspace(-0,1,100)
interval_bbs = np.vstack([arr[:-1],arr[1:]]).T
plot_mean(range_noinf['big_sell_total_ratio']**2,range_noinf,interval_bbs)

#-------
plot(fac_data['bigvolume_netvolume_ratio'],fac_data)
plot(fac_data['bigsell_smallnet_ratio'],fac_data)

# --------------------------- buy
plot(fac_data['bigbuy_smallnet_ratio'],fac_data)

range_noinf = fac_data[(fac_data['bigbuy_smallnet_ratio']>0) & (fac_data['bigbuy_smallnet_ratio']<np.inf)]
range_noinf.dropna(inplace=True)
plot(range_noinf['bigbuy_smallnet_ratio'],range_noinf,log=True,mid=0)

arr = np.linspace(-5,8,2000)
interval_bbsn = np.vstack([arr[:-1],arr[1:]]).T
plot_mean(range_noinf['bigbuy_smallnet_ratio'],range_noinf,interval_bbsn,log=True)

temp_range = range_noinf[(np.log(range_noinf['bigbuy_smallnet_ratio'])>-2) &(np.log(range_noinf['bigbuy_smallnet_ratio'])<2)]
(temp_range['ret_vwapmid_fur']>0).sum()/len(temp_range)

range_noinf['ret_vwapmid_fur'].mean()
fac_data['ret_vwapmid_fur'].mean()

for mid in range(-4,6):
    print('------------------------\n',mid)
    a1,a2,a3,a4 = ((range_noinf[np.log(range_noinf['bigbuy_smallnet_ratio'])<mid]['ret_vwapmid_fur']>0).sum()/len(range_noinf),
                   (range_noinf[np.log(range_noinf['bigbuy_smallnet_ratio'])>mid]['ret_vwapmid_fur']>0).sum()/len(range_noinf),
                   (range_noinf[np.log(range_noinf['bigbuy_smallnet_ratio'])<mid]['ret_vwapmid_fur']<0).sum()/len(range_noinf),
                   (range_noinf[np.log(range_noinf['bigbuy_smallnet_ratio'])>mid]['ret_vwapmid_fur']<0).sum()/len(range_noinf),)
    print('''pct:
        1: %f  2: %f
        3: %f  4: %f
        left mean: %f  right mean: %f''' % (
        a1,a2,a3,a4,
        range_noinf[range_noinf['bigbuy_smallnet_ratio']<mid]['ret_vwapmid_fur'].mean(),
        range_noinf[range_noinf['bigbuy_smallnet_ratio']>mid]['ret_vwapmid_fur'].mean()))
    print('1/3: ',a1/a3,'  2/4: ',a2/a4)


# --------------------------- sell
plot(fac_data['bigsell_smallnet_ratio'],fac_data)

range_noinf = fac_data[(fac_data['bigsell_smallnet_ratio']>0) & (fac_data['bigsell_smallnet_ratio']<np.inf)]
range_noinf.dropna(inplace=True)
plot(range_noinf['bigsell_smallnet_ratio'],range_noinf,log=True,mid=0)

#range_noinf = fac_data[(fac_data['bigsell_smallnet_ratio']<0) & (fac_data['bigsell_smallnet_ratio']>-np.inf)]
#range_noinf.dropna(inplace=True)
#plot(abs(range_noinf['bigsell_smallnet_ratio']),range_noinf,log=True,mid=0)

arr = np.linspace(-5,8,100)
interval_bbsn = np.vstack([arr[:-1],arr[1:]]).T
plot_mean(range_noinf['bigsell_smallnet_ratio'],range_noinf,interval_bbsn,log=True)

for mid in range(-4,6):
    print('------------------------\n',mid)
    a1,a2,a3,a4 = ((range_noinf[np.log(range_noinf['bigsell_smallnet_ratio'])<mid]['ret_vwapmid_fur']>0).sum()/len(range_noinf),
                   (range_noinf[np.log(range_noinf['bigsell_smallnet_ratio'])>mid]['ret_vwapmid_fur']>0).sum()/len(range_noinf),
                   (range_noinf[np.log(range_noinf['bigsell_smallnet_ratio'])<mid]['ret_vwapmid_fur']<0).sum()/len(range_noinf),
                   (range_noinf[np.log(range_noinf['bigsell_smallnet_ratio'])>mid]['ret_vwapmid_fur']<0).sum()/len(range_noinf),)
    print('''pct:
        1: %f  2: %f
        3: %f  4: %f
        left mean: %f  right mean: %f''' % (
        a1,a2,a3,a4,
        range_noinf[range_noinf['bigsell_smallnet_ratio']<mid]['ret_vwapmid_fur'].mean(),
        range_noinf[range_noinf['bigsell_smallnet_ratio']>mid]['ret_vwapmid_fur'].mean()))
    print('1/3: ',a1/a3,'  2/4: ',a2/a4)
    
    
#111111111111
def normalize(data):
    print(data.mean(),data.std())
    return (data - data.mean())/data.std()

range_noinf = fac_data[(fac_data['bigbuy_smallnet_ratio']>-np.inf) & (fac_data['bigbuy_smallnet_ratio']<np.inf)]
range_noinf.dropna(inplace=True)
plot(normalize(range_noinf['bigbuy_smallnet_ratio']),range_noinf,mid=0)

# --------------------------------
range_noinf = fac_data.dropna()
range_noinf = range_noinf[range_noinf['net_volume']>0]
plot(range_noinf['net_volume'],range_noinf,log=True)

range_noinf = fac_data.dropna()
range_noinf = range_noinf[range_noinf['net_volume']<0]
plot(abs(range_noinf['net_volume']),range_noinf,log=True)


range_noinf = fac_data.dropna()
range_noinf = range_noinf[range_noinf['small_net_volume']>0]
plot(range_noinf['small_net_volume'],range_noinf,log=True)





range_noinf = fac_data[(fac_data['bigbuy_smallnet_ratio']<0) & (fac_data['bigbuy_smallnet_ratio']<np.inf)]
plot(abs(range_noinf['bigbuy_smallnet_ratio']),range_noinf,log=True,mid=5)






fac_data['small_net_volume'] = fac_data['net_volume'] - (fac_data['big_buy_volume_lag'] - fac_data['big_sell_volume_lag'])
fac_data['big_net_volume'] = fac_data['big_buy_volume_lag'] - fac_data['big_sell_volume_lag']
fac_data['bigvolume_netvolume_ratio'] = fac_data['big_net_volume']/fac_data['net_volume']
fac_data['bigbuy_smallnet_ratio'] = fac_data['big_buy_volume_lag']/fac_data['small_net_volume']
fac_data['bigsell_smallnet_ratio'] = fac_data['big_sell_volume_lag']/fac_data['small_net_volume']



# ---------------------------------------------------------------------------------
daily = pd.read_csv('daily/201809StockDailyData.csv')
daily['InstrumentId'] = daily['WindCode'].apply(lambda x:int(x[:6]))
daily_SH = daily[daily['WindCode'].apply(lambda x:True if 'SH' in x else False)]
today = daily_SH[daily_SH['TradingDay']==20180903]
today['free_cap'] = today['VWAvgPrice']*today['NonRestrictedShares']/today['SplitFactor']

#temp = fac_data.merge(today[['InstrumentId','MarketCapAFloat']],on = ['InstrumentId'],how = 'left')
temp = fac_data.merge(today[['InstrumentId','free_cap']],on = ['InstrumentId'],how = 'left')
# temp.rename(columns={'vwap_lag_x':'vwap_lag'},inplace=True)

conditions_price = [temp['vwap_lag']>20,
                    (temp['vwap_lag']<=20) & (temp['vwap_lag']>10),
                    (temp['vwap_lag']<=10) & (temp['vwap_lag']>5),
                    (temp['vwap_lag']<=5)]              
cons_price = ['>20','10 - 20','5 - 10','<5']

conditions_cap = [temp['free_cap']>1e10,
              (temp['free_cap']<1e10) & (temp['free_cap']>5e9),
              (temp['free_cap']<5e9) & (temp['free_cap']>3e9),
              (temp['free_cap']<3e9) & (temp['free_cap']>1.5e9),
              (temp['free_cap']<1.5e9)]
cons_cap = ['>1e10','5e9 - 1e10','3e9 - 5e9','1.5e9 - 3e9','<1.5e9']


fac_range = temp.dropna()

arr = np.linspace(-0,1,100)
interval_bst = np.vstack([arr[:-1],arr[1:]]).T
plot_mean(fac_range['big_sell_total_ratio'],fac_range,interval_bst)
plot_mean(fac_range['big_buy_total_ratio'],fac_range,interval_bst)

arr = np.linspace(0,1,100)
interval_bst = np.vstack([arr[:-1],arr[1:]]).T
plot_mean(fac_range['big_imbalance'],fac_range,interval_bst)

plot_mean(fac_range['trade_volume_imbalance'],fac_range,interval_bst)




# -----------------------------------------------------

plot(fac_data['small_net_volume'],fac_data)



fac_data['small_net_volume'] = fac_data['net_volume'] - (fac_data['big_buy_volume_lag'] - fac_data['big_sell_volume_lag'])














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

    





