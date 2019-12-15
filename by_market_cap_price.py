# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:15:02 2019

挂单净增量因子
1、按市值跟价格分类研究
2、一些个股

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

fac_path = 'fac_1m_all/20180903/'
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
today = daily_SH[daily_SH['TradingDay']==20180903].copy()
today['free_cap'] = today['VWAvgPrice']*today['NonRestrictedShares']/today['SplitFactor']

#temp = fac_data.merge(today[['InstrumentId','MarketCapAFloat']],on = ['InstrumentId'],how = 'left')
temp = fac_data.merge(today[['InstrumentId','free_cap']],on = ['InstrumentId'],how = 'left')


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

# --------------------
def plot(temp,factor,log=False,col=None,mid=0,quadratic=False,addv=None,addh=None,save=None,independent='ret_vwapmid_fur'):
    if log:
        temp = np.log(temp)
    elif quadratic:
        temp = temp*temp
    plt.axhline(y=0,color='r',linewidth=0.5)
    #plt.axhline(y=0.001,color='r')
    #plt.axhline(y=-0.001,color='r')
    plt.axvline(x=mid,color='r',linewidth=0.5)
    if addv:
        plt.axvline(x=addv,color='m',linewidth=0.7)
    if addh:
        plt.axhline(y=addh,color='m',linewidth=0.7)    
    plt.scatter(temp,factor[independent],linewidths=0.001,alpha=0.5)
    if col:
        plt.title(col)
    plt.xlim(min(temp)*0.9,max(temp)*1.1)
    plt.ylim(min(factor[independent])-0.0005,max(factor[independent])+0.0005)
    #plt.plot([-10000,10000],[0,0],color='r')
    plt.grid(True)
    if save:
        plt.savefig(save)
    plt.show()
    print('''pct:
        1: %f  2: %f
        3: %f  4: %f
        ret>0 in sample: %f
        left mean: %f  right mean: %f''' % (
        (factor[temp<mid][independent]>0).sum()/len(factor),
        (factor[temp>mid][independent]>0).sum()/len(factor),
        (factor[temp<mid][independent]<0).sum()/len(factor),
        (factor[temp>mid][independent]<0).sum()/len(factor),
        (factor[independent]>0).sum()/len(factor),
        factor[temp<mid][independent].mean(),
        factor[temp>mid][independent].mean()))
    print('right:',(temp>mid).sum()/len(factor))
    print('left: ',(temp<mid).sum()/len(factor))
        
def plot_mean(factor,temp,intervals,log=False,show=True,label=None,quadratic=False,independent='ret_vwapmid_fur'):
    x=[]
    y=[]
    if log:
        factor = np.log(factor)
    elif quadratic:
        factor = factor*factor
    for interval in intervals:
        temp_interval = temp[(factor>interval[0])&(factor<=interval[1])]
        if not temp_interval.empty:
            y.append(temp_interval[independent].mean())
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



def all_plot(fac,df,mid=0,log=False,col=None,quadratic=False,addv=None,addh=None,independent='ret_vwapmid_fur'):
    plot(fac,df,mid=mid,log=log,col=col,quadratic=quadratic,addv=addv,addh=addh,independent=independent)
    if log:
        arr = np.linspace(min(np.log(fac)),max(np.log(fac)),100)
    else:
        arr = np.linspace(min(fac),max(fac),100)
    itv = np.vstack([arr[:-1],arr[1:]]).T
    plot_mean(fac,df,itv,log=log,quadratic=quadratic,independent=independent)

#------------------

########## by market cap
######### sell
len(fac_range['InstrumentId'].unique())
fac_range = temp[(temp['resist_sell_lag']>0) & (temp['resist_sell_lag']<np.inf)]
all_plot(fac_range['resist_sell_lag'],fac_range,log=True)
all_plot(fac_range['resist_sell_lag'],fac_range,log=True,addv=-3.5,addh=0.012)

fac_range = temp[(temp['resist_buy_lag']>0) & (temp['resist_buy_lag']<np.inf)]
all_plot(fac_range['resist_buy_lag'],fac_range,log=True)
all_plot(fac_range['resist_buy_lag'],fac_range,log=True,addv=-3.5,addh=0.012)

spec = fac_range[(np.log(fac_range['resist_sell_lag'])<-3.5) &
                 (fac_range['ret_vwapmid_fur']>0.012)]

print(set(list(spec['InstrumentId'])))




for i in range(len(conditions_cap)):
    con = conditions_cap[i]
    fac_range = temp[con]
    range_noinf = fac_range[(fac_range['resist_sell_lag']>0) & (fac_range['resist_sell_lag']<np.inf)]
    #plot(range_noinf['resist_sell_lag'],range_noinf,1,'sell, price '+cons_cap[i])
    all_plot(range_noinf['resist_sell_lag'],range_noinf,log=True,addv=-3.5,addh=0.012,col = 'sell, cap '+cons_cap[i])
    print('''
          num stocks:   %f
          ave spread:   %f
          ave price:    %f
          spread/price: %f''' % (
          len(range_noinf['InstrumentId'].unique()),
          (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean(),
          range_noinf['vwap_lag'].mean(),
          (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean()/range_noinf['vwap_lag'].mean()))


######### buy
for i in range(len(conditions_price)):
    con = conditions_price[i]
    fac_range = temp[con]
    range_noinf = fac_range[(fac_range['resist_buy_lag']>0) & (fac_range['resist_buy_lag']<np.inf)]
    plot(range_noinf['resist_buy_lag'],range_noinf,1,'buy, price '+cons_price[i])
    print('''
          num stocks:   %f
          ave spread:   %f
          ave price:    %f
          spread/price: %f''' % (
          len(range_noinf['InstrumentId'].unique()),
          (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean(),
          range_noinf['vwap_lag'].mean(),
          (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean()/range_noinf['vwap_lag'].mean()))

# ------------------------------------------------------
for i in range(len(conditions_price)):
    con = conditions_price[i]
    fac_range = temp[con]
    range_noinf = fac_range[(fac_range['resist_sell_lag']>0) & (fac_range['resist_sell_lag']<np.inf)]
    #plot(range_noinf['resist_sell_lag'],range_noinf,1,'sell, price '+cons_price[i])
    all_plot(range_noinf['resist_sell_lag'],range_noinf,log=True,addv=-3.5,addh=0.012,col = 'sell, price '+cons_price[i])
    print('''
          num stocks:   %f
          ave spread:   %f
          ave price:    %f
          spread/price: %f''' % (
          len(range_noinf['InstrumentId'].unique()),
          (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean(),
          range_noinf['vwap_lag'].mean(),
          (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean()/range_noinf['vwap_lag'].mean()))


######### buy
for i in range(len(conditions_price)):
    con = conditions_price[i]
    fac_range = temp[con]
    range_noinf = fac_range[(fac_range['resist_buy_lag']>0) & (fac_range['resist_buy_lag']<np.inf)]
    all_plot(range_noinf['resist_buy_lag'],range_noinf,log = 1,col = 'buy, price '+cons_price[i])
    print('''
          num stocks:   %f
          ave spread:   %f
          ave price:    %f
          spread/price: %f''' % (
          len(range_noinf['InstrumentId'].unique()),
          (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean(),
          range_noinf['vwap_lag'].mean(),
          (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean()/range_noinf['vwap_lag'].mean()))


try:
    os.mkdir('img')
except:
    pass
fac_range = temp[(temp['resist_sell_lag']>0) & (temp['resist_sell_lag']<np.inf)]
fac_range = fac_range.sort_values('free_cap')
fac_range = fac_range.sort_values('vwap_lag',ascending=False)
for Id in fac_range['InstrumentId'].unique()[31:60]: 
    Id = 603813
    range_noinf = fac_range[fac_range['InstrumentId']==Id]
    plot(range_noinf['resist_sell_lag'],range_noinf,log=True,save='img/'+str(Id)+'.png')
    print('id: ',Id)
    print('market cap: ',range_noinf['free_cap'].iloc[0]/1e8)
    print('price: ',range_noinf['vwap_lag'].iloc[-1])


fac_id = temp[temp['InstrumentId']==603045]
fac_id = fac_id[fac_id['ret_vwapmid_fur']<-0.004]



fac_range['midprice']
fac_range['BuyPrice1_lag']


600771






def todf(fac_name,fac_range,mid=0):
    d={}
    mid=0
    ids = fac_range['InstrumentId'].unique()
    for i in range(len(ids)):
        Id = ids[i]
        range_noinf = fac_range[fac_range['InstrumentId']==Id]
        if len(range_noinf)<500:
            continue
        a1 = (range_noinf[np.log(range_noinf[fac_name].values)<mid]['ret_vwapmid_fur']>0).sum()/len(range_noinf)
        a2 = (range_noinf[np.log(range_noinf[fac_name].values)>mid]['ret_vwapmid_fur']>0).sum()/len(range_noinf)
        a3 = (range_noinf[np.log(range_noinf[fac_name].values)<mid]['ret_vwapmid_fur']<0).sum()/len(range_noinf)
        a4 = (range_noinf[np.log(range_noinf[fac_name].values)>mid]['ret_vwapmid_fur']<0).sum()/len(range_noinf)
        d[Id]={}
        d[Id]['left ratio'] = a1/(a1+a3)
        d[Id]['right ratio'] = a2/(a2+a4)
        d[Id]['ratio diff'] = d[Id]['left ratio'] - d[Id]['right ratio']
        d[Id]['>0 all sample'] = (range_noinf['ret_vwapmid_fur']>0).sum()/len(range_noinf)
        d[Id]['left mean'] = range_noinf[np.log(range_noinf[fac_name])<mid]['ret_vwapmid_fur'].mean()
        d[Id]['right mean'] = range_noinf[np.log(range_noinf[fac_name])>mid]['ret_vwapmid_fur'].mean()
        d[Id]['diff'] = d[Id]['left mean'] - d[Id]['right mean']
        d[Id]['sample mean'] = range_noinf['ret_vwapmid_fur'].mean()
        d[Id]['Id'] = Id
        d[Id]['market cap'] = range_noinf['free_cap'].iloc[0]/1e8
        d[Id]['vwap'] = range_noinf['vwap_lag'].iloc[-1]
        if not i%10:
            print(i,end=',')
    
    return pd.DataFrame.from_dict(d,orient='index')

plt.scatter(np.log(df['market cap']),df['left ratio'])

plt.scatter(np.log(df['market cap']),df['left ratio'] - df['right ratio'],alpha=0.5)
plt.axhline(y=0,linewidth=0.7,color='r')

plt.scatter(np.log(df['market cap']),df['diff'],alpha=0.5)
plt.axhline(y=0,linewidth=0.7,color='r')
plt.ylim([-0.01,0.01])

#--------
fac_range = temp[(temp['resist_sell_lag']>0) & (temp['resist_sell_lag']<np.inf)]
df = todf('resist_sell_lag',fac_range)


all_plot(df['market cap'],df,independent='diff',log=True)
all_plot(df['market cap'],df,independent='ratio diff',log=True)

all_plot(df['vwap'],df,independent='diff',log=True)
all_plot(df['vwap'],df,independent='ratio diff',log=True)

#---------
df = todf('resist_buy_lag',fac_range)
all_plot(df['market cap'],df,independent='diff',log=True)
all_plot(df['market cap'],df,independent='ratio diff',log=True)

all_plot(df['vwap'],df,independent='diff',log=True)
all_plot(df['vwap'],df,independent='ratio diff',log=True)


for i in range(1112,1102,-1):
    print((i-2)/100)










######### sell
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

for i in range(len(conditions_price)):
    for j in range(len(conditions_cap)):
        con = conditions_price[i] & conditions_cap[j]
        cons = cons_price[i] + '+' + cons_cap[j]
        fac_range = temp[con]
        range_noinf = fac_range[(fac_range['resist_sell_lag']>0) & (fac_range['resist_sell_lag']<np.inf)]
        plot_mean(range_noinf['resist_sell_lag'],range_noinf,interval_res,log=True,label=cons +', '+str(len(range_noinf['InstrumentId'].unique())

######### buy
for i in range(len(conditions_price)):
    for j in range(len(conditions_cap)):
        con = conditions_price[i] & conditions_cap[j]
        cons = cons_price[i] + '+' + cons_cap[j]
        fac_range = temp[con]
        range_noinf = fac_range[(fac_range['resist_buy_lag']>0) & (fac_range['resist_buy_lag']<np.inf)]
        plot(range_noinf['resist_buy_lag'],range_noinf,1,'buy, price '+cons)
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
        range_noinf = fac_range[(fac_range['resist_buy_lag']>0) & (fac_range['resist_buy_lag']<np.inf)]
        plot_mean(range_noinf['resist_buy_lag'],range_noinf,interval_res,log=True,label=cons +', '+str(len(range_noinf['InstrumentId'].unique())))
            

summary = {}
# for con in conditions: # con = conditions[4]; cons[4]
for k in range(len(conditions_price)):
    for j in range(len(conditions_cap)):
        con = conditions_price[k] & conditions_cap[j]
        cons = cons_price[k] + '+' + cons_cap[j]
        fac_range = temp[con]
        print(len(fac_range),len(fac_range['InstrumentId'].unique()))
        n = len(fac_range['InstrumentId'].unique())
        
        #--------------------------
        d={}
        ############### range,sell
        range_noinf = fac_range[(fac_range['resist_sell_lag']>0) & (fac_range['resist_sell_lag']<1)]    
        for i in range(7):
            temp_range_noinf = range_noinf[np.log(range_noinf['resist_sell_lag'])<-i]
#            temp_range_noinf['ret_vwapmid_fur'].hist(bins=100)
#            plt.show()          
#            print('''%d
#                  median:      %f
#                  mean:        %f
#                  std:         %f
#                  >0 pct:      %f
#                  spread:      %f
#                  >1/2spread:  %f
#                  >-1/2spread: %f''' % (-i,temp_range_noinf['ret_vwapmid_fur'].median(),
#                  temp_range_noinf['ret_vwapmid_fur'].mean(),
#                  temp_range_noinf['ret_vwapmid_fur'].std(),
#                  (temp_range_noinf['ret_vwapmid_fur']>0).sum()/len(temp_range_noinf),
#                  (temp_range_noinf['SellPrice1']-temp_range_noinf['BuyPrice1']).mean()/temp_range_noinf['vwap_lag'].mean(),
#                  (temp_range_noinf['ret_vwapmid_fur']>0.5*(temp_range_noinf['SellPrice1']-temp_range_noinf['BuyPrice1']).mean()/temp_range_noinf['vwap_lag'].mean()).sum()/len(temp_range_noinf),
#                  (temp_range_noinf['ret_vwapmid_fur']>-0.5*(temp_range_noinf['SellPrice1']-temp_range_noinf['BuyPrice1']).mean()/temp_range_noinf['vwap_lag'].mean()).sum()/len(temp_range_noinf)))
#            
            d[(-i,cons+',sell')]=dict(zip(('median','mean','std','>0 pct','95% VaR','spread ret','>1/2spread','>-1/2spread',
              'ave signals','num stock'),(temp_range_noinf['ret_vwapmid_fur'].median(),
                  temp_range_noinf['ret_vwapmid_fur'].mean(),
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
#            temp_range_noinf['ret_vwapmid_fur'].hist(bins=100)
#            plt.show()
#            print('''%d
#                  median:      %f
#                  mean:        %f
#                  std:         %f
#                  >0 pct:      %f
#                  spread:      %f
#                  >1/2spread:  %f
#                  >-1/2spread: %f''' % (i,temp_range_noinf['ret_vwapmid_fur'].median(),
#                  temp_range_noinf['ret_vwapmid_fur'].mean(),
#                  temp_range_noinf['ret_vwapmid_fur'].std(),
#                  (temp_range_noinf['ret_vwapmid_fur']>0).sum()/len(temp_range_noinf),
#                  (temp_range_noinf['SellPrice1']-temp_range_noinf['BuyPrice1']).mean()/temp_range_noinf['vwap_lag'].mean(),
#                  (temp_range_noinf['ret_vwapmid_fur']>0.5*(temp_range_noinf['SellPrice1']-temp_range_noinf['BuyPrice1']).mean()/temp_range_noinf['vwap_lag'].mean()).sum()/len(temp_range_noinf),
#                  (temp_range_noinf['ret_vwapmid_fur']>-0.5*(temp_range_noinf['SellPrice1']-temp_range_noinf['BuyPrice1']).mean()/temp_range_noinf['vwap_lag'].mean()).sum()/len(temp_range_noinf)))
#            
            d[(i,cons+',buy')]=dict(zip(('median','mean','std','>0 pct','95% VaR','spread ret','>1/2spread','>-1/2spread',
              'ave signals','num stock'),(temp_range_noinf['ret_vwapmid_fur'].median(),
                  temp_range_noinf['ret_vwapmid_fur'].mean(),
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
d={}
for k in range(len(conditions_price)):
    for j in range(len(conditions_cap)):
        con = conditions_price[k] & conditions_cap[j]
        cons = cons_price[k] + '+' + cons_cap[j]
        fac_range = temp[con]
        range_noinf = fac_range[(fac_range['resist_sell_lag']>0) & (fac_range['resist_sell_lag']<np.inf)]     
        #print(cons,':   ',np.corrcoef(np.log(range_noinf['resist_sell_lag']),range_noinf['ret_vwapmid_fur'])[0][1])
        d[cons] = np.corrcoef(np.log(range_noinf['resist_sell_lag']),range_noinf['ret_vwapmid_fur'])[0][1]
print(pd.DataFrame.from_dict(d,orient='index').sort_values(0))
        
# buy
d={}
for k in range(len(conditions_price)):
    for j in range(len(conditions_cap)):
        con = conditions_price[k] & conditions_cap[j]
        cons = cons_price[k] + '+' + cons_cap[j]
        fac_range = temp[con]
        range_noinf = fac_range[(fac_range['resist_buy_lag']>0) & (fac_range['resist_buy_lag']<np.inf)]     
        #print(cons,':   ',np.corrcoef(np.log(range_noinf['resist_buy_lag']),range_noinf['ret_vwapmid_fur'])[0][1])
        d[cons] = np.corrcoef(np.log(range_noinf['resist_buy_lag']),range_noinf['ret_vwapmid_fur'])[0][1]
print(pd.DataFrame.from_dict(d,orient='index').sort_values(0))



print(list(fac_data.columns))
  
    
    
















      
        