# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 16:53:08 2019

其他几个日期的情况


@author: yuxiang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from time import time
import os
import gc
import statsmodels.api as sm
import statsmodels.formula.api as smf
%matplotlib inline

# gc.collect()

file_path = 'fac_1m_all'
days = os.listdir(file_path)
file_names = os.listdir(file_path+'/'+days[0])

daily = pd.read_csv('daily/201809StockDailyData.csv')
daily['InstrumentId'] = daily['WindCode'].apply(lambda x:int(x[:6]))
daily_SH = daily[daily['WindCode'].apply(lambda x:True if 'SH' in x else False)]
today = daily_SH[daily_SH['TradingDay']==int(days[-1])].copy()
today['free_cap'] = today['VWAvgPrice']*today['NonRestrictedShares']/today['SplitFactor']
today['VWAvg'] = today['VWAvgPrice']/today['SplitFactor']
today = today.sort_values('VWAvg',ascending=False)
Ids = today['InstrumentId'].unique()

def get_df(file,file_path=file_path,days=days):
    df = []
    for directory in days:
        path_temp = file_path+'/'+directory
        if file in os.listdir(path_temp):
           df.append(pd.read_csv(path_temp+'/'+file,index_col=0))
    if not df:
        return pd.DataFrame()
    df = pd.concat(df)
    df['datetime']=pd.to_datetime(df['datetime'],format='%Y-%m-%d %H:%M:%S')
    return df

def todf(fac_name,range_noinf,d,Id,mid=0):
    mid=0
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

# -------------------------------
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
    plt.ylim(min(factor[independent])-0.005,max(factor[independent])+0.005)
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
    
#-------------------

d_sell={}
d_buy={}
d_vwap={}
for i in range(len(file_names)):
    file = file_names[i]
    Id = int(file[:6])
    df = get_df(file,file_path,days)
    if df.empty:
        continue
    df = df.merge(today[['InstrumentId','free_cap','VWAvgPrice']],on = ['InstrumentId'],how = 'left')
    
    fac_range = df[(df['resist_sell_lag']>0) & (df['resist_sell_lag']<np.inf)]
    if not fac_range.empty:        
        todf('resist_sell_lag',fac_range,d_sell,Id)

    fac_range = df[(df['resist_buy_lag']>0) & (df['resist_buy_lag']<np.inf)]
    if not fac_range.empty:        
        todf('resist_buy_lag',fac_range,d_buy,Id)

    df['ret_vwapmid_lag'] = 2*df['vwap_lag']/(df['BuyPrice1_lag']+df['SellPrice1_lag'])
    todf('ret_vwapmid_lag',df,d_vwap,Id)
    if not i%10:
        print(i,end=',')

df_sell = pd.DataFrame.from_dict(d_sell,orient='index')
df_buy = pd.DataFrame.from_dict(d_buy,orient='index')
df_vwap = pd.DataFrame.from_dict(d_vwap,orient='index')


all_plot(df['market cap'],df,independent='diff',log=True)
all_plot(df['market cap'],df,independent='ratio diff',log=True)

all_plot(df['vwap'],df,independent='diff',log=True)
all_plot(df['vwap'],df,independent='ratio diff',log=True)


all_plot(df_buy['market cap'],df_buy,independent='diff',log=True)
all_plot(df_buy['market cap'],df_buy,independent='ratio diff',log=True)
all_plot(df_buy['vwap'],df_buy,independent='diff',log=True)
all_plot(df_buy['vwap'],df_buy,independent='ratio diff',log=True)

all_plot(df_vwap['market cap'],df_vwap,independent='diff',log=True)
all_plot(df_vwap['market cap'],df_vwap,independent='ratio diff',log=True)
all_plot(df_vwap['vwap'],df_vwap,independent='diff',log=True)
all_plot(df_vwap['vwap'],df_vwap,independent='ratio diff',log=True)


#---------------
X = np.log(df.dropna()[['vwap','market cap']].values)
y = df.dropna()['ratio diff'].values
#y = df.dropna()['diff'].values
model = sm.OLS(y,X)
results = model.fit()
results.summary()

X = np.log(df.dropna()[['vwap']].values)
y = df.dropna()['ratio diff'].values
#y = df.dropna()['diff'].values
model = sm.OLS(y,X)
results = model.fit()
results.summary()
#----------
X = np.log(df_buy.dropna()[['vwap','market cap']].values)
y = df_buy.dropna()['ratio diff'].values
#y = df_buy.dropna()['diff'].values
model = sm.OLS(y,X)
results = model.fit()
results.summary()

X = np.log(df_buy.dropna()[['vwap']].values)
y = df_buy.dropna()['ratio diff'].values
#y = df_buy.dropna()['diff'].values
model = sm.OLS(y,X)
results = model.fit()
results.summary()

#----------
ind_name = 'diff'
X = np.log(df_vwap.dropna()[['vwap','market cap']].values)
y = df_vwap.dropna()[ind_name].values
model = sm.OLS(y,X)
results = model.fit()
results.summary()

X = np.log(df_vwap.dropna()[['vwap']].values)
y = df_vwap.dropna()[ind_name].values
model = sm.OLS(y,X)
results = model.fit()
results.summary()

X = np.log(df_vwap.dropna()[['market cap']].values)
y = df_vwap.dropna()[ind_name].values
model = sm.OLS(y,X)
results = model.fit()
results.summary()



for Id in Ids[50:80]:
    file = str(Id)+'.csv'
    df = get_df(file,file_path,days)
#    if df.empty:
#        continue
    df = df.merge(today[['InstrumentId','free_cap','VWAvg']],on = ['InstrumentId'],how = 'left')
    fac_name = 'resist_buy_lag'
    range_noinf = df[(df[fac_name]>0) & (df[fac_name]<np.inf)]
    plot(range_noinf[fac_name],range_noinf,log=True)
    #all_plot(range_noinf[fac_name],range_noinf,log=True)
    print('id: ',Id)
    print('market cap: ',df['free_cap'].iloc[0]/1e8)
    print('price: ',df['vwap_lag'].iloc[-1])
    print('-----------------')


fac_id = df[df['ret_vwapmid_fur']>0.005]

np.exp(-1)
np.log(2)

all_plot(df['act_vol_skew_lag'],df)

        
Id = 600612


603096,603180,600278,603228,603192,603595,603156,  600612

603019


transaction_file_path = "E:/Level2/XSHG_Transaction/201809/"

trans_l = os.listdir(transaction_file_path)
transaction = pd.read_csv(transaction_file_path+trans_l[0])
transaction['datetime'] = pd.to_datetime(transaction['Date'].astype(str)+transaction['Time'].astype(str).str.zfill(9),format='%Y%m%d%H%M%S%f')
transaction['TradeDirection'] = ((transaction['BuyOrderId']>transaction['SellOrderId']).astype(int)*2-1).astype(int)
transaction['ActOrderId'] = (transaction['BuyOrderId']*(transaction['TradeDirection']+1)*0.5 - transaction['SellOrderId']*(transaction['TradeDirection']-1)*0.5).astype(int)
transaction['NetVolume'] = (transaction['TradeVolume']*transaction['TradeDirection']).astype(int)

temp_trans = transaction[(transaction['datetime']>dt.datetime(2018,9,3,9,40)) &
                         (transaction['datetime']<dt.datetime(2018,9,3,9,45))]
temp_trans = temp_trans[temp_trans['InstrumentId']==600612]



grouped = temp_trans.groupby(['ActOrderId'])
temp_trans_grouped = grouped.last()
temp_trans_grouped[['TradeAmount','TradeVolume','NetVolume']] = grouped.sum()[['TradeAmount','TradeVolume','NetVolume']]
temp_trans_grouped['TradePrice'] = temp_trans_grouped['TradeAmount']/temp_trans_grouped['TradeVolume']


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









