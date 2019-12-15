# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 10:34:51 2019

new order data.
analysis.

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
output_file_path = dir_name+'/'+str(date)+'/'

window = dt.timedelta(seconds=60)

file_path = output_file_path
#fac_path = 'fac_5m/20180903/'
l = os.listdir(file_path)
fac_data = []
for i in range(len(l)): # i=0;
    file = l[i]
    snap = pd.read_csv(snapshot_file_path+file)
    snap = snap.drop_duplicates(['Time','InstrumentId'],keep='last')
    snap['datetime'] = pd.to_datetime(snap['Date'].astype(str)+snap['Time'].astype(str).str.zfill(9),format='%Y%m%d%H%M%S%f')
    snap = snap.reset_index(drop=True)  
    snap_formerge = snap[['datetime','BuyPrice1','BuyVolume1',
                          'SellPrice1','SellVolume1']]

    df = pd.read_csv(file_path+file,index_col=0)
    df = df.reset_index()
    df = df.rename(columns = {'index':'datetime'})
    df['datetime']=pd.to_datetime(df['datetime'],format='%Y-%m-%d %H:%M:%S')
    
    df = df.merge(snap_formerge,on='datetime',how='left')
    df['ret_vwapmid_lag'] = np.log(2*df['traded_all_vwap']/(df['BuyPrice1']+df['SellPrice1']))
    df_fur = df.copy()
    df_fur['datetime'] -= window
    df_fur = df_fur.rename(columns={'ret_vwapmid_lag':'ret_vwapmid_fur'})
    df = df.merge(df_fur[['datetime','ret_vwapmid_fur']],on='datetime',how='left')
    fac_data.append(df)
    if not i % 100:
        print(i)


fac_data = pd.concat(fac_data)
fac_data = fac_data.reset_index()
fac_data['datetime']=pd.to_datetime(fac_data['datetime'],format='%Y-%m-%d %H:%M:%S')

daily = pd.read_csv('E:/Level2/daily/201809StockDailyData.csv')
daily['InstrumentId'] = daily['WindCode'].apply(lambda x:int(x[:6]))
daily_SH = daily[daily['WindCode'].apply(lambda x:True if 'SZ' in x else False)]
today = daily_SH[daily_SH['TradingDay']==20180903].copy()
today['free_cap'] = today['VWAvgPrice']*today['NonRestrictedShares']/today['SplitFactor']
fac_data_m = fac_data.merge(today[['InstrumentId','free_cap','VWAvgPrice','TurnoverVolume','NonRestrictedShares']],on = ['InstrumentId'],how = 'left')


list(df.columns)


Ids = fac_data['InstrumentId'].unique()

fac_data.columns

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

##############


list(fac_data.columns)
range_noinf = fac_data[fac_data['ret_vwapmid_fur']<0.5]

all_plot(range_noinf['trade_volume'],range_noinf,log=True)


fac = 'new_buy_vol_above_5'
fac_range = fac_data[fac_data['ret_vwapmid_fur']<0.5]
range_noinf = fac_range.dropna(subset=[fac,'ret_vwapmid_fur'])
range_noinf = range_noinf[range_noinf[fac]>0]
all_plot(range_noinf[fac],range_noinf,log=True)

fac = 'new_buy_vol_above_lowest'
fac_range = fac_data[fac_data['ret_vwapmid_fur']<0.5]
range_noinf = fac_range.dropna(subset=[fac,'ret_vwapmid_fur'])
range_noinf = range_noinf[range_noinf[fac]>0]
all_plot(range_noinf[fac],range_noinf,log=True)

fac = 'cancelled_buy_vol_above_lowest'
fac_range = fac_data[fac_data['ret_vwapmid_fur']<0.5]
range_noinf = fac_range.dropna(subset=[fac,'ret_vwapmid_fur'])
range_noinf = range_noinf[range_noinf[fac]>0]
all_plot(range_noinf[fac],range_noinf,log=True)

fac = 'cancelled_sell_vol_above_lowest'
range_noinf = fac_range.dropna(subset=[fac,'ret_vwapmid_fur'])
range_noinf = range_noinf[range_noinf[fac]>0]
all_plot(range_noinf[fac],range_noinf,log=True)
#####################

fac1 = 'new_buy_vol_above_lowest'
fac2 = 'cancelled_buy_vol_above_lowest'
fac_range = fac_data[fac_data['ret_vwapmid_fur']<0.5].copy()
fac_range = fac_range.dropna(subset=[fac1,fac2,'ret_vwapmid_fur'])
fac_range = fac_range[(fac_range[fac1]>0) & (fac_range[fac2]>0)]
fac_range['newbuy5_cancelled'] = np.log(fac_range[fac1]/fac_range[fac2])
all_plot(fac_range['newbuy5_cancelled'],fac_range)

fac1 = 'net_new_buy_vol_above_lowest'
fac2 = 'traded_sell_vol'
fac_range = fac_data[fac_data['ret_vwapmid_fur']<0.5].copy()
fac_range = fac_range.dropna(subset=[fac1,fac2,'ret_vwapmid_fur'])
fac_range = fac_range[(fac_range[fac1]>0) & (fac_range[fac2]>0)]
fac_range['newbuy5_cancelled'] = np.log(fac_range[fac1]/fac_range[fac2])
all_plot(fac_range['newbuy5_cancelled'],fac_range)




(fac_data[fac1]<0).sum()/len(fac_data)
(fac_data['new_buy_vol_above_lowest']<0).sum()

fac_range['ret_vwapmid_fur'].mean()
(fac_range['ret_vwapmid_fur']>0).sum()/len(fac_range)



range_noinf['ret_vwapmid_fur'].hist(bins=100)


#################################################

fac1 = 'net_new_buy_vol_above_lowest'
fac2 = 'traded_sell_vol'


Ids = fac_data['InstrumentId'].unique()
for Id in Ids[100:130]:
    fac_range = fac_data[fac_data['ret_vwapmid_fur']<0.5].copy()
    fac_range = fac_range[fac_range['InstrumentId']==Id]
    if len(fac_range)<200:
        continue
    fac_range = fac_range.dropna(subset=[fac1,fac2,'ret_vwapmid_fur'])
    fac_range = fac_range[(fac_range[fac1]>0) & (fac_range[fac2]>0)]
    fac_range['newbuy5_cancelled'] = np.log(fac_range[fac1]/fac_range[fac2])
    print(Id,fac_range['traded_all_vwap'].mean())
    all_plot(fac_range['newbuy5_cancelled'],fac_range)



fac_range = fac_data[fac_data['ret_vwapmid_fur']<0.5].copy()
fac_range = fac_range.dropna(subset=['new_all_vwap_above_5','traded_all_vwap'])

all_plot(np.log(fac_range['traded_all_vwap']/fac_range['new_all_vwap_above_5']),fac_range)


for Id in Ids[100:130]:
    fac_range = fac_data[fac_data['ret_vwapmid_fur']<0.5].copy()
    fac_range = fac_range[fac_range['InstrumentId']==Id]
    fac_range = fac_range[(fac_range['SellPrice1']!=0) & (fac_range['BuyPrice1']!=0) & (fac_range['new_sell_vwap_above_5']!=0)]
    fac_range = fac_range[np.log(fac_range['new_sell_vwap_above_5']/fac_range['SellPrice1'])>-0.1]
    #all_plot(np.log(fac_range['new_sell_vwap_above_5']/fac_range['SellPrice1']),fac_range)
    if len(fac_range)<1000:
        continue
    plot(np.log(fac_range['new_sell_vwap_above_5']/fac_range['SellPrice1']),fac_range)

# above sellvwap?

fac1 = 'net_new_buy_vol_above_5'
fac2 = 'net_new_sell_vol_above_5'
fac_range = fac_data[fac_data['ret_vwapmid_fur']<0.5].copy()
fac_range = fac_range.dropna(subset=[fac1,fac2,'ret_vwapmid_fur'])
fac_range = fac_range[(fac_range[fac1]>0) & (fac_range[fac2]>0)]
fac_range['newbuy5_cancelled'] = np.log(fac_range[fac1]/fac_range[fac2])
print(Id,fac_range['traded_all_vwap'].mean())
plot(fac_range['newbuy5_cancelled'],fac_range)

fac1 = 'net_new_buy_vol_above_5'
fac2 = 'traded_sell_vol'
fac_range = fac_data[fac_data['ret_vwapmid_fur']<0.5].copy()
fac_range = fac_range.dropna(subset=[fac1,fac2,'ret_vwapmid_fur'])
fac_range = fac_range[(fac_range[fac1]>0) & (fac_range[fac2]>0)]
fac_range['newbuy5_cancelled'] = np.log(fac_range[fac1]/fac_range[fac2])
print(Id,fac_range['traded_all_vwap'].mean())
plot(fac_range['newbuy5_cancelled'],fac_range)


list(fac_data.columns)


fac1 = 'net_new_sell_vol_above_5'
fac2 = 'traded_buy_vol'
fac_range = fac_data[fac_data['ret_vwapmid_fur']<0.5].copy()
fac_range = fac_range.dropna(subset=[fac1,fac2,'ret_vwapmid_fur'])
fac_range = fac_range[(fac_range[fac1]>0) & (fac_range[fac2]>0)]
fac_range['newbuy5_cancelled'] = np.log(fac_range[fac1]/fac_range[fac2])
print(Id,fac_range['traded_all_vwap'].mean())
plot(fac_range['newbuy5_cancelled'],fac_range)


fac1 = 'new_sell_vol_above_5'
fac2 = 'traded_buy_vol'
fac_range = fac_data[fac_data['ret_vwapmid_fur']<0.5].copy()
fac_range = fac_range.dropna(subset=[fac1,fac2,'ret_vwapmid_fur'])
fac_range = fac_range[(fac_range[fac1]>0) & (fac_range[fac2]>0)]
fac_range['newbuy5_cancelled'] = np.log(fac_range[fac1]/fac_range[fac2])
print(Id,fac_range['traded_all_vwap'].mean())
plot(fac_range['newbuy5_cancelled'],fac_range)

fac1 = 'cancelled_buy_vol_above_lowest'
fac2 = 'cancelled_sell_vol_above_lowest'
fac_range = fac_data[fac_data['ret_vwapmid_fur']<0.5].copy()
fac_range = fac_range.dropna(subset=[fac1,fac2,'ret_vwapmid_fur'])
fac_range = fac_range[(fac_range[fac1]>0) & (fac_range[fac2]>0)]
fac_range['newbuy5_cancelled'] = np.log(fac_range[fac1]/fac_range[fac2])
print(Id,fac_range['traded_all_vwap'].mean())
plot(fac_range['newbuy5_cancelled'],fac_range)

fac1 = 'cancelled_sell_vol_above_lowest'
fac2 = 'net_new_sell_vol_above_lowest'
fac_range = fac_data[fac_data['ret_vwapmid_fur']<0.5].copy()
fac_range = fac_range.dropna(subset=[fac1,fac2,'ret_vwapmid_fur'])
fac_range = fac_range[(fac_range[fac1]>0) & (fac_range[fac2]>0)]
fac_range['newbuy5_cancelled'] = np.log(fac_range[fac1]/fac_range[fac2])
print(Id,fac_range['traded_all_vwap'].mean())
all_plot(fac_range['newbuy5_cancelled'],fac_range)

fac1 = 'cancelled_buy_vol_above_lowest'
fac2 = 'net_new_buy_vol_above_lowest'
fac_range = fac_data[fac_data['ret_vwapmid_fur']<0.5].copy()
fac_range = fac_range.dropna(subset=[fac1,fac2,'ret_vwapmid_fur'])
fac_range = fac_range[(fac_range[fac1]>0) & (fac_range[fac2]>0)]
fac_range['newbuy5_cancelled'] = np.log(fac_range[fac1]/fac_range[fac2])
print(Id,fac_range['traded_all_vwap'].mean())
all_plot(fac_range['newbuy5_cancelled'],fac_range)


fac1 = 'cancelled_sell_vol_above_lowest'
fac2 = 'new_sell_vol_above_lowest'
fac_range = fac_data[fac_data['ret_vwapmid_fur']<0.5].copy()
fac_range = fac_range.dropna(subset=[fac1,fac2,'ret_vwapmid_fur'])
fac_range = fac_range[(fac_range[fac1]>0) & (fac_range[fac2]>0)]
fac_range['newbuy5_cancelled'] = np.log(fac_range[fac1]/fac_range[fac2])
print(Id,fac_range['traded_all_vwap'].mean())
plot(fac_range['newbuy5_cancelled'],fac_range)

fac1 = 'cancelled_buy_vol_above_lowest'
fac2 = 'traded_buy_vol'
fac_range = fac_data[fac_data['ret_vwapmid_fur']<0.5].copy()
fac_range = fac_range.dropna(subset=[fac1,fac2,'ret_vwapmid_fur'])
fac_range = fac_range[(fac_range[fac1]>0) & (fac_range[fac2]>0)]
fac_range['newbuy5_cancelled'] = np.log(fac_range[fac1]/fac_range[fac2])
print(Id,fac_range['traded_all_vwap'].mean())
plot(fac_range['newbuy5_cancelled'],fac_range)


fac1 = 'cancelled_buy_vol_above_5'
fac2 = 'traded_buy_vol'
fac_range = fac_data[fac_data['ret_vwapmid_fur']<0.5].copy()
fac_range = fac_range.dropna(subset=[fac1,fac2,'ret_vwapmid_fur'])
fac_range = fac_range[(fac_range[fac1]>0) & (fac_range[fac2]>0)]
fac_range['newbuy5_cancelled'] = np.log(fac_range[fac1]/fac_range[fac2])
print(Id,fac_range['traded_all_vwap'].mean())
plot(fac_range['newbuy5_cancelled'],fac_range)

#-----------------
fac1 = 'cancelled_buy_vol_above_5'
fac_range = fac_data[fac_data['ret_vwapmid_fur']<0.5].copy()
fac_range = fac_range.dropna(subset=[fac1,'ret_vwapmid_fur'])
fac_range = fac_range[fac_range[fac1]>0]
plot(fac_range[fac1],fac_range,log=True)
#all_plot(fac_range[fac1],fac_range,log=True)
# --
temp = fac_range[(np.log(fac_range[fac1])>14) & (fac_range['ret_vwapmid_fur']>0.01)]
fac_id = fac_range[fac_range['InstrumentId'] == 979]
plot(fac_id[fac1],fac_id,log=True)

plot(fac_range[fac_range['InstrumentId']!=979][fac1],fac_range[fac_range['InstrumentId']!=979],log=True)


fac_data_m['turnover'] = fac_data_m['TurnoverVolume']/fac_data_m['NonRestrictedShares']

fac_data_m['turnover'].drop_duplicates().hist(bins=100)

fac_data_m['turnover'].drop_duplicates().quantile(0.8)

fac1 = 'cancelled_buy_vol_above_5'
fac_turnover = fac_data_m[fac_data_m['turnover']>0.01]
fac_range = fac_turnover[fac_turnover['ret_vwapmid_fur']<0.5].copy()
fac_range = fac_range.dropna(subset=[fac1,'ret_vwapmid_fur'])
fac_range = fac_range[fac_range[fac1]>0]
plot(fac_range[fac1]/fac_range['NonRestrictedShares'],fac_range,log=True)

fac2 = 'traded_all_vol'
fac_range = fac_range.dropna(subset=[fac2,'ret_vwapmid_fur'])
fac_range = fac_range[fac_range[fac2]>0]
plot(fac_range[fac1]/fac_range[fac2],fac_range,log=True)

plot(fac_range['traded_all_vol'],fac_range,log=True)

##########
fac_range = fac_data_m[fac_data_m['new_buy_vol_above_lowest']>0]
(fac_range['new_buy_vol_above_5']/fac_range['new_buy_vol_above_lowest']).mean()
(fac_range['new_buy_vol_above_5']/fac_range['new_buy_vol_above_lowest']).median()


temp = fac_range['new_buy_vol_above_5']/fac_range['new_buy_vol_above_lowest']
(temp[temp<5]).hist(bins=100)
plot(fac_range[fac1]/fac_range['NonRestrictedShares'],fac_range,log=True)


fac_range = fac_data_m[(fac_data_m['new_sell_vol_above_lowest']>0) & (fac_data_m['traded_buy_vol']>0) &
                       ((fac_data_m['new_sell_vol_above_5']>0))]
fac_range = fac_range[fac_range['ret_vwapmid_fur']<0.5].copy()
temp = fac_range['new_sell_vol_above_5']/fac_range['new_sell_vol_above_lowest']
fac_range = fac_range[temp>2]
plot(fac_range['new_sell_vol_above_lowest']/fac_range['traded_buy_vol'],fac_range,log = True)

#########################
fac_range = fac_data_m[(fac_data_m['net_new_sell_vol_above_lowest']>0) & (fac_data_m['traded_buy_vol']>0) &
                       ((fac_data_m['net_new_sell_vol_above_5']>0))]
fac_range = fac_range[fac_range['ret_vwapmid_fur']<0.5].copy()
temp = fac_range['net_new_sell_vol_above_5']/fac_range['net_new_sell_vol_above_lowest']
fac_range = fac_range[temp<1.5]
all_plot(fac_range['net_new_sell_vol_above_lowest']/fac_range['traded_buy_vol'],fac_range,log = True)


fac_range = fac_data_m[(fac_data_m['net_new_sell_vol_above_lowest']>0) & (fac_data_m['traded_buy_vol']>0) &
                       ((fac_data_m['net_new_sell_vol_above_5']>0))]
fac_range = fac_range[fac_range['ret_vwapmid_fur']<0.5].copy()
temp = fac_range['net_new_sell_vol_above_5']/fac_range['net_new_sell_vol_above_lowest']
fac_range = fac_range[temp>2]
all_plot(fac_range['net_new_sell_vol_above_lowest']/fac_range['traded_buy_vol'],fac_range,log = True)


len(Ids)

fac = 'new_sell_vol_above_5'
fac_range = fac_data_m[(fac_data_m[fac]>0) & (fac_data_m['traded_buy_vol']>0)]
fac_range = fac_range[fac_range['ret_vwapmid_fur']<0.5].copy()
temp = fac_range['new_sell_vol_above_5']/fac_range[fac]
fac_range = fac_range[temp<1.5]
plot(fac_range[fac]/fac_range['traded_buy_vol'],fac_range,log = True)

list(fac_range.columns)
fac_range.shape


fac_range = fac_data_m[(fac_data_m['new_sell_vol_above_lowest']>0) & (fac_data_m['new_sell_vol_above_5']>0)]
fac_range = fac_range[fac_range['ret_vwapmid_fur']<0.5].copy()
plot(fac_range['new_sell_vol_above_lowest']/fac_range['new_sell_vol_above_5'],fac_range)


fac_range = fac_data_m[(fac_data_m['new_buy_vol_above_lowest']>0) & (fac_data_m['traded_buy_vol']>0)]
fac_range = fac_range[fac_range['ret_vwapmid_fur']<0.5].copy()
temp = fac_range['new_buy_vol_above_lowest']/fac_range['new_buy_vol_above_5']
fac_range = fac_range[temp<1]
plot(fac_range['new_buy_vol_above_lowest']/fac_range['new_buy_vol_above_5'],fac_range)


fac_range['new_buy_vol_above_lowest']/fac_range['new_sell_vol_above_5']


# --
增量跟净增量比较
挂单增量在


fac1 = 'cancelled_buy_vol_above_5'
fac2 = 'cancelled_sell_vol_above_5'
fac_range = fac_data[fac_data['ret_vwapmid_fur']<0.5].copy()
fac_range = fac_range.dropna(subset=[fac1,fac2,'ret_vwapmid_fur'])
fac_range = fac_range[(fac_range[fac1]>0) & (fac_range[fac2]>0)]
fac_range['newbuy5_cancelled'] = np.log(fac_range[fac1]/fac_range[fac2])
print(Id,fac_range['traded_all_vwap'].mean())
plot(fac_range['newbuy5_cancelled'],fac_range)



list(fac_data_m.columns)

temp = fac_data_m['cancelled_buy_vol_above_5'] + fac_data_m['cancelled_sell_vol_above_5']
plt.scatter(np.log(fac_data_m['traded_all_vol']),np.log(temp),alpha=0.5)

fac_range = fac_data_m[(fac_data_m['net_new_buy_vol_above_lowest']>0) & (fac_data_m['traded_buy_vol']>0)]
fac_range['fac'] = np.log(fac_data_m['net_new_buy_vol_above_lowest']/fac_data_m['traded_buy_vol'])

plt.scatter(np.log(fac_data_m['net_new_buy_vol_above_lowest']),np.log(fac_data_m['net_new_sell_vol_above_lowest']),alpha=0.5)

################################################## 

for i in range(1,6):
    temp = fac_range.copy()
    temp['datetime'] += dt.timedelta(seconds=30*i)
    fac_range = fac_range.merge(temp[['datetime','fac']].rename(columns={'fac':'fac_-'+str(30*i)}),on='datetime',how='left')

gc.collect()


