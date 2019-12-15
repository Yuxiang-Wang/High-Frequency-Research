# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 08:59:30 2019

算大小单比例因子，这里的计算用的还是当天的成交量的分位数


@author: yuxiang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from time import time
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
%matplotlib inline

file_path = "E:/Level2/XSHG_Transaction/201809/"
year = 2018
month = 9
day = 3
file = "20180903.csv"
snapshot = pd.read_csv("E:/Level2/XSHG_SnapShot/201809/20180903.csv")
transaction = pd.read_csv(file_path + file)

snapshot = snapshot.drop_duplicates()
snapshot = snapshot.drop_duplicates(['Time','InstrumentId'],keep='last')
snapshot['datetime'] = pd.to_datetime(snapshot['Date'].astype(str)+snapshot['Time'].astype(str).str.zfill(9),format='%Y%m%d%H%M%S%f')
snapshot = snapshot.reset_index(drop=True)


transaction['datetime'] = pd.to_datetime(transaction['Date'].astype(str)+transaction['Time'].astype(str).str.zfill(9),format='%Y%m%d%H%M%S%f')
#transaction = transaction[transaction['datetime']>dt.datetime(year,month,day,9,30,0,0)]
transaction['TradeDirection'] = ((transaction['BuyOrderId']>transaction['SellOrderId']).astype(int)*2-1).astype(int)
transaction['ActOrderId'] = (transaction['BuyOrderId']*(transaction['TradeDirection']+1)*0.5 - transaction['SellOrderId']*(transaction['TradeDirection']-1)*0.5).astype(int)
transaction['NetVolume'] = (transaction['TradeVolume']*transaction['TradeDirection']).astype(int)
transaction = transaction.reset_index()


window = dt.timedelta(minutes=1)
am = dt.datetime(year,month,day,11,30,0,0)-window
pm1 = dt.datetime(year,month,day,13,00,0,0)+2*window
pm = dt.datetime(year,month,day,14,57,0,0)-window


df_all = pd.DataFrame()
ids = list(set(transaction['InstrumentId']))
for j in range(int(len(ids)*0.5)): # Id=600000
#for j in range(int(len(ids)*0.5),len(ids)):
    Id = ids[j]
    transaction_id = transaction[transaction['InstrumentId']==Id]
    snapshot_id = snapshot[snapshot['InstrumentId']==Id]
    t = snapshot_id[snapshot_id['datetime']>=dt.datetime(year,month,day,9,30,0,0)]['datetime'].iloc[0]
    d1={}
    while t < pm:  # t = dt.datetime(year,month,day,9,40,0,0)
        if (t>am) & (t<pm1):
            t = snapshot_id[snapshot_id['datetime']>=pm1]['datetime'].iloc[0]
            continue
        temp = transaction_id[(transaction_id['datetime']>t) & (transaction_id['datetime']<t+window)]
        temp_lag = transaction_id[(transaction_id['datetime']>t-window) & (transaction_id['datetime']<=t)]

        if temp.empty or temp_lag.empty:
            t+=window
            #t+=dt.timedelta(seconds=1)
            continue


        d1[t]={}
        d1[t]['vwap_fur'] = sum(temp['TradeAmount'])/sum(temp['TradeVolume'])
        d1[t]['vwap_lag'] = sum(temp_lag['TradeAmount'])/sum(temp_lag['TradeVolume'])
        d1[t]['total_volume_lag'] = sum(temp_lag['TradeVolume'])

        temp_actbuy = temp_lag[temp_lag['TradeDirection']==1]
        temp_actsell = temp_lag[temp_lag['TradeDirection']==-1]
        d1[t]['buy_volume_lag'] = sum(temp_actbuy['TradeVolume'])
        d1[t]['sell_volume_lag'] = sum(temp_actsell['TradeVolume'])
        d1[t]['buy_ave_vol_lag'] = np.log(temp_lag.groupby('BuyOrderId').sum()['TradeVolume'].mean())
        d1[t]['sell_ave_vol_lag'] = np.log(temp_lag.groupby('SellOrderId').sum()['TradeVolume'].mean())

        grouped_actvol = temp_lag.groupby('ActOrderId').sum()
        temp_big = grouped_actvol[grouped_actvol['TradeVolume']>transaction_id.groupby('ActOrderId').sum()['TradeVolume'].quantile(0.8)]
        temp_small = grouped_actvol[grouped_actvol['TradeVolume']<=transaction_id.groupby('ActOrderId').sum()['TradeVolume'].quantile(0.8)]
        
        d1[t]['big_volume_lag'] = sum(temp_big['TradeVolume'])
        d1[t]['act_vol_skew_lag'] = np.power((grouped_actvol['NetVolume'] - grouped_actvol['NetVolume'].mean())/grouped_actvol['NetVolume'].std(),3).sum()
        
        temp_buybig = temp_big[temp_big['TradeDirection']==1]
        temp_sellbig = temp_big[temp_big['TradeDirection']==-1]
        d1[t]['big_buy_volume_lag'] = sum(temp_buybig['TradeVolume'])        
        d1[t]['big_sell_volume_lag'] = sum(temp_sellbig['TradeVolume'])

        temp_supbig = grouped_actvol[grouped_actvol['TradeVolume']>transaction_id.groupby('ActOrderId').sum()['TradeVolume'].quantile(0.95)]
        temp_supbuybig = temp_supbig[temp_supbig['TradeDirection']==1]
        temp_supsellbig = temp_supbig[temp_supbig['TradeDirection']==-1]
        d1[t]['supbig_buy_volume_lag'] = sum(temp_supbuybig['TradeVolume'])        
        d1[t]['supbig_sell_volume_lag'] = sum(temp_supsellbig['TradeVolume'])

        t+=window
        #---------------

        
        #t+=dt.timedelta(seconds=1)
    df = pd.DataFrame.from_dict(d1,orient='index')
    df['InstrumentId'] = Id
    df_all = pd.concat([df_all,df])
    if not j%10:
        print(j,end=',')

df_all = df_all.reset_index()
df_all.rename(columns={'index':'datetime'},inplace=True)
df_all.to_csv('big_small_1.csv')
df_all.to_csv('big_small_2.csv')

#-----------------------------------------
def plot(temp,factor=temp,log=False,col=None):
    if log:
        temp = np.log(temp)
    plt.axhline(y=0,color='r')
    #plt.axhline(y=0.001,color='r')
    #plt.axhline(y=-0.001,color='r')
    plt.axvline(x=0,color='r')
    plt.scatter(temp,factor['ret_vwapmid_fur'])
    if col:
        plt.title(col)
    plt.xlim(min(temp)*0.9,max(temp)*1.1)
    plt.ylim(min(factor['ret_vwapmid_fur'])-0.0005,max(factor['ret_vwapmid_fur'])+0.0005)
    #plt.plot([-10000,10000],[0,0],color='r')
    plt.show()

df1 = pd.read_csv('df6_merged.csv',index_col=0)
df1['datetime']=pd.to_datetime(df1['datetime'],format='%Y-%m-%d %H:%M:%S')

big_small_1 = pd.read_csv('big_small_1.csv',index_col=0)
big_small_2 = pd.read_csv('big_small_2.csv',index_col=0)

big_small = pd.concat([big_small_1,big_small_2])
big_small['datetime']=pd.to_datetime(big_small['datetime'],format='%Y-%m-%d %H:%M:%S')

temp = df1.merge(big_small,on=['InstrumentId','datetime'])

temp['big_total_ratio_lag'] = temp['big_volume_lag']/temp['total_volume_lag']
temp['net_volume'] = temp['buy_volume_lag'] - temp['sell_volume_lag']



# ----------------------------
plot(temp['net_volume']/temp['trade_volume_lag'].mean())
plot(temp['net_volume']/temp['trade_volume_lag'])
temp[temp['net_volume']/temp['trade_volume_lag']>0.9]['ret_vwapmid_fur'].quantile(0.5)
plot(temp['net_volume'])
temp[(temp['net_volume']>500000) & (temp['net_volume']<1000000)]['ret_vwapmid_fur'].quantile(0.4)

plot(temp['net_volume_lag'],temp)
temp_netvolume = temp[temp['net_volume_lag']>100000]
print((temp_netvolume['ret_vwapmid_fur']>0.0005).sum()/len(temp_netvolume))


plot(temp['big_volume_lag']/temp['trade_volume_lag'],col = 'big_volume/total_volume')

plot(temp['big_buy_volume_lag']/temp['sell_volume_lag'],log=1,col = 'big_volume/total_volume')



temp_bigvolume = temp[temp['big_volume_lag']>0]
plot(temp_bigvolume['big_volume_lag'],temp_bigvolume,1,'big_volume_log')

plot(temp['big_buy_volume_lag'],temp)
temp[temp['big_buy_volume_lag']>500000]['ret_vwapmid_fur'].quantile(0.5)

plot(temp['big_buy_volume_lag']/temp['big_sell_volume_lag'],temp)
temp[temp['big_buy_volume_lag']/temp['big_sell_volume_lag']>40]['ret_vwapmid_fur'].quantile(0.9)

plt.scatter(temp['big_buy_volume_lag']/temp['big_sell_volume_lag'],temp['ret_vwapmid_fur'])
plot(np.log(temp['big_buy_volume_lag']/temp['big_sell_volume_lag']),temp,0,'big buy/big sell')

temp_bigvolume = temp[(temp['big_volume_lag']>0) & ()]



plot(temp['big_buy_volume_lag']/temp['trade_volume_lag'],temp)


temp_netvolume = temp[(temp['net_volume']<500000) & (temp['net_volume']>0)]
plot(temp_netvolume['big_buy_volume_lag'],temp_netvolume)
plot(temp_netvolume['big_buy_volume_lag']-temp_netvolume['big_sell_volume_lag'],temp_netvolume)
plot(temp_netvolume['big_buy_volume_lag']/temp_netvolume['big_sell_volume_lag'],temp_netvolume)
plot(temp_netvolume['buy_volume_lag']/(temp_netvolume['buy_volume_lag']+temp_netvolume['sell_volume_lag']),temp_netvolume)

plot(temp['act_vol_skew_lag'])
temp[temp['act_vol_skew_lag']>0]['ret_vwapmid_fur'].quantile(0.5)
plot(temp['supbig_buy_volume_lag'])
temp[temp['supbig_buy_volume_lag']>50000]['ret_vwapmid_fur'].quantile(0.5)
plot(temp['supbig_buy_volume_lag']/temp[''])

plot(temp['supbig_sell_volume_lag'])

temp_netvolume = temp[(temp['net_volume']<temp['net_volume'].quantile(0.8)) & (temp['net_volume']>temp['net_volume'].quantile(0.5))]
temp_netvolume = temp_netvolume[temp_netvolume['buy_volume_lag']!=0]
temp_netvolume['big_buy_ratio_lag'] = temp_netvolume['big_buy_volume_lag']/temp_netvolume['buy_volume_lag']
plot(temp_netvolume['big_buy_ratio_lag'],temp_netvolume)
temp_netvolume = temp_netvolume[(temp_netvolume['big_buy_ratio_lag']>0) & (temp_netvolume['big_buy_ratio_lag']<0.4)]
print(temp_netvolume['ret_vwapmid_fur'].mean(),temp_netvolume['ret_vwapmid_fur'].quantile(0.7))
temp_netvolume['ret_vwapmid_fur'].hist(bins=50)


temp_netvolume = temp[(temp['net_volume']<temp['net_volume'].quantile(0.8)) & (temp['net_volume']>temp['net_volume'].quantile(0.5))]
temp_netvolume = temp_netvolume[temp_netvolume['sell_volume_lag']!=0]
temp_netvolume['big_sell_ratio_lag'] = temp_netvolume['big_sell_volume_lag']/temp_netvolume['sell_volume_lag']
plot(temp_netvolume['big_sell_ratio_lag'],temp_netvolume)
temp_netvolume = temp_netvolume[(temp_netvolume['big_sell_ratio_lag']>0) & (temp_netvolume['big_sell_ratio_lag']<0.4)]
print(temp_netvolume['ret_vwapmid_fur'].mean(),temp_netvolume['ret_vwapmid_fur'].quantile(0.7))
temp_netvolume['ret_vwapmid_fur'].hist(bins=50)


#---------------------------------


l=[]
for file in ['incre_4.csv','incre_7.csv','incre_8.csv','incre_9.csv']:
    l.append(pd.read_csv(file,index_col=0).dropna())
incre_all = pd.concat(l)
incre_all['datetime']=pd.to_datetime(incre_all['datetime'],format='%Y-%m-%d %H:%M:%S')

df_merged = temp.merge(incre_all,on=['InstrumentId','datetime'])


plot(np.log(df_merged['resist2']),df_merged)


plot(df_merged['incre_buy'] - df_merged['total_sell_vol_lag'],df_merged)
plot(df_merged['incre_buy'],df_merged)
plot(df_merged['incre_sell'],df_merged)
plot(df_merged['incre_buy'] - df_merged['incre_sell'],df_merged)







temp_merged = df_merged[df_merged['incre_buy'] - df_merged['total_sell_vol_lag'] >0]
plot(temp_merged['big_buy_volume_lag']/temp_merged['buy_volume_lag'],temp_merged)

temp_merged.columns

down(temp_merged['BuyPrice1']+temp_merged['SellPrice1']) - temp_merged['lowest_price_lag']

temp_merged.columns




# --------------------------------------
temp_supbig = temp[temp['supbig_buy_volume_lag']!=0]
print(temp_supbig['ret_vwapmid_fur'].mean(), temp_supbig['ret_vwapmid_fur'].quantile(0.6))




# ------------------------------------------
def quantile_cut(down,up,factor):
    for col in factor.columns:
        if col in ['InstrumentId','datetime','ret_highest_fur','ret_lowest_fur','ret_vwapmid_fur']:
            continue
        factor.loc[factor[col]>factor[col].quantile(up),col] = factor[col].quantile(up)
        factor.loc[factor[col]<factor[col].quantile(down),col] = factor[col].quantile(down)
def normalize(factor):
    for col in factor.columns:
        if col in ['InstrumentId','datetime','ret_highest_fur','ret_lowest_fur','ret_vwapmid_fur']:
            continue
        factor.loc[:,col] = (factor[col] - factor[col].mean())/factor[col].std()

df_reg = df_merged[(df_merged['total_sell_vol_lag']!=0 ) & (df_merged['incre_buy']>0)].copy()
df_reg['resist'] = df_reg['incre_buy'] - df_reg['total_sell_vol_lag']
df_reg['resist2'] = np.log(df_reg['incre_buy']/df_reg['total_sell_vol_lag'])
df_reg['resist3'] = df_reg['incre_buy']-df_reg['incre_sell']
df_reg.loc[:,'volume_imbalance'] = df_reg['BuyVolume1']/(df_reg['BuyVolume1']+df_reg['SellVolume1'])
quantile_cut(0.01,0.99,df_reg)
normalize(df_reg)

df_reg.columns

cols = ['trade_volume_imbalance', 'volume_imbalance',
       'total_volume_lag', 'buy_volume_lag', 'sell_volume_lag',
       'buy_ave_vol_lag', 'sell_ave_vol_lag', 'big_volume_lag',
       'act_vol_skew_lag', 'big_buy_volume_lag', 'big_sell_volume_lag',
       'supbig_buy_volume_lag', 'supbig_sell_volume_lag',
       'big_total_ratio_lag', 'net_volume', 'incre_buy', 'incre_sell','resist3','resist2','resist']

d={}
for col in cols:
    X = df_reg[col].values
    y = df_reg['ret_vwapmid_fur'].values
    
    X = sm.add_constant(X)
    model = sm.OLS(y,X)
    
    results = model.fit() 
    d[col] = results.params[1],results.pvalues[1]
    
coefd = pd.DataFrame.from_dict(d,orient='index')

# ------------------
fac = ['trade_volume_imbalance','net_volume','volume_imbalance','resist2','act_vol_skew_lag']  # ,'resist2','act_vol_skew_lag'
X = df_reg[fac].values
y = df_reg['ret_vwapmid_fur'].values

X = sm.add_constant(X)
model = sm.OLS(y,X)

results = model.fit() 
results.summary()
# ------------------    

for col in fac:
    plot(df_reg[col],df_reg,col=col)


df_reg['volume_imbalance']

(df_reg['resist3']>45).sum()

df_reg[df_reg['resist3']>45]


