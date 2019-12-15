# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 13:25:21 2019

big order influence.
vwap change after big order.

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


transaction_file_path = "E:/Level2/XSHG_Transaction/201809/"
snapshot_file_path = "E:/Level2/XSHG_SnapShot/201809/"

transaction_files = os.listdir(transaction_file_path)
for i in transaction_files:
    if '.csv' not in i:
        transaction_files.remove(i)
snapshot_files = os.listdir(snapshot_file_path)
for i in snapshot_files:
    if '.csv' not in i:
        snapshot_files.remove(i)

dir_name = 'fac_1m_big_vwap_pricerange'
try:
    os.mkdir(dir_name)
except:
    pass
window = dt.timedelta(minutes=1)


transaction = pd.read_csv(transaction_file_path + transaction_files[0])
transaction['datetime'] = pd.to_datetime(transaction['Date'].astype(str)+transaction['Time'].astype(str).str.zfill(9),format='%Y%m%d%H%M%S%f')
#transaction = transaction[transaction['datetime']>dt.datetime(year,month,day,9,30,0,0)]
transaction['TradeDirection'] = ((transaction['BuyOrderId']>transaction['SellOrderId']).astype(int)*2-1).astype(int)
transaction['ActOrderId'] = (transaction['BuyOrderId']*(transaction['TradeDirection']+1)*0.5 - transaction['SellOrderId']*(transaction['TradeDirection']-1)*0.5).astype(int)
transaction['NetVolume'] = (transaction['TradeVolume']*transaction['TradeDirection']).astype(int)
transaction = transaction.reset_index()

trans_grouped = transaction[['InstrumentId','datetime','TradeDirection','ActOrderId']].groupby(['InstrumentId','ActOrderId']).last()
trans_grouped[['NetVolume','TradeVolume','TradeAmount']] =transaction.groupby(['InstrumentId','ActOrderId']).sum()[['NetVolume','TradeVolume','TradeAmount']]
trans_grouped['TradePrice'] = trans_grouped['TradeAmount']/trans_grouped['TradeVolume']

quantile_group = trans_grouped.groupby(level=0)
quantile_last = quantile_group.quantile(0.8)['TradeVolume']

for i in range(2,7): #i=0; 3 ; i=1
    file = transaction_files[i]
    print('start',file)
    year = int(file[:4])
    month = int(file[4:6])
    day = int(file[6:8])
    try:
        os.mkdir(dir_name+'/'+file[:8])
    except:
        pass
    output_file_path = (dir_name+'/'+file[:8]+'/')

    snapshot = pd.read_csv(snapshot_file_path+file)
    transaction = pd.read_csv(transaction_file_path + file)


    snapshot = snapshot.drop_duplicates()
    snapshot = snapshot.drop_duplicates(['Time','InstrumentId'],keep='last')
    snapshot['datetime'] = pd.to_datetime(snapshot['Date'].astype(str)+snapshot['Time'].astype(str).str.zfill(9),format='%Y%m%d%H%M%S%f')
    snapshot = snapshot.reset_index(drop=True)  
#    snapshot_formerge = snapshot[['InstrumentId','datetime','BuyPrice1','BuyVolume1',
#                              'BuyPrice2','BuyVolume2','BuyPrice3','BuyVolume3',
#                              'BuyPrice4','BuyVolume4','BuyPrice5','BuyVolume5',
#                              'BuyPrice6','BuyVolume6','BuyPrice7','BuyVolume7',
#                              'BuyPrice8','BuyVolume8','BuyPrice9','BuyVolume9',
#                              'BuyPrice10','BuyVolume10','SellPrice1','SellVolume1',
#                              'SellPrice2','SellVolume2','SellPrice3','SellVolume3',
#                              'SellPrice4','SellVolume4','SellPrice5','SellVolume5',
#                              'SellPrice6','SellVolume6','SellPrice7','SellVolume7',
#                              'SellPrice8','SellVolume8','SellPrice9','SellVolume9',
#                              'SellPrice10','SellVolume10']]
    snapshot_formerge = snapshot[['InstrumentId','datetime','BuyPrice1','BuyVolume1',
                                  'SellPrice1','SellVolume1']]
    
    transaction['datetime'] = pd.to_datetime(transaction['Date'].astype(str)+transaction['Time'].astype(str).str.zfill(9),format='%Y%m%d%H%M%S%f')
    #transaction = transaction[transaction['datetime']>dt.datetime(year,month,day,9,30,0,0)]
    transaction['TradeDirection'] = ((transaction['BuyOrderId']>transaction['SellOrderId']).astype(int)*2-1).astype(int)
    transaction['ActOrderId'] = (transaction['BuyOrderId']*(transaction['TradeDirection']+1)*0.5 - transaction['SellOrderId']*(transaction['TradeDirection']-1)*0.5).astype(int)
    transaction['NetVolume'] = (transaction['TradeVolume']*transaction['TradeDirection']).astype(int)
    transaction = transaction.reset_index()
    
    trans_grouped = transaction[['InstrumentId','datetime','TradeDirection','ActOrderId']].groupby(['InstrumentId','ActOrderId']).last()
    trans_grouped[['NetVolume','TradeVolume','TradeAmount']] =transaction.groupby(['InstrumentId','ActOrderId']).sum()[['NetVolume','TradeVolume','TradeAmount']]
    trans_grouped['TradePrice'] = trans_grouped['TradeAmount']/trans_grouped['TradeVolume']
    trans_grouped['HighestPrice'] = transaction[['InstrumentId','TradePrice','ActOrderId']].groupby(['InstrumentId','ActOrderId']).max()['TradePrice']
    trans_grouped['lowestPrice'] = transaction[['InstrumentId','TradePrice','ActOrderId']].groupby(['InstrumentId','ActOrderId']).min()['TradePrice']
    quantile_group = trans_grouped.groupby(level=0)
    quantile_new = quantile_group.quantile(0.8)['TradeVolume']


    print('read snapshot, transaction file')
    
    #am = dt.datetime(year,month,day,11,30,0,0)-window
    #pm1 = dt.datetime(year,month,day,13,00,0,0)+2*window
    #pm = dt.datetime(year,month,day,14,57,0,0)-window
    am = dt.datetime(year,month,day,11,30,0,0)
    pm1 = dt.datetime(year,month,day,13,0,0,0)+window
    pm = dt.datetime(year,month,day,14,56,0,0)
    
    
    df_all = pd.DataFrame()
    ids = list(set(transaction['InstrumentId']))
    print('total: ',len(ids))
   

    for j in range(len(ids)): # j=0,len(ids)
        #dtime = [];j=0;dtime.append(time())
        Id = ids[j]
        if Id not in quantile_last.index:
            continue
        #transaction_id = transaction[transaction['InstrumentId']==Id]
        #snapshot_id = snapshot[snapshot['InstrumentId']==Id]
        snapshot_formerge_id = snapshot_formerge[snapshot_formerge['InstrumentId']==Id]
        #snapshot_formerge_lag2_id = snapshot_formerge_lag2[snapshot_formerge_lag2['InstrumentId']==Id]
        trans_grouped_id = trans_grouped.loc[Id]
        t = snapshot_formerge_id[snapshot_formerge_id['datetime']>=dt.datetime(year,month,day,9,30,0,0)]['datetime'].iloc[0]
        d1={}
        
        #dtime.append(time())
        while t < pm:  # t = dt.datetime(year,month,day,9,40,0,0)
            if (t>am) & (t<pm1):
                t = snapshot_formerge_id[snapshot_formerge_id['datetime']>=pm1]['datetime'].iloc[0]
                continue
            
            d1[t]={}
            
            temp_sec = trans_grouped_id[(trans_grouped_id['datetime']>t) & (trans_grouped_id['datetime']<=t+dt.timedelta(seconds=3))]
            temp_sec_big = temp_sec[temp_sec['TradeVolume']>=quantile_last.loc[Id]]
            temp_sec_big = temp_sec_big[temp_sec_big['TradeDirection']==1]

            if temp_sec.empty:
                t+=dt.timedelta(seconds=3)
                continue
            d1[t]['vwap'] = temp_sec['TradeAmount'].values.sum()/temp_sec['TradeVolume'].values.sum()
            d1[t]['trade_volume'] = temp_sec['TradeVolume'].values.sum()
            
            d1[t]['sell_volume'] = (-(temp_sec['TradeDirection'].values-1)*0.5*temp_sec['TradeVolume'].values).sum()
            d1[t]['sell_vwap'] = (-(temp_sec['TradeDirection'].values-1)*0.5*temp_sec['TradeAmount'].values).sum()/d1[t]['sell_volume']
            
            d1[t]['buy_volume'] = ((temp_sec['TradeDirection'].values+1)*0.5*temp_sec['TradeVolume'].values).sum()
            d1[t]['buy_vwap'] = ((temp_sec['TradeDirection'].values+1)*0.5*temp_sec['TradeAmount'].values).sum()/d1[t]['buy_volume']


            if not temp_sec_big.empty:
                d1[t]['big_buy_vwap'] = temp_sec_big['TradeAmount'].values.sum()/temp_sec_big['TradeVolume'].values.sum()                 
                d1[t]['big_buy_volume'] = temp_sec_big['TradeVolume'].values.sum()
                d1[t]['highest_buy'] = temp_sec_big['HighestPrice'].values.max()
                d1[t]['lowest_buy'] = temp_sec_big['lowestPrice'].values.min()
            
                
            t+=dt.timedelta(seconds=3)
        
        df = pd.DataFrame.from_dict(d1,orient='index')
        if df.empty:
            continue
        df['InstrumentId'] = Id
        df = df.reset_index()
        df.rename(columns={'index':'datetime'},inplace=True)
        df = df.merge(snapshot_formerge_id,on=['InstrumentId','datetime'],how='left')
        
        #------------
        #dtime.append(time())
        
        df['midprice'] = 0.5*(df['BuyPrice1'].values + df['SellPrice1'].values)
       
        #dtime.append(time())

        df.reset_index(drop=True,inplace=True)
        
        df.to_csv(output_file_path+str(Id)+'.csv')
        if not j%10:
            print(j,end=',')
    
    quantile_last = quantile_new.copy()
    print('\n',file,'done')
    gc.collect()

file = transaction_files[1]
output_file_path = (dir_name+'/'+file[:8]+'/')
file_path = output_file_path
#fac_path = 'fac_5m/20180903/'
l = os.listdir(file_path)
fac_data = []
for i in range(len(l)): # i=0;
    file = l[i]
    fac_data.append(pd.read_csv(file_path+file,index_col=0))
fac_data = pd.concat(fac_data)
fac_data['datetime']=pd.to_datetime(fac_data['datetime'],format='%Y-%m-%d %H:%M:%S')
#fac_data['datetime']+=dt.timedelta(seconds=3)
fac_data['spread'] = fac_data['highest_buy'] - fac_data['lowest_buy']


df = fac_data.copy()
d={}
sec = 15
window = dt.timedelta(seconds=sec)
nl = np.arange(0,sec,3,dtype=int)
interval = int(sec/3)


l = list(map(lambda x:x*3+3,[-interval-1]+list(range(0,41,interval))))
for i in l:#i=l[0]
    temp_df = df.copy()

    for n in nl: # n=nl[0]    
        diff_time = i+n
        temp_df2 = df.copy()
        temp_df2['datetime']-= dt.timedelta(seconds=int(diff_time))
        temp_df2.rename(columns={
                'sell_volume':'sell_volume_'+str(n),
                'sell_vwap':'sell_vwap_'+str(n),
                'buy_volume':'buy_volume_'+str(n),
                'buy_vwap':'buy_vwap_'+str(n),
                'trade_volume':'trade_volume_'+str(n),
                'vwap':'vwap_'+str(n)},inplace=True)
        temp_df = temp_df.merge(temp_df2[['datetime','InstrumentId',
                                          'sell_volume_'+str(n),'sell_vwap_'+str(n),
                                          'buy_volume_'+str(n),'buy_vwap_'+str(n),
                                          'trade_volume_'+str(n),'vwap_'+str(n)]],how='left',on=['datetime','InstrumentId'])

        del temp_df2
        gc.collect()
        temp_df.loc[:,['sell_volume_'+str(n),'sell_vwap_'+str(n),
                     'buy_volume_'+str(n),'buy_vwap_'+str(n),
                     'trade_volume_'+str(n),'vwap_'+str(n)]] = temp_df.loc[:,['sell_volume_'+str(n),'sell_vwap_'+str(n),
                                                             'buy_volume_'+str(n),'buy_vwap_'+str(n),
                                                             'trade_volume_'+str(n),'vwap_'+str(n)]].replace(np.nan,0)
        print(n,end=',')
        
    temp_volume = temp_df['buy_volume_'+str(nl[0])].values.copy().astype(int)
    temp_amount = temp_df['buy_volume_'+str(nl[0])].values * temp_df['buy_vwap_'+str(nl[0])].values
    for n in nl[1:]:
        temp_volume += temp_df['buy_volume_'+str(n)].values.copy().astype(int)
        temp_amount += temp_df['buy_volume_'+str(n)].values * temp_df['buy_vwap_'+str(n)].values
    df['buy_vwap_window_'+str(i)] = temp_amount/temp_volume
     
    temp_volume = temp_df['sell_volume_'+str(nl[0])].values.copy().astype(int)
    temp_amount = temp_df['sell_volume_'+str(nl[0])].values * temp_df['sell_vwap_'+str(nl[0])].values
    for n in nl[1:]:
        temp_volume += temp_df['sell_volume_'+str(n)].values.copy().astype(int)
        temp_amount += temp_df['sell_volume_'+str(n)].values * temp_df['sell_vwap_'+str(n)].values
    df['sell_vwap_window_'+str(i)] = temp_amount/temp_volume
           
    temp_volume = temp_df['trade_volume_'+str(nl[0])].values.copy().astype(int)
    temp_amount = temp_df['trade_volume_'+str(nl[0])].values * temp_df['vwap_'+str(nl[0])].values
    for n in nl[1:]:
        temp_volume += temp_df['trade_volume_'+str(n)].values.copy().astype(int)
        temp_amount += temp_df['trade_volume_'+str(n)].values * temp_df['vwap_'+str(n)].values
    df['vwap_window_'+str(i)] = temp_amount/temp_volume
    gc.collect()
    print(i,'done')


np.zeros(3)/np.zeros(3)



df.to_csv('temp_15.csv')


l = list(map(lambda x:x*3+3,[-interval-1]+list(range(0,41,interval))))
for i in range(len(l)-1):  # i=-10
    df['ret_buy_vwap_'+str(l[i])+'_'+str(l[i+1])] = np.log(df['buy_vwap_window_'+str(l[i+1])]/df['buy_vwap_window_'+str(l[i])])
    df['ret_sell_vwap_'+str(l[i])+'_'+str(l[i+1])] = np.log(df['sell_vwap_window_'+str(l[i+1])]/df['sell_vwap_window_'+str(l[i])])
    df['ret_vwap_'+str(l[i])+'_'+str(l[i+1])] = np.log(df['vwap_window_'+str(l[i+1])]/df['vwap_window_'+str(l[i])])

df.columns
cols = list(df.columns)[-27:]
['ret_buy_vwap_-30_3', 'ret_sell_vwap_-30_3', 'ret_vwap_-30_3',
       'ret_buy_vwap_3_33', 'ret_sell_vwap_3_33', 'ret_vwap_3_33',
       'ret_buy_vwap_33_63', 'ret_sell_vwap_33_63', 'ret_vwap_33_63',
       'ret_buy_vwap_63_93', 'ret_sell_vwap_63_93', 'ret_vwap_63_93',
       'ret_buy_vwap_93_123', 'ret_sell_vwap_93_123', 'ret_vwap_93_123']

df.isna().sum()


fac = df.copy()
#fac = fac.dropna(subset = cols+['big_buy_volume'])
fac = fac.dropna(subset = ['big_buy_volume'])
fac.isna().sum()

(fac>0).sum()

d={}
for Id in fac['InstrumentId'].unique():
    fac_id = fac[fac['InstrumentId']==Id]
    if len(fac_id)<100:
        continue
    d[Id]={}
    for col in cols:
        d[Id][col] = fac_id[col].mean()
fac_df = pd.DataFrame.from_dict(d,orient='index')

d = {}
for col in np.array(cols).reshape(int(size(cols)/3),3).T.flatten():
    d[col] = fac_df[col].mean()
pd.DataFrame.from_dict(d,orient='index')

############################
fac = df.copy()
fac = fac[fac['spread']>0.02]

fac.isna().sum()

(fac>0).sum()

d={}
for Id in fac['InstrumentId'].unique():
    fac_id = fac[fac['InstrumentId']==Id]
    if len(fac_id)<100:
        continue
    d[Id]={}
    for col in cols:
        d[Id][col] = fac_id[col].mean()
fac_df = pd.DataFrame.from_dict(d,orient='index')

d = {}
for col in np.array(cols).reshape(int(size(cols)/3),3).T.flatten():
    d[col] = fac_df[col].mean()
pd.DataFrame.from_dict(d,orient='index')
#####################

volume up and vwap up











############
fac2 = df.copy()
fac2 = fac2.dropna(subset=cols)
d={}
for Id in fac2['InstrumentId'].unique():
    fac2_id = fac2[fac2['InstrumentId']==Id]
    if len(fac2_id)<100:
        continue
    d[Id]={}
    for col in cols:
        d[Id][col] = fac2_id[col].mean()
fac2_df = pd.DataFrame.from_dict(d,orient='index')

d = {}
for col in np.array(cols).reshape(int(size(cols)/3),3).T.flatten():
    d[col] = fac2_df[col].mean()
pd.DataFrame.from_dict(d,orient='index') 
# --------------

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

# ----------------
fac = df.copy()
fac = fac.dropna(subset = cols+['big_buy_volume'])
fac2 = fac2.dropna(subset=cols)
d={}
for Id in fac['InstrumentId'].unique():
    fac_id = fac[fac['InstrumentId']==Id]
    fac2_id = fac2[fac2['InstrumentId']==Id]
    if len(fac_id)<100:
        continue
    d[Id]={}
    for col in cols:
        d[Id][col] = fac_id[col].mean()
fac_df = pd.DataFrame.from_dict(d,orient='index')

for col in cols:
    all_plot(np.log(fac['big_buy_volume']),fac,independent=col,col=col)

# ----------------------------------
daily = pd.read_csv('daily/201809StockDailyData.csv')
daily['InstrumentId'] = daily['WindCode'].apply(lambda x:int(x[:6]))
daily_SH = daily[daily['WindCode'].apply(lambda x:True if 'SH' in x else False)]
today = daily_SH[daily_SH['TradingDay']==20180903].copy()
today['free_cap'] = today['VWAvgPrice']*today['NonRestrictedShares']/today['SplitFactor']

temp = fac.merge(today[['InstrumentId','free_cap','VWAvgPrice']],on = ['InstrumentId'],how = 'left')

conditions_price = [temp['VWAvgPrice']>20,
                    (temp['VWAvgPrice']<=20) & (temp['VWAvgPrice']>10),
                    (temp['VWAvgPrice']<=10) & (temp['VWAvgPrice']>5),
                    (temp['VWAvgPrice']<=5)]              
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

# -------------------------------------
col = 'big_buy_volume'
for i in range(len(conditions_cap)):
    con = conditions_cap[i]
    range_noinf = temp[con]
    #plot(range_noinf['resist_sell_lag'],range_noinf,1,'sell, price '+cons_cap[i])
    all_plot(range_noinf[col],range_noinf,mid=8,log=True,addv=-3.5,addh=0.012,col = 'ret_vwap_3_33'+' '+cons_cap[i],independent = 'ret_vwap_3_33')
    print('''
          num stocks:   %f''' % (
          len(range_noinf['InstrumentId'].unique())))

col = 'big_buy_volume'
for i in range(len(conditions_price)):
    con = conditions_price[i]
    range_noinf = temp[con]
    #plot(range_noinf['resist_sell_lag'],range_noinf,1,'sell, price '+cons_price[i])
    all_plot(range_noinf[col],range_noinf,log=True,addv=-3.5,addh=0.012,col = 'ret_vwap_3_33'+' '+cons_price[i],independent = 'ret_vwap_3_33')
    print('''
          num stocks:   %f''' % (
          len(range_noinf['InstrumentId'].unique())))
# -------------------------------------
fac = df.copy()
fac = fac.dropna(subset = cols+['big_buy_volume'])
fac2 = fac2.dropna(subset=cols)
d={}
for Id in fac['InstrumentId'].unique():
    fac_id = fac[fac['InstrumentId']==Id]
    fac2_id = fac2[fac2['InstrumentId']==Id]
    if len(fac_id)<100:
        continue
    d[Id]={}
    for col in cols:
        temp_col = fac_id[col] - fac2_id[col].mean()
#        d[Id]['> pct'] = (temp_col>0).sum()/len(temp_col)
#        d[Id]['diff prob >0 pct'] = (fac_id[col]>0).sum()/len(fac_id) - (fac2_id[col]>0).sum()/len(fac2_id)
        d[Id][col] = temp_col.mean()
#        d[Id]['big mean'] = fac_id[col].mean()
#        d[Id]['sample mean'] = fac2_id[col].mean()
#        d[Id]['big order num'] = len(fac_id)
#        d[Id]['total sample num'] = len(fac2_id)
fac_df = pd.DataFrame.from_dict(d,orient='index')

fac_df.mean()
(fac_df>0).sum()/len(fac_df)
# -------------------------------------
fac = df.copy()
fac = fac.dropna(subset = cols+['big_buy_volume'])
fac2 = fac2.dropna(subset=cols)
d={}
for Id in fac['InstrumentId'].unique():
    fac_id = fac[fac['InstrumentId']==Id]
    fac2_id = fac2[fac2['InstrumentId']==Id]
    if len(fac_id)<100:
        continue
    d[Id]={}
    col = 'ret_vwap_3_33'
    temp_col = fac_id[col] - fac2_id[col].mean()
    d[Id]['> pct'] = (temp_col>0).sum()/len(temp_col)
    d[Id]['diff prob >0 pct'] = (fac_id[col]>0).sum()/len(fac_id) - (fac2_id[col]>0).sum()/len(fac2_id)
    d[Id]['diff mean'] = temp_col.mean()
    d[Id]['big mean'] = fac_id[col].mean()
    d[Id]['sample mean'] = fac2_id[col].mean()
    d[Id]['big order num'] = len(fac_id)
    d[Id]['total sample num'] = len(fac2_id)
fac_df = pd.DataFrame.from_dict(d,orient='index')

print((fac_df['diff mean']>0).sum()/len(fac_df),
      (fac_df['diff prob >0 pct']>0).sum()/len(fac_df),
      fac_df['> pct'].mean())

fac_df['diff mean'].mean()
fac_df['diff prob >0 pct'].mean()

(fac_df['> pct']>0.60).sum()/len(fac_df)



# ----------------------------------
fac_df = fac_df.reset_index()
fac_df = fac_df.rename(columns={'index':'InstrumentId'})
fac_df = fac_df.merge(today[['InstrumentId','free_cap','VWAvgPrice']],on = ['InstrumentId'],how = 'left')


all_plot(fac_df['free_cap'],fac_df,log=True,independent='diff mean')
all_plot(fac_df['VWAvgPrice'],fac_df,log=True,independent='diff mean')

all_plot(fac_df['free_cap'],fac_df,log=True,independent='> pct')
all_plot(fac_df['VWAvgPrice'],fac_df,log=True,independent='> pct')


for col in cols:
    all_plot(np.log(fac['big_buy_volume']),fac,independent=col,col=col)


# -----------------------------------------------------









    

d={}
for Id in df['InstrumentId'].unique(): # Id=600525
    d[Id]={}
    temp_df = df[df['InstrumentId']==Id].copy()
    temp_df = temp_df.dropna(subset=['big_buy_volume'])
    if len(temp_df)<100:
        continue

    for i in range(0,31,5):#i=1
 
        temp_df = df[df['InstrumentId']==Id].copy()
        temp_df2 = temp_df.copy()
        temp_df2['datetime'] -= dt.timedelta(seconds=3*(i-1))
        temp_df2 = temp_df2.rename(columns={'midprice':'midprice_'+str(3*i)})
        temp_df = temp_df.merge(temp_df2[['datetime','midprice_'+str(3*i)]],how='left',on='datetime')
        
        for n in nl:
            temp_df2=temp_df.copy()
            temp_df2 = temp_df2.rename(columns={'trade_volume':'trade_volume_'+str(n),
                                                'vwap':'vwap_'+str(n)})
            temp_df2['datetime']-=dt.timedelta(seconds=int(n)+3*i)
            temp_df = temp_df.merge(temp_df2[['datetime','trade_volume_'+str(n),'vwap_'+str(n)]],
                                              how='left',on='datetime')       

        temp_df['vwap_'+str(3*i)] = temp_df.apply(cal_ret,axis=1)
        temp_df['ret_vwapmid_fur']=np.log(temp_df['vwap_'+str(3*i)]/temp_df['midprice_'+str(3*i)])
        
        if i==0:
            temp_df = temp_df.dropna(subset=['ret_vwapmid_fur'])
            d[Id]['ret_all'] = temp_df['ret_vwapmid_fur'].mean()
            d[Id]['ret_prob_all'] = (temp_df['ret_vwapmid_fur']>0).sum()/len(temp_df)
        
        temp_df = temp_df.dropna(subset=['big_buy_volume','ret_vwapmid_fur'])
        d[Id]['ret_'+str(3*i)] = temp_df['ret_vwapmid_fur'].mean()
        d[Id]['ret_prob_'+str(3*i)] = (temp_df['ret_vwapmid_fur']>0).sum()/len(temp_df)

    print(len(d),end=',')


res = pd.DataFrame.from_dict(d,orient='index')

res = res.dropna()
for i in range(0,101,15): # 
    print(i, res['ret_'+str(i)].mean(), res['ret_prob_'+str(i)].mean())


res2 = pd.DataFrame()
for i in range(0,100,15): # 
    res2['diff_'+str(i)] = res['ret_'+str(i)] - res['ret_all']
    res2['diff_prob_'+str(i)] = res['ret_prob_'+str(i)] - res['ret_prob_all']

for i in range(0,100,15): # 
    print(i,(res2['diff_'+str(i)]>0).sum()/len(res2))

for i in range(0,100,15): # 
    print(i,(res2['diff_prob_'+str(i)]>0).sum()/len(res2))


plt.hist(res2['diff_45'],bins=100)

      
np.diff(dtime)
t4-t3
%timeit temp_df.dropna(subset=['big_volume','ret_vwapmid_fur'])

%timeit dtime.append(time())









df = fac_data.copy()
for i in range(1,101): # i=1
    det = dt.timedelta(seconds=i*3)
    temp_df = fac_data[['datetime','InstrumentId','vwap']].copy()
    temp_df = temp_df.rename(columns={'vwap':'vwap_'+str(3*i)})
    temp_df['datetime']-=det
    df = df.merge(temp_df,how='left',on=['datetime','InstrumentId'])
    print(gc.collect())

#temp = df[df['InstrumentId']==600000]
#
#ind = np.arange(1,31).reshape(6,5)*3
#for ar in ind: # ar=ind[0]
#    cols = list(map(lambda x:'vwap_'+str(x),ar))
#    df[cols].mean(axis=1)
#temp = temp.dropna()


temp_df = df[df['InstrumentId']==600000]

cols = ['datetime','vwap','midprice']
for i in range(10,101,5): # 
    cols = cols[:3]
    cols.append('vwap_'+(str(3*i)))
    cols.append('vwap_'+(str(3*(i-5))))
    temp = temp_df[cols]
    temp = temp.dropna()
    temp['ret_vwapmid_fur'] = np.log(temp[cols[-2]]/temp[cols[-1]])

        
    print('''
          mean:          %f
          0.5 quantile:  %f
          std:           %f
          >0 pct:        %f
          ''' % (
          temp['ret_vwapmid_fur'].mean(),temp['ret_vwapmid_fur'].quantile(0.5),
          temp['ret_vwapmid_fur'].std(),(temp['ret_vwapmid_fur']>0).sum()/len(temp)))

d={}
for Id in df['InstrumentId'].unique():
    temp_df = df[df['InstrumentId']==Id]
    d[Id]={}

    cols = ['datetime','big_volume','midprice']
    for i in range(10,101,5): # 
        cols = cols[:3]
        cols.append('vwap_'+(str(3*i)))
        cols.append('vwap_'+(str(3*(i-5))))
        temp = temp_df[cols].copy()
        temp['ret_vwapmid_fur'] = np.log(temp[cols[-2]]/temp[cols[-1]])
        
        temp = temp.dropna(subset=['ret_vwapmid_fur'])
        d[Id][str((i-5)*3)+' all'] = temp['ret_vwapmid_fur'].mean()
        d[Id][str((i-5)*3) + ' >0 pct all'] = (temp['ret_vwapmid_fur']>0).sum()/len(temp)
        
        temp = temp.dropna(subset = ['big_volume'])
        d[Id][str((i-5)*3)+' big'] = temp['ret_vwapmid_fur'].mean()
        d[Id][str((i-5)*3) + '>0 pct big'] = (temp['ret_vwapmid_fur']>0).sum()/len(temp)
    
    if not len(d)%10:
        print(len(d),end=',')

ana_df = pd.DataFrame.from_dict(d,orient='index')
for i in range(5,96,5): # 
    ana_df[str(3*i)+' diff'] = ana_df[str(i*3)+' big'] - ana_df[str(i*3)+' all']
    ana_df[str(3*i)+' diff prob'] = ana_df[str(i*3)+'>0 pct big'] - ana_df[str(i*3)+' >0 pct all']


ana_df = ana_df.dropna()
for i in range(5,96,5): # 
    print(i*3, ana_df[str(i*3)+' diff'].mean(), ana_df[str(i*3)+' diff prob'].mean())


    
    
    
    
    
    
    
