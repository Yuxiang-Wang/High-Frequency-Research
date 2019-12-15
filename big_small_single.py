# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 09:22:29 2019

大小单因子，算了几个个股

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

# gc.collect()

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

#for i in range(3): #i=0
        
#dir_name = 'fac_30s_bigsmall'
dir_name = 'temp600000'
window = dt.timedelta(minutes=1)
for i in range(8): #i=0
    file = transaction_files[i]
    print('start',file)
    year = int(file[:4])
    month = int(file[4:6])
    day = int(file[6:8])
    try:
        os.mkdir(dir_name)
    except:
        pass
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
    
    
    transaction['datetime'] = pd.to_datetime(transaction['Date'].astype(str)+transaction['Time'].astype(str).str.zfill(9),format='%Y%m%d%H%M%S%f')
    #transaction = transaction[transaction['datetime']>dt.datetime(year,month,day,9,30,0,0)]
    transaction['TradeDirection'] = ((transaction['BuyOrderId']>transaction['SellOrderId']).astype(int)*2-1).astype(int)
    transaction['ActOrderId'] = (transaction['BuyOrderId']*(transaction['TradeDirection']+1)*0.5 - transaction['SellOrderId']*(transaction['TradeDirection']-1)*0.5).astype(int)
    transaction['NetVolume'] = (transaction['TradeVolume']*transaction['TradeDirection']).astype(int)
    transaction = transaction.reset_index()
    
    trans_grouped = transaction[['InstrumentId','datetime','TradeDirection','ActOrderId']].groupby(['InstrumentId','ActOrderId']).last()
    trans_grouped['NetVolume'] =transaction.groupby(['InstrumentId','ActOrderId']).sum()['NetVolume']
    trans_grouped['TradeVolume'] = transaction.groupby(['InstrumentId','ActOrderId']).sum()['TradeVolume']
    
    print('file ready')
    
    #am = dt.datetime(year,month,day,11,30,0,0)-window
    #pm1 = dt.datetime(year,month,day,13,00,0,0)+2*window
    #pm = dt.datetime(year,month,day,14,57,0,0)-window
    am = dt.datetime(year,month,day,11,30,0,0)
    pm1 = dt.datetime(year,month,day,13,00,0,0)+window
    pm = dt.datetime(year,month,day,14,57,0,0)
    
    
    df_all = pd.DataFrame()
    ids = list(set(transaction['InstrumentId']))
    print('total: ',len(ids))
   
#    for j in range(len(ids)): # j=0,len(ids)
#    for j in range(len(ids)): # j=0,len(ids)
#        Id = ids[j]
    for Id in [603918,603958]:
        transaction_id = transaction[transaction['InstrumentId']==Id]
        snapshot_id = snapshot[snapshot['InstrumentId']==Id]
        trans_grouped_id = trans_grouped.loc[Id]
        t = snapshot_id[snapshot_id['datetime']>=dt.datetime(year,month,day,9,30,0,0)]['datetime'].iloc[0]
        d1={}
        
        while t < pm:  # t = dt.datetime(year,month,day,9,40,0,0)
            if (t>am) & (t<pm1):
                t = snapshot_id[snapshot_id['datetime']>=pm1]['datetime'].iloc[0]
                continue
            
            temp_lag = transaction_id[(transaction_id['datetime']>t-window) & (transaction_id['datetime']<=t)]
            temp_grouped_lag = trans_grouped_id[(trans_grouped_id['datetime']>t-window) & (trans_grouped_id['datetime']<=t)]
            
            #if temp.empty or temp_lag.empty or temp_lag2.empty:
            if temp_lag.empty:
                #t+=window
                t+=dt.timedelta(seconds=3)
                continue
            
            d1[t]={}
            d1[t]['trade_volume_lag'] = temp_lag['TradeVolume'].sum()
            d1[t]['vwap_lag'] = temp_lag['TradeAmount'].sum()/d1[t]['trade_volume_lag']     
            d1[t]['buy_volume_lag'] = temp_lag[temp_lag['TradeDirection']==1]['TradeVolume'].sum()
            d1[t]['sell_volume_lag'] = temp_lag[temp_lag['TradeDirection']==-1]['TradeVolume'].sum()

            temp_big = temp_grouped_lag[temp_grouped_lag['TradeVolume']>trans_grouped_id['TradeVolume'].quantile(0.8)]
            temp_small = temp_grouped_lag[temp_grouped_lag['TradeVolume']<trans_grouped_id['TradeVolume'].quantile(0.5)]
            d1[t]['big_volume_lag'] = temp_big['TradeVolume'].sum()
            d1[t]['small_volume_lag'] = temp_small['TradeVolume'].sum()
            
            temp_buybig = temp_big[temp_big['TradeDirection']==1]
            temp_sellbig = temp_big[temp_big['TradeDirection']==-1]
            d1[t]['big_buy_volume_lag'] = sum(temp_buybig['TradeVolume'])        
            d1[t]['big_sell_volume_lag'] = sum(temp_sellbig['TradeVolume'])
            temp_buysmall = temp_small[temp_small['TradeDirection']==1]
            temp_sellsmall = temp_small[temp_small['TradeDirection']==-1]
            d1[t]['small_buy_volume_lag'] = sum(temp_buysmall['TradeVolume'])        
            d1[t]['small_sell_volume_lag'] = sum(temp_sellsmall['TradeVolume'])


            t+=dt.timedelta(seconds=3)
    
        df = pd.DataFrame.from_dict(d1,orient='index')
        df['InstrumentId'] = Id
        #df_all = pd.concat([df_all,df])
        df = df.reset_index()
        df.rename(columns={'index':'datetime'},inplace=True)
        
        df.to_csv(output_file_path+str(Id)+'.csv')
        if not j%10:
            print(j,end=',')
    print('\n',file,'done')



l = os.listdir(dir_name)


for i in range(len(l)): # i=0
    file_names = os.listdir(dir_name+'/'+l[i])
#    try:
#        os.mkdir("fac_5m/"+l[i])
#    except:
#        pass
    output_file_path = (dir_name+'/'+l[i]+'/')

    print(l[i],'start')
    snapshot = pd.read_csv(snapshot_file_path+l[i]+'.csv')
    snapshot = snapshot.drop_duplicates(['Time','InstrumentId'],keep='last')
    snapshot['datetime'] = pd.to_datetime(snapshot['Date'].astype(str)+snapshot['Time'].astype(str).str.zfill(9),format='%Y%m%d%H%M%S%f')
    snapshot = snapshot.reset_index(drop=True)
    snapshot_formerge = snapshot[['InstrumentId','datetime','BuyPrice1','BuyVolume1',
                              'BuyPrice2','BuyVolume2','BuyPrice3','BuyVolume3',
                              'BuyPrice4','BuyVolume4','BuyPrice5','BuyVolume5',
                              'BuyPrice6','BuyVolume6','BuyPrice7','BuyVolume7',
                              'BuyPrice8','BuyVolume8','BuyPrice9','BuyVolume9',
                              'BuyPrice10','BuyVolume10','SellPrice1','SellVolume1',
                              'SellPrice2','SellVolume2','SellPrice3','SellVolume3',
                              'SellPrice4','SellVolume4','SellPrice5','SellVolume5',
                              'SellPrice6','SellVolume6','SellPrice7','SellVolume7',
                              'SellPrice8','SellVolume8','SellPrice9','SellVolume9',
                              'SellPrice10','SellVolume10']]
    snapshot_formerge_lag2 = snapshot_formerge.copy()
    snapshot_formerge_lag2['datetime'] = snapshot_formerge_lag2['datetime']+window
    
    print('read snapshot')
    
    new_name = []
    for col in snapshot_formerge_lag2.columns:
        if col in ['InstrumentId','datetime']:
            new_name.append(col)
        else:
            new_name.append(col+'_lag')
    snapshot_formerge_lag2.columns = new_name

    for j in range(len(file_names)): # file = file_names[0];
        file = file_names[j]
        if file == '600000.csv':
            continue
        df = pd.read_csv(dir_name+'/'+l[i]+'/'+file,index_col=0)
        df['datetime']=pd.to_datetime(df['datetime'],format='%Y-%m-%d %H:%M:%S')
        
        df_lag = df.copy()
        df_lag['datetime'] = df_lag['datetime']+window
        new_name = []
        for col in df_lag.columns:
            if col in ['InstrumentId','datetime']:
                new_name.append(col)
            else:
                new_name.append(col+'2')
        df_lag.columns = new_name
        df = df.merge(df_lag,on=['InstrumentId','datetime'],how='left')
        
        df_fur = df[['InstrumentId','datetime','vwap_lag','trade_volume_lag']].copy()
        df_fur['datetime'] = df_fur['datetime']-window
        df_fur.columns = ['InstrumentId','datetime','vwap_fur','trade_volume_fur']
        df = df.merge(df_fur,on = ['InstrumentId','datetime'],how='left') 

        df = df.merge(snapshot_formerge,on=['InstrumentId','datetime'],how='left')
        df = df.merge(snapshot_formerge_lag2,on=['InstrumentId','datetime'],how='left')

        df['midprice'] = 0.5*(df['BuyPrice1'] + df['SellPrice1'])
        df['ret_vwapmid_fur'] = np.log(df['vwap_fur']/df['midprice'])
        #df['trade_volume_imbalance_lag'] = df['total_buy_vol_lag']/df['trade_volume_lag']
        #df['volume_imbalance'] = df['BuyVolume1']/(df['SellVolume1']+df['BuyVolume1'])
        
#        df['incre_buy_lag'] = df.apply(incre_buy,axis=1)
#        df['incre_sell_lag'] = df.apply(incre_sell,axis=1)
#        
#        df['new_buy_lag'] = df['incre_buy_lag']/df['total_sell_vol_lag']
#        df['new_sell_lag'] = df['incre_sell_lag']/df['total_buy_vol_lag']
        
        df.dropna(inplace=True)
        df.reset_index(drop=True,inplace=True)
        
        df.to_csv(output_file_path+file)
        
        print(j,end=',')


df = []
for i in range(len(l)): # i=0
    file_names = os.listdir(dir_name+'/'+l[i]) 
    df.append(pd.read_csv(dir_name+'/'+l[i]+'/'+file_names[0]))

df = pd.concat(df)
df['net_volume'] = df['buy_volume_lag'] - df['sell_volume_lag']

# ---------------------------------------------
def plot(temp,factor,log=False,col=None,mid=0):
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
    plt.axvline(x=0,color='r',linewidth=0.8)
    plt.axhline(y=0,color='r',linewidth=0.8)
    plt.grid(True)
    if label:
        plt.legend(labels=[label])
    if show:
        plt.show()
    #print('zero point',x[list(map(abs,y)).index(min(list(map(abs,y))))])
# ---------------------------------------------
    
plot(df['net_volume'],df)

#daily = pd.read_csv('daily/201809StockDailyData.csv')
#daily['InstrumentId'] = daily['WindCode'].apply(lambda x:int(x[:6]))
#daily_SH = daily[daily['WindCode'].apply(lambda x:True if 'SH' in x else False)]
#daily[daily['InstrumentId']==600000]

df['big_total_ratio'] = df['big_volume_lag']/df['trade_volume_lag']
df['small_total_ratio'] = df['small_volume_lag']/df['trade_volume_lag']
df['big_small_ratio'] = df['big_volume_lag']/df['small_volume_lag']

df['buy_total_ratio'] = df['buy_volume_lag']/df['trade_volume_lag']
df['sell_total_ratio'] = df['sell_volume_lag']/df['trade_volume_lag']



df['big_buy_total_ratio'] = df['big_buy_volume_lag']/df['trade_volume_lag']
df['small_buy_total_ratio'] = df['small_buy_volume_lag']/df['trade_volume_lag']
df['big_buy_buytotal_ratio'] = df['big_buy_volume_lag']/df['buy_volume_lag']
df['small_buy_buytotal_ratio'] = df['small_buy_volume_lag']/df['buy_volume_lag']
df['big_buy_small_buy_ratio'] = df['big_buy_volume_lag']/df['small_buy_volume_lag']

df['big_sell_total_ratio'] = df['big_sell_volume_lag']/df['trade_volume_lag']
df['small_sell_total_ratio'] = df['small_sell_volume_lag']/df['trade_volume_lag']
df['big_sell_selltotal_ratio'] = df['big_sell_volume_lag']/df['sell_volume_lag']
df['small_sell_selltotal_ratio'] = df['small_sell_volume_lag']/df['sell_volume_lag']
df['big_sell_small_sell_ratio'] = df['big_sell_volume_lag']/df['small_sell_volume_lag']

df['big_volume_spread'] = df['big_buy_volume_lag'] - df['big_sell_volume_lag']
df['small_volume_spread'] = df['small_buy_volume_lag'] - df['small_sell_volume_lag']

df['big_spread_total_spread'] = df['big_volume_spread']/df['net_volume']
df['small_spread_total_spread'] = df['small_volume_spread']/df['net_volume']

list(df.columns)

arr = np.linspace(0,1,100)
zero_one_itv = np.vstack([arr[:-1],arr[1:]]).T

def all_plot(fac_name,df,itv,mid=0,log=False):
    plot(df[fac_name],df,mid=mid,log=log)
    plot_mean(df[fac_name],df,itv,log=log)

all_plot('big_total_ratio',df,zero_one_itv,0.8)
all_plot('small_total_ratio',df,zero_one_itv,0.1)

temp_df = df[(~df['big_small_ratio'].isna()) & (df['big_small_ratio']<np.inf) & (df['big_small_ratio']>0)]
arr = np.linspace(-1,8,100)
negone_eight_itv = np.vstack([arr[:-1],arr[1:]]).T
all_plot('big_small_ratio',temp_df,negone_eight_itv,log=True,mid=2)

all_plot('big_buy_total_ratio',df,zero_one_itv,mid=0.7)     #
all_plot('buy_total_ratio',df,zero_one_itv,mid=0.5)         #
all_plot('small_buy_total_ratio',df,zero_one_itv,mid=0.2)   #
all_plot('big_buy_buytotal_ratio',df,zero_one_itv,mid=0.2)
all_plot('small_buy_buytotal_ratio',df,zero_one_itv,mid=0.2)
temp_df = df[(~df['big_buy_small_buy_ratio'].isna()) & (df['big_buy_small_buy_ratio']<np.inf) & (df['big_buy_small_buy_ratio']>0)]
all_plot('big_buy_small_buy_ratio',temp_df,negone_eight_itv,mid=0.2,log=True)


all_plot('big_sell_total_ratio',df,zero_one_itv,mid=0.7)     #
all_plot('sell_total_ratio',df,zero_one_itv,mid=0.1)         #
all_plot('small_sell_total_ratio',df,zero_one_itv,mid=0.2)   #
all_plot('big_sell_selltotal_ratio',df,zero_one_itv,mid=0.8)
all_plot('small_sell_selltotal_ratio',df,zero_one_itv,mid=0.2)
temp_df = df[(~df['big_sell_small_sell_ratio'].isna()) & (df['big_sell_small_sell_ratio']<np.inf) & (df['big_sell_small_sell_ratio']>0)]
all_plot('big_sell_small_sell_ratio',temp_df,negone_eight_itv,mid=0.2,log=True)

(df['ret_vwapmid_fur']>0).sum()/len(df)/(1-(df['ret_vwapmid_fur']>0).sum()/len(df))
0.158/0.190

all_plot('big_volume_spread',df,zero_one_itv,mid=50000)
all_plot('small_volume_spread',df,zero_one_itv,mid=4500)

temp_df = df[(~df['big_spread_total_spread'].isna()) & (df['big_spread_total_spread']<np.inf) & (df['big_spread_total_spread']>0)]
all_plot('big_spread_total_spread',temp_df,zero_one_itv,log=True)
#-------
temp_df = df[(df['net_volume']>0) & (df['big_spread_total_spread']>0)]
arr = np.linspace(-4,4,100)
negfour_four_itv = np.vstack([arr[:-1],arr[1:]]).T
all_plot('big_spread_total_spread',temp_df,negfour_four_itv,log=True)

temp_df2 = temp_df[(np.log(temp_df['big_spread_total_spread'])<1) & (np.log(temp_df['big_spread_total_spread'])>-1)]
all_plot('big_spread_total_spread',temp_df2,negfour_four_itv,log=True,mid=0.1)
#---------
temp_df = df[(df['net_volume']<0) & (df['big_spread_total_spread']<0)].copy()
temp_df['big_spread_total_spread'] = abs(temp_df['big_spread_total_spread'])
arr = np.linspace(-4,4,100)
negfour_four_itv = np.vstack([arr[:-1],arr[1:]]).T
all_plot('big_spread_total_spread',temp_df,negfour_four_itv,log=True)
#-------
temp_df = df[(df['net_volume']<0) & (df['big_spread_total_spread']>0)].copy()
temp_df['big_spread_total_spread'] = abs(temp_df['big_spread_total_spread'])
arr = np.linspace(-4,4,100)
negfour_four_itv = np.vstack([arr[:-1],arr[1:]]).T
all_plot('big_spread_total_spread',temp_df,negfour_four_itv,log=True)
#-------
temp_df = df[(df['net_volume']>0) & (df['big_spread_total_spread']<0)].copy()
temp_df['big_spread_total_spread'] = abs(temp_df['big_spread_total_spread'])
arr = np.linspace(-4,4,100)
negfour_four_itv = np.vstack([arr[:-1],arr[1:]]).T
all_plot('big_spread_total_spread',temp_df,negfour_four_itv,log=True)

# -----------------------
temp_df = df[(~df['small_spread_total_spread'].isna()) & (df['small_spread_total_spread']<np.inf) & (df['small_spread_total_spread']>0)]
all_plot('small_spread_total_spread',temp_df,zero_one_itv,log=True)
#-------
temp_df = df[(df['net_volume']>0) & (df['small_spread_total_spread']>0)]
arr = np.linspace(-4,4,100)
negfour_four_itv = np.vstack([arr[:-1],arr[1:]]).T
all_plot('small_spread_total_spread',temp_df,negfour_four_itv,log=True)

temp_df2 = temp_df[(np.log(temp_df['small_spread_total_spread'])<1) & (np.log(temp_df['small_spread_total_spread'])>-1)]
all_plot('small_spread_total_spread',temp_df2,negfour_four_itv,log=True,mid=0.1)
#---------
temp_df = df[(df['net_volume']<0) & (df['small_spread_total_spread']<0)].copy()
temp_df['small_spread_total_spread'] = abs(temp_df['small_spread_total_spread'])
arr = np.linspace(-4,4,100)
negfour_four_itv = np.vstack([arr[:-1],arr[1:]]).T
all_plot('small_spread_total_spread',temp_df,negfour_four_itv,log=True)
#-------
temp_df = df[(df['net_volume']<0) & (df['small_spread_total_spread']>0)].copy()
temp_df['small_spread_total_spread'] = abs(temp_df['small_spread_total_spread'])
arr = np.linspace(-4,4,100)
negfour_four_itv = np.vstack([arr[:-1],arr[1:]]).T
all_plot('small_spread_total_spread',temp_df,negfour_four_itv,log=True)
#-------
temp_df = df[(df['net_volume']>0) & (df['small_spread_total_spread']<0)].copy()
temp_df['small_spread_total_spread'] = abs(temp_df['small_spread_total_spread'])
arr = np.linspace(-4,4,100)
negfour_four_itv = np.vstack([arr[:-1],arr[1:]]).T
all_plot('small_spread_total_spread',temp_df,negfour_four_itv,log=True)

# ------------------------------


df['small_volume_spread'].hist(bins=100)

temp_df = df[(df['small_volume_spread'].quantile(0.7)>df['small_volume_spread']) &
             (df['small_volume_spread'].quantile(0.3)<df['small_volume_spread'])]
temp_df = temp_df[temp_df['big_volume_spread']>0]
arr = np.linspace(5,11,100)
itv = np.vstack([arr[:-1],arr[1:]]).T
all_plot('big_volume_spread',temp_df,itv,log=True)
temp_df['big_volume_spread'].hist(bins=100)

temp_df = df[df['net_volume']>0]
all_plot('net_volume',temp_df,itv,log=True)

df['big_volume_spread'].hist(bins=100)



daily = pd.read_csv('daily/201809StockDailyData.csv')
daily['InstrumentId'] = daily['WindCode'].apply(lambda x:int(x[:6]))

daily[daily['MarketCapAFloat']<3e9]

daily[daily['InstrumentId']==600000]



603918
market cap < 3e9, price > 20. 
'big_buy_small_buy_ratio'
big_spread_total_spread_ratio

603958
small_volume_spread

