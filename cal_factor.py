# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 15:40:53 2019

挂单净增量因子计算

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
for i in range(8,11): #i=0
    file = transaction_files[i]
    print('start',file)
    year = int(file[:4])
    month = int(file[4:6])
    day = int(file[6:8])
    try:
        os.mkdir("fac_5m_1")
    except:
        pass
    try:
        os.mkdir("fac_5m_1/"+file[:8])
    except:
        pass
    output_file_path = ("fac_5m_1/"+file[:8]+'/')

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
    trans_grouped = transaction[['InstrumentId','datetime','ActOrderId']].groupby(['InstrumentId','ActOrderId']).last()
    trans_grouped['NetVolume'] =transaction.groupby(['InstrumentId','ActOrderId']).sum()['NetVolume']
    
    print('file ready')
    window = dt.timedelta(minutes=5)
    #am = dt.datetime(year,month,day,11,30,0,0)-window
    #pm1 = dt.datetime(year,month,day,13,00,0,0)+2*window
    #pm = dt.datetime(year,month,day,14,57,0,0)-window
    am = dt.datetime(year,month,day,11,30,0,0)
    pm1 = dt.datetime(year,month,day,13,00,0,0)+window
    pm = dt.datetime(year,month,day,14,57,0,0)
    
    
    df_all = pd.DataFrame()
    ids = list(set(transaction['InstrumentId']))
    print('total: ',len(ids))
    
    for j in range(len(ids)): # j=0,len(ids)
        Id = ids[j]
        transaction_id = transaction[transaction['InstrumentId']==Id]
        snapshot_id = snapshot[snapshot['InstrumentId']==Id]
        trans_grouped_id = trans_grouped.loc[Id]
        t = snapshot_id[snapshot_id['datetime']>=dt.datetime(year,month,day,9,30,0,0)]['datetime'].iloc[0]
        d1={}
        
        while t < pm:  # t = dt.datetime(year,month,day,9,40,0,0)
            if (t>am) & (t<pm1):
                t = snapshot_id[snapshot_id['datetime']>=pm1]['datetime'].iloc[0]
                continue
            
            #temp = transaction_id[(transaction_id['datetime']>t) & (transaction_id['datetime']<=t+window)]
            temp_lag = transaction_id[(transaction_id['datetime']>t-window) & (transaction_id['datetime']<=t)]
            temp_grouped_lag = trans_grouped_id[(trans_grouped_id['datetime']>t-window) & (trans_grouped_id['datetime']<=t)]
            #temp_lag2 = transaction_id[(transaction_id['datetime']>t-2*window) & (transaction_id['datetime']<=t-window)]
            
            #if temp.empty or temp_lag.empty or temp_lag2.empty:
            if temp_lag.empty:
                #t+=window
                t+=dt.timedelta(seconds=3)
                continue
            
            d1[t]={}
            #d1[t]['vwap_fur'] = temp['TradeAmount'].sum()/temp['TradeVolume'].sum()
            #------
            trade_volume = temp_lag['TradeVolume']
            d1[t]['net_volume_lag'] = temp_lag['NetVolume'].sum()
            d1[t]['trade_volume_lag'] = trade_volume.sum()
            d1[t]['vwap_lag'] = temp_lag['TradeAmount'].sum()/d1[t]['trade_volume_lag']
            d1[t]['lowest_price_lag'] = min(temp_lag['TradePrice'])
            d1[t]['highest_price_lag'] = max(temp_lag['TradePrice'])
                    
    #        temp_actbuy = temp_lag[temp_lag['TradeDirection']==1]
    #        temp_actsell = temp_lag[temp_lag['TradeDirection']==-1]
    #        d1[t]['total_buy_vol_lag'] = sum(temp_actbuy['TradeVolume'])
    #        d1[t]['total_sell_vol_lag'] = sum(temp_actsell['TradeVolume'])
            d1[t]['total_buy_vol_lag'] = ((temp_lag['TradeDirection']+1)*0.5*trade_volume).sum()
            d1[t]['total_sell_vol_lag'] = -((temp_lag['TradeDirection']-1)*0.5*trade_volume).sum()
            
    #        grouped_actvol = temp_lag.groupby('ActOrderId').sum()
    #        d1[t]['act_vol_skew_lag'] = np.power((grouped_actvol['NetVolume'] - grouped_actvol['NetVolume'].mean()),3).mean()/np.power(grouped_actvol['NetVolume'].std(),3)
    
            net_volume = temp_grouped_lag['NetVolume']
            d1[t]['act_vol_skew_lag'] = np.power((net_volume - net_volume.mean()),3).mean()/np.power(net_volume.std(),3)
              
            # ---------------------------
            
    #        temptime3_1 = time()
    #        d1[t]['net_volume_lag2'] = temp_lag2['NetVolume'].sum()
    #        d1[t]['trade_volume_lag2'] = temp_lag2['TradeVolume'].sum()
    #        d1[t]['vwap_lag2'] = temp_lag2['TradeAmount'].sum()/d1[t]['trade_volume_lag2']
    #        d1[t]['lowest_price_lag2'] = min(temp_lag2['TradePrice'])
    #        d1[t]['highest_price_lag2'] = max(temp_lag2['TradePrice'])
    #        
    #        temp_actbuy = temp_lag2[temp_lag2['TradeDirection']==1]
    #        temp_actsell = temp_lag2[temp_lag2['TradeDirection']==-1]
    #        d1[t]['total_buy_vol_lag2'] = sum(temp_actbuy['TradeVolume'])
    #        d1[t]['total_sell_vol_lag2'] = sum(temp_actsell['TradeVolume'])
    #        d1[t]['total_buy_vol_lag2'] = ((temp_lag2['TradeDirection']+1)*0.5*temp_lag2['TradeVolume']).sum()
    #        d1[t]['total_sell_vol_lag2'] = -((temp_lag2['TradeDirection']-1)*0.5*temp_lag2['TradeVolume']).sum()
    #        
    #        grouped_actvol = temp_lag2.groupby('ActOrderId').sum()
    #        d1[t]['act_vol_skew_lag2'] = np.power((grouped_actvol['NetVolume'] - grouped_actvol['NetVolume'].mean()),3).mean()/np.power(grouped_actvol['NetVolume'].std(),3)
    #        temptime3.append(time() - temptime3_1)
            #t+=window
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



# ----------------------------------

def incre_buy(x):
    vol2 = 0
    vol1 = 0
    for i in map(str,range(1,11)):
        if x['BuyPrice'+i]>=x['lowest_price_lag']:
            vol2 += x['BuyVolume'+i]
        else:
            break
    for i in map(str,range(1,11)):
        if x['BuyPrice'+i+'_lag']>=x['lowest_price_lag']:
            vol1 += x['BuyVolume'+i+'_lag']
        else:
            break
    return vol2 + x['total_sell_vol_lag'] - vol1
        
 
       
def incre_sell(x):
    vol2 = 0
    vol1 = 0
    for i in map(str,range(1,11)):
        if x['SellPrice'+i]<=x['highest_price_lag']:
            vol2 += x['SellVolume'+i]
        else:
            break
    for i in map(str,range(1,11)):
        if x['SellPrice'+i+'_lag']<=x['highest_price_lag']:
            vol1 += x['SellVolume'+i+'_lag']
        else:
            break
    return vol2 + x['total_buy_vol_lag'] - vol1

l = os.listdir('fac_5m_1')

try:
    os.mkdir("fac_5m")
except:
    pass

for i in range(len(l)): # i=0
    file_names = os.listdir('fac_5m_1/'+l[i])
    try:
        os.mkdir("fac_5m/"+l[i])
    except:
        pass
    output_file_path = ("fac_5m/"+l[i]+'/')

    snapshot = pd.read_csv(snapshot_file_path+l[i]+'.csv')
    snapshot = snapshot.drop_duplicates()
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
    
    new_name = []
    for col in snapshot_formerge_lag2.columns:
        if col in ['InstrumentId','datetime']:
            new_name.append(col)
        else:
            new_name.append(col+'_lag')
    snapshot_formerge_lag2.columns = new_name

    for j in range(len(file_names)): # file = file_names[0];
        file = file_names[j]
        df = pd.read_csv('fac_5m_1/'+l[i]+'/'+file,index_col=0)
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
        df['trade_volume_imbalance_lag'] = df['total_buy_vol_lag']/df['trade_volume_lag']
        df['volume_imbalance'] = df['BuyVolume1']/(df['SellVolume1']+df['BuyVolume1'])
        
        df['incre_buy_lag'] = df.apply(incre_buy,axis=1)
        df['incre_sell_lag'] = df.apply(incre_sell,axis=1)
        
        df['new_buy_lag'] = df['incre_buy_lag']/df['total_sell_vol_lag']
        df['new_sell_lag'] = df['incre_sell_lag']/df['total_buy_vol_lag']
        
        df.dropna(inplace=True)
        df.reset_index(drop=True,inplace=True)
        
        df.to_csv(output_file_path+file)
        
        print(j,end=',')


df[df['ret_vwapmid_fur']>0.002].sum()

temp = df.corr()['ret_vwapmid_fur']

# --------------------------------

file_names = os.listdir(output_file_path)

l=[]
for j in range(len(file_names)): # j=0;
    file = file_names[j]
    l.append(pd.read_csv(output_file_path+file,index_col=0))
    if not j%50:
        print(j,end=',')
df = pd.concat(l)
df['datetime']=pd.to_datetime(df['datetime'],format='%Y-%m-%d %H:%M:%S')

def plot(temp,factor,log=False,col=None):
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

columns = pd.DataFrame(df.columns)

plot(df['incre_buy_lag'],df,col='incre_buy')

plot(df['incre_sell_lag'],df,col='incre_sell')

plot(df['resist_buy_lag'],df,1,col='incre_buy/total_sell')

plot(df['resist_sell_lag'],df,1,col='incre_sell/total_buy')

plot(df['resist_buy_lag'] / df['resist_sell_lag'],df,1,col='incre_cmp')

temp  = df[(df['incre_buy_lag']>0)&(df['incre_sell_lag']>0)].copy()

temp = df.copy()
temp['cmp'] = np.log(temp['resist_buy_lag']/temp['resist_sell_lag'])
temp['cmp'] = np.log(temp['resist_buy_lag']-temp['resist_sell_lag'])
temp.dropna(inplace=True)
temp.shape
temp_a = temp[temp['cmp']>0]
temp_a[temp_a['cmp']==-np.inf]



(temp_a[temp_a['cmp']>3]['ret_vwapmid_fur']>0.002).sum()/len(temp_a[temp_a['cmp']>3])

((df['ret_vwapmid_fur']>0.002).sum())/len(df)

plot(temp['cmp'],temp)

plot(df['volume_imbalance'],df,col='volume_imbalance')

plot(df['net_volume_lag'],df,log = 1,col='net_volume')

plot(df['net_volume_lag']/df['trade_volume_lag'],df,col='net_volume')

temp = df[df['net_volume_lag']!=0]
plot((temp['incre_buy_lag']/temp['net_volume_lag']).astype(float),temp,1)

plot(df['act_vol_skew_lag'],df,col='volume_skew')

plot((df['trade_volume_lag']).astype(int),df,1)

temp = df[df['net_volume_lag']<0].copy()
plot(temp['incre_buy_lag'],temp,1)



############



file_names = os.listdir(output_file_path)

l=[]
for j in range(len(file_names)): # j=0;
    file = file_names[j]
    l.append(pd.read_csv(output_file_path+file,index_col=0))
    if not j%50:
        print(j,end=',')
df = pd.concat(l)
df['datetime']=pd.to_datetime(df['datetime'],format='%Y-%m-%d %H:%M:%S')













