# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 16:29:07 2019

big order influence.
30+3+30+1

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

dir_name = 'fac_1m_big'
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


#for i in range(3): #i=0; 2
#for i in range(3,6): #i=0; 3
for i in range(5,7): #i=0; 3 ; i=1
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
    quantile_new = quantile_group.quantile(0.8)['TradeVolume']


    print('read snapshot, transaction file')
    
    #am = dt.datetime(year,month,day,11,30,0,0)-window
    #pm1 = dt.datetime(year,month,day,13,00,0,0)+2*window
    #pm = dt.datetime(year,month,day,14,57,0,0)-window
    am = dt.datetime(year,month,day,11,30,0,0)
    pm1 = dt.datetime(year,month,day,13,1,0,0)+window
    pm = dt.datetime(year,month,day,14,56,0,0)
    
    
    df_all = pd.DataFrame()
    ids = list(set(transaction['InstrumentId']))
    print('total: ',len(ids))
   
    
#    for j in range(len(ids)): # j=0,len(ids)
    for j in range(len(ids)): # j=0,len(ids)
#    for j in range(20): # j=0,len(ids)
        #dtime = [];j=0;dtime.append(time())
#    for j in range(650,len(ids)):    
        Id = ids[j]
        if Id not in quantile_last.index:
            continue
        transaction_id = transaction[transaction['InstrumentId']==Id]
        snapshot_id = snapshot[snapshot['InstrumentId']==Id]
        snapshot_formerge_id = snapshot_formerge[snapshot_formerge['InstrumentId']==Id]
        #snapshot_formerge_lag2_id = snapshot_formerge_lag2[snapshot_formerge_lag2['InstrumentId']==Id]
        trans_grouped_id = trans_grouped.loc[Id]
        t = snapshot_id[snapshot_id['datetime']>=dt.datetime(year,month,day,9,31,0,0)]['datetime'].iloc[0]
        d1={}
        
        #dtime.append(time())
        while t < pm:  # t = dt.datetime(year,month,day,9,40,0,0)
            if (t>am) & (t<pm1):
                t = snapshot_id[snapshot_id['datetime']>=pm1]['datetime'].iloc[0]
                continue
            
            t2 = t+dt.timedelta(seconds=30)
            d1[t2] = {}            
            
            temp_sec = trans_grouped_id[(trans_grouped_id['datetime']>t-dt.timedelta(seconds=3)) & (trans_grouped_id['datetime']<=t)]
            temp_sec_big = temp_sec[temp_sec['TradeVolume']>=quantile_last.loc[Id]]

            temp_ret = transaction_id[(transaction_id['datetime']>t+dt.timedelta(seconds=30)) & (transaction_id['datetime']<=t+dt.timedelta(seconds=90))]         
            if temp_ret.empty:
                t+=dt.timedelta(seconds=3)
                continue
            else:
                d1[t2]['vwap_fur'] = temp_ret['TradeAmount'].values.sum()/temp_ret['TradeVolume'].values.sum()
                d1[t2]['volume_fur'] = temp_ret['TradeVolume'].values.sum()
                buy_order_after = temp_ret[temp_ret['TradeDirection']==1]
                if not buy_order_after.empty:
                    d1[t2]['buy_volume_fur'] = buy_order_after['TradeVolume'].values.sum()
                    d1[t2]['buy_vwap_fur'] = buy_order_after['TradeAmount'].values.sum()/buy_order_after['TradeVolume'].values.sum()
                sell_order_after = temp_ret[temp_ret['TradeDirection']==-1]
                if not sell_order_after.empty:
                    d1[t2]['sell_volume_fur'] = sell_order_after['TradeVolume'].values.sum()
                    d1[t2]['sell_vwap_fur'] = sell_order_after['TradeAmount'].values.sum()/sell_order_after['TradeVolume'].values.sum()
            
            if temp_sec_big.empty:
                t+=dt.timedelta(seconds=3)
                continue
            else:
                # big order
                d1[t2]['big_order_volume'] = temp_sec_big['TradeVolume'].values.sum()
                d1[t2]['big_order_vwap'] = temp_sec_big['TradeAmount'].values.sum()/temp_sec_big['TradeVolume'].values.sum()
                big_order_buy = temp_sec_big[temp_sec_big['TradeDirection']==1]
                if not big_order_buy.empty:
                    d1[t2]['big_buy_order_volume'] = big_order_buy['TradeVolume'].values.sum()
                    d1[t2]['big_buy_order_vwap'] = big_order_buy['TradeAmount'].values.sum()/big_order_buy['TradeVolume'].values.sum()
                big_order_sell = temp_sec_big[temp_sec_big['TradeDirection']==-1]
                if not big_order_sell.empty:
                    d1[t2]['big_sell_order_volume'] = big_order_sell['TradeVolume'].values.sum()
                    d1[t2]['big_sell_order_vwap'] = big_order_sell['TradeAmount'].values.sum()/big_order_sell['TradeVolume'].values.sum()
                            
                
            temp_lag = transaction_id[(transaction_id['datetime']>t-dt.timedelta(seconds=33)) & (transaction_id['datetime']<=t-dt.timedelta(seconds=3))]
            temp_fur = transaction_id[(transaction_id['datetime']>t) & (transaction_id['datetime']<=t+dt.timedelta(seconds=30))]
                      
            if temp_lag.empty or temp_fur.empty:
                t+=dt.timedelta(seconds=3)
                continue
            
            # before
            d1[t2]['volume_before_big_order'] = temp_lag['TradeVolume'].values.sum()
            d1[t2]['vwap_before_big_order'] = temp_lag['TradeAmount'].values.sum()/temp_lag['TradeVolume'].values.sum()
            buy_order_before = temp_lag[temp_lag['TradeDirection']==1]
            if not buy_order_before.empty:
                d1[t2]['buy_volume_before_big_order'] = buy_order_before['TradeVolume'].values.sum()
                d1[t2]['buy_vwap_before_big_order'] = buy_order_before['TradeAmount'].values.sum()/buy_order_before['TradeVolume'].values.sum()
            sell_order_before = temp_lag[temp_lag['TradeDirection']==-1]
            if not sell_order_before.empty:
                d1[t2]['sell_volume_before_big_order'] = sell_order_before['TradeVolume'].values.sum()
                d1[t2]['sell_vwap_before_big_order'] = sell_order_before['TradeAmount'].values.sum()/sell_order_before['TradeVolume'].values.sum()

            # after                
            d1[t2]['volume_after_big_order'] = temp_fur['TradeVolume'].values.sum()
            d1[t2]['vwap_after_big_order'] = temp_fur['TradeAmount'].values.sum()/temp_fur['TradeVolume'].values.sum()
            buy_order_after = temp_fur[temp_fur['TradeDirection']==1]
            if not buy_order_after.empty:
                d1[t2]['buy_volume_after_big_order'] = buy_order_after['TradeVolume'].values.sum()
                d1[t2]['buy_vwap_after_big_order'] = buy_order_after['TradeAmount'].values.sum()/buy_order_after['TradeVolume'].values.sum()
            sell_order_after = temp_fur[temp_fur['TradeDirection']==-1]
            if not sell_order_after.empty:
                d1[t2]['sell_volume_after_big_order'] = sell_order_after['TradeVolume'].values.sum()
                d1[t2]['sell_vwap_after_big_order'] = sell_order_after['TradeAmount'].values.sum()/sell_order_after['TradeVolume'].values.sum()

            # ret                        
            t+=dt.timedelta(seconds=3)
        #dtime.append(time())
        
        df = pd.DataFrame.from_dict(d1,orient='index')
        if df.empty:
            continue
        df['InstrumentId'] = Id
        #df_all = pd.concat([df_all,df])
        df = df.reset_index()
        df.rename(columns={'index':'datetime'},inplace=True)
        df = df.merge(snapshot_formerge_id,on=['InstrumentId','datetime'],how='left')
        
        #------------
        #dtime.append(time())
        
        df['midprice'] = 0.5*(df['BuyPrice1'].values + df['SellPrice1'].values)
        df['ret_vwapmid_fur'] = np.log(df['vwap_fur'].values/df['midprice'].values)
       
        #dtime.append(time())

        df.reset_index(drop=True,inplace=True)
        
        df.to_csv(output_file_path+str(Id)+'.csv')
        if not j%10:
            print(j,end=',')
    
    quantile_last = quantile_new.copy()
    print('\n',file,'done')
    gc.collect()




np.diff(np.array(dtime))




#-------------------------------

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



#----------------------------------------------
list(df.columns)




temp.isna().sum()



#file = transaction_files[4]
file = os.listdir(dir_name)
file_path = dir_name+'/'+file[1]+'/'
#fac_path = 'fac_5m/20180903/'
l = os.listdir(file_path)
fac_data = []
for i in range(len(l)): # i=0;
    file = l[i]
    fac_data.append(pd.read_csv(file_path+file,index_col=0))
fac_data = pd.concat(fac_data)
fac_data['datetime']=pd.to_datetime(fac_data['datetime'],format='%Y-%m-%d %H:%M:%S')


print(fac_data.shape)
temp = fac_data.dropna(subset=['big_buy_order_volume'])
temp = fac_data.dropna(subset=['big_sell_order_volume'])
temp = fac_data.dropna()


temp = temp[temp['big_sell_order_volume'].isna()]
print(temp.shape)

len(temp['InstrumentId'].unique())
all_plot(np.log(temp['vwap_after_big_order']/temp['vwap_before_big_order']), temp)

all_plot(np.log(temp['buy_vwap_after_big_order']/temp['buy_vwap_before_big_order']), temp)
all_plot(np.log(temp['sell_vwap_after_big_order']/temp['sell_vwap_before_big_order']), temp)
all_plot(np.log(temp['sell_vwap_after_big_order']/temp['buy_vwap_before_big_order']), temp)

all_plot(,temp)

all_plot(np.log(temp['vwap_after_big_order']/temp['big_order_vwap'])-np.log(temp['big_order_vwap']/temp['vwap_before_big_order']),temp)

all_plot(np.log(temp['big_order_volume']/temp['volume_before_big_order']),temp)

all_plot(np.log(temp['volume_after_big_order']/temp['big_order_volume']),temp)



# ------------------------------------

daily = pd.read_csv('daily/201809StockDailyData.csv')
daily['InstrumentId'] = daily['WindCode'].apply(lambda x:int(x[:6]))
daily_SH = daily[daily['WindCode'].apply(lambda x:True if 'SH' in x else False)]
today = daily_SH[daily_SH['TradingDay']==20180903].copy()
today['free_cap'] = today['VWAvgPrice']*today['NonRestrictedShares']/today['SplitFactor']

#temp = fac_data.merge(today[['InstrumentId','MarketCapAFloat']],on = ['InstrumentId'],how = 'left')
temp = fac_data.merge(today[['InstrumentId','free_cap','VWAvgPrice']],on = ['InstrumentId'],how = 'left')

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
#--------------------------------------------

fac_range = temp.dropna(subset=['big_buy_order_volume'])
fac_range = temp.dropna(subset=['big_buy_order_volume','vwap_after_big_order','vwap_before_big_order','ret_vwapmid_fur'])

fac_range = temp.dropna(subset=['big_buy_order_volume'])
fac_range = temp.dropna(subset=['big_buy_order_volume','vwap_after_big_order','vwap_before_big_order'])
fac_range[['vwap_after_big_order','vwap_before_big_order','ret_vwapmid_fur']].isna().sum()

fac_range = temp.dropna(subset=['big_buy_order_volume','vwap_after_big_order','vwap_before_big_order','ret_vwapmid_fur'])
all_plot(np.log(fac_range['vwap_after_big_order']/fac_range['vwap_before_big_order']),fac_range,log=False)



spec = fac_range[(np.log(fac_range['resist_sell_lag'])<-3.5) &
                 (fac_range['ret_vwapmid_fur']>0.012)]

print(set(list(spec['InstrumentId'])))






for i in range(len(conditions_cap)):
    con = conditions_cap[i]
    fac_range = temp[con]
    range_noinf = fac_range.dropna(subset=['big_buy_order_volume','ret_vwapmid_fur',
                                           'vwap_after_big_order','vwap_before_big_order'])
    #plot(range_noinf['resist_sell_lag'],range_noinf,1,'sell, price '+cons_cap[i])
    all_plot(np.log(range_noinf['vwap_after_big_order']/range_noinf['vwap_before_big_order']),range_noinf,log=False,col = 'cap '+cons_cap[i])
    print('''
          num stocks:   %f
          ave spread:   %f
          ave price:    %f
          spread/price: %f''' % (
          len(range_noinf['InstrumentId'].unique()),
          (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean(),
          range_noinf['VWAvgPrice'].mean(),
          (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean()/range_noinf['VWAvgPrice'].mean()))

for i in range(len(conditions_price)):
    con = conditions_price[i]
    fac_range = temp[con]
    range_noinf = fac_range.dropna(subset=['big_buy_order_volume','ret_vwapmid_fur',
                                           'vwap_after_big_order','vwap_before_big_order'])
    #plot(range_noinf['resist_sell_lag'],range_noinf,1,'sell, price '+cons_price[i])
    all_plot(np.log(range_noinf['vwap_after_big_order']/range_noinf['vwap_before_big_order']),range_noinf,log=False,col = 'price '+cons_price[i])
    print('''
          num stocks:   %f
          ave spread:   %f
          ave price:    %f
          spread/price: %f''' % (
          len(range_noinf['InstrumentId'].unique()),
          (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean(),
          range_noinf['VWAvgPrice'].mean(),
          (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean()/range_noinf['VWAvgPrice'].mean()))

temp[temp['InstrumentId'] == 603032]

for i in range(len(conditions_cap)): # i=3
    con = conditions_cap[i]
    fac_range = temp[con]
    fac_range_ret = fac_range.dropna(subset=['ret_vwapmid_fur'])
    fac_range = fac_range.dropna(subset=['big_buy_order_volume','vwap_after_big_order','vwap_before_big_order','ret_vwapmid_fur'])
    fac_range = fac_range[(fac_range['vwap_after_big_order']!=0) & 
                          (fac_range['vwap_before_big_order']!=0)]
    for Id in fac_range['InstrumentId'].unique(): # Id = 600525
        #Id = 600929
        range_noinf = fac_range[fac_range['InstrumentId']==Id]
        range_ret = fac_range_ret[fac_range_ret['InstrumentId']==Id]
        if len(range_noinf)>500 or len(range_noinf)<300:
            continue
        try:
            #all_plot(np.log(range_noinf['vwap_after_big_order']/range_noinf['vwap_before_big_order']),range_noinf,log=False,addv=-3.5,addh=0.012,col = 'cap '+cons_cap[i])
            plot(np.log(range_noinf['vwap_after_big_order']/range_noinf['vwap_before_big_order']),range_noinf,log=False,addv=-3.5,addh=0.012,col = 'cap '+cons_cap[i])
            print('''
                  num stocks:   %f
                  ave spread:   %f
                  ave price:    %f
                  spread/price: %f''' % (
                  len(range_noinf['InstrumentId'].unique()),
                  (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean(),
                  range_noinf['VWAvgPrice'].mean(),
                  (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean()/range_noinf['VWAvgPrice'].mean()))
            print('ret>0,all:',(range_ret['ret_vwapmid_fur']>0).sum()/len(range_ret))
            print('ret,all:',range_ret['ret_vwapmid_fur'].mean())
            print('id: ',Id)
            print('market cap: ',range_noinf['free_cap'].iloc[0]/1e8)
            print('price: ',range_noinf['VWAvgPrice'].iloc[-1])
        except:
            print(Id,'error')
            pass

# --------------------------
for i in range(len(conditions_cap)): # i=1
    con = conditions_cap[i]
    fac_range = temp[con]
    fac_range_ret = fac_range.dropna(subset=['ret_vwapmid_fur'])
    fac_range = fac_range.dropna(subset=['big_buy_order_volume','buy_vwap_after_big_order','buy_vwap_before_big_order','ret_vwapmid_fur'])
    for Id in fac_range['InstrumentId'].unique()[0:20]: # 
        range_noinf = fac_range[fac_range['InstrumentId']==Id]
        if len(range_noinf)<100:
            continue
        try:
            all_plot(np.log(range_noinf['buy_vwap_after_big_order']/range_noinf['buy_vwap_before_big_order']),range_noinf,log=False,addv=-3.5,addh=0.012,col = 'cap '+cons_cap[i])
            print('''
                  num stocks:   %f
                  ave spread:   %f
                  ave price:    %f
                  spread/price: %f''' % (
                  len(range_noinf['InstrumentId'].unique()),
                  (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean(),
                  range_noinf['VWAvgPrice'].mean(),
                  (range_noinf['SellPrice1']-range_noinf['BuyPrice1']).mean()/range_noinf['VWAvgPrice'].mean()))
            print('ret>0,all:',(fac_range_ret['ret_vwapmid_fur']>0).sum()/len(fac_range_ret))
            print('ret,all:',fac_range_ret['ret_vwapmid_fur'].mean())
            print('id: ',Id)
            print('market cap: ',range_noinf['free_cap'].iloc[0]/1e8)
            print('price: ',range_noinf['VWAvgPrice'].iloc[-1])
        except:
            print(Id,'error')
            pass

# ----------------------------
range_noinf = temp.dropna(subset=['big_buy_order_volume','buy_vwap_after_big_order','buy_vwap_before_big_order','ret_vwapmid_fur'])
all_plot(np.log(range_noinf['buy_vwap_after_big_order']/range_noinf['buy_vwap_before_big_order']),range_noinf,log=False,addv=-3.5,addh=0.012)

range_noinf = temp.dropna(subset=['big_buy_order_volume','sell_vwap_after_big_order','sell_vwap_before_big_order','ret_vwapmid_fur'])
all_plot(np.log(range_noinf['sell_vwap_after_big_order']/range_noinf['sell_vwap_before_big_order']),range_noinf,log=False,addv=-3.5,addh=0.012)

range_noinf = temp.dropna(subset=['big_buy_order_volume','vwap_after_big_order','vwap_before_big_order','ret_vwapmid_fur'])
all_plot(np.log(range_noinf['vwap_after_big_order']/range_noinf['vwap_before_big_order']),range_noinf,log=False,addv=-3.5,addh=0.012)

d={}
for Id in temp['InstrumentId'].unique():
    fac_range = temp[temp['InstrumentId']==Id]
    fac_range = fac_range.dropna(subset = ['ret_vwapmid_fur'])
    range_noinf = fac_range.dropna(subset = ['big_buy_order_volume','vwap_after_big_order','vwap_before_big_order','ret_vwapmid_fur']).copy()
    if len(range_noinf)<100:
        continue
    d[Id] = {}
    range_noinf['fac'] = np.log(range_noinf['vwap_after_big_order']/range_noinf['vwap_before_big_order'])
    d[Id]['pct fac >0'] = (range_noinf['fac']>0).sum()/len(range_noinf)
    d[Id]['right mean'] = range_noinf[range_noinf['fac']>0]['ret_vwapmid_fur'].mean()
    d[Id]['left mean'] = range_noinf[range_noinf['fac']<0]['ret_vwapmid_fur'].mean()
    d[Id]['mean'] = range_noinf['ret_vwapmid_fur'].mean()
    d[Id]['all mean'] = fac_range['ret_vwapmid_fur'].mean()

    d[Id]['diff mean'] = d[Id]['right mean'] - d[Id]['left mean']
    d[Id]['diff mean right all'] = d[Id]['right mean'] - d[Id]['all mean']
    d[Id]['diff mean left all'] = d[Id]['left mean'] - d[Id]['all mean']
    d[Id]['diff mean all'] = d[Id]['mean'] - d[Id]['all mean']
    
    d[Id]['right prob'] = (range_noinf[range_noinf['fac']>0]['ret_vwapmid_fur']>0).sum()/(range_noinf['fac']>0).sum()
    d[Id]['left prob'] = (range_noinf[range_noinf['fac']<0]['ret_vwapmid_fur']>0).sum()/(range_noinf['fac']<0).sum()
    d[Id]['prob'] = (range_noinf['ret_vwapmid_fur']>0).sum()/len(range_noinf)
    d[Id]['all prob'] = (fac_range['ret_vwapmid_fur']>0).sum()/len(fac_range)

    d[Id]['diff prob right all'] = d[Id]['right prob'] - d[Id]['all prob']
    d[Id]['diff prob left all'] = d[Id]['left prob'] - d[Id]['all prob']
    d[Id]['diff prob all'] = d[Id]['prob'] - d[Id]['all prob'] 
    d[Id]['diff prob'] = d[Id]['right prob'] - d[Id]['left prob'] 

    d[Id]['num_all'] = len(fac_range)
    d[Id]['num_bigbuy'] = len(range_noinf)
    d[Id]['cap'] = range_noinf['free_cap'].mean()
    d[Id]['price'] = range_noinf['VWAvgPrice'].mean()    
df = pd.DataFrame.from_dict(d,orient='index')

print(df.shape)
df_nona = df.dropna()
print(df_nona.shape)

plt.hist(df_nona['diff mean'],bins=100)
(df_nona['diff mean']>0).sum()/len(df_nona)
all_plot(np.log(df_nona['cap']),df_nona,independent='diff mean')
all_plot(np.log(df_nona['price']),df_nona,independent='diff mean')

plt.hist(df_nona['diff mean right all'],bins=100)
(df_nona['diff mean right all']>0).sum()/len(df_nona)
all_plot(np.log(df_nona['cap']),df_nona,independent='diff mean right all')
all_plot(np.log(df_nona['price']),df_nona,independent='diff mean right all')

df_nona['diff mean right all'].mean()

(df_nona['diff mean left all']>0).sum()/len(df_nona)
plt.hist(df_nona['pct fac >0'],bins=100)
df_nona['pct fac >0'].mean()
df_nona['pct fac >0'].median()


plt.hist(df_nona['diff mean all'],bins=100)
(df_nona['diff mean all']>0).sum()/len(df_nona)
all_plot(np.log(df_nona['cap']),df_nona,independent='diff mean all')
all_plot(np.log(df_nona['price']),df_nona,independent='diff mean all')


df_nona['diff mean all'].mean()

(df['all mean']!=df['left mean']).sum()

df['right mean'].mean()
df['mean'].mean()
df['all mean'].mean()


df['right mean'].mean() - df['left mean'].mean()
df['right mean'].mean() - df['mean'].mean()

plt.hist(df_nona['diff prob'],bins=100)
(df_nona['diff prob']>0).sum()/len(df_nona)
all_plot(np.log(df_nona['cap']),df_nona,independent='diff prob')
all_plot(np.log(df_nona['price']),df_nona,independent='diff prob')

plt.hist(df_nona['diff prob right all'],bins=100)
(df_nona['diff prob right all']>0).sum()/len(df_nona)
all_plot(np.log(df_nona['cap']),df_nona,independent='diff prob right all')
all_plot(np.log(df_nona['price']),df_nona,independent='diff prob right all')

df_nona['diff prob right all'].mean()

(df_nona['diff prob left all']>0).sum()/len(df_nona)


plt.hist(df_nona['diff prob all'],bins=100)
(df_nona['diff prob all']>0).sum()/len(df_nona)
all_plot(np.log(df_nona['cap']),df_nona,independent='diff prob all')
all_plot(np.log(df_nona['price']),df_nona,independent='diff prob all')





plt.hist(df_nona['diff prob left all'],bins=100)
(df_nona['diff prob left all']>0).sum()/len(df_nona)
all_plot(np.log(df_nona['cap']),df_nona,independent='diff prob left all')
all_plot(np.log(df_nona['price']),df_nona,independent='diff prob left all')


df_nona_temp = df_nona[df_nona['diff']>0]


#---------------------------------------------


600038, 600050

for i in range(len(conditions_cap)): # i=0
    con = conditions_cap[i]
    fac_range = temp[con]
    range_noinf = fac_range.dropna(subset=['big_buy_order_volume','vwap_after_big_order','vwap_before_big_order','ret_vwapmid_fur'])
    print(range_noinf['ret_vwapmid_fur'].mean(),fac_range['ret_vwapmid_fur'].dropna().mean(),range_noinf['ret_vwapmid_fur'].mean() - fac_range['ret_vwapmid_fur'].dropna().mean())

# -0--------------------------------
d = {}
for Id in temp['InstrumentId'].unique():
    fac_range = temp[temp['InstrumentId']==Id]
    fac_range = fac_range.dropna(subset = ['ret_vwapmid_fur'])
    range_noinf = fac_range.dropna(subset=['big_buy_order_volume'])
    if range_noinf.empty:
        continue
    d[Id] = {}
    d[Id]['ret_bigbuy'] = range_noinf['ret_vwapmid_fur'].mean()
    d[Id]['ret_all'] = fac_range['ret_vwapmid_fur'].mean()
    d[Id]['diff'] = range_noinf['ret_vwapmid_fur'].mean() - fac_range['ret_vwapmid_fur'].mean()
    d[Id]['num_all'] = len(fac_range)
    d[Id]['num_bigbuy'] = len(range_noinf)
    d[Id]['cap'] = range_noinf['free_cap'].mean()
    d[Id]['price'] = range_noinf['VWAvgPrice'].mean()

df = pd.DataFrame.from_dict(d,orient='index')

plt.hist(df['diff'],bins=100)
(df['diff']>0).sum()/len(df)
all_plot(np.log(df['cap']),df,independent='diff')
all_plot(np.log(df['price']),df,independent='diff')

(df['ret_bigbuy']>0).sum()/len(df)
(df['ret_all']>0).sum()/len(df)


df_temp = df[df['diff']>0]


# ------------------------------------------------


d = {}
for Id in temp['InstrumentId'].unique():
    fac_range = temp[temp['InstrumentId']==Id]
    fac_range = fac_range.dropna(subset = ['ret_vwapmid_fur'])
    range_noinf = fac_range.dropna(subset=['big_buy_order_volume'])
    if range_noinf.empty:
        continue
    d[Id] = {}
    d[Id]['ret_bigbuy'] = range_noinf['ret_vwapmid_fur'].mean()
    d[Id]['ret_all'] = fac_range['ret_vwapmid_fur'].mean()
    d[Id]['diff'] = range_noinf['ret_vwapmid_fur'].mean() - fac_range['ret_vwapmid_fur'].mean()
    d[Id]['num_all'] = len(fac_range)
    d[Id]['num_bigbuy'] = len(range_noinf)
    d[Id]['cap'] = range_noinf['free_cap'].mean()
    d[Id]['price'] = range_noinf['VWAvgPrice'].mean()

# -------------------------------------------------
    
range_noinf = temp.dropna(subset=['big_buy_order_volume','vwap_after_big_order','ret_vwapmid_fur'])
all_plot(range_noinf['buy_volume_after_big_order']/range_noinf['volume_after_big_order'],range_noinf,log=False,addv=-3.5,addh=0.012)



