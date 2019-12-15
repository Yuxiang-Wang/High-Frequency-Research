# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 10:32:46 2019

1、常规因子互相的相关性
2、筛选几个因子做了回归

@author: yuxiang
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import os
import gc
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LinearRegression
%matplotlib inline

def get_daily_corr(file_path):
    file_names = sorted(os.listdir(file_path))
    
    d_corr = {}
    d_nind = {}
    corr_mean = pd.read_csv(file_path+file_names[0],index_col=0)
    for factor in corr_mean.index:
        d_nind[factor]=1
    
    for i in range(1,len(file_names)):  # file = file_names[1];
        file = file_names[i]
        temp = pd.read_csv(file_path+file,index_col=0)
        d_corr[int(file[:6])]=temp
        temp.dropna(inplace=True)
        corr_mean = corr_mean.add(temp,fill_value=0)
    
        for factor in temp.index:
            if factor in d_nind.keys():
                d_nind[factor]+=1
            else:
                d_nind[factor]=1
        if not i%50:
            print(i)
    
    nind = pd.DataFrame.from_dict(d_nind,orient='index')
    corr_mean = corr_mean.divide(nind.T.values[0],axis='index')
    return corr_mean

corr_0903 = get_daily_corr('XSHG_corr/0903/')
corr_0904 = get_daily_corr('XSHG_corr/0904/')
corr_0903.to_csv('corr_mean_0903.csv')
#===============================
factor_file_path = 'XSHG_factor/0903/'
files_factor = sorted(os.listdir(factor_file_path))

def quantile_cut(down,up,factor):
    for col in factor.columns:
        if col in ['InstrumentId','datetime','ret_highest_fur','ret_lowest_fur','ret_vwap_fur']:
            continue
        factor.loc[factor[col]>factor[col].quantile(0.99),col] = factor[col].quantile(0.99)
        factor.loc[factor[col]<factor[col].quantile(0.01),col] = factor[col].quantile(0.01)
def normalize(factor):
    for col in factor.columns:
        if col in ['InstrumentId','datetime','ret_highest_fur','ret_lowest_fur','ret_vwap_fur']:
            continue
        factor.loc[:,col] = (factor[col] - factor[col].mean())/factor[col].std()

# -------------------------------------------------
# console 1, vwap
try:
    os.mkdir('XSHG_corr/reg_0903/')
except:
    pass

output_path = 'XSHG_corr/reg_0903/'
error=[]
for i in range(len(files_factor)): # file=files_factor[0]
    file = files_factor[i]
    factor = pd.read_csv(factor_file_path+file)
    if factor.empty:
        continue
    quantile_cut(0.01,0.99,factor)    
    normalize(factor)
    y = factor['ret_vwap_fur'].values
    d = {}
    for col in factor.columns: # col = factor.columns[3]
        if col in ['InstrumentId','datetime','ret_highest_fur','ret_lowest_fur','ret_vwap_fur']:
            continue
        try:
            X = factor[col].values
            X = sm.add_constant(X)
            model = sm.OLS(y,X)
            results = model.fit()
            d[col] = results.params[1],results.tvalues[1],results.pvalues[1]
        except:
            error.append((file,col))
            pass
    pv2_normalized = pd.DataFrame.from_dict(d,orient='index')    
    if pv2_normalized.empty:
        continue
    pv2_normalized.columns = ['coef','t_value','p_value']    
    pv2_normalized.to_csv(output_path+file)
    print(i,end=',')

#results.summary()
#plt.scatter(factor['weighted_midprice_1_vwappre_ratio'],factor['ret_vwap_fur'])



# console 2, highest
try:
    os.mkdir('XSHG_corr/reg_highest_0903/')
except:
    pass

output_path = 'XSHG_corr/reg_highest_0903/'
error=[]
for i in range(len(files_factor)): # file=files_factor[0]
    file = files_factor[i]
    factor = pd.read_csv(factor_file_path+file)
    if factor.empty:
        continue
    quantile_cut(0.01,0.99,factor)    
    normalize(factor)
    y = factor['ret_highest_fur'].values
    d = {}    
    for col in factor.columns: # col = factor.columns[3]
        if col in ['InstrumentId','datetime','ret_highest_fur','ret_lowest_fur','ret_vwap_fur']:
            continue
        try:
            X = factor[col].values
            X = sm.add_constant(X)
            model = sm.OLS(y,X)
            results = model.fit()
            d[col] = results.params[1],results.tvalues[1],results.pvalues[1]
        except:
            error.append((file,col))
            pass        
    pv2_normalized = pd.DataFrame.from_dict(d,orient='index')   
    if pv2_normalized.empty:
        continue
    pv2_normalized.columns = ['coef','t_value','p_value']    
    pv2_normalized.to_csv(output_path+file)
    print(i,end=',')


# console 3, lowest
try:
    os.mkdir('XSHG_corr/reg_lowest_0903/')
except:
    pass

output_path = 'XSHG_corr/reg_lowest_0903/'
error=[]
for i in range(len(files_factor)): # file=files_factor[0]
    file = files_factor[i]
    factor = pd.read_csv(factor_file_path+file)
    if factor.empty:
        continue
    quantile_cut(0.01,0.99,factor)    
    normalize(factor)
    y = factor['ret_lowest_fur'].values
    d = {}    
    for col in factor.columns: # col = factor.columns[3]
        if col in ['InstrumentId','datetime','ret_highest_fur','ret_lowest_fur','ret_vwap_fur']:
            continue
        try:
            X = factor[col].values
            X = sm.add_constant(X)
            model = sm.OLS(y,X)
            results = model.fit()
            d[col] = results.params[1],results.tvalues[1],results.pvalues[1]
        except:
            error.append((file,col))
            pass        
    pv2_normalized = pd.DataFrame.from_dict(d,orient='index')
    if pv2_normalized.empty:
        continue
    pv2_normalized.columns = ['coef','t_value','p_value']    
    pv2_normalized.to_csv(output_path+file)
    print(i,end=',')

# =========================================

file_path = 'XSHG_corr/reg_highest_0903/'
files_name = os.listdir(file_path)

reg_mean = pd.read_csv(file_path+files_name[0],index_col=0)
reg_pct = (reg_mean['p_value']<0.01).astype(int)
d = {}
for ind in reg_mean.index:
    d[ind] = 1
for file in files_name[1:]: # file=files_name[0]; file='600817.csv'
    reg = pd.read_csv(file_path+file,index_col=0)
    reg.dropna(inplace=True)
    if reg.empty:
        continue
    #reg_mean = reg_mean.add(reg,fill_value=0)
    reg_pct = reg_pct.add((reg['p_value']<0.01).astype(int),fill_value=0)
    
    for ind in reg.index:
        if ind in d.keys():
            d[ind]+=1
        else:
            d[ind]=1

sorted(d.values())
nind = pd.DataFrame.from_dict(d,orient='index').values
reg_pct/=nind.reshape(-1)

list_fac = list(reg_pct[reg_pct>=0.75].index)

# ========================================= highest
file_path = 'XSHG_factor/0903/'
files_name = os.listdir(file_path)

factor = pd.read_csv(file_path+files_name[0])
fac_corr_mean = factor[list_fac].corr()

n=1
for j in range(1,len(files_name)): # j=0;
    file = files_name[j]
    factor = pd.read_csv(file_path+file)
    if factor.empty:
        continue
    temp = factor[list_fac].corr()
    if temp.isna().sum().sum():
        continue
    n+=1
    fac_corr_mean = fac_corr_mean.add(temp,)
    print(j,end=',')

fac_corr_mean/=n

waitlist = ['midprice_1_highestpre_ratio','weighted_buy_1_highestpre_ratio',
            'weighted_sell_1_highestpre_ratio','midprice_2_highestpre_ratio',
            'midprice_3_highestpre_ratio']
final_fac = list(fac_corr_mean.index)
for i in waitlist:
    final_fac.remove(i)
fac_corr_mean_1 = fac_corr_mean.loc[final_fac,final_fac]

waitlist = ['highest_fur','buy_sell_volume_ratio_log_1','volume_spread_level_ratio_1',
            'volume_imbalance_log_1','weighted_buy_1_vwappre_ratio','midprice_1_vwappre_ratio',
            'weighted_buy_2_vwappre_ratio','weighted_sell_1_vwappre_ratio']
for i in waitlist:
    final_fac.remove(i)
fac_corr_mean_2 = fac_corr_mean.loc[final_fac,final_fac]

waitlist = ['weighted_buy_3_highestpre_ratio','acc_buy_sell_volume_ratio_log_2',
            'acc_volume_imbalance_log_2','acc_volume_spread_2','acc_volume_imbalance_3',
            'acc_buy_sell_volume_ratio_log_3','acc_volume_imbalance_log_3']
for i in waitlist:
    final_fac.remove(i)
fac_corr_mean_3 = fac_corr_mean.loc[final_fac,final_fac]

waitlist = ['weighted_sell_2_mid_ratio','weighted_sell_4_mid_ratio','weighted_sell_5_mid_ratio',
            'weighted_sell_7_mid_ratio','weighted_sell_8_mid_ratio','weighted_sell_9_mid_ratio']
for i in waitlist:
    final_fac.remove(i)
fac_corr_mean_4 = fac_corr_mean.loc[final_fac,final_fac]

waitlist = ['high_minus_low_lag','high_vwap_ratio_lag','bidaskspread_vwap_ratio',
            'bid_ask_spread','midprice_2_lowestpre_ratio','midprice_3_lowestpre_ratio',
            'weighted_midprice_1_lowestpre_ratio']
for i in waitlist:
    final_fac.remove(i)
fac_corr_mean_5 = fac_corr_mean.loc[final_fac,final_fac]

waitlist = ['acc_sell_volume_4','weighted_buy_2_highestpre_ratio',]
for i in waitlist:
    final_fac.remove(i)
fac_corr_mean_6 = fac_corr_mean.loc[final_fac,final_fac]

final_corr_mean = fac_corr_mean_6.copy()
len(final_corr_mean)
len(final_fac)

final_corr_mean.to_csv('0903_final_factor.csv')

#plt.scatter(factor['weighted_midprice_1_highestpre_ratio'],factor['ret_highest_fur'])
#plt.scatter(factor['weighted_midprice_2_highestpre_ratio'],factor['ret_highest_fur'])
#plt.scatter(factor['weighted_midprice_1_vwappre_ratio'],factor['ret_highest_fur'])
#
#
#(factor[factor['high_minus_low_lag']<-1]['ret_highest_fur']>0.002).sum() / (factor['high_minus_low_lag']>-1).sum()
#high_minus_low_lag, ret_highest_fur; scatter
#pv2_normalized = pv2_normalized.sort_index()

# ==========
file_path = 'XSHG_factor/0903/'
files_name = os.listdir(file_path)

factor = pd.read_csv(file_path+files_name[0])
factor['datetime']=pd.to_datetime(factor['datetime'],format='%Y-%m-%d %H:%M:%S')
quantile_cut(0.01,0.99,factor)
normalize(factor)

# OLS
X = factor[final_fac].values
y = factor['ret_highest_fur'].values

X = sm.add_constant(X)
model = sm.OLS(y,X)

results = model.fit()
results.summary()

def myOLS(fac_list,ret):
    X = factor[fac_list].values
    y = factor[ret].values
    
    X = sm.add_constant(X)
    model = sm.OLS(y,X)
    
    results = model.fit()
    return results


resuls = myOLS(final_fac,'ret_highest_fur')
results.summary()
results.pvalues
non_significant = []
for i in range(len(results.pvalues)):
    if results.pvalues[i]>0.1:
        non_significant.append(final_fac[i-1])

#---
temp  = final_fac.copy()
for i in non_significant:
    temp.remove(i)

results_2 = myOLS(temp,'ret_highest_fur')
results_2.summary()
#--- 
temp3  = final_fac.copy()
temp3.remove(final_fac[8])
temp3.remove(final_fac[10])
results_3 = myOLS(temp3,'ret_highest_fur')
results_3.summary() 

temp4 = temp3.copy()
temp4.remove(temp3[4])
results_4 = myOLS(temp4,'ret_highest_fur')
results_4.summary() 

temp5 = temp4.copy()
temp5.remove(temp4[2])
results_5 = myOLS(temp5,'ret_highest_fur')
results_5.summary() 

temp6 = temp5.copy()
temp6.remove(temp5[9])
results_6 = myOLS(temp6,'ret_highest_fur')
results_6.summary() 

temp7 = temp6.copy()
temp7.remove(temp6[10])
results_7 = myOLS(temp7,'ret_highest_fur')
results_7.summary() #***

removed_fac_ind = []
removed_fac = []
for i in range(len(final_fac)):
    if final_fac[i] not in temp7:
        removed_fac_ind.append(i)
        removed_fac.append(final_fac[i])
removed_fac_ind
removed_fac

temp8  = final_fac.copy()
temp8.remove(final_fac[0])
temp8.remove(final_fac[2])
temp8.remove(final_fac[4])
temp8.remove(final_fac[15])
results_8 = myOLS(temp8,'ret_highest_fur')
results_8.summary() 


# Lasso

d={}
for alpha in [1e-6,1e-5,1e-4,0.001,0.01,0.1,1.0]: # alpha =0.0001
    model = Lasso(alpha=alpha, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=10000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')

    X = factor[final_fac].values
    y = factor['ret_highest_fur'].values
    model.fit(X,y)
    
    d[alpha] = model.coef_

np.where(d[1e-05]==0)
for i in np.where(d[1e-05]==0)[0]:
    print(final_fac[i])

# LassoCV
model = LassoCV(eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, normalize=False, precompute='auto', max_iter=10000, tol=0.0001, copy_X=True, cv=10, verbose=False, n_jobs=None, positive=False, random_state=None, selection='cyclic')
X = factor[final_fac][:100].values
y = factor['ret_highest_fur'][:100].values
model.fit(X,y)

model.coef_
model.alpha_
model.alphas_
model.dual_gap_
model.n_iter_

factor['datetime'][100]




np.where(model.coef_==0)
for i in np.where(model.coef_==0)[0]:
    print(final_fac[i])


# ==============================

fac_reg = factor[['datetime']+temp7+['ret_highest_fur']]
time = fac_reg['datetime'][0]
mtime1 = dt.datetime(year=time.year,month=time.month,day=time.day,hour=9,minute=30)
mtime2 = dt.datetime(year=time.year,month=time.month,day=time.day,hour=11,minute=30)
mtime3 = dt.datetime(year=time.year,month=time.month,day=time.day,hour=13,minute=00)
mtime4 = dt.datetime(year=time.year,month=time.month,day=time.day,hour=14,minute=57)

d={}
for i in range(len(fac_reg['datetime'])): # i=100;
    time = fac_reg['datetime'][i]
    if time - dt.timedelta(minutes=43)<mtime1:
        continue
    if time - dt.timedelta(minutes=43)<mtime3 and time>mtime2-dt.timedelta(minutes=1):
        continue
    if time>mtime4-dt.timedelta(minutes=1):
        break
    train = fac_reg[(fac_reg['datetime']<time-dt.timedelta(minutes=1)) & (fac_reg['datetime']>time-dt.timedelta(minutes=41))]   
    X_train = train[temp7].values
    y_train = train['ret_highest_fur'].values
    model = LassoCV(eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, normalize=False, precompute='auto', max_iter=10000, tol=0.0001, copy_X=True, cv=5, verbose=False, n_jobs=None, positive=False, random_state=None, selection='cyclic')
    model.fit(X_train,y_train)
#    model = sm.OLS(y_train,X_train)
#    results = model.fit()
    
    test = fac_reg[fac_reg['datetime']==time]
    X_test = test[temp7].values
    d[time]={}
    d[time]['real'] = test['ret_highest_fur'].values[0]
#    d[time]['pred'] = results.predict(X_test)[0]
    d[time]['pred'] = model.predict(X_test)[0]
        
    if not i%100:
        print(time,end=',')

model.coef_
pred_20 = pred.copy()
pred_30 = pred.copy()
pred_60 = pred.copy()
pred = pd.DataFrame.from_dict(d,orient='index')

pred[pred['pred']>0.0020]


X_temp = pred['pred'].values
X_temp = sm.add_constant(X_temp)
y_temp = pred['real'].values
model_temp = sm.OLS(y_temp,X_temp)
results = model_temp.fit()
results.summary()

plt.scatter(pred.sort_values('pred')['pred'],pred.sort_values('pred')['real'])
plt.plot([0,0.0025],[0,0.0025],color='r')
plt.xlim([-0.001,0.003])
plt.ylim([-0.001,0.008])
plt.xlabel('pred')
plt.ylabel('real')

(pred['pred']>0.0020).sum()
((pred['pred']>0.0020) & (pred['real']>0.0020)).sum()


(pred['pred']>0.0025).sum()
(pred['real']>0.0025).sum()
((pred['pred']>0.0025) & (pred['real']>0.0025)).sum()


(pred['pred']<0.0025).sum()
((pred['pred']<0.0025) & (pred['real']<0.0025)).sum()

(pred['real']>0.0025).sum()

plt.plot(pred['real'])

plt.plot(pred['pred'])



X = factor[final_fac].values
y = factor['ret_highest_fur'].values

X = sm.add_constant(X)
model = sm.OLS(y,X)

results = model.fit()
results.summary()

# 只对收益率高的部分回归



# ==================================
factor = pd.read_csv(file_path+files_name[0])
factor['datetime']=pd.to_datetime(factor['datetime'],format='%Y-%m-%d %H:%M:%S')
quantile_cut(0.01,0.99,factor)

factor_high = factor[factor['ret_vwap_fur']>0.002]

d={}
d['ret_vwap_fur']={}
for col in factor.columns.drop(['InstrumentId','datetime']):
    d['ret_vwap_fur'][col] = np.corrcoef(factor['ret_vwap_fur'],factor[col])[0][1]
corr_mean_temp = pd.DataFrame.from_dict(d)


l = abs(corr_mean_temp).sort_values('ret_vwap_fur',ascending=False).index.values[2:]

np.where(l=='total_trade_volume_lag')
np.where(l=='trade_volume_spread_lag')
for i in l:
    if 'trade_volume' in i:
        print(i)

for i in range(71,81): #i=210
    plt.scatter(factor[l[i]],factor['ret_vwap_fur'])
    plt.scatter(factor_high[l[i]],factor_high['ret_vwap_fur'],c='r')
    plt.xlim(min(factor[l[i]])*0.9,max(factor[l[i]])*1.1)
    plt.ylim(min(factor['ret_vwap_fur'])-0.0005,max(factor['ret_vwap_fur'])+0.0005)
#    plt.xlim(min(factor_high[l[i]])*0.9,max(factor_high[l[i]])*1.1)
#    plt.ylim(min(factor_high['ret_vwap_fur'])-0.0005,max(factor_high['ret_vwap_fur'])+0.0005)
    plt.title(l[i])
    plt.show()



abs(corr_mean_temp).sort_values('ret_highest_fur',ascending=False)['ret_highest_fur'][70]


