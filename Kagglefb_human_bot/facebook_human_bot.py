
# coding: utf-8

# In[2]:

## importing libraries
import sys
import pandas as pd
import numpy as np
import os
from __future__ import division  ### to make sure that division results in float numbers


# In[3]:

## setting the working directory
os.chdir('/local/common/pravengs/dwpublish/MISC/Facebook_Human_Bot')
os.getcwd()


# In[7]:

##loading the data
test_data=pd.read_csv('test.csv') ##test data
train_data=pd.read_csv('train.csv') ## train data
bids_data=pd.read_csv('bids.csv') ## bids data


# In[8]:

## rows and cols of dataframes
print (test_data.shape,train_data.shape,bids_data.shape)


# In[9]:

## finding the count of unique items in each columns of dataframes
test_cols= test_data.columns
train_cols=train_data.columns
bids_cols= bids_data.columns


# In[10]:

uc_test=[]
uc_train=[]
uc_bids=[]
for i in test_cols:
    uc_test.append((i,len(test_data[i].unique())))

for i in train_cols:
    uc_train.append((i,len(train_data[i].unique())))

for i in bids_cols:
    uc_bids.append((i,len(bids_data[i].unique())))


# In[373]:

## printing the unique counts agsinst each variable in that datasets
print 'Test'
print uc_test
print '\n'
print 'Train'
print uc_train
print '\n'
print 'Bids'
print uc_bids


# In[374]:

print test_data.dtypes
print train_data.dtypes
print bids_data.dtypes


# In[12]:

bids_data.head(5)


# In[379]:

## merging the train with bids and test with bids
train=pd.merge(train_data, bids_data, left_on='bidder_id', right_on='bidder_id', how='left')
test=pd.merge(test_data, bids_data, left_on='bidder_id', right_on='bidder_id', how='left')


# In[380]:

print train.shape
print test.shape


# In[381]:

##event rate of train data
print 'Frequencies'
print train_data.outcome.value_counts()
print '\n'
print 'Proportions'
print train_data.outcome.value_counts()/len( train_data.outcome)


# In[17]:

## finding the count of unique items in each columns of dataframes
test_col= test.columns
train_col=train.columns


uc_test1=[]
uc_train1=[]

for i in test_col:
    uc_test1.append((i,len(test[i].unique())))

for i in train_col:
    uc_train1.append((i,len(train[i].unique())))


# In[18]:

## printing the unique counts against each variable in that datasets
print 'Test'
print uc_test1
print '\n'
print 'Train'
print uc_train1


# In[458]:

## Crosstab of outcome and merchandise
pd.crosstab(index=train['merchandise'], columns=train['outcome'], margins=True)


# In[467]:

train_ct1=pd.crosstab(index=train['merchandise'], columns=train['outcome']).apply(lambda r: r/r.sum(), axis=1) 


# In[468]:

train_ct1.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# In[460]:

train.outcome.value_counts()/len(train.outcome)


# In[471]:

train.merchandise.value_counts()


# In[477]:

train.merchandise.value_counts()


# In[472]:

##New feature creation in train
train['is_sporting_goods']=train['merchandise'].apply(lambda x:1 if x=='sporting goods' else 0)
train['is_mobile']=train['merchandise'].apply(lambda x:1 if x=='mobile' else 0)
train['is_home_goods']=train['merchandise'].apply(lambda x:1 if x=='home goods ' else 0)
train['is_office_equipment']=train['merchandise'].apply(lambda x:1 if x=='office equipment' else 0)
train['is_furniture']=train['merchandise'].apply(lambda x:1 if x=='furniture' else 0)
train['is_computers']=train['merchandise'].apply(lambda x:1 if x=='computers' else 0)
train['is_books_music']=train['merchandise'].apply(lambda x:1 if x=='books and music' else 0)
train['is_auto_parts']=train['merchandise'].apply(lambda x:1 if x=='auto parts' else 0)
train['is_clothing']=train['merchandise'].apply(lambda x:1 if x=='clothing' else 0)


# In[478]:

##New feature creation in test
test['is_sporting_goods']=test['merchandise'].apply(lambda x:1 if x=='sporting goods' else 0)
test['is_mobile']=test['merchandise'].apply(lambda x:1 if x=='mobile' else 0)
test['is_home_goods']=test['merchandise'].apply(lambda x:1 if x=='home goods ' else 0)
test['is_office_equipment']=test['merchandise'].apply(lambda x:1 if x=='office equipment' else 0)
test['is_furniture']=test['merchandise'].apply(lambda x:1 if x=='furniture' else 0)
test['is_computers']=test['merchandise'].apply(lambda x:1 if x=='computers' else 0)
test['is_books_music']=test['merchandise'].apply(lambda x:1 if x=='books and music' else 0)
test['is_auto_parts']=test['merchandise'].apply(lambda x:1 if x=='auto parts' else 0)
test['is_clothing']=test['merchandise'].apply(lambda x:1 if x=='clothing' else 0)


# In[475]:

print train['is_sporting_goods'].value_counts()
print train['is_mobile'].value_counts()
print train['is_home_goods'].value_counts()
print train['is_office_equipment'].value_counts()
print train['is_furniture'].value_counts()
print train['is_computers'].value_counts()
print train['is_books_music'].value_counts()
print train['is_auto_parts'].value_counts()
print train['is_clothing'].value_counts()


# In[474]:

pd.crosstab(index=train['merchandise'], columns=train['outcome'])


# In[22]:

# sorting the merged datasets by the variable time in ascending order 
train_sort=train.sort(['bidder_id','payment_account','address','auction','bid_id','time'])
test_sort=test.sort(['bidder_id','payment_account','address','auction','bid_id','time'])


# In[30]:

# creating the time lag variable in train
train_sort['time_lag1']=train_sort.groupby(['bidder_id','payment_account','address','auction'])['time'].shift(1)


# In[33]:

# creating the time lag difference variable in train
train_sort['time_diff1']=train_sort['time']-train_sort['time_lag1']


# In[407]:

# creating the time lag variable in test
test_sort['time_lag1']=test_sort.groupby(['bidder_id','payment_account','address','auction'])['time'].shift(1)


# In[408]:

# creating the time lag differnce variable in test
test_sort['time_diff1']=test_sort['time']-test_sort['time_lag1']


# In[36]:

print train_sort.time_diff1.describe()
print train_sort.time.describe()
print train_sort.time_lag1.describe()


# In[410]:

print test_sort.time_diff1.describe()
print test_sort.time.describe()
print test_sort.time_lag1.describe()


# In[39]:

print train_sort.time_diff1.isnull().sum()/len(train_sort.time_diff1)
print len(train_sort.time_diff1)
print len(train_sort.time)


# In[411]:

print test_sort.time_diff1.isnull().sum()/len(test_sort.time_diff1)
print len(test_sort.time_diff1)
print len(test_sort.time)


# In[31]:

train_sort.shape


# In[252]:

#replacing the NaN with zero in train
train_sort.replace(np.nan, 0,inplace=True)
print ""


# In[298]:

## creating a groupby datasets for train
gp_train_time=train_sort.groupby(['bidder_id','payment_account','address','auction','outcome'])


# In[413]:

## creating a groupby datasets for test
gp_test_time=test_sort.groupby(['bidder_id','payment_account','address','auction'])


# In[299]:

f_time={'time_diff1':{'time_diff_avg':'mean'}}


# In[300]:

#Finding the mean of the average time differnce per acution in train
gp_train_time_agg=gp_train_time.agg(f_time)


# In[414]:

#Finding the mean of the average time differnce per acution in test
gp_test_time_agg=gp_test_time.agg(f_time)


# In[301]:

gp_train_time_agg.shape


# In[415]:

gp_test_time_agg.shape


# In[302]:

gp_train_time_agg.columns


# In[304]:

# dropping the unwanted levels in train
gp_train_time_agg.columns=gp_train_time_agg.columns.droplevel([0])
## resetting the index to bring it to level with all columns
train_time_agg=gp_train_time_agg.reset_index()


# In[417]:

# dropping the unwanted levels in test
gp_test_time_agg.columns=gp_test_time_agg.columns.droplevel([0])
## resetting the index to bring it to level with all columns
test_time_agg=gp_test_time_agg.reset_index()


# In[418]:

train_time_agg.columns


# In[419]:

test_time_agg.columns


# In[308]:

## fucntion dictiponary for finding the mean, min, max, median of avg_time_differnce
f_time_diff={'time_diff_avg':
             {'mean_time_diff_avg':'mean'
              ,'min_time_diff_avg':'min'
              ,'max_time_diff_avg':'max'
              ,'median_time_diff_avg':'median'
             }}


# In[309]:

#Applying the aggregate functions for train
train_time_diff_var_df=train_time_agg.groupby(['bidder_id','payment_account','address','outcome']).agg(f_time_diff)


# In[421]:

#Applying the aggregate functions for test
test_time_diff_var_df=test_time_agg.groupby(['bidder_id','payment_account','address']).agg(f_time_diff)


# In[311]:

## dropping the unwanted levels in train
train_time_diff_var_df.columns=train_time_diff_var_df.columns.droplevel([0])
train_time_diff_var_df=train_time_diff_var_df.reset_index()


# In[423]:

## dropping the unwanted levels in test
test_time_diff_var_df.columns=test_time_diff_var_df.columns.droplevel([0])
test_time_diff_var_df=test_time_diff_var_df.reset_index()


# In[277]:

## histogram for time difference  in Train
plt.figure();
train_time_agg.time_diff_avg.plot(kind='hist',bins=100)
#plt.title(i)
plt.show()


# In[427]:

## histogram for time difference in Test
plt.figure();
test_time_agg.time_diff_avg.plot(kind='hist',bins=100)
#plt.title(i)
plt.show()


# In[488]:

gp_train=train.groupby(['bidder_id','payment_account','address','outcome'])


# In[489]:

gp_test=test.groupby(['bidder_id','payment_account','address'])


# In[490]:

## creating aggregate functions for columns concerned
f1 ={'bid_id':{'bid_id_count':'count','bid_id_unq':'nunique'}}
f2 ={'auction':{'auction_count':'count','auction_unq':'nunique'}}
f3 ={'merchandise':{'merchandise_count':'count','merchandise_unq':'nunique'}}
f4 ={'device':{'device_count':'count','device_unq':'nunique'}}
f5 ={'country':{'country_count':'count','country_unq':'nunique'}}
f6 ={'ip':{'ip_count':'count','ip_unq':'nunique'}}
f7 ={'url':{'url_count':'count','url_unq':'nunique'}}
f8={'is_sporting_goods':{'is_sporting_goods_count':'sum'}}
f9={'is_mobile':{'is_mobile_count':'sum'}}
f10={'is_home_goods':{'is_home_goods_count':'sum'}}
f11={'is_office_equipment':{'is_office_equipment_count':'sum'}}
f12={'is_furniture':{'is_furniture_count':'sum'}}
f13={'is_computers':{'is_computers_count':'sum'}}
f14={'is_books_music':{'is_books_music_count':'sum'}}
f15={'is_auto_parts':{'is_auto_parts_count':'sum'}}
f16={'is_clothing':{'is_clothing_count':'sum'}}


# In[491]:

## putting all the fucntions in a big dictionary
f={}
for d in [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16]:
    f.update(d)


# In[493]:

## applying the fcuntions to train and test
gp_agg_train=gp_train.agg(f)
gp_agg_test=gp_test.agg(f)


# In[498]:

## dropping the unwanted levels in train and test
gp_agg_train.columns=gp_agg_train.columns.droplevel([0])
gp_agg_test.columns=gp_agg_test.columns.droplevel([0])


# In[500]:

## resetting the index to bring it to level with all columns
train_agg=gp_agg_train.reset_index()
test_agg=gp_agg_test.reset_index()


# In[505]:

##function to prevent the division by zero
def ifnull(var, val):
    if var is None or var is 0:
        return val
    return var


# In[593]:

### TRAIN
## more feature creation from frequency variables 
#per acuction variables num: unq, denom :unq
train_agg['unq_dev_per_unq_auction']=train_agg['device_unq']/ifnull(train_agg['auction_unq'],1)
train_agg['unq_url_per_unq_auction']=train_agg['url_unq']/ifnull(train_agg['auction_unq'],1)
train_agg['unq_country_per_unq_auction']=train_agg['country_unq']/ifnull(train_agg['auction_unq'],1)
train_agg['unq_bid_per_unq_auction']=train_agg['bid_id_unq']/ifnull(train_agg['auction_unq'],1)
train_agg['unq_ip_per_unq_auction']=train_agg['ip_unq']/ifnull(train_agg['auction_unq'],1)

#per acuction variables num: all, denom :unq
train_agg['dev_per_unq_auction']=train_agg['device_count']/ifnull(train_agg['auction_unq'],1)
train_agg['url_per_unq_auction']=train_agg['url_count']/ifnull(train_agg['auction_unq'],1)
train_agg['country_per_unq_auction']=train_agg['country_count']/ifnull(train_agg['auction_unq'],1)
train_agg['bid_per_unq_auction']=train_agg['bid_id_count']/ifnull(train_agg['auction_unq'],1)
train_agg['ip_per_unq_auction']=train_agg['ip_count']/ifnull(train_agg['auction_unq'],1)

## bid device country url ip per unq auction (unq doublets per unq acution)
train_agg['unq_bid_unq_country_unq_auction']=(train_agg['bid_id_unq']/ifnull(train_agg['country_unq'],1))/ifnull(train_agg['auction_unq'],1)
train_agg['unq_bid_unq_device_unq_auction']=(train_agg['bid_id_unq']/ifnull(train_agg['device_unq'],1))/ifnull(train_agg['auction_unq'],1)
train_agg['unq_bid_unq_ip_unq_auction']=(train_agg['bid_id_unq']/ifnull(train_agg['ip_unq'],1))/ifnull(train_agg['auction_unq'],1)
train_agg['unq_bid_unq_url_unq_auction']=(train_agg['bid_id_unq']/ifnull(train_agg['url_unq'],1))/ifnull(train_agg['auction_unq'],1)

train_agg['unq_device_unq_country_unq_auction']=(train_agg['device_unq']/ifnull(train_agg['country_unq'],1))/ifnull(train_agg['auction_unq'],1)
train_agg['unq_device_unq_ip_unq_auction']=(train_agg['device_unq']/ifnull(train_agg['ip_unq'],1))/ifnull(train_agg['auction_unq'],1)
train_agg['unq_device_unq_url_unq_auction']=(train_agg['device_unq']/ifnull(train_agg['url_unq'],1))/ifnull(train_agg['auction_unq'],1)

train_agg['unq_country_unq_ip_unq_auction']=(train_agg['country_unq']/ifnull(train_agg['ip_unq'],1))/ifnull(train_agg['auction_unq'],1)
train_agg['unq_country_unq_url_unq_auction']=(train_agg['country_unq']/ifnull(train_agg['url_unq'],1))/ifnull(train_agg['auction_unq'],1)

train_agg['unq_ip_unq_url_unq_auction']=(train_agg['ip_unq']/ifnull(train_agg['url_unq'],1))/ifnull(train_agg['auction_unq'],1)

## bid device country url ip per unq auction (doublets per unq acution)
train_agg['bid_country_unq_auction']=(train_agg['bid_id_count']/ifnull(train_agg['country_count'],1))/ifnull(train_agg['auction_unq'],1)
train_agg['bid_device_unq_auction']=(train_agg['bid_id_count']/ifnull(train_agg['device_count'],1))/ifnull(train_agg['auction_unq'],1)
train_agg['bid_ip_unq_auction']=(train_agg['bid_id_count']/ifnull(train_agg['ip_count'],1))/ifnull(train_agg['auction_unq'],1)
train_agg['bid_url_unq_auction']=(train_agg['bid_id_count']/ifnull(train_agg['url_count'],1))/ifnull(train_agg['auction_unq'],1)

train_agg['device_country_unq_auction']=(train_agg['device_count']/ifnull(train_agg['country_count'],1))/ifnull(train_agg['auction_unq'],1)
train_agg['device_ip_unq_auction']=(train_agg['device_count']/ifnull(train_agg['ip_count'],1))/ifnull(train_agg['auction_unq'],1)
train_agg['device_url_unq_auction']=(train_agg['device_count']/ifnull(train_agg['url_count'],1))/ifnull(train_agg['auction_unq'],1)

train_agg['country_ip_unq_auction']=(train_agg['country_count']/ifnull(train_agg['ip_count'],1))/ifnull(train_agg['auction_unq'],1)
train_agg['country_url_unq_auction']=(train_agg['country_count']/ifnull(train_agg['url_count'],1))/ifnull(train_agg['auction_unq'],1)

train_agg['ip_url_unq_auction']=(train_agg['ip_count']/ifnull(train_agg['url_count'],1))/ifnull(train_agg['auction_unq'],1)


#per unq auction merchandise types

train_agg['is_sporting_goods_per_unq_auction']=train_agg['is_sporting_goods_count']/ifnull(train_agg['auction_unq'],1)
train_agg['is_mobile_per_unq_auction']=train_agg['is_mobile_count']/ifnull(train_agg['auction_unq'],1)
train_agg['is_home_goods_per_unq_auction']=train_agg['is_home_goods_count']/ifnull(train_agg['auction_unq'],1)
train_agg['is_office_equipment_per_unq_auction']=train_agg['is_office_equipment_count']/ifnull(train_agg['auction_unq'],1)
train_agg['is_furniture_per_unq_auction']=train_agg['is_furniture_count']/ifnull(train_agg['auction_unq'],1)
train_agg['is_computers_per_unq_auction']=train_agg['is_computers_count']/ifnull(train_agg['auction_unq'],1)
train_agg['is_books_music_per_unq_auction']=train_agg['is_books_music_count']/ifnull(train_agg['auction_unq'],1)
train_agg['is_auto_parts_per_unq_auction']=train_agg['is_auto_parts_count']/ifnull(train_agg['auction_unq'],1)
train_agg['is_clothing_per_unq_auction']=train_agg['is_clothing_count']/ifnull(train_agg['auction_unq'],1)

#per  auction merchandise types
train_agg['is_sporting_goods_per_auction']=train_agg['is_sporting_goods_count']/ifnull(train_agg['auction_count'],1)
train_agg['is_mobile_per_auction']=train_agg['is_mobile_count']/ifnull(train_agg['auction_count'],1)
train_agg['is_home_goods_per_auction']=train_agg['is_home_goods_count']/ifnull(train_agg['auction_count'],1)
train_agg['is_office_equipment_per_auction']=train_agg['is_office_equipment_count']/ifnull(train_agg['auction_count'],1)
train_agg['is_furniture_per_auction']=train_agg['is_furniture_count']/ifnull(train_agg['auction_count'],1)
train_agg['is_computers_per_auction']=train_agg['is_computers_count']/ifnull(train_agg['auction_count'],1)
train_agg['is_books_music_per_auction']=train_agg['is_books_music_count']/ifnull(train_agg['auction_count'],1)
train_agg['is_auto_parts_per_auction']=train_agg['is_auto_parts_count']/ifnull(train_agg['auction_count'],1)
train_agg['is_clothing_per_auction']=train_agg['is_clothing_count']/ifnull(train_agg['auction_count'],1)


# In[594]:

## TEST
## more feature creation from frequency variables 
#per acuction variables num: unq, denom :unq
test_agg['unq_dev_per_unq_auction']=test_agg['device_unq']/ifnull(test_agg['auction_unq'],1)
test_agg['unq_url_per_unq_auction']=test_agg['url_unq']/ifnull(test_agg['auction_unq'],1)
test_agg['unq_country_per_unq_auction']=test_agg['country_unq']/ifnull(test_agg['auction_unq'],1)
test_agg['unq_bid_per_unq_auction']=test_agg['bid_id_unq']/ifnull(test_agg['auction_unq'],1)
test_agg['unq_ip_per_unq_auction']=test_agg['ip_unq']/ifnull(test_agg['auction_unq'],1)

#per acuction variables num: all, denom :unq
test_agg['dev_per_unq_auction']=test_agg['device_count']/ifnull(test_agg['auction_unq'],1)
test_agg['url_per_unq_auction']=test_agg['url_count']/ifnull(test_agg['auction_unq'],1)
test_agg['country_per_unq_auction']=test_agg['country_count']/ifnull(test_agg['auction_unq'],1)
test_agg['bid_per_unq_auction']=test_agg['bid_id_count']/ifnull(test_agg['auction_unq'],1)
test_agg['ip_per_unq_auction']=test_agg['ip_count']/ifnull(test_agg['auction_unq'],1)

## bid device country url ip per unq auction (unq doublets per unq acution)
test_agg['unq_bid_unq_country_unq_auction']=(test_agg['bid_id_unq']/ifnull(test_agg['country_unq'],1))/ifnull(test_agg['auction_unq'],1)
test_agg['unq_bid_unq_device_unq_auction']=(test_agg['bid_id_unq']/ifnull(test_agg['device_unq'],1))/ifnull(test_agg['auction_unq'],1)
test_agg['unq_bid_unq_ip_unq_auction']=(test_agg['bid_id_unq']/ifnull(test_agg['ip_unq'],1))/ifnull(test_agg['auction_unq'],1)
test_agg['unq_bid_unq_url_unq_auction']=(test_agg['bid_id_unq']/ifnull(test_agg['url_unq'],1))/ifnull(test_agg['auction_unq'],1)

test_agg['unq_device_unq_country_unq_auction']=(test_agg['device_unq']/ifnull(test_agg['country_unq'],1))/ifnull(test_agg['auction_unq'],1)
test_agg['unq_device_unq_ip_unq_auction']=(test_agg['device_unq']/ifnull(test_agg['ip_unq'],1))/ifnull(test_agg['auction_unq'],1)
test_agg['unq_device_unq_url_unq_auction']=(test_agg['device_unq']/ifnull(test_agg['url_unq'],1))/ifnull(test_agg['auction_unq'],1)

test_agg['unq_country_unq_ip_unq_auction']=(test_agg['country_unq']/ifnull(test_agg['ip_unq'],1))/ifnull(test_agg['auction_unq'],1)
test_agg['unq_country_unq_url_unq_auction']=(test_agg['country_unq']/ifnull(test_agg['url_unq'],1))/ifnull(test_agg['auction_unq'],1)

test_agg['unq_ip_unq_url_unq_auction']=(test_agg['ip_unq']/ifnull(test_agg['url_unq'],1))/ifnull(test_agg['auction_unq'],1)

## bid device country url ip per unq auction (doublets per unq acution)
test_agg['bid_country_unq_auction']=(test_agg['bid_id_count']/ifnull(test_agg['country_count'],1))/ifnull(test_agg['auction_unq'],1)
test_agg['bid_device_unq_auction']=(test_agg['bid_id_count']/ifnull(test_agg['device_count'],1))/ifnull(test_agg['auction_unq'],1)
test_agg['bid_ip_unq_auction']=(test_agg['bid_id_count']/ifnull(test_agg['ip_count'],1))/ifnull(test_agg['auction_unq'],1)
test_agg['bid_url_unq_auction']=(test_agg['bid_id_count']/ifnull(test_agg['url_count'],1))/ifnull(test_agg['auction_unq'],1)

test_agg['device_country_unq_auction']=(test_agg['device_count']/ifnull(test_agg['country_count'],1))/ifnull(test_agg['auction_unq'],1)
test_agg['device_ip_unq_auction']=(test_agg['device_count']/ifnull(test_agg['ip_count'],1))/ifnull(test_agg['auction_unq'],1)
test_agg['device_url_unq_auction']=(test_agg['device_count']/ifnull(test_agg['url_count'],1))/ifnull(test_agg['auction_unq'],1)

test_agg['country_ip_unq_auction']=(test_agg['country_count']/ifnull(test_agg['ip_count'],1))/ifnull(test_agg['auction_unq'],1)
test_agg['country_url_unq_auction']=(test_agg['country_count']/ifnull(test_agg['url_count'],1))/ifnull(test_agg['auction_unq'],1)

test_agg['ip_url_unq_auction']=(test_agg['ip_count']/ifnull(test_agg['url_count'],1))/ifnull(test_agg['auction_unq'],1)


#per unq auction merchandise types

test_agg['is_sporting_goods_per_unq_auction']=test_agg['is_sporting_goods_count']/ifnull(test_agg['auction_unq'],1)
test_agg['is_mobile_per_unq_auction']=test_agg['is_mobile_count']/ifnull(test_agg['auction_unq'],1)
test_agg['is_home_goods_per_unq_auction']=test_agg['is_home_goods_count']/ifnull(test_agg['auction_unq'],1)
test_agg['is_office_equipment_per_unq_auction']=test_agg['is_office_equipment_count']/ifnull(test_agg['auction_unq'],1)
test_agg['is_furniture_per_unq_auction']=test_agg['is_furniture_count']/ifnull(test_agg['auction_unq'],1)
test_agg['is_computers_per_unq_auction']=test_agg['is_computers_count']/ifnull(test_agg['auction_unq'],1)
test_agg['is_books_music_per_unq_auction']=test_agg['is_books_music_count']/ifnull(test_agg['auction_unq'],1)
test_agg['is_auto_parts_per_unq_auction']=test_agg['is_auto_parts_count']/ifnull(test_agg['auction_unq'],1)
test_agg['is_clothing_per_unq_auction']=test_agg['is_clothing_count']/ifnull(test_agg['auction_unq'],1)

#per  auction merchandise types
test_agg['is_sporting_goods_per_auction']=test_agg['is_sporting_goods_count']/ifnull(test_agg['auction_count'],1)
test_agg['is_mobile_per_auction']=test_agg['is_mobile_count']/ifnull(test_agg['auction_count'],1)
test_agg['is_home_goods_per_auction']=test_agg['is_home_goods_count']/ifnull(test_agg['auction_count'],1)
test_agg['is_office_equipment_per_auction']=test_agg['is_office_equipment_count']/ifnull(test_agg['auction_count'],1)
test_agg['is_furniture_per_auction']=test_agg['is_furniture_count']/ifnull(test_agg['auction_count'],1)
test_agg['is_computers_per_auction']=test_agg['is_computers_count']/ifnull(test_agg['auction_count'],1)
test_agg['is_books_music_per_auction']=test_agg['is_books_music_count']/ifnull(test_agg['auction_count'],1)
test_agg['is_auto_parts_per_auction']=test_agg['is_auto_parts_count']/ifnull(test_agg['auction_count'],1)
test_agg['is_clothing_per_auction']=test_agg['is_clothing_count']/ifnull(test_agg['auction_count'],1)


# In[597]:

##merging the new features in train
train3=pd.merge(train_agg,train_time_diff_var_df
                , left_on=['bidder_id', 'payment_account', 'address', 'outcome']
                , right_on=['bidder_id', 'payment_account', 'address', 'outcome']
                , how='left')


# In[598]:

##merging the new features in test
test3=pd.merge(test_agg,test_time_diff_var_df
                , left_on=['bidder_id', 'payment_account', 'address' ]
                , right_on=['bidder_id', 'payment_account', 'address' ]
                , how='left')


# In[602]:

#replacing inf and nan with zero in train
train3.replace([np.inf, -np.inf,np.nan], 0,inplace=True)
print  "\n";


# In[603]:

#replacing inf and nan with zero in test
test3.replace([np.inf, -np.inf,np.nan], 0,inplace=True)
print  "\n";


# In[876]:

## plotting the ROC Curve

## AUC for train 
fpr, tpr, thresholds = roc_curve(train3['outcome'], train_pred_prob)
roc_auc = auc(fpr, tpr)
print "Area under the ROC curve : %f" % roc_auc
print fpr,tpr,thresholds                                        


# In[663]:

# Plot ROC curve
plt.clf()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()


# In[873]:

### putting it all together in a function
from sklearn import ensemble
from sklearn.metrics import roc_curve, auc
from sklearn import cross_validation

def rf_model(X,y,train_ds, test_ds):
    
    ## initial random forest classifier with full train data
    clf = ensemble.RandomForestClassifier(n_estimators=1000,random_state=9876,n_jobs=16)
    clf.fit(train_ds[X], train_ds[y])
    
    ## predicting class on train and test
    train_pred_class=clf.predict(train_ds[X])
    test_pred_class=clf.predict(test_ds[X])

    ## predicting probabilities on train and test
    train_pred_prob=clf.predict_proba(train_ds[X])
    test_pred_prob=clf.predict_proba(test_ds[X])

    ##taking the probabilities for predicted class=1 (2 nd column in the array)
    train_pred_prob=train_pred_prob[:,1]
    test_pred_prob=test_pred_prob[:,1]
    
    ## creating the confusion matrix for train
    cf_mat_train=pd.crosstab(train_ds[y], train_pred_class, rownames=['actual'], colnames=['preds'])

    ## train metrics
    rf_train_err=(cf_mat_train.iloc[0,1]+cf_mat_train.iloc[1,0])/(cf_mat_train.iloc[0,1]+cf_mat_train.iloc[1,0]+cf_mat_train.iloc[0,0]+cf_mat_train.iloc[1,1])  ## error rate
    rf_train_acc=(cf_mat_train.iloc[0,0]+cf_mat_train.iloc[1,1])/(cf_mat_train.iloc[0,1]+cf_mat_train.iloc[1,0]+cf_mat_train.iloc[0,0]+cf_mat_train.iloc[1,1]) ## accuracy
    rf_train_recall =cf_mat_train.iloc[1,1]/(cf_mat_train.iloc[1,1]+cf_mat_train.iloc[1,0]) ###recall  or hit rate or tpr or sensitivity
    rf_train_spc=cf_mat_train.iloc[0,0]/(cf_mat_train.iloc[0,0]+cf_mat_train.iloc[0,1])  ##tnr or specificity
    rf_train_prec=cf_mat_train.iloc[1,1]/(cf_mat_train.iloc[1,1]+cf_mat_train.iloc[0,1]) ### precision  or positive predicted value(ppv) 
    rf_train_npv =cf_mat_train.iloc[0,0]/(cf_mat_train.iloc[0,0]+cf_mat_train.iloc[1,0]) ###negative predicted value
    rf_train_fpr =cf_mat_train.iloc[0,1]/(cf_mat_train.iloc[0,0]+cf_mat_train.iloc[0,1]) ###false positive rate or fall out  
    rf_train_fdr =cf_mat_train.iloc[0,1]/(cf_mat_train.iloc[0,1]+cf_mat_train.iloc[1,1]) ###false discovery rate
    rf_train_fnr =cf_mat_train.iloc[1,0]/(cf_mat_train.iloc[1,0]+cf_mat_train.iloc[1,1]) ###false negative rate

    train_met_dict={
        "accuracy":rf_train_acc*100
        ,"error":rf_train_err*100
        ,"precision":rf_train_prec*100
        ,"recall":rf_train_recall*100
        ,"FDR":rf_train_fdr*100
        ,"FNR":rf_train_fnr*100
    }
    ## feature importance
    feat_index = np.argsort(clf.feature_importances_)[::-1] ## sorting the indices of feature imporatnce in decending order
    fet_imp = clf.feature_importances_[feat_index] ##using the descending sorted index and arranging the feature importance array 
    
    fet_imp_names = [X[i] for i in feat_index] ## collecting the feature names from the index
    
    ##Putting the sorted feature importance and feature names in a dataframe
    d = {'v_imp_names': pd.Series(fet_imp_names),
         'v_imp_values': pd.Series(fet_imp)
        }
    v_imp_df = pd.DataFrame(d)
    
    #train AUC
    fpr, tpr, thresholds = roc_curve(train_ds[y], train_pred_prob)
    roc_auc = auc(fpr, tpr)
    ret_dict={"train_pred_class":train_pred_class
              ,"test_pred_class":test_pred_class
              ,"train_pred_prob":train_pred_prob
              ,"test_pred_prob":test_pred_prob
              ,"cf_mat_train":cf_mat_train
              ,"train_met_dict":train_met_dict
              ,"v_imp_df":v_imp_df
              ,"train_auc":roc_auc
              ,"fpr_auc":fpr
              ,"tpr_auc":tpr
              ,"thresholds_auc":thresholds
             }
    return ret_dict


# In[874]:

fet1=train3.columns[4:].tolist() ##should be passed as list
y='outcome' ## should be passed as a string


# In[875]:

rf1=rf_model(fet1,y,train3,test3)


# In[761]:

rf1['train_auc']


# In[763]:

rf1['cf_mat_train']


# In[764]:

rf1['v_imp_df']


# In[783]:

##event rate of train data
print 'Train Actual Event'
print train3.outcome.value_counts()
print train3.outcome.value_counts()/len( train3.outcome)
print '\n'
print 'TRAIN predited events'
print pd.DataFrame(rf1['train_pred_class']).iloc[:,0].value_counts()
print pd.DataFrame(rf1['train_pred_class']).iloc[:,0].value_counts()/len(pd.DataFrame(rf1['train_pred_class']).iloc[:,0])
print '\n'

print 'TEST Predicted events'
print pd.DataFrame(rf1['test_pred_class']).iloc[:,0].value_counts()
print pd.DataFrame(rf1['test_pred_class']).iloc[:,0].value_counts()/len(pd.DataFrame(rf1['test_pred_class']).iloc[:,0])
print '\n'


# In[773]:

## creating a second features from only the imporatne variables from First features
fet2=rf1['v_imp_df']['v_imp_names'][rf1['v_imp_df']['v_imp_values']>=0.01].tolist()


# In[774]:

rf2=rf_model(fet2,y,train3,test3)


# In[775]:

rf2['train_auc']


# In[776]:

rf2['cf_mat_train']


# In[784]:

##event rate of train data
print 'Train Actual Event'
print train3.outcome.value_counts()
print train3.outcome.value_counts()/len( train3.outcome)
print '\n'
print 'TRAIN predited events'
print pd.DataFrame(rf2['train_pred_class']).iloc[:,0].value_counts()
print pd.DataFrame(rf2['train_pred_class']).iloc[:,0].value_counts()/len(pd.DataFrame(rf2['train_pred_class']).iloc[:,0])
print '\n'

print 'TEST Predicted events'
print pd.DataFrame(rf2['test_pred_class']).iloc[:,0].value_counts()
print pd.DataFrame(rf2['test_pred_class']).iloc[:,0].value_counts()/len(pd.DataFrame(rf2['test_pred_class']).iloc[:,0])
print '\n'


# In[770]:

## creating a third features from only the imporatne variables from First features
fet3=rf1['v_imp_df']['v_imp_names'][rf1['v_imp_df']['v_imp_values']>=0.02].tolist()


# In[785]:

rf3=rf_model(fet3,y,train3,test3)


# In[786]:

print 'Train Actual Event'
print train3.outcome.value_counts()
print train3.outcome.value_counts()/len( train3.outcome)
print '\n'
print 'TRAIN predited events'
print pd.DataFrame(rf3['train_pred_class']).iloc[:,0].value_counts()
print pd.DataFrame(rf3['train_pred_class']).iloc[:,0].value_counts()/len(pd.DataFrame(rf3['train_pred_class']).iloc[:,0])
print '\n'

print 'TEST Predicted events'
print pd.DataFrame(rf3['test_pred_class']).iloc[:,0].value_counts()
print pd.DataFrame(rf3['test_pred_class']).iloc[:,0].value_counts()/len(pd.DataFrame(rf3['test_pred_class']).iloc[:,0])
print '\n'


# In[787]:

## creating a second features from only the imporatne variables from third set of features
## doing feature selection using the variable importance in Random forests
fet4=rf3['v_imp_df']['v_imp_names'][rf3['v_imp_df']['v_imp_values']>=0.02].tolist()


# In[788]:

rf4=rf_model(fet4,y,train3,test3)


# In[790]:

rf4['v_imp_df']


# In[789]:

##event rate of train data
print 'Train Actual Event'
print train3.outcome.value_counts()
print train3.outcome.value_counts()/len( train3.outcome)
print '\n'
print 'TRAIN predited events'
print pd.DataFrame(rf4['train_pred_class']).iloc[:,0].value_counts()
print pd.DataFrame(rf4['train_pred_class']).iloc[:,0].value_counts()/len(pd.DataFrame(rf4['train_pred_class']).iloc[:,0])
print '\n'

print 'TEST Predicted events'
print pd.DataFrame(rf4['test_pred_class']).iloc[:,0].value_counts()
print pd.DataFrame(rf4['test_pred_class']).iloc[:,0].value_counts()/len(pd.DataFrame(rf4['test_pred_class']).iloc[:,0])
print '\n'


# In[804]:

## submission 
sub_7=pd.concat([test3['bidder_id'],pd.DataFrame(rf4['test_pred_prob'])],axis=1)
sub_7.columns=['bidder_id','prediction']
sub_7.to_csv('submission_7.csv',sep=',',index=None)

