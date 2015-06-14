
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

print (test_data.shape,train_data.shape,bids_data.shape)## rows and cols of dataframes


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


# In[13]:

## checking the presence of bidder_ids from test and train data in the bids_data
bidder_id_list=bids_data.bidder_id.unique()
bidder_id_list_train=train_data.bidder_id.unique()
bidder_id_list_test=test_data.bidder_id.unique()


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


# ## the event rate is 5.11%, so the data is a skewed data and hence may have to apply downsampling techniques

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

pd.crosstab(index=train['merchandise'], columns=train['outcome'], margins=True)


# In[467]:

train_ct1=pd.crosstab(index=train['merchandise'], columns=train['outcome']).apply(lambda r: r/r.sum(), axis=1) 


# In[468]:

train_ct1.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


# In[469]:

train_ct2=pd.crosstab(index=train['country'], columns=train['outcome']).apply(lambda r: r/r.sum(), axis=1)
train_ct2.plot(kind='bar', stacked=True, color=['red','blue'], grid=False)


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


# In[476]:

1901529+1169724


# In[474]:

pd.crosstab(index=train['merchandise'], columns=train['outcome'])


# In[ ]:




# In[18]:

train.head(5)


# In[19]:

test.head(5)


# In[352]:

train.columns


# In[22]:

# sorting the merged datasets by the variable time in ascending order 
train_sort=train.sort(['bidder_id','payment_account','address','auction','bid_id','time'])
test_sort=test.sort(['bidder_id','payment_account','address','auction','bid_id','time'])


# In[23]:

train_sort.head(5)


# In[406]:

test_sort.head(5)


# In[29]:

train_sort.columns


# In[30]:

# creating the time lag variable in train
train_sort['time_lag1']=train_sort.groupby(['bidder_id','payment_account','address','auction'])['time'].shift(1)


# In[33]:

# creating the time lag differnce variable in train
train_sort['time_diff1']=train_sort['time']-train_sort['time_lag1']


# In[407]:

# creating the time lag variable in test
test_sort['time_lag1']=test_sort.groupby(['bidder_id','payment_account','address','auction'])['time'].shift(1)


# In[408]:

# creating the time lag differnce variable in test
test_sort['time_diff1']=test_sort['time']-test_sort['time_lag1']


# In[182]:

print train.shape
print train_sort.shape


# In[409]:

print test.shape
print test_sort.shape


# In[180]:

train_sort.head(3)


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


# In[253]:

train_sort.head(5)


# ####  temporary codes
# na_df_train=train_sort[:][train_sort.time_diff1.isnull()==True]
# nonna_df_train=train_sort[:][train_sort.time_diff1.isnull()==False]

# print na_df_train.shape
# print nonna_df_train.shape

# print 124199+2947054
# print train_sort.shape

# 124199/3071253

# na_gp_train_time_wip=na_df_train.groupby(['bidder_id','payment_account','address','auction'])

# nonna_gp_train_time_wip=nonna_df_train.groupby(['bidder_id','payment_account','address','auction'])

# f_time_wip={'bid_id':{'bid_id_count':'count','bid_id_unq':'nunique'}}

# na_gp_train_time_wip_agg=na_gp_train_time_wip.agg(f_time_wip)

# nona_gp_train_time_wip_agg=nonna_gp_train_time_wip.agg(f_time_wip)

# # dropping the unwanted levels in train
# na_gp_train_time_wip_agg.columns=na_gp_train_time_wip_agg.columns.droplevel([0])
# ## resetting the index to bring it to level with all columns
# na_gp_train_time_wip_agg=na_gp_train_time_wip_agg.reset_index()

# # dropping the unwanted levels in train
# nona_gp_train_time_wip_agg.columns=nona_gp_train_time_wip_agg.columns.droplevel([0])
# ## resetting the index to bring it to level with all columns
# nona_gp_train_time_wip_agg=nona_gp_train_time_wip_agg.reset_index()

# na_gp_train_time_wip_agg.head(4)

# nona_gp_train_time_wip_agg.head(4)

# na_gp_train_time_wip_agg.describe()

# nona_gp_train_time_wip_agg.describe()

# ##############################################################

# In[298]:

gp_train_time=train_sort.groupby(['bidder_id','payment_account','address','auction','outcome'])


# In[413]:

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


# In[303]:

gp_train_time_agg.head(5)


# In[416]:

gp_test_time_agg.head(5)


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

## fucntion dictoiany for finding the mean, min, max, median of avg_time_differnce
f_time_diff={'time_diff_avg':
             {'mean_time_diff_avg':'mean'
              ,'min_time_diff_avg':'min'
              ,'max_time_diff_avg':'max'
              ,'median_time_diff_avg':'median'
             }}


# In[307]:

train_time_agg.columns


# In[420]:

test_time_agg.columns


# In[309]:

train_time_diff_var_df=train_time_agg.groupby(['bidder_id','payment_account','address','outcome']).agg(f_time_diff)


# In[421]:

test_time_diff_var_df=test_time_agg.groupby(['bidder_id','payment_account','address']).agg(f_time_diff)


# In[310]:

train_time_diff_var_df.head(5)


# In[422]:

test_time_diff_var_df.head(5)


# In[311]:

## dropping the unwanted levels in train
train_time_diff_var_df.columns=train_time_diff_var_df.columns.droplevel([0])
train_time_diff_var_df=train_time_diff_var_df.reset_index()


# In[423]:

## dropping the unwanted levels in test
test_time_diff_var_df.columns=test_time_diff_var_df.columns.droplevel([0])
test_time_diff_var_df=test_time_diff_var_df.reset_index()


# In[312]:

train_agg=gp_agg_train.reset_index()


# In[424]:

print train_time_diff_var_df.shape
print train_agg.shape
print train_data.shape


# In[425]:

print test_time_diff_var_df.shape
print test_agg.shape
print test_data.shape


# In[315]:

train_agg.head(4)


# In[314]:

train_time_diff_var_df.head(5)


# In[319]:

train_agg.shape


# In[320]:

train_time_diff_var_df.shape


# In[276]:

train_time_agg.time_diff_avg.describe()


# In[426]:

test_time_agg.time_diff_avg.describe()


# In[278]:

train_time_agg.columns


# In[279]:

train_time_agg.shape


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


# In[324]:

train.head(5)


# In[377]:

train.shape


# In[484]:

#dropping time for the time being in train
train_1=train.drop('time',axis=1)


# In[485]:

#dropping time for the time being in test
test_1=test.drop('time',axis=1)


# In[486]:

test_1.columns


# In[487]:

train_1.columns


# In[488]:

gp_train=train_1.groupby(['bidder_id','payment_account','address','outcome'])


# In[489]:

gp_test=test_1.groupby(['bidder_id','payment_account','address'])


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


# In[492]:

f


# In[493]:

gp_agg_train=gp_train.agg(f)
gp_agg_test=gp_test.agg(f)


# In[494]:

gp_agg_train.columns


# In[495]:

gp_agg_test.columns


# In[496]:

gp_agg_train.head(4)


# In[497]:

gp_agg_test.head(4)


# In[498]:

## dropping the unwanted levels in train and test
gp_agg_train.columns=gp_agg_train.columns.droplevel([0])
gp_agg_test.columns=gp_agg_test.columns.droplevel([0])


# In[499]:

gp_agg_train.head(5)


# In[395]:

gp_agg_test.head(5)


# In[500]:

## resetting the index to bring it to level with all columns
train_agg=gp_agg_train.reset_index()
test_agg=gp_agg_test.reset_index()


# In[501]:

train_agg.head(5)


# In[502]:

test_agg.head(5)


# In[43]:

train_agg.columns


# In[141]:

train_agg.head(5)


# In[503]:

test_agg.columns


# In[504]:

train_agg.fillna(0,inplace=True)
test_agg.fillna(0,inplace=True)
print "hello" ## printing this to acvoid the dataset being written to the area


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


# In[595]:

print  test_agg.shape
print test_agg.columns


# In[596]:

print train_agg.shape
print train_agg.columns


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


# In[805]:

train3.describe()


# In[600]:

test3.shape


# In[514]:

train3.columns


# In[515]:

test3.columns


# In[350]:

train3.head(4)


# In[432]:

test3.head(4)


# In[602]:

#replacing inf and nan with zero
train3.replace([np.inf, -np.inf,np.nan], 0,inplace=True)
print  "\n";


# In[603]:

#replacing inf and nan with zero
test3.replace([np.inf, -np.inf,np.nan], 0,inplace=True)
print  "\n";


# In[347]:

train_agg.columns


# In[73]:

pd.set_option('display.max_columns', None)
train_agg.head(4)


# In[143]:

test_agg.head(4)


# In[352]:

train3.head(4)


# In[62]:

len(train_agg.columns)


# In[519]:

train3.iloc[:,4:].describe()


# In[520]:

test3.iloc[:,4:].describe()


# In[151]:

##basic plotting

get_ipython().magic(u'pylab inline')
import matplotlib.pyplot as plt 
matplotlib.style.use('ggplot')


# In[152]:

## histogram for Train 
for i in train_agg.columns[14:]:
    plt.figure();
    #plotting the 95 percentile
    p95=np.percentile( train_agg[i],95)
    train_agg[i][train_agg[i]<=p95].plot(kind='hist')
    plt.title(i)
    plt.show()


# In[153]:

## histogram for Test 
for i in test_agg.columns[14:]:
    plt.figure();
    p95=np.percentile( test_agg[i],95)
    test_agg[i][test_agg[i]<=p95].plot(kind='hist')
    plt.title(i)
    plt.show()


# In[858]:

## initial random forest classifier with full train data
from sklearn import ensemble,preprocessing
clf = ensemble.RandomForestClassifier(n_estimators=1000,random_state=9876,n_jobs=16)
X, y = (train3[features],train3.outcome)
clf.fit(X, y,sample_weight=preprocessing.balance_weights(y))


# In[ ]:

## initial random forest classifier with full train data
from sklearn import ensemble
clf = ensemble.RandomForestClassifier(n_estimators=1000,random_state=9876,n_jobs=16)
X, y = train3[features],train3.outcome
clf.fit(X, y)


# In[628]:

features


# In[629]:

len(features)


# In[859]:

## predicting class on train
train_pred_class=clf.predict(train3[features])


# In[860]:

## predicting class on test
test_pred_class=clf.predict(test3[features])


# In[861]:

## predicting probabilities on train
train_pred_prob=clf.predict_proba(train3[features])


# In[862]:

## predicting probabilities on test
test_pred_prob=clf.predict_proba(test3[features])


# In[863]:

##taking the probabilities for predicted class=1 (2 nd column in the array)
train_pred_prob=train_pred_prob[:,1]
test_pred_prob=test_pred_prob[:,1]


# In[864]:

train_pred_prob.shape


# In[579]:

type(train_pred_prob)


# In[865]:

## creating the confusion matrix for train
cf_mat_train=pd.crosstab(train3['outcome'], train_pred_class, rownames=['actual'], colnames=['preds'])


# In[866]:

cf_mat_train


# In[638]:

## checking the event rates for the predicted class
train_pred_class=pd.DataFrame(train_pred_class)
test_pred_class=pd.DataFrame(test_pred_class)


# In[639]:

##event rate of train data
print 'Train Actual Event'
print train_data.outcome.value_counts()
print train_data.outcome.value_counts()/len( train_data.outcome)
print '\n'
print 'TRAIN predited events'
print train_pred_class.iloc[:,0].value_counts()
print train_pred_class.iloc[:,0].value_counts()/len(train_pred_class.iloc[:,0])
print '\n'
print 'TEST Predicted events'
print test_pred_class.iloc[:,0].value_counts()
print test_pred_class.iloc[:,0].value_counts()/len(test_pred_class.iloc[:,0])
print '\n'


# In[90]:

0.983617-0.96771
0.983617-0.948833


# In[640]:

### train metrics

rf_train_err=(cf_mat_train.iloc[0,1]+cf_mat_train.iloc[1,0])/(cf_mat_train.iloc[0,1]+cf_mat_train.iloc[1,0]+cf_mat_train.iloc[0,0]+cf_mat_train.iloc[1,1])  ## error rate
rf_train_acc=(cf_mat_train.iloc[0,0]+cf_mat_train.iloc[1,1])/(cf_mat_train.iloc[0,1]+cf_mat_train.iloc[1,0]+cf_mat_train.iloc[0,0]+cf_mat_train.iloc[1,1]) ## accuracy
rf_train_recall =cf_mat_train.iloc[1,1]/(cf_mat_train.iloc[1,1]+cf_mat_train.iloc[1,0]) ###recall  or hit rate or tpr or sensitivity
rf_train_spc=cf_mat_train.iloc[0,0]/(cf_mat_train.iloc[0,0]+cf_mat_train.iloc[0,1])  ##tnr or specificity
rf_train_prec=cf_mat_train.iloc[1,1]/(cf_mat_train.iloc[1,1]+cf_mat_train.iloc[0,1]) ### precision  or positive predicted value(ppv) 
rf_train_npv =cf_mat_train.iloc[0,0]/(cf_mat_train.iloc[0,0]+cf_mat_train.iloc[1,0]) ###negative predicted value
rf_train_fpr =cf_mat_train.iloc[0,1]/(cf_mat_train.iloc[0,0]+cf_mat_train.iloc[0,1]) ###false positive rate or fall out  
rf_train_fdr =cf_mat_train.iloc[0,1]/(cf_mat_train.iloc[0,1]+cf_mat_train.iloc[1,1]) ###false discovery rate
rf_train_fnr =cf_mat_train.iloc[1,0]/(cf_mat_train.iloc[1,0]+cf_mat_train.iloc[1,1]) ###false negative rate


# In[618]:

print('TRAIN',"error: ",rf_train_err*100,"Precision",rf_train_prec*100,"Recall: ",rf_train_recall*100,"FDR: ",rf_train_fdr*100,"FNR: ",rf_train_fnr*100)


# In[641]:

## feature importance
feat_index = np.argsort(clf.feature_importances_)[::-1] ## sorting the indices of feature imporatnce in decending order
fet_imp = clf.feature_importances_[feat_index] ##using the descending sorted index and arranging the feature importance array 


# In[642]:

feat_index


# In[643]:

fet_imp_names =features[feat_index]


# In[644]:

fet_imp_names


# In[645]:

##Putting the sorted feature importance and feature names in a dataframe
d = {'v_imp_names': pd.Series(fet_imp_names),
     'v_imp_values': pd.Series(fet_imp),
    }
v_imp_df = pd.DataFrame(d)


# In[646]:

v_imp_df


# In[662]:

## plotting the ROC Curve

## AUC for train 
fpr, tpr, thresholds = roc_curve(train3['outcome'], train_pred_prob)
roc_auc = auc(fpr, tpr)
print "Area under the ROC curve : %f" % roc_auc
                                        


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


# In[802]:

## Fifth submission
sub_5=pd.concat([test3['bidder_id'],pd.DataFrame(rf2['test_pred_prob'])],axis=1)
sub_5.columns=['bidder_id','prediction']
sub_5.to_csv('submission_5.csv',sep=',',index=None)


# In[803]:

## Sixth submission
sub_6=pd.concat([test3['bidder_id'],pd.DataFrame(rf3['test_pred_prob'])],axis=1)
sub_6.columns=['bidder_id','prediction']
sub_6.to_csv('submission_6.csv',sep=',',index=None)


# In[804]:

## Seventh submission ## have to submit (have to submit)
sub_7=pd.concat([test3['bidder_id'],pd.DataFrame(rf4['test_pred_prob'])],axis=1)
sub_7.columns=['bidder_id','prediction']
sub_7.to_csv('submission_7.csv',sep=',',index=None)


# test_list=[]
# train_list=[]
# for item in bidder_id_list:
#     for tr in bidder_id_list_train:
#         if tr==item:
#             train_list.append(tr)
#     for tes in bidder_id_list_test:
#          if tes==item:
#             test_list.append(tes)

# ##printing the count (if the bidder_id count matches 
# ##then they are all biddder_id in test and traunb are present in bids_data)
# print (len(bidder_id_list_train),len(bidder_id_list_test),len(bidder_id_list))
# print (len(train_list),len(test_list))

# delta_train= len(bidder_id_list_train)-len(train_list)
# delta_test=len(bidder_id_list_test)-len(test_list)
# print ('Delta Train',delta_train)
# print ('Delta Test',delta_test)

# from the output above we can see that there are 29 bidder ids which are there in train dataset and not in bids dataset and 70 bidder ids which are there in test dataset and not in bids dataset

# ##summarising the time variable in bids_dataset
# bids_data.time.describe()

# from the above results we can see that the time is a obfuscated number hence we should go with the lag differnce of time 

# # printing head of the data in the terminal
# ! head -10 bids.csv

# In[102]:

## initial submission
sub_1=pd.concat([test_agg['bidder_id'],pd.DataFrame(test_pred_prob)],axis=1)


# In[103]:

sub_1.head(5)


# In[175]:

##second submission (eventhough the model is overfitting)
sub_2=pd.concat([test_agg['bidder_id'],pd.DataFrame(test_pred_prob)],axis=1)


# In[454]:

## third submission
sub_3=pd.concat([test3['bidder_id'],pd.DataFrame(test_pred_prob)],axis=1)


# In[555]:

## Fourth submission
sub_4=pd.concat([test3['bidder_id'],pd.DataFrame(test_pred_prob)],axis=1)


# In[556]:

sub_4.head(5)


# In[455]:

sub_3.head(5)


# In[104]:

sub_1.columns=['bidder_id','prediction']


# In[178]:

sub_2.columns=['bidder_id','prediction']


# In[456]:

sub_3.columns=['bidder_id','prediction']


# In[557]:

sub_4.columns=['bidder_id','prediction']


# In[105]:

sub_1.to_csv('submission_1.csv',sep=',',index=None)


# In[179]:

sub_2.to_csv('submission_2.csv',sep=',',index=None)


# In[559]:

sub_3.to_csv('submission_3.csv',sep=',',index=None)


# In[560]:

sub_4.to_csv('submission_4.csv',sep=',',index=None)


# In[106]:

sub_1.shape


# In[868]:

train3.to_csv('train3.txt',sep='\t',index=None)


# In[869]:

test3.to_csv('test3.txt',sep='\t',index=None)


# ## To do
# ##Features/variables needed in the data set for analysis
# ##bidder
# ##no of acutions per bidder --- done
# ##unique no of acutions per bidder ---done
# ##no of devices per bidder --done
# ##unique no off devices per bidder---done
# ##count of country --done
# ##unq no of country --done
# 
# ####order by time variables and find the lag of time and lag differnce (need to do after brain storming)
# #### bids/auction --single bid vs multiple bids
# ## average bids per auction --done
# ## max bid per auction --have to do
# ## min bids per auction--have to do
# ## time lag in each bid ---done 
#     #### created four time related variables min, max,mean, median of the average first lag of time differnce per acution per bidder id
# ## number of ip address per auction-- done
# ## number of url references per auction---done
# ## average number of devices  per auction---done
# ##
# 
# #TO DO
# ## some min, max and median varaibles for other features (if possible)
# ## log time variables
# ## Feature selection
# ## test train subdivision
# ##other algos (SVM,....)
# 
# 

# ####ROUGH WORK####

# ##############

# In[ ]:



