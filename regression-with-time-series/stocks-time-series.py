from pathlib import Path
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 12)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.4f}'.format

#[]
#

path = Path(os.getcwd())
base_dir = path.parent.parent.parent

data_in = os.path.join(str(base_dir), "Desktop/Rain-data/stocks-sp500/raw/")

stocks = pd.read_csv(os.path.join(data_in,'ready_sp500_45_cos.csv'))

sp_500 = pd.read_csv(os.path.join(data_in,'ready_sp500_index.csv'))

#[]
# to datetime
stocks['ref.date'] = pd.to_datetime(stocks['ref.date'])
sp_500['ref.date'] = pd.to_datetime(sp_500['ref.date'])

#[]
#

stocks.columns = stocks.columns.str.strip().str.replace('.','_')
sp_500.columns = sp_500.columns.str.strip().str.replace('.','_')
sp_500.columns

#[]
#
sp_500 #already sorted
stocks.head()

#[]
# separate the data I want do NTAP
stocks_ntap = stocks.loc[stocks.ticker == 'NTAP']
stocks_ntap.head()

#[]
# same amount of data in same exact order
list(sp_500['ref_date']) == list(stocks_ntap['ref_date'])

#[]
# double checking the merge operations
# sys.path.append(str(base_dir) + 'git-repos/data-cleaning/udfs-and-classes/helper-functions/')
# import combineaggfunctions as caf

#[]
#
stocks_ntap.isna().sum()
sp_500.isna().sum()

#[]
#
gspc_ntap = pd.merge(sp_500,stocks_ntap, how='inner', left_on='ref_date', right_on='ref_date', suffixes=('_gspc','_ntap'))
gspc_ntap.info()
gspc_ntap.columns.tolist()

#[]
# S&P defintely has trend so better to take percentage differences as a measure
# plus this corresponds to the returns
gspc_ntap.plot(x='ref_date',y='price_close_gspc')
plt.show()

#[]
#
gspc_ntap.plot(x='ref_date',y='price_close_ntap')
plt.show()

#[]
# the scale is totally different which is one reason to take differences
plt.plot(gspc_ntap['ref_date'],gspc_ntap['price_close_gspc'], color='m')
plt.plot(gspc_ntap['ref_date'],gspc_ntap['price_close_ntap'],color='c')
plt.show()

#[]
# since we're going to do our analysis with returns based on monthly frequency we'll filter our some irrelevant dates
gspc_ntap = gspc_ntap.loc[(gspc_ntap['ref_date'] >= '1997-12-31') & (gspc_ntap['ref_date'] <= '2018-12-31'),:]

#[]
#
gspc_ntap.loc[:,'ref_month'] = gspc_ntap.loc[:,'ref_date'].dt.month
gspc_ntap['ref_year'] = gspc_ntap.loc[:,'ref_date'].dt.year

#[]
#
gspc_ntap.to_pickle(os.path.join(str(base_dir),'git-repos/data-analysis-business-economics-policy/data-sets/gspc_ntap-stock-data'))
#[]
#since data is in order i can group by year and month and just keep the last value
stocks_month = gspc_ntap.groupby(['ref_year','ref_month'],as_index=False).last()

#[]
# getting the stock closing price from the previous date
stocks_month.loc[:,'price_close_gspc_lag_1'] = stocks_month['price_close_gspc'].shift()
stocks_month.loc[:,'price_close_ntap_lag_1'] = stocks_month['price_close_ntap'].shift()

#[]
#
stocks_month.columns
stocks_month.head()

#[]
# getting the relative differences and the percent differencs
# ((stocks_month['price_close_gspc'] - stocks_month.loc[:,'price_close_gspc_lag_1']) / stocks_month.loc[:,'price_close_gspc_lag_1']) * 100
stocks_month.loc[:, 'rel_diff_gspc'] = stocks_month['price_close_gspc'] - stocks_month.loc[:,'price_close_gspc_lag_1']
stocks_month.loc[:, 'per_diff_gspc'] = (stocks_month['rel_diff_gspc'] / stocks_month.loc[:,'price_close_gspc_lag_1']) * 100

#[]
#
stocks_month.loc[:, 'rel_diff_ntap'] = stocks_month['price_close_ntap'] - stocks_month.loc[:,'price_close_ntap_lag_1']
stocks_month.loc[:, 'per_diff_ntap'] = (stocks_month['rel_diff_ntap'] / stocks_month.loc[:,'price_close_ntap_lag_1']) * 100
#[]
#
stocks_month.to_pickle(os.path.join(str(base_dir),'git-repos/data-analysis-business-economics-policy/data-sets/gspc_ntap-stock-data_month'))
#[]
#
stocks_month.head()

#[]
#
ax = stocks_month.plot(x='ref_date',y='per_diff_gspc',color='r')
ax.axhline(y=0,color='b')
plt.show()
# positive trend knocked out by percent differences
stocks_month['rel_diff_gspc'].mean() # definetly has a positive trean

#[]
#
ax = stocks_month.plot(x='ref_date',y='per_diff_ntap',color='r')
ax.axhline(y=0,color='b')
plt.show()
stocks_month['rel_diff_ntap'].mean() # slight positive trend

#[]
# ntap looks a lot more volatile but hard to see any relationships
plt.plot(stocks_month['ref_date'],stocks_month['per_diff_gspc'], color='m',label = 'sp500')
plt.plot(stocks_month['ref_date'],stocks_month['per_diff_ntap'],color='c', label = 'ntap')
plt.axhline(y=0,color='b')
plt.legend()
plt.show()

#[]
#
stocks_month[['per_diff_gspc','per_diff_ntap']].describe().T

#[]
# we could have checked for seasonality prior but now when plotting we'll know for sure whether there is a good chance
# of seasonality or not with the percentage differences
stocks_month_2017_2018 = stocks_month.loc[(stocks_month['ref_date'] >= '2017-01-01') & (stocks_month['ref_date'] <= '2018-12-31') , ['ref_date','per_diff_gspc','per_diff_ntap']]

fig, ax = plt.subplots()
ax.plot(stocks_month_2017_2018['ref_date'], stocks_month_2017_2018['per_diff_gspc'], color='m',label = 'sp500')
ax.plot(stocks_month_2017_2018['ref_date'], stocks_month_2017_2018['per_diff_ntap'],color='c', label = 'ntap')
ax.axhline(y=0,color='b')
for tick in ax.get_xticklabels():
    tick.set_rotation(45)
# ax.set_xticklabels(labels= ax.get_xticks() ,rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

#[]
#
stocks_month = pd.read_pickle(os.path.join(str(base_dir),'git-repos/data-analysis-business-economics-policy/data-sets/gspc_ntap-stock-data_month'))
import statsmodels.formula.api as smf

reg_model_month = smf.ols("per_diff_ntap ~ per_diff_gspc", data=stocks_month)
reg_model_month_results = reg_model_month.fit()
reg_model_month_results.summary()

#[]
# when the sp_500 doesn't change I can't say with 95% confidence that the ntap stocks tend to go up
# there could very well be no change at all in this case
# however for every 1 percent increase in the sp_500 stock there is an associated increase of 1.97 in the ntap
# and zero is not included in the 95% CI [1.4992,2.4472]
reg_model_month_results_summary = reg_model_month_results.get_robustcov_results(cov_type='HC1').summary()

#[]
# persiting model, and results
import joblib
joblib.dump(reg_model_month,"reg_model_month")
joblib.dump(reg_model_month_results_summary,"reg_model_month_results_summary")

#[]
# now using season dummies
reg_model_month_encoded = smf.ols("per_diff_ntap ~ per_diff_gspc + C(ref_month)", data=stocks_month)
reg_model_month_encoded_results = reg_model_month_encoded.fit()
reg_model_month_encoded_results.summary()
# the extremely high SE's indicate that there would be a vary low chance of seasonality

#[]
#
fig, ax = plt.subplots()
ax.scatter(stocks_month['per_diff_gspc'],stocks_month['per_diff_ntap'])
plt.show()

#[]
# shows how the percentages tend to increase with each other
sns.regplot(stocks_month['per_diff_gspc'], stocks_month['per_diff_ntap'], robust=True, ci=95, color='r')
plt.show()

#[]
#

stocks_month.loc[:, 'predictions'] = reg_model_month.fit().predict(stocks_month)

stocks_month.loc[:,['per_diff_ntap','predictions','per_diff_gspc']]

#[]
#
from sklearn.metrics import mean_squared_error
model_rmse = mean_squared_error(stocks_month['per_diff_ntap'][1:], stocks_month['predictions'][1:], squared=True)
model_rmse
#[]
# our 2nd model with the season dummies has a lower rmse however this probably due to overfitting the data
model2_rmse = mean_squared_error(stocks_month['per_diff_ntap'][1:], reg_model_month_encoded_results.predict(stocks_month)[1:])
model2_rmse



# **future edits****
# use newey west SE to check for serial correlation
# lag some of the dependent variables to deal with serial correlation and also potententially change regression coefficients
# check for lagged associations by lagging the independant variable
# check for non linear patterns through bin scatter
# test for random walk
# try a daily frequency model and use a binary variable for changes that occured over gaps (1 day+)
joblib.dump(stocks_month,"reg_model_month_predictions")

stocks_month['per_diff_ntap']
['ticker_gspc',
 'ref_date',
 'price_open_gspc',
 'price_close_gspc',
 'price_adjusted_gspc',
 'price_low_gspc',
 'price_high_gspc',
 'volume_gspc',
 'ticker_ntap',
 'price_open_ntap',
 'price_close_ntap',
 'price_adjusted_ntap',
 'price_low_ntap',
 'price_high_ntap',
 'volume_ntap']
