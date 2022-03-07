import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from google.oauth2 import service_account
from google.cloud import bigquery
import configparser

#[]
#
KEY_PATH = "/mnt/c/Users/Ron/git-repos/yelp-data/gourmanddwh-f75384f95e86.json"
CREDS = service_account.Credentials.from_service_account_file(KEY_PATH)
client = bigquery.Client(credentials=CREDS, project=CREDS.project_id)


#[]
#
transactions_file = open('business_transactions.sql','r')
tran_query =  transactions_file.read()

business_transactions_dataframe = (
    client.query(tran_query)
    .result()
    .to_dataframe()
)

#[]
#
business_transactions_dataframe.to_parquet('business_transactions.snappy.parquet', 'pyarrow','snappy', partition_cols=['transactions_list'])

#[]
#
holding_file = open('sql_scripts/business_daily_holding.sql')
holding_query = holding_file.read()

holding_dataframe : pd.DataFrame = (
    client.query(holding_query)
    .result()
    .to_dataframe()
)
#[]
#
holding_dataframe.to_parquet('bus_holdings.snappy.parquet', 'pyarrow','snappy', partition_cols=['CloseDate'])


#[]
#
cg_df = pd.read_parquet('cg_est', engine='pyarrow')
bh_df: pd.DataFrame = pd.read_parquet('bus_holdings.snappy.parquet', engine='pyarrow')
bus_cats_df = pd.read_parquet('bus_cats', engine='pyarrow')
bt_df = pd.read_parquet('business_transactions.snappy.parquet', engine='pyarrow')

#[]
#
bt_df

#[]
#
most_recent_bh_df = bh_df.drop_duplicates(subset='BusinessName', keep='last')

#[]
#
most_recent_bh_df = most_recent_bh_df.reset_index(drop=True)
most_recent_bh_df

#[]
#
most_recent_bh_df.hist('ReviewCount', bins=10)

#[]
#
plt.figure(figsize=(15, 10))
sns.kdeplot(most_recent_bh_df['ReviewCount'])
plt.show()

#[]
#
most_recent_bh_df['ReviewCount'].value_counts(normalize=True ,bins=10)

#[]
#
most_recent_bh_tran_df = pd.merge(left=most_recent_bh_df, right=bt_df, on='BusinessKey', how='inner')
most_recent_bh_tran_df
#[]
#
sns.set_theme()
plt.figure(figsize=(15, 10))
sns.scatterplot(x=most_recent_bh_tran_df['transactioncounts'], y=most_recent_bh_tran_df['ReviewCount'], x_jitter=.30)


#[]
#
most_recent_bh_df['ReviewCount'].nsmallest(10)

#[]
#
most_recent_bh_df_sin_cero = most_recent_bh_tran_df[most_recent_bh_tran_df['ReviewCount'] > 0 ]

#[]
#
most_recent_bh_df_sin_cero.loc[:, 'ln_review_count'] = np.log(most_recent_bh_df_sin_cero['ReviewCount'])

#[]
#
sns.set_theme()
plt.figure(figsize=(15, 10))
sns.scatterplot(x=most_recent_bh_df_sin_cero['transactioncounts'], y=most_recent_bh_df_sin_cero['ln_review_count'], x_jitter=.30)

#[]
#
sns.set_theme()
fig, axes = plt.subplots(1,2, figsize=(20, 10))

sns.scatterplot(x=most_recent_bh_df_sin_cero['transactioncounts'], y=most_recent_bh_df_sin_cero['ReviewCount'], x_jitter=.30, ax=axes[0])

sns.scatterplot(x=most_recent_bh_df_sin_cero['transactioncounts'], y=most_recent_bh_df_sin_cero['ln_review_count'], x_jitter=.30,ax=axes[1] )
axes[0].set(title='Before Transformation')
axes[1].set(title='After Transformation', xlabel=f'{axes[1].get_ylabel()} (ln)')

#[]
#
X_tc = most_recent_bh_df_sin_cero['transactioncounts']
X_tc_with_constant = sm.add_constant(X_tc)

y = most_recent_bh_df_sin_cero['ln_review_count']

univariate_lin_model = sm.OLS(y, X_tc_with_constant)
results = univariate_lin_model.fit(cov_type='HC1')

#[]
#
results.summary()

#[]
#
sns.set_theme()
plt.figure(figsize=(15, 10))
sns.scatterplot(x=most_recent_bh_df_sin_cero['total_review_cnt_delta'], y=most_recent_bh_df_sin_cero['ReviewCount'], x_jitter=.30)

#[]
#
sns.set_theme()
plt.figure(figsize=(15, 10))
sns.scatterplot(y=most_recent_bh_df_sin_cero['total_review_cnt_delta'], x=most_recent_bh_df_sin_cero['ReviewCount'], x_jitter=.30)

#[]
#
most_recent_bh_df_sin_cero['total_review_cnt_delta'].isna().sum()

#[]
#
most_recent_bh_df_sin_cero_sin_na = most_recent_bh_df_sin_cero[~ most_recent_bh_df_sin_cero['total_review_cnt_delta'].isna()]
most_recent_bh_df_sin_cero_sin_na.shape
#[]
#
X_rc = most_recent_bh_df_sin_cero_sin_na['ln_review_count']
X_rc_with_constant = sm.add_constant(X_rc)

y2 = most_recent_bh_df_sin_cero_sin_na['total_review_cnt_delta']

univariate_lin_model_trcd_lnrc = sm.OLS(y2, X_rc_with_constant)
univariate_lin_model_trcd_lnrc_results = univariate_lin_model_trcd_lnrc.fit(cov_type='HC1')

#[]
#
univariate_lin_model_trcd_lnrc_results.summary()

#[]
#
sns.set_theme()
plt.figure(figsize=(15, 10))
sns.regplot(y=most_recent_bh_df_sin_cero['total_review_cnt_delta'], 
x=most_recent_bh_df_sin_cero['ln_review_count'], x_jitter=.30,
color='red', marker='x', lowess=True)

#[]
#
most_recent_bh_df_sin_cero.loc[:, 'BusinessRating'] = most_recent_bh_df_sin_cero.loc[:, 'BusinessRating'].astype(float)
sns.set_theme()
plt.figure(figsize=(15, 10))
sns.scatterplot(x=most_recent_bh_df_sin_cero['BusinessRating'], 
y=most_recent_bh_df_sin_cero['ReviewCount'], x_jitter=.30)



#[]
#
most_recent_bh_df_sin_cero.loc[:, 'BusinessRating_squared'] = most_recent_bh_df_sin_cero.loc[:, 'BusinessRating'] ** 2

#[]
#
X_br_brsquared = most_recent_bh_df_sin_cero.loc[:, ['BusinessRating_squared', 'BusinessRating']]
X_br_brsquared_with_constant = sm.add_constant(X_br_brsquared)
y
univariate_lin_model_rc_br = sm.OLS(y, X_br_brsquared_with_constant)
univariate_lin_model_rc_br_results = univariate_lin_model_rc_br.fit(cov_type='HC1')

#[]
#
univariate_lin_model_rc_br_results.summary()

#[]
#
X_br = most_recent_bh_df_sin_cero.loc[:, 'BusinessRating']
X_br_with_constant = sm.add_constant(X_br)
y
univariate_lin_model_rc_br1 = sm.OLS(y, X_br_with_constant)
univariate_lin_model_rc_br1_results = univariate_lin_model_rc_br1.fit(cov_type='HC1')

univariate_lin_model_rc_br1_results.summary()

#[]
#
sns.set_theme()
plt.figure(figsize=(15, 10))
sns.regplot(
    x=most_recent_bh_df_sin_cero['BusinessRating'],
    y=most_recent_bh_df_sin_cero['ln_review_count'], x_jitter=.10, order=2,
    line_kws={'color':'cyan'})

plt.title('higher-order polynomial of explanatory variable')

#[]
#
