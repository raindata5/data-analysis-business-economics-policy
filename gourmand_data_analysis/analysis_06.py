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
