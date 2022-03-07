# regression ln(ReviewCounts) on rating for review counts 
    # regression ln(ReviewCounts) on rating and top 5 ccats or not 
    # omitted variable bias through difference (try product method?) (conclusions based on this bias?)
    # checkout interaction of rating on delivery
    # y-y plot to visualize and identify over or under predictions

#[]
#
from typing import Union
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
sys.path.append('..')
from packages import utils
from stargazer.stargazer import Stargazer

#[]
#
cg_df = pd.read_parquet('cg_est', engine='pyarrow')
bh_df: pd.DataFrame = pd.read_parquet('bus_holdings.snappy.parquet', engine='pyarrow')
bus_cats_df = pd.read_parquet('bus_cats', engine='pyarrow')
bt_df = pd.read_parquet('business_transactions.snappy.parquet', engine='pyarrow')


#[]
#
most_recent_bh_tran_df = utils.most_recent_businesses_df(df1 = bh_df, df2= bt_df, on= 'BusinessKey', remove_na=True)
most_recent_bh_df_sin_cero = most_recent_bh_tran_df[most_recent_bh_tran_df['ReviewCount'] > 0 ]
most_recent_bh_df_sin_cero.loc[:, 'ln_review_count'] = np.log(most_recent_bh_df_sin_cero['ReviewCount'])
sns.set_theme()
fig, axes = plt.subplots(1,2, figsize=(20, 10))

sns.scatterplot(x=most_recent_bh_df_sin_cero['transactioncounts'], y=most_recent_bh_df_sin_cero['ReviewCount'], x_jitter=.30, ax=axes[0])

sns.scatterplot(x=most_recent_bh_df_sin_cero['transactioncounts'], y=most_recent_bh_df_sin_cero['ln_review_count'], x_jitter=.30,ax=axes[1] )
axes[0].set(title='Before Transformation')
axes[1].set(title='After Transformation', ylabel=f'{axes[1].get_ylabel()} (ln)')

#[]
#

X_tc = most_recent_bh_df_sin_cero['transactioncounts']
X_tc_with_constant = sm.add_constant(X_tc)

y = most_recent_bh_df_sin_cero['ln_review_count']

univariate_lin_model = sm.OLS(y, X_tc_with_constant)
results = univariate_lin_model.fit(cov_type='HC1')
results.summary()

#[]
#

most_recent_bh_df_sin_cero.loc[:,'delivery'] = np.where(most_recent_bh_df_sin_cero.loc[:,'transactions_list'].str.contains('delivery'), 1 , 0)
#[]
#
most_recent_bh_df_sin_cero.corr()['delivery']

# group on business name , custom apply where make a list of the categories, variable = 0 if bc in top 5 list then make variable one
#[]
#

#[]
#
top_5_cat_list = bus_cats_df['BusinessCategoryName'].value_counts(normalize=True).head(5).index.tolist()

#[]
#
from google.oauth2 import service_account
from google.cloud import bigquery

#[]
#
KEY_PATH = "/mnt/c/Users/Ron/git-repos/yelp-data/gourmanddwh-f75384f95e86.json"
CREDS = service_account.Credentials.from_service_account_file(KEY_PATH)
client = bigquery.Client(credentials=CREDS, project=CREDS.project_id)
#[]
#

cat_file = open('sql_scripts/business_category_location_revised.sql')
cat_query = cat_file.read()

cat_df : pd.DataFrame = (
    client.query(cat_query)
    .result()
    .to_dataframe()
)

cat_df.to_parquet('cat_df.snappy.parquet', 'pyarrow','snappy', partition_cols=['StateName'])

#[]
#
cat_df = pd.read_parquet('cat_df.snappy.parquet', engine='pyarrow')

#[]
#
most_recent_bh_df_sin_cero_cat_df = pd.merge(left=most_recent_bh_df_sin_cero, right=cat_df, left_on='BusinessKey', right_on='BusinessKey', how='inner')
most_recent_bh_df_sin_cero_cat_df.shape

most_recent_bh_df_sin_cero_cat_df['business_cat_list'] = most_recent_bh_df_sin_cero_cat_df['business_cat_list'].str.split(', ')
most_recent_bh_df_sin_cero_cat_df
#[]
#
def row_lvl_comparison(row, target_col:str,  target_list: list,  col_name_edit: Union[str, None] = None):
    the_dict = {}
    the_dict[f'{col_name_edit}_match'] =  0 
    the_dict[f'{col_name_edit}_matches'] =  0
    for item in row[target_col]:
        if item in target_list:
            the_dict[f'{col_name_edit}_match'] = 1
            the_dict[f'{col_name_edit}_matches'] += 1
        else:
            continue
    
    return pd.Series(the_dict)

#[]
#
cats_series = most_recent_bh_df_sin_cero_cat_df.apply(row_lvl_comparison,target_col='business_cat_list',target_list=top_5_cat_list,col_name_edit='top_5_cat',axis=1)
most_recent_bh_df_sin_cero_cat_df = pd.concat([most_recent_bh_df_sin_cero_cat_df, cats_series], axis=1)

#[]
#
most_recent_bh_df_sin_cero_cat_df['top_5_cat_match'].value_counts()

#[]
#
sns.set_theme()
plt.figure(figsize=(15, 10))
sns.lmplot(x="transactioncounts", y="ln_review_count", hue="top_5_cat_match", data=most_recent_bh_df_sin_cero_cat_df, x_jitter=.10)

#[]
#
X_tc_cat_bin = most_recent_bh_df_sin_cero_cat_df[['transactioncounts', 'top_5_cat_match']]
X_tc_cat_bin_with_constant = sm.add_constant(X_tc_cat_bin)

y = most_recent_bh_df_sin_cero_cat_df['ln_review_count']

multivariate_lin_model = sm.OLS(y, X_tc_cat_bin_with_constant)
mlm_results = multivariate_lin_model.fit(cov_type='HC1')
mlm_results.summary()

#[]
#
Stargazer([results, mlm_results])

#[]
#
X_tc_with_constant = sm.add_constant(most_recent_bh_df_sin_cero_cat_df['transactioncounts'])
simple_lin_reg_tc_cat_bin = sm.OLS(most_recent_bh_df_sin_cero_cat_df['top_5_cat_match'], X_tc_with_constant)
slrtcb_results = simple_lin_reg_tc_cat_bin.fit(cov_type='HC1')

Stargazer([results, mlm_results, slrtcb_results])

#[]
#
most_recent_bh_df_sin_cero_cat_df.PaymentLevelName.value_counts()
#[]
#

payment_lvl_dummies = pd.get_dummies(most_recent_bh_df_sin_cero_cat_df.PaymentLevelName).drop('Low', axis=1)
payment_lvl_dummies

#[]
#
most_recent_bh_df_sin_cero_cat_dummies_df = pd.merge(most_recent_bh_df_sin_cero_cat_df, payment_lvl_dummies, how='inner', left_index=True, right_index=True)
most_recent_bh_df_sin_cero_cat_dummies_df

#[]
#
payment_lvls = most_recent_bh_df_sin_cero_cat_dummies_df.PaymentLevelName.unique().tolist()
reg_dict = {}
reg_list_tuples: list = []
for payment_lvl in payment_lvls:
    cut_df = most_recent_bh_df_sin_cero_cat_dummies_df.loc[most_recent_bh_df_sin_cero_cat_dummies_df.PaymentLevelName == payment_lvl, :]
    y_cut = cut_df['ln_review_count']
    X_cut_with_constant = sm.add_constant(cut_df['transactioncounts'])
    cut_model = sm.OLS(y_cut, X_cut_with_constant)
    cut_model_results = cut_model.fit(cov_type='HC1')

    reg_dict[payment_lvl] = cut_model_results
    reg_tuple = (payment_lvl, cut_model_results,)
    reg_list_tuples.append(reg_tuple)

#[]
#
most_recent_bh_df_sin_cero_cat_dummies_df['tc_by_Unknown'] = most_recent_bh_df_sin_cero_cat_dummies_df['transactioncounts'] * most_recent_bh_df_sin_cero_cat_dummies_df['Unknown']

X_tc_dummies_inter = most_recent_bh_df_sin_cero_cat_dummies_df[['transactioncounts', 'High', 'Unknown', 'Very High', 'Very Low', 'tc_by_Unknown']]
X_tc_dummies_inter_with_constant = sm.add_constant(X_tc_dummies_inter)
multivariate_inter_model = sm.OLS(y, X_tc_dummies_inter_with_constant)
multivariate_inter_model_results = multivariate_inter_model.fit(cov_type='HC1')
multivariate_inter_model_results.summary()

#[]
#
stargazer_inter = Stargazer([reg_dict[p_lvl] for p_lvl in payment_lvls] + [multivariate_inter_model_results])
stargazer_inter.custom_columns([p_lvl for p_lvl in payment_lvls] + ["All"], [1, 1, 1, 1, 1, 1])

#[]
#
most_recent_bh_df_sin_cero_cat_dummies_with_cons_df = sm.add_constant(most_recent_bh_df_sin_cero_cat_dummies_df)
cols = ['const', 'transactioncounts', 'High', 'Unknown', 'Very High', 'Very Low', 'tc_by_Unknown']
unknown_bus = most_recent_bh_df_sin_cero_cat_dummies_with_cons_df.loc[most_recent_bh_df_sin_cero_cat_dummies_with_cons_df['PaymentLevelName'] == 'Unknown', cols]
low_bus = most_recent_bh_df_sin_cero_cat_dummies_with_cons_df.loc[most_recent_bh_df_sin_cero_cat_dummies_with_cons_df['PaymentLevelName'] == 'Low', cols]


unknown_pred = multivariate_inter_model_results.get_prediction(unknown_bus).summary_frame()[["mean", "mean_se"]]
unknown_pred.columns = ["fit", "fit_se"]

unknown_bus_pred_df = pd.concat([unknown_bus, unknown_pred], axis=1)
unknown_bus_pred_df["CIup"]=unknown_bus_pred_df["fit"]+2*unknown_bus_pred_df["fit_se"]
unknown_bus_pred_df["CIlo"]=unknown_bus_pred_df["fit"]-2*unknown_bus_pred_df["fit_se"]

low_pred = multivariate_inter_model_results.get_prediction(low_bus).summary_frame()[["mean", "mean_se"]]
low_pred.columns = ["fit", "fit_se"]

low_bus_pred_df = pd.concat([low_bus, low_pred], axis=1)
low_bus_pred_df["CIup"]=low_bus_pred_df["fit"]+2*low_bus_pred_df["fit_se"]
low_bus_pred_df["CIlo"]=low_bus_pred_df["fit"]-2*low_bus_pred_df["fit_se"]

#[]
#
sns.set_theme()
fig, ax = plt.subplots(1, 1, figsize=(15, 10))

sns.lineplot(
    x=unknown_bus_pred_df['transactioncounts'],
    y=unknown_bus_pred_df['fit'], label= "Unknown-cost Businesses", color='grey', ax=ax)

sns.lineplot(
    x=unknown_bus_pred_df['transactioncounts'],
    y=unknown_bus_pred_df['CIlo'], color='grey', linestyle='--', ax=ax)

sns.lineplot(
    x=unknown_bus_pred_df['transactioncounts'],
    y=unknown_bus_pred_df['CIup'],  color='grey', linestyle='--', ax=ax)

sns.lineplot(
    x=low_bus_pred_df['transactioncounts'],
    y=low_bus_pred_df['fit'], label= "Low-cost Businesses", color='blue', ax=ax)

sns.lineplot(
    x=low_bus_pred_df['transactioncounts'],
    y=low_bus_pred_df['CIlo'], color='blue', linestyle='--', ax=ax)

sns.lineplot(
    x=low_bus_pred_df['transactioncounts'],
    y=low_bus_pred_df['CIup'],  color='blue', linestyle='--', ax=ax)



plt.legend()