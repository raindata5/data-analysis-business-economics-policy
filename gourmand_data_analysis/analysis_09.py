# think about why I would want to change the bus rating into a binary variable
    # how do I decide between choosing a lpm or using logit and probit
    # after having gotten this/these models how do I go about evaluating it
        # how to do so graphically?
    # how do I interpret the coefficients?


import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
sys.path.append('..')
from packages import utils
import plotly.express as px
from stargazer.stargazer import Stargazer
#[]
#
cg_df = pd.read_parquet('cg_est', engine='pyarrow')
bh_df: pd.DataFrame = pd.read_parquet('bus_holdings.snappy.parquet', engine='pyarrow')
cat_df = pd.read_parquet('cat_df.snappy.parquet', engine='pyarrow')
bt_df = pd.read_parquet('business_transactions.snappy.parquet', engine='pyarrow')

#[]
#
most_recent_bh_tran_df = utils.most_recent_businesses_df(df1 = bh_df, df2= bt_df, on= 'BusinessKey', remove_na=True)

#[]
#
most_recent_bh_tran_df['good_business'] = np.where(most_recent_bh_tran_df['BusinessRating'] >= 4, 1, 0)
#[]
#

most_recent_bh_tran_df['good_business'].mean()

#[]
#
plt.figure(figsize=(15,10))
sns.scatterplot(x=most_recent_bh_tran_df['total_review_cnt_delta'], y=most_recent_bh_tran_df['good_business'])
plt.show()

#[]
#
payment_lvl_dummies = pd.get_dummies(cat_df.PaymentLevelName).drop('Low', axis=1)
payment_lvl_dummies['unknown_flag'] = np.where(payment_lvl_dummies.Unknown == 1, 1, 0)

#[]
#
payment_lvl_dummies_bk_df = pd.concat([cat_df[['BusinessKey', 'cat_counts']], payment_lvl_dummies], axis=1)
most_recent_bh_tran_w_dummies_df = pd.merge(left=most_recent_bh_tran_df, right=payment_lvl_dummies_bk_df, how='inner', on='BusinessKey')


#[]
#
X = sm.add_constant(most_recent_bh_tran_w_dummies_df[['transactioncounts', 'High', 'Unknown', 'Very High', 'Very Low']])
y = most_recent_bh_tran_df['good_business']

lpm_model = sm.OLS(y, X)
lpm_model_results = lpm_model.fit(cov_type='HC1')
lpm_model_results.summary()

#[]
#
most_recent_bh_tran_w_dummies_df['lpm_pred'] = lpm_model_results.predict(X)

fig = px.histogram(most_recent_bh_tran_w_dummies_df, x="lpm_pred", color="good_business")
fig.show()


#[]
#
print(f"bias of predictions: {most_recent_bh_tran_w_dummies_df['lpm_pred'].mean()}")

print(f"relative frequency of good businesses:{most_recent_bh_tran_w_dummies_df['good_business'].mean()}")

#[]
#
good_businesses = most_recent_bh_tran_w_dummies_df.loc[most_recent_bh_tran_w_dummies_df['good_business'] == 1, 'lpm_pred']
bad_businesses = most_recent_bh_tran_w_dummies_df.loc[most_recent_bh_tran_w_dummies_df['good_business'] == 0, 'lpm_pred']

sum_stats = {}
sum_stats['gb_mean'] = good_businesses.mean()
sum_stats['gb_median'] = good_businesses.median()
sum_stats['bb_mean'] = bad_businesses.mean()
sum_stats['bb_median'] = bad_businesses.median()

sum_stats_df = pd.Series(sum_stats)
sum_stats_df.T

#[]
#
pred_bins_10 = pd.cut(most_recent_bh_tran_w_dummies_df['lpm_pred'], bins=10)
most_recent_bh_tran_w_dummies_df['pred_bins_10'] = pred_bins_10
calibration_curve_df = most_recent_bh_tran_w_dummies_df.groupby(['pred_bins_10'], as_index=False)[['good_business', 'lpm_pred']].mean()
calibration_curve_df

#[]
#

fig, axes = plt.subplots(1,1, figsize=(20, 10))

sns.lineplot(x=calibration_curve_df['lpm_pred'], y=calibration_curve_df['good_business'], ax=axes)

axes.axline([0, 0], [1, 1])
axes.set_xlabel("Bins of Predicted Probabilities")
axes.set_ylabel("Actual Probability")

#[]
#
logit_model = sm.Logit(y, X)
logit_model_results = logit_model.fit()
logit_model_results.summary()

#[]
# default link is the logit link
logit_model_man = sm.GLM(y,X, family=sm.families.Binomial(sm.genmod.families.links.logit()))
logit_model_man_results = logit_model_man.fit()
logit_model_man_results.summary()

#[]
#
logit_model_marg_diff_results = logit_model_results.get_margeff()

#[]
#
logit_model_marg_diff_results.summary()


#[]
#
print(lpm_model_results.summary(), logit_model_marg_diff_results.summary(), end='\n')

#[]
#
most_recent_bh_tran_w_dummies_df['logit_pred'] = logit_model_man_results.predict(X)

fig = px.histogram(most_recent_bh_tran_w_dummies_df, x="logit_pred", color="good_business")
fig.show()
#[]
#
good_businesses_logit = most_recent_bh_tran_w_dummies_df.loc[most_recent_bh_tran_w_dummies_df['good_business'] == 1, 'logit_pred']
bad_businesses_logit = most_recent_bh_tran_w_dummies_df.loc[most_recent_bh_tran_w_dummies_df['good_business'] == 0, 'logit_pred']

sum_stats_logit = {}
sum_stats_logit['gb_mean'] = good_businesses_logit.mean()
sum_stats_logit['gb_median'] = good_businesses_logit.median()
sum_stats_logit['bb_mean'] = bad_businesses_logit.mean()
sum_stats_logit['bb_median'] = bad_businesses_logit.median()

sum_stats_df2 = pd.Series(sum_stats_logit)
sum_stats_df2.T

#[]
#

pd.concat([sum_stats_df, sum_stats_df2], axis=1)

#[]
#

logit_pred_bins_10 = pd.cut(most_recent_bh_tran_w_dummies_df['logit_pred'], bins=10)
most_recent_bh_tran_w_dummies_df['logit_pred_bins_10'] = logit_pred_bins_10
logit_calibration_curve_df = most_recent_bh_tran_w_dummies_df.groupby(['logit_pred_bins_10'], as_index=False)[['good_business', 'lpm_pred']].mean()

fig, axes = plt.subplots(1,1, figsize=(20, 10))

sns.lineplot(x=calibration_curve_df['lpm_pred'], y=calibration_curve_df['good_business'], label= "LPM", color='red', ax=axes)
sns.lineplot(x=logit_calibration_curve_df['lpm_pred'], y=logit_calibration_curve_df['good_business'], label= "Logit", color='green', ax=axes)

plt.legend()
axes.axline([0, 0], [1, 1])
axes.set_xlabel("Bins of Predicted Probabilities")
axes.set_ylabel("Actual Probability")

#[]
#
logit_brier_score = ((most_recent_bh_tran_w_dummies_df['logit_pred'] - most_recent_bh_tran_w_dummies_df['good_business']) **2).sum() / most_recent_bh_tran_w_dummies_df['logit_pred'].shape[0]
lpm_brier_score = ((most_recent_bh_tran_w_dummies_df['lpm_pred'] - most_recent_bh_tran_w_dummies_df['good_business']) **2).sum() / most_recent_bh_tran_w_dummies_df['lpm_pred'].shape[0]

#[]
#
print(f'logit: {logit_brier_score}', f'LPM: {lpm_brier_score}', sep='\n')