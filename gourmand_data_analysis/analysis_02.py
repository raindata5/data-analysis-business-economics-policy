from google.oauth2 import service_account
from google.cloud import bigquery
import configparser
import pandas as pd
import numpy as np
import random
from collections import Counter
pd.options.display.float_format = '{:,.4f}'.format

KEY_PATH = "/mnt/c/Users/Ron/git-repos/yelp-data/gourmanddwh-f75384f95e86.json"
CREDS = service_account.Credentials.from_service_account_file(KEY_PATH)
client = bigquery.Client(credentials=CREDS, project=CREDS.project_id)

cg_file = open('sql_scripts/county_growth_est.sql','r')
county_growth_query =  cg_file.read()
cg_dataframe: pd.DataFrame = (
    client.query(county_growth_query)
    .result()
    .to_dataframe()
)

cg_dataframe.to_parquet('cg_est', 'pyarrow','snappy', partition_cols=['StateName'])


holding_file = open('sql_scripts/business_daily_holding.sql')
holding_query = holding_file.read()

holding_dataframe : pd.DataFrame = (
    client.query(holding_query)
    .result()
    .to_dataframe()
)
holding_dataframe.shape
holding_dataframe.info()

holding_dataframe.to_parquet('bus_holdings', 'pyarrow','snappy', partition_cols=['CloseDate'])



bus_cat_file = open('sql_scripts/business_category_location.sql')
bus_cat_query = bus_cat_file.read()

bus_cat_dataframe: pd.DataFrame = (
    client.query(bus_cat_query)
    .result()
    .to_dataframe()
)

bus_cat_dataframe.to_parquet('bus_cats', 'pyarrow','snappy', partition_cols=['StateName'])

#[]
# reading in data
cg_df = pd.read_parquet('cg_est', engine='pyarrow')
bh_df: pd.DataFrame = pd.read_parquet('bus_holdings', engine='pyarrow')

#[]
# idea is to found the chain with the largest amount of business
# from here check and see the days the businesses had a gain in reviews oabove some threshold (binary variable)
# then from here look to generalize and see what one could expect in the future
# look at limitations and tradeoffs



#[]
# looking at days with the highest number of business holdings
holdings_count_df = bh_df.groupby(['CloseDate'], as_index=False)['BusinessName'].count()
holdings_count_df.sort_values('BusinessName', ascending=False)

#[]
# Subway and Mcdonald's have the highest number of holdings and therefore businesses, we'll choose Subway and Mcdonald's
bus_count_chn_df = bh_df.groupby(['ChainName'], as_index=False)['BusinessName'].count()
bus_count_chn_df.sort_values('BusinessName', ascending=False).head(10)

#[]
#
# 21494 like so
bh_df.ChainName.str.lower().str.contains(r'mcdonald[\']?s').sum()
# 3 different variations
bh_df.loc[bh_df.ChainName.str.lower().str.contains(r'mcdonald[\']?s'), 'ChainName'].value_counts()


mcdonalds = bh_df.loc[bh_df.ChainName.str.lower().str.contains(r'mcdonald[\']?s'), :].reset_index(drop=True)
mcdonalds.head()
mcdonalds.shape
subway = bh_df.loc[bh_df.ChainName == 'Subway'].reset_index(drop=True)
subway.head()
#[]

# look to get percentage rating increases then get rid of null
subway.columns
subway.head()

#[]
#
ordered_subway = subway.sort_values(['BusinessName', 'CloseDate'], ascending=True).reset_index(drop=True)
ordered_subway

ordered_mcdonalds = mcdonalds.sort_values(['BusinessName', 'CloseDate'], ascending=True).reset_index(drop=True)
ordered_mcdonalds

#[]
# could have used pandas shift if data didn't already have lags
ordered_subway.loc[:, 'review_count_abs_diff_lag_1'] = (ordered_subway.ReviewCount - ordered_subway.previous_review_cnt)
ordered_subway.loc[:, 'review_count_perc_diff_lag_1'] = (ordered_subway['review_count_abs_diff_lag_1'] / ordered_subway.previous_review_cnt) ** 100
ordered_subway.sort_values('review_count_perc_diff_lag_1', ascending=False)
ordered_subway['review_count_perc_diff_lag_1'].value_counts(sort=True)
ordered_subway['review_count_perc_diff_lag_1'].astype(float).value_counts(sort=True)
ordered_subway['review_count_perc_diff_lag_1'].value_counts(bins=10)
# percent differences while useful here may not be of much help so I will stick with abosolute values
ordered_subway['review_count_abs_diff_lag_1'].value_counts(sort=True)

# now the same with mcdonald's
ordered_mcdonalds.loc[:, 'review_count_abs_diff_lag_1'] = (ordered_mcdonalds.ReviewCount - ordered_mcdonalds.previous_review_cnt)
ordered_mcdonalds.loc[:, 'review_count_perc_diff_lag_1'] = (ordered_mcdonalds['review_count_abs_diff_lag_1'] / ordered_mcdonalds.previous_review_cnt) ** 100
ordered_mcdonalds['review_count_abs_diff_lag_1'].value_counts(sort=True)


#[]
# now we'll take the mean value of those days where a any of chain's businesses had a review count increase of 1 or higher
# in other words the relative frequency of this occurring since this is a binary variable
ordered_subway_sin_na = ordered_subway.dropna(subset=['review_count_abs_diff_lag_1'])
ordered_subway_sin_na.loc[:, 'review_count_abs_diff_lag_1_gte_1'] = ordered_subway_sin_na['review_count_abs_diff_lag_1'] >= 1
subway_daily_rc_diff_mean = ordered_subway_sin_na['review_count_abs_diff_lag_1_gte_1'].mean(skipna=True)
subway_daily_rc_diff_mean # .003 rounded

ordered_mcdonalds_sin_na = ordered_mcdonalds.dropna(subset=['review_count_abs_diff_lag_1'])
ordered_mcdonalds_sin_na.loc[:, 'review_count_abs_diff_lag_1_gte_1'] = ordered_mcdonalds_sin_na['review_count_abs_diff_lag_1'] >= 1
mcdonalds_daily_rc_diff_mean = ordered_mcdonalds_sin_na['review_count_abs_diff_lag_1_gte_1'].mean(skipna=True)
mcdonalds_daily_rc_diff_mean # .014 rounded


#[]
# now we can begin with bootstrap sampling
sample_means_subway = []
sample_means_mcdonalds = []
# random.choices allow us to do it with replacement
for i in range(10):
    ixs = random.choices(population = ordered_mcdonalds_sin_na.index.tolist() ,k=ordered_mcdonalds_sin_na.shape[0])
    count_dict = Counter(ixs)
    print(count_dict.most_common(1))

#[]
#
sample_means_subway = []
sample_means_mcdonalds = []
#rn generator
random.seed(42)

for i in range(10000):
    #seems this way would be less efficient but nonetheless viable
    # mcd_ixs = random.choices(population = ordered_mcdonalds_sin_na.index.tolist() ,k=ordered_mcdonalds_sin_na.shape[0])
    # mcd_sample_mean = ordered_mcdonalds_sin_na.loc[mcd_ixs,'review_count_abs_diff_lag_1_gte_1'].mean()
    # sample_means_mcdonalds.append(mcd_sample_mean)

    # subway_ixs = random.choices(population = ordered_subway_sin_na.index.tolist() ,k=ordered_subway_sin_na.shape[0])
    # subway_sample_mean = ordered_subway_sin_na.loc[subway_ixs,'review_count_abs_diff_lag_1_gte_1'].mean()
    # sample_means_subway.append(subway_sample_mean)

    mcd_ixs = random.choices(population = ordered_mcdonalds_sin_na.review_count_abs_diff_lag_1_gte_1.tolist() ,k=ordered_mcdonalds_sin_na.shape[0])
    mcd_sample_mean = np.mean(mcd_ixs)
    sample_means_mcdonalds.append(mcd_sample_mean)

    subway_ixs = random.choices(population = ordered_subway_sin_na.review_count_abs_diff_lag_1_gte_1.tolist() ,k=ordered_subway_sin_na.shape[0])
    subway_sample_mean = np.mean(subway_ixs)
    sample_means_subway.append(subway_sample_mean)

#[]
#
sample_means_subway_mean = np.mean(sample_means_subway)

sample_means_mcdonalds_mean = np.mean(sample_means_mcdonalds)

#[]
#
subway_se = np.std(sample_means_subway)
subway_se # 0.00035

mcdonalds_se = np.std(sample_means_mcdonalds)
mcdonalds_se # 0.00081

#[]
#
print(f'Subway CI with 95% confidence {subway_daily_rc_diff_mean - (subway_se * 2):.5f} <---> {subway_daily_rc_diff_mean + (subway_se * 2):.5f} with ', end=' ')
print(f'and bootstrap se estimate of {subway_se:.5f}')
# Subway CI with 95% confidence 0.00231 <---> 0.00371 with  and bootstrap se estimate of 0.00035
print(f'Mcdonald\'s CI with 95% confidence {mcdonalds_daily_rc_diff_mean - (mcdonalds_se * 2):.5f} <---> {mcdonalds_daily_rc_diff_mean + (mcdonalds_se * 2):.5f} with', end=' ')
print(f'and bootstrap se estimate of {mcdonalds_se:.5f}')
# Mcdonald's CI with 95% confidence 0.01204 <---> 0.01526 with and bootstrap se estimate of 0.00081

#[]
# finish with the SE formula (1/sqrt(n)) * std

subway_formula_se = (1 / np.sqrt(ordered_subway_sin_na.shape[0])) * np.std(ordered_subway_sin_na.review_count_abs_diff_lag_1_gte_1)
subway_formula_se

mcdonalds_formula_se = (1 / np.sqrt(ordered_mcdonalds_sin_na.shape[0])) * np.std(ordered_mcdonalds_sin_na.review_count_abs_diff_lag_1_gte_1)
mcdonalds_formula_se

print(f'Subway CI with 95% confidence {subway_daily_rc_diff_mean - (subway_formula_se * 2):.5f} <---> {subway_daily_rc_diff_mean + (subway_formula_se * 2):.5f} with ', end=' ')
print(f'and se formula estimate of {subway_formula_se:.5f}')
# Subway CI with 95% confidence 0.00230 <---> 0.00372 with  and se formula estimate of 0.00035

print(f'Mcdonald\'s CI with 95% confidence {mcdonalds_daily_rc_diff_mean - (mcdonalds_formula_se * 2):.5f} <---> {mcdonalds_daily_rc_diff_mean + (mcdonalds_formula_se * 2):.5f} with', end=' ')
print(f'and se formula estimate of {mcdonalds_formula_se:.5f}')
# Mcdonald's CI with 95% confidence 0.01204 <---> 0.01526 with and se formula estimate of 0.00080

#[]
# add another hypothesis test as a difference of two means
# why this worked and conclusion