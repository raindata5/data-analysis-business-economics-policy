from typing import Union
from datetime import date, datetime
import pandas as pd
import numpy as np
from scipy import stats
import joblib
pd.options.display.float_format = '{:,.4f}'.format

#[]
# in a previous analysis we determined that it would be better to invest in Mcdonalds , now with this in mind what if we have 
# an addtional threshold by which we will decide whether or not it is a good idea to invest in Mcdonald's? Something like the
# rate of a review count change greater than 5% over a period of 30 days for each business location?

#[]
#
bh_df: pd.DataFrame = pd.read_parquet('bus_holdings', engine='pyarrow')
mcdonalds = bh_df.loc[bh_df.ChainName.str.lower().str.contains(r'mcdonald[\']?s'), :].reset_index(drop=True)

#[]
# get businesses with 30 days of data more or less
#
mcdonalds.info()
mcdonalds['CloseDate'] = pd.to_datetime(mcdonalds['CloseDate']).dt.date

mcdonalds['CloseDate'].min()
mcdonalds['CloseDate'].max()

mcdonalds_one_month = mcdonalds.loc[mcdonalds['CloseDate'].between(left=date(2022, 1,9), right=date(2022, 2,9), inclusive='left')]


bus_instance_counts = mcdonalds_one_month.groupby(by=['BusinessName'], as_index=False)['CloseDate'].count() 
bus_instance_counts_gte_30 = bus_instance_counts.loc[bus_instance_counts.CloseDate >= 30]
bus_instance_counts_gte_30

mcd_bus_instances_gte_30 = mcdonalds[mcdonalds['BusinessName'].isin(bus_instance_counts_gte_30['BusinessName'])]
mcd_bus_instances_gte_30.head()
mcd_bus_instances_gte_30.shape

#[]
# 
mcd_bus_instances_gte_30_sorted = mcd_bus_instances_gte_30.sort_values(by=['BusinessName', 'CloseDate'], ascending=True)
mcd_bus_instances_gte_30_sorted

#[]
# take first and last and find the total difference then make sure above a threshold
# let's say we want at least 5% of business to have a review count change of 15 percent or higher

first_last_review_cnts = mcd_bus_instances_gte_30_sorted.groupby(['BusinessName'], as_index=False).agg({'CloseDate': ['first','last'], 'ReviewCount': ['first','last']})
first_last_review_cnts['relative_change'] = ((first_last_review_cnts['ReviewCount']['last'] - first_last_review_cnts['ReviewCount']['first']) / first_last_review_cnts['ReviewCount']['first']) * 100
first_last_review_cnts['relative_change']
first_last_review_cnts['relative_change_gte_15'] = np.where(first_last_review_cnts['relative_change'] >= 15, 1, 0)
relative_change_gte_15_stat = first_last_review_cnts['relative_change_gte_15'].mean() * 100
relative_change_gte_15_stat # 14 %

#[]
# our goal is to have relative_change_gte_15_stat - 5% > 0  so we can make the following hypothesis test
# H0: relative_change_gte_15_stat - 5% <= 0
# Ha: relative_change_gte_15_stat - 5% > 0

# we can do a two-sided test and then split the p-value in 2 since we are only concerning with whether the popmean would positive
# after subtracting our threshold and we already have acheived a statistic greater than our threshold
# otherwise if we got a value lower than the null from the get-go (beginning) there would have been no need to continue
test_result = stats.ttest_1samp(a=first_last_review_cnts['relative_change_gte_15'], popmean=.05, nan_policy="omit", alternative='two-sided')
test_result
test_result.pvalue / 2 # 0.013651847138385078

#[]
# Here is a method of doing the equivalent in scipy but specfically for a one-sided test
test_result2 = stats.ttest_1samp(a=first_last_review_cnts['relative_change_gte_15'], popmean=.05, nan_policy="omit", alternative='greater')
test_result2

# so in this case one could say that we could invest in Mcdonald's since we can expect that their businesses' review counts
# do increase by 15% more than 5% of the days 
# only if it weren't for the fact that our sample size is pretty small so this would require us to lower
# our level of signficance and so if we did so at 1% then we would end up not rejecting the null hypothesis
# or we could conduct further analysis 

#[]
# Another question we could ask is whether this is the sort of thing we could observe across businesses, so is Mcdonald's even on
# a higher playing field? perhaps the standard we gave it is too low?


bh_df['CloseDate'] = pd.to_datetime(bh_df['CloseDate']).dt.date

bh_df_one_month = bh_df.loc[bh_df['CloseDate'].between(left=date(2022, 1,9), right=date(2022, 2,9), inclusive='left')]

bh_df_instance_counts = bh_df_one_month.groupby(by=['BusinessName'], as_index=False)['CloseDate'].count() 
bh_df_instance_counts_gte_30 = bh_df_instance_counts.loc[bh_df_instance_counts.CloseDate >= 30]
bh_df_instance_counts_gte_30

#[]
#
bh_df_bus_instances_gte_30 = bh_df[bh_df['BusinessName'].isin(bh_df_instance_counts_gte_30['BusinessName'])]
bh_df_bus_instances_gte_30.head()
bh_df_bus_instances_gte_30.shape

#[]
#
bh_df_bus_instances_gte_30_sorted = bh_df_bus_instances_gte_30.sort_values(by=['BusinessName', 'CloseDate'], ascending=True)
bh_df_bus_instances_gte_30_sorted

first_last_review_cnts_bh_df = bh_df_bus_instances_gte_30_sorted.groupby(['BusinessName'], as_index=False).agg({'CloseDate': ['first','last'], 'ReviewCount': ['first','last']})
first_last_review_cnts_bh_df['relative_change'] = ((first_last_review_cnts_bh_df['ReviewCount']['last'] - first_last_review_cnts_bh_df['ReviewCount']['first']) / first_last_review_cnts_bh_df['ReviewCount']['first']) * 100
first_last_review_cnts_bh_df['relative_change']

#[]
#

first_last_review_cnts_bh_df['relative_change_gte_15'] = np.where(first_last_review_cnts_bh_df['relative_change'] >= 15, 1, 0)
relative_change_gte_15_stat_bh_df = first_last_review_cnts_bh_df['relative_change_gte_15'].mean() * 100
relative_change_gte_15_stat_bh_df # 4.319371727748691

#[]
# it's a bit below 5% , so it's in the null hypothesis and we can see that with the t-statistic and there's not much reason to conduct the hypothesis test
# nonetheless we can carry it out and see what happens
test_result3_t_stat = (first_last_review_cnts_bh_df['relative_change_gte_15'].mean() - 0.05) / (first_last_review_cnts_bh_df['relative_change_gte_15'].std() / np.sqrt(first_last_review_cnts_bh_df['relative_change_gte_15'].shape[0]))
test_result3_t_stat


test_result3 = stats.ttest_1samp(a=first_last_review_cnts_bh_df['relative_change_gte_15'], popmean=.05, nan_policy="omit", alternative='greater')
test_result3


# now we have even more confirmation with p-value

#[]
# having tested one hypothesis at a time we can also test multiple while also keeping in mind the caveats
# here we are going to see how this applies to the top 10 most populous states for the states in the bottom 10 in terms of 
# population
cg_df = pd.read_parquet('cg_est', engine='pyarrow')

state_counts = cg_df.groupby(['StateName'], as_index=False)['EstimatedPopulation'].sum()
state_counts_sorted = state_counts.sort_values(by='EstimatedPopulation', ascending=True)
#[] we'll get rid of district of columbia and it included with virginia wouldn't bump it up either way
state_counts_sorted_wo_dc = state_counts_sorted.loc[state_counts_sorted.StateName != 'District of Columbia'].reset_index(drop=True)

top10_low10 = state_counts_sorted_wo_dc[(state_counts_sorted_wo_dc.index < 10)  | (state_counts_sorted_wo_dc.index > 40)]
top10_low10.shape

#[]
# top10_low10.unique()

bus_cats_df: pd.DataFrame = pd.read_parquet('bus_cats', engine='pyarrow')

bus_cats_df_uq_bus = bus_cats_df.groupby(['BusinessName']).first()

bus_cats_df_uq_bus_states = bus_cats_df_uq_bus.loc[bus_cats_df_uq_bus['StateName'].isin(values=top10_low10.StateName.unique())] 
bus_cats_df_uq_bus_states.shape
bh_df_loc_states = pd.merge(left=bh_df, right=bus_cats_df_uq_bus_states, on='BusinessName', how='inner', suffixes=(None, '_right'))
bh_df_loc_states.shape # (929550, 21)
bh_df_loc_states.info()

#[]
#
bh_df_loc_states_cut =  bh_df_loc_states.loc[bh_df_loc_states['CloseDate'].between(left=date(2022, 1,9), right=date(2022, 2,9), inclusive='left')]


bh_df_loc_states_cut_sorted = bh_df_loc_states_cut.sort_values(by=['BusinessName', 'CloseDate'], ascending=True)
bh_df_loc_states_cut_sorted

#[]
#
first_last_review_cnts_loc_states = bh_df_loc_states_cut_sorted.groupby(['BusinessName'], as_index=False).agg({'CloseDate': ['first','last'], 'ReviewCount': ['first','last'], 'StateName': ['first']})

first_last_review_cnts_loc_states['relative_change'] = ((first_last_review_cnts_loc_states['ReviewCount']['last'] - first_last_review_cnts_loc_states['ReviewCount']['first']) / first_last_review_cnts_loc_states['ReviewCount']['first']) * 100
first_last_review_cnts_loc_states['relative_change']

first_last_review_cnts_loc_states['relative_change_gte_15'] = np.where(first_last_review_cnts_loc_states['relative_change'] >= 15, 1, 0)

#[]
#
first_last_review_cnts_loc_states['StateName1'] = first_last_review_cnts_loc_states['StateName']['first']
def group_t_stat_and_mean(x: pd.Series, a_number: Union[float, int], the_alternative: str= 'two-sided') -> pd.Series:
    stats_dict = {}
    stats_dict['stn_dev'] = np.std(x)
    stats_dict['array_mean'] = x.mean()
    stats_dict['array_count'] = x.shape[0]
    stats_dict['stn_err'] = stats_dict['stn_dev'] / np.sqrt(stats_dict['array_count']) 
    stats_dict['array_tstat'] = (stats_dict['array_mean'] - a_number) / stats_dict['stn_err']
    stats_dict['ttest'] = stats.ttest_1samp(a=x, popmean=a_number, nan_policy="omit", alternative=the_alternative)
    return pd.Series(stats_dict)

multiple_hypothesis_test_results = first_last_review_cnts_loc_states.groupby(by=['StateName1'], as_index=False, observed=True)['relative_change_gte_15'].apply(group_t_stat_and_mean, .05, 'greater')

joblib.dump(multiple_hypothesis_test_results, 'multiple_hypothesis_test_results_dump')
joblib.dump(top10_low10, 'top10_low10_states_dump')

#[]
# come back and label top 10 businesses and bottom 10 then interpret
# joblib.load('multiple_hypothesis_test_results')
