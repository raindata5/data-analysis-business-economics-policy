import pandas as pd
import numpy as np
import random
from collections import Counter
from pathlib import Path
import os
from statsmodels.stats.api import CompareMeans, DescrStatsW
pd.options.display.float_format = '{:,.4f}'.format
# between Florida and New York are there differences between rating or review percentage differences
# would take the sums of the current day minus the sums of the previous day divided by the sums of the previous day ? (take the per capita stats based on population data we have)
# h0 : 0
# ha : != 0 (i.e. 2 sided)

#[]
#

cg_df: pd.DataFrame = pd.read_parquet('cg_est', engine='pyarrow')
bh_df: pd.DataFrame = pd.read_parquet('bus_holdings', engine='pyarrow')
bus_cats_df: pd.DataFrame = pd.read_parquet('bus_cats', engine='pyarrow')

#[]
# isolating california and Florida
florida_newyork = cg_df.loc[cg_df['StateName'].isin(values=['Florida', 'New York'])]
florida_newyork

#[]
# getting population counts as estimated in the census data
florida_newyork_pop_sums = florida_newyork.groupby(['StateName'], as_index=False)['EstimatedPopulation'].sum()
florida_newyork_pop_sums.sort_values(by='EstimatedPopulation', ascending=False).head()

#[]
# These estimates are very similar to the actual population estimates currently
florida_pop = florida_newyork_pop_sums.loc[florida_newyork_pop_sums.StateName == 'Florida', 'EstimatedPopulation']
florida_pop

newyork_pop = florida_newyork_pop_sums.loc[florida_newyork_pop_sums.StateName == 'New York', 'EstimatedPopulation']
newyork_pop

#[]
# There is some missing data here in bh_df so I will need join this with another df
bh_df.info()
bus_cats_df.info()

#[]
#
bus_cats_df_uq_bus = bus_cats_df.groupby(['BusinessName']).first()
bus_cats_df_uq_bus.shape

#[]
# unfortunately can't join on int column such as businesskey
bus_cats_df_uq_bus_newyork_fl = bus_cats_df_uq_bus.loc[bus_cats_df_uq_bus['StateName'].isin(values=['Florida', 'New York'])] 
bus_cats_df_uq_bus_newyork_fl.shape
bh_df_loc_fl_newyork = pd.merge(left=bh_df, right=bus_cats_df_uq_bus_newyork_fl, on='BusinessName', how='inner', suffixes=(None, '_right'))
bh_df_loc_fl_newyork.shape # (1982344, 21)
bh_df_loc_fl_newyork.info()

#[]
# due to restrictions such as a mask requirement , could this lead to a difference
# in the amount of traffic a restaurant receives?
bh_df_loc_fl = bh_df_loc_fl_newyork.loc[bh_df_loc_fl_newyork.StateName == 'Florida']
bh_df_loc_newyork = bh_df_loc_fl_newyork.loc[bh_df_loc_fl_newyork.StateName == 'New York']
bh_df_loc_fl.shape
bh_df_loc_newyork.shape

#[]
# will first get rid of the first day a business appears since that will have a null value for a business review count change
bh_df_loc_fl.isna().sum()
bh_df_loc_fl_sin_na = bh_df_loc_fl.dropna(axis=0, subset=['previous_review_cnt']).reset_index(drop=True)
bh_df_loc_fl_sin_na.isna().sum()

bh_df_loc_newyork_sin_na = bh_df_loc_newyork.dropna(axis=0, subset=['previous_review_cnt']).reset_index(drop=True)

#[]
# H0: avg_review_count_difference(Florida) - avg_review_count_difference(New York) = 0
# Ha: avg_review_count_difference(Florida) - avg_review_count_difference(New York) != 0 

fl_review_diff_mean = bh_df_loc_fl_sin_na['abs_review_diff'].mean()
newyork_review_diff_mean = bh_df_loc_newyork_sin_na['abs_review_diff'].mean()

mean_diff = fl_review_diff_mean - newyork_review_diff_mean
mean_diff

#[]
# Now we can conduct a hypothesis test and carry it out at a significance level of 5%
cm = CompareMeans(DescrStatsW(bh_df_loc_fl_sin_na['abs_review_diff']), DescrStatsW(bh_df_loc_newyork_sin_na['abs_review_diff']))
summary = cm.summary(usevar='unequal')
summary.as_text()
summary.as_html()
data_df = pd.DataFrame(data=summary.data)
data_df
# 0                   coef    std err          t   P>|t|     [0.025     0.975]
# 1  subset #1      0.0161      0.004      4.093   0.000      0.008      0.024

#[]
# Here we can see the results of the test with  a few different notes
# First and foremost the standard error formula is dependent on whether the observations are independent across the frequency in the data.
# in our case this corresponds to days . Now in this data we took daily differences so that in theory should help a bit with this
# dependency across days by lowering it
# apart from that in terms of our results the tests from stats.models offers us a couple of alternatives on which to base 
# a conclusion
# If we want to use simply the t-statistic then this is more than enough evidence to allow us to reject the null at a 5%
# significance level since it is at 4 i.e. far away from 2. We could also use the p-value as well which tells us the probability
# we'll get a statistic higher than what we observed . In this case it shows as 0.000 probably beecause it's rounding but this
# as well is not close to 5%. 
# Finally using the confidence interval we can see that zero is not included inside of it although it is close.
# This could pose an issue as the results of our hypothesis test is more definitive the more samples we have and the farther away the 
# true value of the statiistic is from zero

# continue here and restate question
# unfortunately review count isn't the best variable to measure customer traffic as customer could have food delivered and subvert 
# any mask requirement (We do have data on which business offer a delivery option so this could be an interesting robustness check)
