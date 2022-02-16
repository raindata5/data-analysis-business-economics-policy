from datetime import date
import pandas as pd
import numpy as np
from scipy import stats
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

mcdonalds_one_month = mcdonalds.loc[mcdonalds['CloseDate'].between(left=date(2022, 1,9), right=date(2022, 2,9), inclusive='both')]


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
relative_change_gte_15_stat # 9 %

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