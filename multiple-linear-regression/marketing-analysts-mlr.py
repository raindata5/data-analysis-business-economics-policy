# create a MLR with multiple qualitative variables
# try a similar method with the hotel data set
# data from https://gabors-data-analysis.com/datasets/   hotels-europe
import pandas as pd
import numpy as np
import pyreadstat
import sys
pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 35)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.2f}'.format

#[]
#
import os
from pathlib import Path
path = Path(os.getcwd())
base_dir = path.parent.parent.parent
data_in = os.path.join(str(base_dir),"data-sets/")

hotels_stata, metastata = pyreadstat.read_dta(os.path.join(data_in,'hotels-europe_features.dta'), encoding = "latin1")
prices_stata, metastata = pyreadstat.read_dta(os.path.join(data_in,'hotels-europe_price.dta'), encoding = "latin1")

# hotels_csv = pd.read_csv(os.path.join(data_in,"hotels-europe_features.csv"))

#[]
#
metastata.variable_value_labels.keys() #no labels present
hotels_stata.info()
prices_stata.info()

#[]
# going to merge data on hotelid

hotels_europe = hotels_stata.merge(prices_stata,left_on='hotel_id', right_on='hotel_id')
hotels_europe.info()

#[]
#

def if_not_na(x):
    out = {}
    out['num_miss'] = x.isna().sum()
    out['count'] = x.count()
    return pd.Series(out)

# hotels_europe[['country','city_actual']]



countries = hotels_europe.groupby(['country'])['rating'].apply(if_not_na).unstack()
countries.sort_values('num_miss', ascending=True) # france has a good proportion of values to missing at least in the rating column

#[]
#
hotels_france = hotels_europe.loc[hotels_europe['country']=='France']

cities = hotels_france.groupby(['city_actual'])['stars'].apply(if_not_na).unstack()
cities.sort_values('num_miss', ascending=True)
cities.sort_values('count', ascending=False)

cities.sort_values('count', ascending=False).head(10)
# going to choose Roissy-en-France

#[]
#
ref_france = hotels_europe.loc[(hotels_europe['country']=='France') & (hotels_europe['city_actual']== 'Roissy-en-France')]
ref_france.head()
ref_france.shape

#[]
# going to narrow our data down to hotels

ref_france.accommodation_type.value_counts()
ref_hotel_f = ref_france.loc[ref_france.accommodation_type == 'Hotel']
ref_hotel_f.shape

#[]
#




ref_hotel_f.columns.to_list()
ref_hotel_f = ref_hotel_f.loc[ref_hotel_f.nnights== 1.0] #getting just the observations where the number of nights is 1
ref_hotel_f.shape

#[]
#
ref_hotel_f.stars.value_counts(normalize=True)
ref_hotel_f.reset_index(drop=True,inplace=True)

from sklearn.model_selection import StratifiedShuffleSplit
shuffler = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in shuffler.split(ref_hotel_f,ref_hotel_f['stars']):
    ref_hotel_strat_train = ref_hotel_f.loc[train_index]
    ref_hotel_strat_test = ref_hotel_f.loc[test_index]

ref_hotel_strat_train.shape
ref_hotel_strat_test.shape

ref_hotel_strat_train.stars.value_counts(normalize=True)
ref_hotel_strat_test.stars.value_counts(normalize=True)
ref_hotel = ref_hotel_strat_train.copy()

#[]
#
ref_hotel.info()
ref_hotel.head()
from pandas.plotting import scatter_matrix
imp_attributes = ['price','rating','stars','distance']

import matplotlib.pyplot as plt
scatter_matrix(ref_hotel[imp_attributes], figsize=(12,8))
plt.show()
#price seems skewed and the distance seem rather too close to make much of a distance
#this seems to show in the scatter plot
#[]
#
import seaborn as sns
sns.boxplot(ref_hotel['price'])
plt.show()

ref_hotel['price'].plot.hist()
plt.show()
ref_hotel['price'].describe()

#[]
# seems best to take the log of price
ref_hotel['ln_price'] = np.log(ref_hotel.price)
ref_hotel_strat_train['ln_price'] = np.log(ref_hotel_strat_train.price)
sns.boxplot(ref_hotel['ln_price'])
plt.show()

ref_hotel['ln_price'].plot.hist()
plt.show()



#[]
# for this distance variable it seems as the distance increases the price goes up perhaps as the hotels begin to toward toward some alternate center
ref_hotel['distance'].describe()
ref_hotel['distance'].plot.hist()
plt.show()
sns.boxplot(ref_hotel['distance'])
plt.show()
ref_hotel['distance'].value_counts()

ref_hotel['distance_cat'] = ref_hotel['distance'].astype('category')

sns.boxplot(y=ref_hotel['distance_cat'], x=ref_hotel['ln_price'] )
plt.show()

#[]
# same pattern present for next distance variable
ref_hotel['distance_alter'].value_counts()
ref_hotel['distance_alter_cat'] = ref_hotel['distance_alter'].astype('category')

sns.boxplot(y=ref_hotel['distance_alter_cat'], x=ref_hotel['ln_price'] )
plt.show()


#[]
# checking out the associations these variables have with ln_price
# there could be multicolinearity between rating and stars so I would have to see how that affects
#the SE's of the coefficients
var= ['rating','stars','distance']
fig , axes = plt.subplots(ncols=1, nrows=3)

axes.ravel()
for v,ax in zip(var ,axes) :
    ax.scatter(y=ref_hotel['ln_price'], x= ref_hotel[v])
    ax.set_xlabel(v)
    ax.set_ylabel('ln_price')
plt.tight_layout()
plt.show()

#[] going to continue with
#
import statsmodels.api as sm
import statsmodels.formula.api as smf

model1 = smf.ols('ln_price ~ rating + C(stars) + distance', data=ref_hotel_strat_train).fit(cov_type='HC1')
model1.summary()

#[]
# y and yhat plot
y_hat = model1.predict()

fig, ax = plt.subplots(1,1)
ax.scatter(x=y_hat, y=ref_hotel_strat_train['ln_price'], color='red',alpha=0.4 )
ax.set_xlabel('predicted ln_price')
ax.set_ylabel('ln_price')
ax.axline([0, 0], [1, 1])
plt.tight_layout()
plt.show()

#[]
# joining the predictions to training set data
ref_hotel_strat_train.reset_index(drop=True,inplace=True)

y_hat = pd.DataFrame(y_hat, columns=['y_hat'])
ref_hotel_strat_train = ref_hotel_strat_train.merge(y_hat , left_index=True, right_index=True)

fig, ax = plt.subplots(1,1)
ax.scatter(x=ref_hotel_strat_train['y_hat'], y=ref_hotel_strat_train['ln_price'], color='red',alpha=0.4 )
ax.set_xlabel('predicted ln_price')
ax.set_ylabel('ln_price')
ax.axline([0, 0], [1, 1])
plt.tight_layout()
plt.show()

#[]
# with the residual we can sort the data by the hotels with the best deal (cheaper than what was predicted for their price)
ref_hotel_strat_train['residual'] = ref_hotel_strat_train['ln_price'] - ref_hotel_strat_train['y_hat']

ref_hotel_strat_train.sort_values('residual', ascending=True).head(10)


#[]
#
ref_hotel_strat_train.to_pickle('views/ref-hotel-strat-train.pkl')
ref_hotel_strat_test.to_pickle('views/ref-hotel-strat-test.pkl')
ref_hotel_f.to_pickle('views/ref-hotel-f.pkl')

# try a quadratic with one of the vars
#test model








ref_hotel['stars'].value_counts()
['hotel_id',
 'country',
 'city_actual',
 'rating_count',
 'distance',
 'center1label',
 'distance_alter',
 'center2label',
 'neighbourhood',
 'city',
 'stars',
 'ratingta',
 'ratingta_count',
 'accommodation_type',
 'rating',
 'price',
 'scarce_room',
 'offer',
 'offer_cat',
 'year',
 'month',
 'weekend',
 'holiday',
 'nnights']
#[]
ref_hotel[['offer','offer_cat']].value_counts()
pd.crosstab(ref_hotel['offer'], ref_hotel['offer_cat'])
