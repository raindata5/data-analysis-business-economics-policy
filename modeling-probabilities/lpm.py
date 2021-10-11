import os
from pathlib import Path
import pyreadstat
import pandas as pd
import numpy as np

pd.set_option('display.width', 200)
pd.set_option('display.max_columns', 35)
pd.set_option('display.max_rows', 200)
pd.options.display.float_format = '{:,.2f}'.format

# #[]
# #
# path = Path(os.getcwd())
# base_dir = path.parent.parent
# data_in = os.path.join(str(base_dir),"data-cleaning/importing-data/data/") #must use dobule quotes with these operations

# nls_data, metastata = pyreadstat.read_dta(os.path.join(data_in,"nls97.dta"), apply_value_formats=True, formats_as_category=True)

# #[]
# #

# nls_data.info()
# nls_data.columns = metastata.column_labels

#[]
#
path = Path(os.getcwd())
base_dir = path.parent
data_in = os.path.join(str(base_dir),"data-sets/")
nls97 = pd.read_csv(os.path.join(data_in,"nls97b.csv"))
# nls97.set_index('personid', inplace=True)
# nls97.reset_index(inplace=True)


#[]
#

nls97.info()

#[]
#
nls97_mar_cat = nls97.dropna(subset=['maritalstatus'],axis=0)
marital_list =['Married', 'Never-married', 'Divorced', 'Separated', 'Widowed']
# marital_dict = {}

for cat in marital_list:
    nls97_mar_cat = nls97_mar_cat.copy()
    nls97_mar_cat.loc[:,cat] = np.where(nls97_mar_cat['maritalstatus'] == cat, 1, 0)
    # marital_dict[cat] = nls97_mar_cat

#[]
#
# marital_dict.keys()
# marital_dict['Married']
import matplotlib.pyplot as plt
nls97_mar_cat.wageincome.plot.hist()
plt.show()
# is super skewed may want to come back and transform to log although log on right side can mess up interpretation

#[]
#
nls97_mar_cat = nls97.dropna(subset=['highestdegree'],axis=0)
nls97_mar_cat.reset_index(drop =True , inplace=True)
nls97_mar_cat.highestdegree.value_counts().index.to_list()

dummy_vars = pd.get_dummies(nls97_mar_cat['highestdegree'])

nls97_mar_cat.merge(dummy_vars, left_index=True, right_index=True).iloc[:5,[18,-1,-2,-3,-4,-5,-6,-7]] #verification
nls97_mar_cat = nls97_mar_cat.merge(dummy_vars, left_index=True, right_index=True)

nls97_mar_cat.columns.to_list()[29:87]
nls97_mar_cat.drop(columns=nls97_mar_cat.columns.to_list()[29:87], inplace=True) #too many columns
#[]
#
# ['2. High School',  '4. Bachelors',  '1. GED',  '0. None',  '3. Associates',  '5. Masters',  '7. Professional',  '6. PhD']
nls97_mar_cat[[]].value_counts(dropna=False)

nls97_mar_cat = nls97_mar_cat.dropna(subset=['wageincome'],axis=0)
nls97_mar_cat.reset_index(drop=True,inplace=True)

import statsmodels.api as sm

X = nls97_mar_cat[['wageincome', '4. Bachelors',  '1. GED',  '0. None',  '3. Associates',  '5. Masters',  '7. Professional',  '6. PhD']]
y = nls97_mar_cat['Married']
X = sm.add_constant(X)

model = sm.OLS(y,X).fit(cov_type='HC1')
model.summary()

#[]
#

y_hat = model.predict()
y_hat = pd.DataFrame(y_hat, columns=['y_hat'])
nls97_mar_cat = nls97_mar_cat.merge(y_hat , left_index=True, right_index=True)

#[]
# distribution of predicted probabilities
those_married =  nls97_mar_cat.loc[nls97_mar_cat['Married'] == 1,['Married','y_hat']]
those_not_married = nls97_mar_cat.loc[nls97_mar_cat['Married'] == 0,['Married','y_hat']]


plt.hist(those_married['y_hat'], color='pink', label='married', histtype='step')
plt.hist(those_not_married['y_hat'], color='blue', label= 'not married', histtype='step')
plt.legend(loc='upper left')
plt.show()
#while those not married tend to have lower probabilities than those married there is far too much overlap
# will calculate briers score after but judging by r-sqaured it would seem low
# despite r-squared not  being a good metric of fit here
#[]
#
nls97_mar_cat

#on return *****
# calculate briers score
    # calc. calibration curve
    # do the rest of the nominal variables in marital status
# calculate probit and logit
#do analysis on ordinal variable (like = 'gov')
#do analysis on survival time variable (like='hrs')



['personid',
 'gender',
 'birthmonth',
 'birthyear',
 'highestgradecompleted',
 'maritalstatus',
 'childathome',
 'childnotathome',
 'wageincome',
 'weeklyhrscomputer',
 'weeklyhrstv',
 'nightlyhrssleep',
 'satverbal',
 'satmath',
 'gpaoverall',
 'gpaenglish',
 'gpamath',
 'gpascience',
 'highestdegree',
 'govprovidejobs',
 'govpricecontrols',
 'govhealthcare',
 'govelderliving',
 'govindhelp',
 'govunemp',
 'govincomediff',
 'govcollegefinance',
 'govdecenthousing',
 'govprotectenvironment',
 'weeksworked00']
