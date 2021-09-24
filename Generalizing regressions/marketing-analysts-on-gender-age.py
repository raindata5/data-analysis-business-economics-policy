import pandas as pd
import numpy as np
import pyreadstat
import json
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures #alternative
import statsmodels.formula.api as smf
import pyarrow

pd.options.display.float_format = '{:,.2f}'.format
# Desktop/Rain data/da-for-bep/da_data_repo/cps-earnings/raw/morg2014.csv

pd.set_option('display.max_columns',8)
pd.set_option('display.width', 80)

#important variables: sex,highestgrade attend,

cps2014 , metastata = pyreadstat.read_dta('data-sets/morg19.dta')

#the variable value labels seem fine so I load the data in with that
metastata.variable_value_labels.keys()

metastata.variable_value_labels['docc00']
metastata.variable_value_labels['paidhre']
metastata.variable_value_labels['hrhtype']

#ihigrdc, earnhr, earnwke, ind cols, uhours (main job) ftpt..,


cps2014 , metastata = pyreadstat.read_dta('data-sets/morg19.dta', apply_value_formats=True,formats_as_category=True)

cps2014.columns = metastata.column_labels


#data dictionary shows male is 1 and female is 2

#important variable to analysis
# ['hhid', 'hrhhid2', 'lineno']
# ['earnwke','uhourse','occ2012','grade92']

#[]
#

cps2014.columns = cps2014.columns.str.lower().str.strip().str.replace(r' ', '_').str.replace('[^0-9a-z_]','')




# having converted column names to the full variable names i can now see which ones to keep
important_variables = ['hhid', 'intmonth', 'stfips', 'weight', 'earnwke',
       'uhourse', 'grade92', 'race', 'ethnic', 'age', 'sex', 'marital',
       'ownchild', 'chldpres', 'prcitshp', 'stfips', 'ind02', 'occ2012',
       'class94', 'unionmme', 'unioncov', 'lfsr94']


cps2014.columns = metastata.column_names

cps2014_analysis = cps2014[important_variables]

#[]
# going to pull the marketing analysts

cps2014_analysis.occ2012.value_counts()
(cps2014_analysis['occ2012'] == 735).sum() #735 coressponds to the code for marketing analysts

cps2014_analysis_analyst = cps2014_analysis.loc[cps2014_analysis['occ2012'] == 735]
cps2014_analysis_analyst.shape # 378 marketing analysts in data

#[]
#
cps2014_analysis_analyst.reset_index(inplace=True)
cps2014_analysis_analyst.info()
cps2014_analysis_analyst.age.value_counts()
cps2014_analysis_analyst.uhourse.value_counts(dropna=False)
#going to drop the nan values and also the -4.00
cps2014_analysis_analyst = cps2014_analysis_analyst[(cps2014_analysis_analyst.uhourse.notna()) & (cps2014_analysis_analyst.uhourse >= 0 )]
#dropping where earnwke is nan (#check if any demographic factors associated with these nulls)

cps2014_analysis_analyst.earnwke.value_counts(dropna=False)
cps2014_analysis_analyst.earnwke.value_counts(dropna=False).sort_index()
#some people have null values on weekly earning but hours worked so we'll exlcude them
cps2014_analysis_analyst.loc[cps2014_analysis_analyst.earnwke.isna(),['earnwke','uhourse']]

cps2014_analysis_analyst.copy().dropna(subset=['earnwke'], inplace=True)

#[]
# create a wage variable based by dividing  weekly earning by hours worked per week
cps2014_analysis_analyst.copy()
cps2014_analysis_analyst.copy()['wage'] = cps2014_analysis_analyst.copy()['earnwke'] / cps2014_analysis_analyst.copy()['uhourse']

cps2014_analysis_analyst.copy().loc[:,'wage'] = cps2014_analysis_analyst.copy().loc[:,'wage'].astype('int64')

#[]
# wage seems right skewed with just one value over $286
cps2014_analysis_analyst['wage'].skew()
cps2014_analysis_analyst['wage'].kurtosis()
cps2014_analysis_analyst['wage'].describe()
cps2014_analysis_analyst['wage'].value_counts(bins=7)
plt.hist(cps2014_analysis_analyst['wage'])
plt.show()
sns.boxplot(cps2014_analysis_analyst['wage'])
plt.show()
#going to take log
cps2014_analysis_analyst = cps2014_analysis_analyst.copy()
cps2014_analysis_analyst['ln_wage'] = np.log(cps2014_analysis_analyst['wage'])

#[]
#
plt.hist(cps2014_analysis_analyst['ln_wage'])
plt.show()

#[]
#making columns where females = 1 and males = 0 (binary variable for regression)
cps2014_analysis_analyst['sex'].value_counts() #checking values beforehand to ensure integrity

cps2014_analysis_analyst['female'] = cps2014_analysis_analyst['sex'].map({1:0,2:1})

#[]
#

def getlm(regresson,regressor):
    Y=regresson
    X = regressor
    X = sm.add_constant(X)
    model = sm.OLS(Y,X).fit(cov_type='HC1')
    return model

lrm = getlm(cps2014_analysis_analyst['ln_wage'],cps2014_analysis_analyst[['female','age']])
lrm_w = getlm(cps2014_analysis_analyst['wage'],cps2014_analysis_analyst[['female','age']])

#[]
# see what happens when female with extremely high wage is removed
(cps2014_analysis_analyst['wage'] < 286).sum() #this outlier

cps2014_analysis_analyst_no = cps2014_analysis_analyst[cps2014_analysis_analyst['wage'] < 286]

lrm_w_no = getlm(cps2014_analysis_analyst_no['wage'],cps2014_analysis_analyst_no[['female','age']])
lrm_w_no.summary()

#[]
# will try a quadratic on age
cps2014_analysis_analyst['age_sq'] = np.power(cps2014_analysis_analyst['age'], 2)

def get_quad_r():
    reg = smf.ols(formula = 'ln_wage~age + age_sq',data =cps2014_analysis_analyst ).fit(cov_type="HC1")
    return reg

lrm_quad = get_quad_r()
lrm_quad.summary()
sns.regplot(y='ln_wage', x='age', data=cps2014_analysis_analyst,lowess=True)
plt.show() #- age squared corresponds to this upside down parabola which is indicating that
#as age increases the wage will go up until a certain point where it will go down

# #r-squared is still kind of low will view data on graph

# sns.regplot( y =['ln_wage'], x=['female','age'], data = cps2014_analysis_analyst)
# plt.show()

#saving to feather file to continue analysis a bit later
cps2014_analysis_analyst.reset_index(inplace=True)
cps2014_analysis_analyst.to_pickle("views/cps2014_analysis_analyst.pkl")
test = pd.read_pickle("views/cps2014_analysis_analyst.pkl")
test.head()

