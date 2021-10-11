import pandas as pd
import numpy as np
import pyreadstat
import json
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
#from sklearn.preprocessing import PolynomialFeatures #alternative
import statsmodels.formula.api as smf
from collections import Counter
from pyod.models.knn import KNN
from sklearn.preprocessing import StandardScaler

# import pyarrow #issues saving the file as a ftr file

# Desktop/Rain data/da-for-bep/da_data_repo/cps-earnings/raw/morg2014.csv

pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_columns',8)
pd.set_option('display.width', 80)
pd.set_option('display.max_rows',50)


#loading in data and the metaobject
#[]
#
cps2014 , metastata = pyreadstat.read_dta('data-sets/morg19.dta')

#the variable value labels seem fine
metastata.variable_value_labels.keys()

metastata.variable_value_labels['docc00']
metastata.variable_value_labels['paidhre']
metastata.variable_value_labels['hrhtype']

#ihigrdc, earnhr, earnwke, ind cols, uhours (main job) ftpt.., #important variables: sex,highestgrade attend,

#[]
#
#loading in the data again with the assigned variable value labels, and assigned column names
cps2014 , metastata = pyreadstat.read_dta('data-sets/morg19.dta', apply_value_formats=True,formats_as_category=True)

cps2014.columns = metastata.column_labels


#data dictionary shows male is 1 and female is 2

#important variable to analysis
# ['hhid', 'hrhhid2', 'lineno']
# ['earnwke','uhourse','occ2012','grade92']

#[]
#

cps2014.columns = cps2014.columns.str.lower().str.strip().str.replace(r' ', '_').str.replace('[^0-9a-z_]','')



#[]
# having converted column names to the full length names i can now see which ones to keep
important_variables = ['hhid', 'intmonth', 'stfips', 'weight', 'earnwke',
       'uhourse', 'grade92', 'race', 'ethnic', 'age', 'sex', 'marital',
       'ownchild', 'chldpres', 'prcitshp', 'stfips', 'ind02', 'occ2012',
       'class94', 'unionmme', 'unioncov', 'lfsr94']

# now going to convert col. names back to original to separate those needed for the moment
cps2014.columns = metastata.column_names

cps2014_analysis = cps2014.loc[:,important_variables]

#[]
#

#[]
# going to pull the marketing analysts

cps2014_analysis.occ2012.value_counts()
(cps2014_analysis['occ2012'] == 735).sum() #735 corresponds to the code for marketing analysts

cps2014_analysis_analyst = cps2014_analysis.loc[cps2014_analysis['occ2012'] == 735]
cps2014_analysis_analyst.shape # 378 marketing analysts in data

#[]
#
cps2014_analysis_analyst.reset_index(inplace=True)
cps2014_analysis_analyst.info()

#[]
# going to analyze the nulls and the data in general to find any logical inconsistencies
cps2014_analysis_analyst.age.value_counts()
cps2014_analysis_analyst.uhourse.value_counts(dropna=False)

#[]
# seems we would want to drop the nan values and also the -4.00 from the uhourse however just to confirm this doesn't produce any sampling bias
# we will inspect these particular observations to see if they tend to share any demographic variables
odd_uhourse = cps2014_analysis_analyst.loc[(cps2014_analysis_analyst.uhourse<0) | (cps2014_analysis_analyst.uhourse.isna())]
odd_uhourse.sample(10,random_state =10)
# reg_uhourse = cps2014_analysis_analyst.loc[(cps2014_analysis_analyst.uhourse>=0) | (cps2014_analysis_analyst.uhourse.notna())]
# reg_uhourse.sample(10,random_state =10)
# it is probably better to go ahead and just compare odd_uhourse to the whole set of marketing analyst since that is considered the "ideal sample"

#[]
# going to check some important demographic variables ;
cps2014_analysis_analyst.info() # many nulls on the ethnic variable and a bit on the union
odd_uhourse.info() #nulls mostly on union variables and weekly earnings

var1 = ['ethnic','unioncov','unionmme','earnwke']
cps2014_analysis_analyst.loc[:,var1].describe(include='all')
odd_uhourse.loc[:,var1].describe(include='all')
cps2014_analysis_analyst.loc[:,'unionmme'].value_counts(normalize=True) # to keep in mind that there is low union membership

#[]
# since ethnic is plagued with null values in both datasets the difference between the majority here isn't significant
# no significant difference with respect to the union variables nor the earnings per week although in odd_uhourse it has a higher minimum salary but this
# doesn't seem to pose any concern

#[]
# going to check one more set of variables
var2 = ['race','sex','age','marital','ownchild','prcitshp']
cps2014_analysis_analyst.loc[:,var2].describe(include='all')
odd_uhourse.loc[:,var2].describe(include='all')
cps2014_analysis_analyst['race'].value_counts(normalize=True)
cps2014_analysis_analyst['sex'].value_counts(normalize=True)
cps2014_analysis_analyst['marital'].value_counts(normalize=True)
cps2014_analysis_analyst['ownchild'].value_counts(normalize=True)

odd_uhourse['race'].value_counts(normalize=True)
odd_uhourse['sex'].value_counts(normalize=True)
odd_uhourse['marital'].value_counts(normalize=True)
odd_uhourse['ownchild'].value_counts(normalize=True)
#   come back to this
#[]
# citizenship majority in both is native,

#going to drop the nan values and also the -4.00
cps2014_analysis_analyst = cps2014_analysis_analyst.loc[(cps2014_analysis_analyst.uhourse.notna()) & (cps2014_analysis_analyst.uhourse >= 0 )]

#[]
#
#dropping where earnwke is nan (#check if any demographic factors associated with these nulls)
cps2014_analysis_analyst.earnwke.value_counts(dropna=False)
cps2014_analysis_analyst.earnwke.value_counts(dropna=False).sort_index()
#[]
#

cps2014_analysis_analyst.loc[cps2014_analysis_analyst.earnwke.isna(),['earnwke','uhourse','class94']]
cps2014_analysis_analyst.loc[cps2014_analysis_analyst.earnwke.notna(),['earnwke','uhourse','class94']]

cps2014_analysis_analyst.loc[cps2014_analysis_analyst.earnwke.isna(),['class94']].value_counts()
cps2014_analysis_analyst.loc[cps2014_analysis_analyst.earnwke.notna(),['class94']].value_counts()

#some people have null values on weekly earning but hours worked so we'll exlcude them however all of those with weekly earnings as null
#are self employed. furthermore, none of those with a value for weekly earnings are declared as self-employed
#but this isn't necessarily important to have for our analysis so we'll drop them

cps2014_analysis_analyst.dropna(subset=['earnwke'], inplace=True)

#[]
# create a wage variable based on dividing weekly earnings by hours worked per week

cps2014_analysis_analyst['wage'] = cps2014_analysis_analyst.loc[:,'earnwke'] / cps2014_analysis_analyst.loc[:,'uhourse']

cps2014_analysis_analyst.loc[:,'wage'] = cps2014_analysis_analyst.loc[:,'wage'].astype('int64')

#[]
# wage seems right skewed with just one value over $286
cps2014_analysis_analyst['wage'].skew()
cps2014_analysis_analyst['wage'].kurtosis()
cps2014_analysis_analyst['wage'].describe()
cps2014_analysis_analyst['wage'].value_counts(bins=7).sort_index()

plt.hist(cps2014_analysis_analyst['wage'])
plt.show()

sns.boxplot(cps2014_analysis_analyst['wage'])
plt.show()
#going to take log

cps2014_analysis_analyst['ln_wage'] = np.log(cps2014_analysis_analyst.loc[:,'wage'])

#[]
#
plt.hist(cps2014_analysis_analyst['ln_wage'])
plt.show()

#[]
#making columns where females = 1 and males = 0 (binary variable for regression)
cps2014_analysis_analyst['sex'].value_counts() #checking values beforehand to ensure integrity
Counter(np.where(cps2014_analysis_analyst['sex']== 1,'1. Male','2. Female' ))
# 2 methods for carrying out the process
cps2014_analysis_analyst['sex2'] = np.where(cps2014_analysis_analyst['sex']== 1,'1. Male','2. Female')
cps2014_analysis_analyst['female'] = cps2014_analysis_analyst.loc[:,'sex'].map({1:0,2:1})

#[]
# grouped boxplots
myplt = sns.boxplot(cps2014_analysis_analyst['sex2'],cps2014_analysis_analyst['wage'],data=cps2014_analysis_analyst)
myplt.set_title('boxplots of wage by sex')
myplt.set_xlabel('sex')
myplt.set_ylabel('wage')
plt.show()

#[]
# violinplot to check the distribution a bit more

myplt = sns.violinplot(x=cps2014_analysis_analyst.loc[:,'class94'],y=cps2014_analysis_analyst['wage'],data=cps2014_analysis_analyst)
myplt.set_title('boxplots of wage by class of worker')
myplt.set_xlabel('worker class')
myplt.set_ylabel('wage')
myplt.set_xticklabels(myplt.get_xticklabels(), rotation=45,horizontalalignment='right')
plt.tight_layout()
plt.show()

myplt = sns.violinplot(y=cps2014_analysis_analyst.loc[:,'class94'],x=cps2014_analysis_analyst['wage'],data=cps2014_analysis_analyst)
myplt.set_title('boxplots of wage by class of worker')
myplt.set_xlabel('wage')
myplt.set_ylabel('worker class')
plt.tight_layout()
plt.show()

myplt = sns.violinplot(cps2014_analysis_analyst['sex2'],cps2014_analysis_analyst['wage'],data=cps2014_analysis_analyst)
myplt.set_title('boxplots of wage by sex')
myplt.set_xlabel('sex')
myplt.set_ylabel('wage')
plt.show()

#[]
# sort of mimicking a violin plot but with more precision
#also will remove the one female extreme value just to get a better view of data in chart
sns.boxplot(y=cps2014_analysis_analyst.loc[cps2014_analysis_analyst['wage']<285,'class94'],x=cps2014_analysis_analyst.loc[cps2014_analysis_analyst['wage']<285,'wage'],data=cps2014_analysis_analyst)
sns.swarmplot(y=cps2014_analysis_analyst.loc[cps2014_analysis_analyst['wage']<285,'class94'],x=cps2014_analysis_analyst.loc[cps2014_analysis_analyst['wage']<285,'wage'],data=cps2014_analysis_analyst,color='.1',size=2)
plt.tight_layout()
plt.show()

cps2014_analysis_analyst.loc[:,'class94'].value_counts()


# cps2014_analysis_analyst.reset_index(inplace=True)
# cps2014_analysis_analyst.to_pickle("cps2014_analysis_analyst2.pkl")
# test = pd.read_pickle("cps2014_analysis_analyst2.pkl")
# test.head()

#[]
# I would have to transform some of the values to get an effective correlation heatmap.
# however marital does seem to have a negative correlation with childpresent which makes sense since beeing married corresponds
# to lower numbers
corr = cps2014_analysis_analyst.corr()
sns.heatmap(corr,xticklabels=corr.columns, yticklabels=corr.columns, cmap='coolwarm')
plt.tight_layout()
plt.show()
corr['marital']

#[]
# the highest-paid workers don't necessarily get paid more however at 40 hours is where it tends to be the highest
plt.figure(6,10)
cps2014_analysis_analyst.plot.scatter(x='uhourse',y='wage')
plt.show()

sns.relplot(x='uhourse',y='wage',data=cps2014_analysis_analyst, hue='marital', style="sex2")
plt.title('wage by hours worked and gender')
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1,2,sharey=True)
sns.regplot(x=cps2014_analysis_analyst.loc[cps2014_analysis_analyst.female==1,'age'] ,y=cps2014_analysis_analyst.loc[cps2014_analysis_analyst.female==1,'chldpres'],data=cps2014_analysis_analyst, ax = axes[0],lowess=True)
axes[0].set_title('females')
sns.regplot(color = 'red', x=cps2014_analysis_analyst.loc[cps2014_analysis_analyst.female==0,'age'] ,y=cps2014_analysis_analyst.loc[cps2014_analysis_analyst.female==0,'chldpres'],data=cps2014_analysis_analyst, ax = axes[1],lowess=True)
axes[1].set_title('males')
# axes[1].set_xticklabels(axes[1].get_xticklabels()) issue with the x tick labels on right plot
plt.show()

#[]
#
var3 = ['wage','earnwke','chldpres','uhourse']
od = cps2014_analysis_analyst.loc[:,var3]

scaler = StandardScaler()
od_standard = scaler.fit_transform(od)


clf = KNN(contamination=0.1)
clf.fit(od_standard)

pred_labels = clf.labels_
d_scores = clf.decision_scores_

#[]
# using KNN with anomaly detection there are apparently 271 outliers and 31 non-outliers
pred = pd.DataFrame(zip(pred_labels,d_scores), columns=['outliers','scores'], index=cps2014_analysis_analyst.index)
pred.sample(10,random_state=10)
pred.outliers.value_counts()

pred.groupby(['outliers'])[['scores']].agg(['min','median','max'])
cps2014_analysis_analyst.join(pred)

#[]
# using KNN for regression
import sklearn.neighbors
X = np.c_[cps2014_analysis_analyst['age']] #turn into 2D array
y = np.c_[cps2014_analysis_analyst['wage']] #turn into 2D array
model = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
model.fit(X,y)
new_x = [[50]]
predicted_y = model.predict(new_x)

plt.scatter(x=cps2014_analysis_analyst['age'], y=cps2014_analysis_analyst['wage'])
plt.scatter(new_x,predicted_y.item(), c='red') #a.item() to change into scalar value
plt.annotate(text='predicted value',xy=(50,predicted_y.item()),xytext=(50,100),arrowprops={'width':2}) #places arrow on graph
plt.show()

#[]
#
import sklearn.linear_model
model2 = sklearn.linear_model.LinearRegression()
model2.fit(X,y)

predicted_y2 = model2.predict(new_x)
plt.scatter(x=cps2014_analysis_analyst['age'], y=cps2014_analysis_analyst['wage'])
plt.scatter(new_x,predicted_y2.item(), c='red') #a.item() to change into scalar value
plt.annotate(text='predicted value',xy=(50,predicted_y2.item()),xytext=(50,100),arrowprops={'width':2}) #places arrow on graph
plt.show()

#[]
# quick regression on age and wage
lm_cps = sm.OLS(cps2014_analysis_analyst['wage'], sm.add_constant(cps2014_analysis_analyst['age'])).fit(cov_type='HC1')
lm_cps.summary()


#[]
# going to find different method for linear piecewise regresion as this is not working
# import statsmodels.formula.api as smf
# from patsy import dmatrices

# y1, X1 = dmatrices('ln_wage ~ lspline(uhourse,40) + female', cps2014_analysis_analyst)
# lps_model = smf.ols(y1, X1)
# lps_model1 = lps_model.fit()
# lps_model.summary()


# run isolation forest
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

#saving to pickle file to continue analysis a bit later
cps2014_analysis_analyst.reset_index(inplace=True)
cps2014_analysis_analyst.to_pickle("cps2014_analysis_analyst.pkl")
test = pd.read_pickle("cps2014_analysis_analyst.pkl")
test.head()

