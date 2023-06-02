# ****To access data from CodeUp MySQL database****:

# user = 'host name'
# password = 'user paswoord'
# host = 'data.codeup.com'
# db = 'zillow'

# def get_db_url(user, host, password, db):
    
#     url = f'mysql+pymysql://{env.user}:{env.password}@{env.host}/{db}'
#     return url


# In[9]:


import pandas as pd
import numpy as np
import env

from scipy import stats
from scipy.stats import pearsonr, spearmanr

import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression 
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures



from math import sqrt
import matplotlib.pyplot as plt
import seaborn as sb


# In[11]:


# Zillow:

##################### *ACQUIRE* ##########################
url = f'mysql+pymysql://{env.user}:{env.password}@{env.host}/zillow'

def get_zillow():
    """ This function pulls information from the mySQL zillow database and returns it as a
    pandas dataframe"""
    url = f'mysql+pymysql://{env.user}:{env.password}@{env.host}/zillow'
    sql = """ select bedroomcnt, bathroomcnt,
calculatedfinishedsquarefeet, fips, lotsizesquarefeet,
 taxvaluedollarcnt, yearbuilt, assessmentyear, taxamount
 from properties_2017 where propertylandusetypeid = '261' limit 1500000 """
    df = pd.read_sql(sql, url)
    return df

#################### *PREPARE* ############################

def prep_zillow(df):
    """ This function prepares/cleans data from the zillow df for splitting"""
    
    # Drops null values from columns
    df= df.dropna()
    
    #Renames columns to something more visual appealing
    df = df.rename(columns= {'bedroomcnt': 'beds', 'bathroomcnt':'baths',
                        'taxamount':'tax_amt', 'lotsizesquarefeet':'lot_size',
                        'calculatedfinishedsquarefeet':'sq_ft',
                        'taxvaluedollarcnt':'tax_val','yearbuilt':'year',
                        })
    
    # Drops duplicate values from df
    # df = df.drop_duplicates(inplace=True)

    # Reassign county names from FIPS data. I'll keep fips column for ease of use with numeric data
    df['county']= df['fips'].replace({6037: 'Los Angeles', 6059: 'Orange County', 6111: 'Ventura'})

    # Get dummy variables for counties
    blah = pd.get_dummies(df['county'], drop_first=False)
    df= pd.concat([df, blah], axis=1)

    # Feature engineering
    
    # New metric that measures yard size
    df['yard_sqft']= df['lot_size'] - df['sq_ft']
    

    
    return df

################### *OUTLIERS* #########################
def remove_outliers(df, columns):
    """This function removes outliers from specified columns using interquartile range"""
    columns = ['beds', 'baths', 'tax_amt', 'lot_size', 'sq_ft', 'tax_val', 'yard_sqft']

    for col in columns:
    # setting floor for my data 
        Q1 = df[col].quantile(0.25)
    # setting ceiling for my data
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        floor = Q1 - 1.5 * IQR
        ceiling = Q3 + 1.5 * IQR
        df = df.loc[(df[col] >= floor) & (df[col] <= ceiling)]
    return df
################### *SPLIT* ###################
def split_zillow(df):
    '''
    take in a DataFrame return train, validate, test split on zillow DataFrame.
    '''
# Reminder: I don't need to stratify in regression. I don't remember why, but Madeleine said 
# it
    train, test = train_test_split(df, test_size=.2, random_state=123)
    train, val = train_test_split(train, 
                                       test_size=.3, 
                                       random_state=123)
    return train, val, test

# ################## *WRANGLE* ################
# def wrangle_zillow():
    
#     train, val, test = prep_zillow(get_zillow())
    
#     return train, val, test


################## *Exploration* ############
def show_corr(train):
    ''' This function will return my spearman correlation table and other visualizations
    that display the distribution of tax value/ the correlation of my metrics'''

    # Correlation table
    corr =train.corr(method='spearman')
    #Distribution of Tax Value
    plt.figure(figsize=(15,20))
    sb.set_style('whitegrid')
    target=sb.displot(train['tax_val'], color='purple')
    target.set(title= 'Tax Value Distribution')
    plt.xticks(ticks = [0, 200000, 400000, 600000, 800000], 
               labels=['$0', '$200', '$400', '$600', '$800+'])
    plt.xticks(fontsize = 14)
    plt.yticks(fontsize = 8)
    plt.xlabel('Tax Value x 100',fontsize= 10)
    plt.ylabel('Number of Properties',fontsize= 10)
    plt.show()
    #Correlation of Metrics to tax value
    train.corr()['tax_val'].sort_values(ascending=False).plot(kind='bar')
    plt.title('Correlation of Metrics with Tax Value')
    plt.show()
    #Heatmap
    corr = train.corr(method='spearman')
    sb.heatmap(corr, cmap='Greens', annot=True, mask = np.triu(corr), annot_kws={"fontsize":8})
    plt.title('Heatmap Visualization of Correlation of Metrics')
    plt.show()
    
    return corr



################## *SCALING* ################
def scaling(train, val, test, toscale):
    """ Takes in train, val, test datasets and scales the data using MinMaxScaler and returns new scaled dataframes"""
    #columns to scale
    toscale = ['beds','baths','sq_ft','yard_sqft','Los Angeles','Orange County','Ventura', 'tax_val']
    #name scaler
    mms= sklearn.preprocessing.MinMaxScaler()
    mms.fit(train[toscale])

    #make copies so I don't interefere with OGs
    val_sc= val.copy()
    test_sc= test.copy()
    train_sc= train.copy()

    train_sc[toscale] = pd.DataFrame(mms.transform(train[toscale]), columns=train[toscale].columns.values).set_index([train.index.values])

    test_sc[toscale] = pd.DataFrame(mms.transform(test[toscale]), columns=test[toscale].columns.values).set_index([test.index.values])

    val_sc[toscale] = pd.DataFrame(mms.transform(val[toscale]), columns=val[toscale].columns.values).set_index([val.index.values])

    return train_sc, val_sc, test_sc
        
##########Evaluation########
def q1_test(train):
    """This function returns a boxplot that displays the relationship between number of 
    bathrooms and home tax value"""

    normbath= train[(train.baths <= train.baths.median())].tax_val
    plusbath= train[(train.baths > train.baths.median())].tax_val
#     train['masbeds'] = np.where(train['beds'] > 3, 1, 0)
#     train['masbanos'] = np.where(train['baths'] > 2,1,0)
#     yes = train[train['masbeds'] == 1].tax_val.mean()
#     no = train[train['masbeds'] == 0].tax_val.mean()
#     yesbth = train[train['masbanos'] == 1].tax_val.mean()
#     nobth = train[train['masbanos'] == 0].tax_val.mean()
    
    
    sb.boxplot(data=train, x='baths', y='tax_val', showfliers=False, linewidth=.5)
    plt.xlabel("Number of Bathrooms", fontsize=12)
    plt.ylabel("Home Tax Value", fontsize=12)
    plt.title("Tax Value and Number of Bathrooms", fontsize=18)
    plt.yticks([200000, 400000, 600000, 800000, 1_000_000], 
               ['$200,000', '$400,000', '$600,000', '$800,000', '$1,000,000'], fontsize=12)
    plt.xticks(fontsize=12)
    # Setting line at 2 to reflect my mode/mean
    plt.axvline(x=4, color='red', linestyle='dashed', linewidth=1)
    plt.show()

    # Stats testing
    a = 0.05
    t, p = stats.ttest_ind(normbath, plusbath)
    if p < a:
        print(f'We reject the null hypothesis.')
        print('Average property value of properties with more than two full bathrooms', "${:,}".format(round(plusbath.mean())))
        print('Average property value of properties with less than three full bathrooms', "${:,}".format(round(normbath.mean())))
    else:
        print('We fail to reject the null hypothesis')

def q2_test(train):
    train['yardcats'] = pd.qcut(train['yard_sqft'], q=4, labels=['Small', 'Medium', 'Large', 'Extra Large'])

    plt.figure(figsize=(30,10))
    sb.violinplot(x='yardcats', y='tax_val', data=train, estimator=np.mean)
    plt.yticks(ticks= [0, 200000, 400000, 600000, 800000, 1_000_000], labels = ['$0', '$200,000', '$400,000', '$600,000', '$800,000', '$1,000,000+'])
    plt.xlabel('Square Footage Range of Yards')
    plt.ylabel('Tax Value of Homes ($)')
    plt.title('Tax Value by Yard Size')
    plt.show()

    a = 0.05
    f, p = stats.f_oneway(train[train['yardcats'] == 'Small']['tax_val'],
    train[train['yardcats'] == 'Medium']['tax_val'],
    train[train['yardcats'] == 'Large']['tax_val'],
    train[train['yardcats'] == 'Extra Large']['tax_val'])
    print("ANOVA results:")
    print("F-statistic:", round(f, 2))
    print("p-value:", round(p, 8))
    if p < a:
        print('We reject the null hypothesis')
    else:
        print('We fail to reject the null hypothesis')


def q3_test(train):

    ventura= train[train.county == 'Ventura'].tax_val.median()
    la = train[train.county == 'Los Angeles'].tax_val.median()
    oc = train[train.county == 'Orange County'].tax_val.median()
    ventura1= train[train.county == 'Ventura']
    la1 = train[train.county == 'Los Angeles']
    oc1 = train[train.county == 'Orange County']
    a = 0.05
    # Plot    
    plt.figure(figsize=(30,10))
    sb.histplot(x='tax_val', hue='county', hue_order=(['Ventura', 'Orange County', 'Los Angeles']), data=train)
    plt.axvline(train['tax_val'].median(), color='black', linewidth=2)
    plt.axvline(x=train[train.county == 'Orange County'].tax_val.median(), color='orange', linestyle='--')
    plt.axvline(x=train[train.county == 'Los Angeles'].tax_val.median(), color='blue',linestyle='--')
    plt.axvline(x=train[train.county == 'Ventura'].tax_val.median(), color='green',linestyle='--')
    plt.xticks(ticks= [0, 200000, 400000, 600000, 800000, 1_000_000], labels = ['$0', '$200,000', '$400,000', '$600,000', '$800,000', '$1,000,000+'])
    plt.xlabel('Tax Value ($)')
    plt.ylabel('Count of Homes')
    plt.title('Tax Value of Homes By County')
    plt.show()

    print(f'Median Tax Value for homes in LA County: ${la}')
    print(f'Median Tax Value for homes in Orange County: ${oc}')
    print(f'Median Tax Value for homes in Ventura County: ${ventura}')
    #Stats test
    f, p = stats.f_oneway(oc1.tax_val, ventura1.tax_val, la1.tax_val)
    if p < a:
        print('We can reject the null hypothesis')
    else:
        print('We fail to reject the null hypothesis')


########### Feature Engineering ############### 
def select_kbest(X, y, k=2):
    """ X: df of independent features
        y: target
        k: number of kbest features to select. defaulted to 2, but can be changed)"""
    
 # make
    kbest= SelectKBest(f_regression, k=k)
    
 # fit
    kbest.fit(X, y)
 # get support
    mask = kbest.get_support()
    return X.columns[mask]

# In[ ]:




######### MODELING ##############
def comp_rmse_mm(y_train, y_val):
    """This function will return a dataframe which allows us to compare the 
    computed rmse mean and median and their differences for the baseline models """
    #Establish Baseline using mean
    baseline_mean= y_train.tax_val.mean()
    baseline_median= y_train.tax_val.median()
    y_train['baseline_mean'] = baseline_mean
    y_train['baseline_median']= baseline_median
    y_val['baseline_mean'] = baseline_mean
    y_val['baseline_median']= baseline_median

    # Calculate rmse for train, val
    rmse_train=mean_squared_error(y_train.tax_val, y_train.baseline_mean, squared=False) 
    rmse_val=mean_squared_error(y_val.tax_val, y_val.baseline_mean, squared=False)
    rmse_train_med=mean_squared_error(y_train.tax_val, y_train.baseline_median, squared=False)
    rmse_val_med=mean_squared_error(y_val.tax_val, y_val.baseline_median, squared=False)

    # Calculate differences 
    difference = rmse_train-rmse_val
    difference_med = rmse_train_med-rmse_val_med
    # Create dataframes for mean and median rmse
    results = pd.DataFrame( data=[{'model' : 'Baseline Mean',
                             'Train RMSE': '${:,.2f}'.format(rmse_train),
                             'Validate RMSE': '${:,.2f}'.format(rmse_val), 
                             'Difference': difference}])
    results_med = pd.DataFrame( data=[{'model' : 'Baseline Median',
                             'Train RMSE': '${:,.2f}'.format(rmse_train_med),
                             'Validate RMSE': '${:,.2f}'.format(rmse_train_med),
                             'Difference': difference_med}])    
    # Concating to compare baseline means and medians with differences
    results = pd.concat([results, results_med])
    return results


def comp_models(X_train, y_train, X_val, y_val):
    '''This model calculates the train/val rmse for Lasso Lars, OLS, and Polynomial
    features models and compares them to the baseline on a concatenated dataframe'''
    
    # Creating my modeling objects
    lr = LinearRegression(normalize=True)
    ll = LassoLars(alpha=1)
    pf= PolynomialFeatures(degree= 5)
    
    # Baseline
    baseline_mean= y_train.tax_val.mean()
    baseline_median= y_train.tax_val.median()
    y_train['baseline_mean'] = baseline_mean
    y_train['baseline_median']= baseline_median
    y_val['baseline_mean'] = baseline_mean
    y_val['baseline_median']= baseline_median
    # Calculate rmse for train, val
    rmse_train=mean_squared_error(y_train.tax_val, y_train.baseline_mean, squared=False) 
    rmse_val=mean_squared_error(y_val.tax_val, y_val.baseline_mean, squared=False)
    rmse_train_med=mean_squared_error(y_train.tax_val, y_train.baseline_median, squared=False)
    rmse_val_med=mean_squared_error(y_val.tax_val, y_val.baseline_median, squared=False)
    # Calculate differences 
    difference = rmse_train-rmse_val
    difference_med = rmse_train_med-rmse_val_med
    # Create dataframes for mean and median rmse
    results = pd.DataFrame( data=[{'model' : 'Baseline Mean',
                             'Train RMSE': '${:,.2f}'.format(rmse_train),
                             'Validate RMSE': '${:,.2f}'.format(rmse_val), 
                             'Difference': difference}])
    results_med = pd.DataFrame( data=[{'model' : 'Baseline Median',
                             'Train RMSE': '${:,.2f}'.format(rmse_train_med),
                             'Validate RMSE': '${:,.2f}'.format(rmse_train_med),
                             'Difference': difference_med}])    
    # Concating to compare baseline means and medians with differences
    base_results = pd.concat([results, results_med])
    
    #PolyFeats
    
    X_train_pf = pf.fit_transform(X_train)
    X_val_pf = pf.transform(X_val)
    
    lr.fit(X_train_pf, y_train['tax_val'])
    y_train['pf_pred'] = lr.predict(X_train_pf)
    pf_rmse_train= round(mean_squared_error(y_train['tax_val'], y_train['pf_pred'], squared=False),2)

    y_val['pf_pred'] = lr.predict(X_val_pf)
    pf_rmse_val= round(mean_squared_error(y_val['tax_val'], y_val['pf_pred'], squared=False),2)
    pfdifference = pf_rmse_train - pf_rmse_val

    pf_r2_train= explained_variance_score(y_train.tax_val, y_train.pf_pred)
    pf_r2_val = explained_variance_score(y_val.tax_val, y_val.pf_pred)
    
    pfmetrics = pd.DataFrame(data = [{ 'model': 'PolyFeats',
                                   'Train RMSE': '${:,.2f}'.format(pf_rmse_train),
                             'Validate RMSE': '${:,.2f}'.format(pf_rmse_val),
                                   'Difference': pfdifference,
                                   'r2 Train': pf_r2_train,
                                   'r2 Validate': pf_r2_val}])
    # OLS
    
    #Fit the model to train data
    lr.fit(X_train, y_train.tax_val)

    # Predict train
    y_train['pred_ols'] = lr.predict(X_train)
    y_val['pred_ols'] = lr.predict(X_val)

    # Evaluate RMSE
    ols_rmse_train = round(mean_squared_error(y_train.tax_val, y_train.pred_ols, squared=False), 2) 
    ols_rmse_val = round(mean_squared_error(y_val.tax_val, y_val.pred_ols, squared=False), 2) 
    olsdifference = ols_rmse_train - ols_rmse_val

    # Evaluate r2
    ols_r2_train= explained_variance_score(y_train.tax_val, y_train.pred_ols)
    ols_r2_val = explained_variance_score(y_val.tax_val, y_val.pred_ols)
    
    olsmetrics = pd.DataFrame(data = [{ 'model': 'OLS',
                                   'Train RMSE': '${:,.2f}'.format(ols_rmse_train),
                             'Validate RMSE': '${:,.2f}'.format(ols_rmse_val),
                                   'Difference': olsdifference,
                                   'r2 Train': ols_r2_train,
                                   'r2 Validate': ols_r2_val}])
    
    #LASSO LARS

        #Fit the model to train data
    ll.fit(X_train, y_train.tax_val)

    # Predict train
    y_train['pred_ll'] = ll.predict(X_train)
    y_val['pred_ll'] = ll.predict(X_val)


    # Evaluate RMSE
    ll_rmse_train = round(mean_squared_error(y_train.tax_val, y_train.pred_ll, squared=False), 2) 
    ll_rmse_val = round(mean_squared_error(y_val.tax_val, y_val.pred_ll, squared=False), 2) 
    lldifference = ll_rmse_train - ll_rmse_val

    # Evaluate r2
    ll_r2_train= explained_variance_score(y_train.tax_val, y_train.pred_ll)
    ll_r2_val = explained_variance_score(y_val.tax_val, y_val.pred_ll)

    llmetrics = pd.DataFrame(data = [{ 'model': 'Lasso Lars',
                                       'Train RMSE': '${:,.2f}'.format(ll_rmse_train),
                                 'Validate RMSE': '${:,.2f}'.format(ll_rmse_val),
                                       'Difference': lldifference,
                                       'r2 Train': ll_r2_train,
                                       'r2 Validate': ll_r2_val}])
    # Final DF
    
    final_results = pd.concat([base_results, llmetrics, olsmetrics, pfmetrics])
    
    return final_results
