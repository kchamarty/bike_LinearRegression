# -*- coding: utf-8 -*-
"""
Created on Thu Oct 17 15:02:23 2019

@author: KC
"""

#############################################################################
# --------------------Importing the requied Packages------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import statsmodels.api as stm
from sklearn import metrics
from sklearn.metrics import r2_score
#############################################################################
    #************** Reading the data into a Pandas DataFrame  ******

fpath = r'C:\Users\KC\Documents\Metro College\DataMining\Project\bike_share_LR'
fname = r'london_merged.csv'
file = '{}\{}'.format(fpath,fname)
bike=pd.read_csv(file)
bike.columns
print(bike.describe())
print(bike.isnull().sum())
'''
timestamp       0
cnt             0
t1              0
t2              0
hum             0
wind_speed      0
weather_code    0
is_holiday      0
is_weekend      0
season          0
''' 
bike.shape


#############################################################################

#cleaning the data

bike_clean=bike.copy()
    #1.---- get Rid of Timestamp
bike_clean.get_values()
bike_clean.timestamp.value_counts
bike_clean.timestamp
bike_clean.drop('timestamp',axis=1,inplace=True)
bike_clean.loc[1].loc['timestamp'].year()

    #2------. check for the nan/NULL values
print(bike.describe())
print(bike.isnull().sum())  #No Null Values
    
    #3. ---- Corelation matrix and HeatMaps.
bike_corr_matrix = bike.corr(method='pearson')
sns.pairplot(bike_clean, kind="reg")
sns.heatmap(bike_corr_matrix) 

#--------------------------------------------------------------------------

#Checking for Outliers and removing them

#-------------- defining function to transform  to Z-scores --------

def ztransform(df,column):
    df_col_mean=df[column].mean()
    df_col_std=df[column].std()
    df[column]=(df[column] - df_col_mean) / df_col_std
    return df[column]

bike_s1=bike_clean.copy()
for column in bike_s1.columns:
    print(column)
    bike_s1[column]= ztransform(bike_s1,column) # converting values to Z-Scores

#bike_s1.shape
#bike.shape

sns.boxplot(data=bike_s1).set_title("Before removing outliers")
    

# ------- Function to remove outliers for X based on Z-Scores, Z>3
def remove_rows_std_outlier(df,column):
    before_count=df[column].count()
    df.drop(df.index[abs(df[column]) > 3], inplace=True)
    after_count=df[column].count()
    count=before_count-after_count
    print('Removing ouliers in column : {}\nNumber of outliers removed : {}'.format(column,count))

#index_list=bike_s1[abs(bike_s1['fixed acidity']) > 3].index
#bike.drop(bike.index[index_list])

for column in bike_s1.columns:
    if column != 'cnt' :
        remove_rows_std_outlier(bike_s1,column)
        
     
sns.boxplot(data=bike_s1).set_title("After removing outliers")

def remove_rows_outlier(df,column):
    df_col_mean=df[column].mean()
    df_col_std=df[column].std()
    df.drop(df.index[abs((df[column] - df_col_mean) / df_col_std) > 3], inplace=True)
    
bike_noout=bike_clean.copy()
for column in bike_noout.columns:
    if column != 'quality' :
        remove_rows_outlier(bike_noout,column)

#######################################################################################

#--------- Linear Regression using OLS    ---------------

    # --- Preparing the Data For linear Regression ------------

bike_lr=bike_noout.copy()
bike_lr=stm.add_constant(bike_lr)
bike_X = bike_lr.drop('cnt',axis=1)
bike_y = bike_lr['cnt'] #Target Variable
bike_X.columns

    # --- Splitting the Data For Training and Testing ------------
    
X_train, X_test, y_train, y_test = train_test_split(bike_X,bike_y, test_size=0.3, random_state=999)


#---------- Model Fitting and Backward Feature Elimination after setting significance level at 0.05 ----------------

def remove_maxpvalcol(drop_col,X_train,X_test):    
    X_train.drop([drop_col],axis=1,inplace=True)
    X_test.drop([drop_col],axis=1,inplace=True)
i=0
while True: 
    OLS = stm.OLS(y_train,X_train)
    OLSR = OLS.fit()
    OLSR_pval_max=OLSR.pvalues.max()
    i+=1
    if OLSR_pval_max > 0.05:
        drop_col=OLSR.pvalues[OLSR.pvalues==OLSR_pval_max].index[0]
        print('For iteration no : {} \n the max pval is for {} column and the value is {}'.format(i,drop_col,OLSR_pval_max))
        print('Dropping column : {}'.format(drop_col)) 
        remove_maxpvalcol(drop_col,X_train,X_test)
    else:
        print('all the pvalues for the selected explanatory set is <0.05')
        break

print(OLSR.summary())

y_pred = OLSR.predict(X_test)
y_train_pred = OLSR.predict(X_train)


#-------------------  Metrics related to  OLSR ------------

plt.title('Comparison of Y values in test and the Predicted values')
plt.ylabel('Test Set')
plt.xlabel('Predicted values')
plt.scatter(y_test,y_pred, marker ='+')

print('MAE : ', metrics.mean_absolute_error(y_test, y_pred))
print('MSE : ', metrics.mean_squared_error(y_test, y_pred))
print('RMSE : ', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Train_R2 : ', OLSR.rsquared)
print('Test_R2 : ' , r2_score(y_test,y_pred))


######################################################################################


# ---------------------- prepare and run k-fold for Linear Regression --------

bike_klr=bike_noout.copy()
bike_klr=stm.add_constant(bike_lr)
bike_X = bike_klr.drop('cnt',axis=1)
bike_y = bike_klr['cnt'] #Target Variable
bike_X.columns

    # ----- Regression and Backward Feature Elimination on entire dataset (for K-Fold purposes).
i=0
while True: 
    OLS = stm.OLS(bike_y,bike_X)
    OLSR = OLS.fit()
    OLSR_pval_max=OLSR.pvalues.max()
    if OLSR_pval_max > 0.05:
        drop_col=OLSR.pvalues[OLSR.pvalues==OLSR_pval_max].index[0]
        print('For iteration no : {} \n the max pval is for {} column and the value is {}'.format(i,drop_col,OLSR_pval_max))
        print('Dropping column : {}'.format(drop_col)) 
        bike_X.drop(drop_col,axis=1,inplace=True)
    else:
        print('all the pvalues for the selected explanatory set is <0.05')
        break
    i+=1
    
#------------ Training and validating model using Kfold where n=10
from sklearn.model_selection import KFold
kf = KFold(n_splits=10,random_state=9, shuffle=False)

test_r2_tot,train_r2_tot,n=0,0,1
for train_index, test_index in kf.split(bike_X):
   #print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = bike_X.iloc[train_index,:], bike_X.iloc[test_index,:]
   y_train, y_test = bike_y.iloc[train_index], bike_y.iloc[test_index]
   OLS = stm.OLS(y_train,X_train)
   OLSR = OLS.fit()
   y_pred = OLSR.predict(X_test)
   rsquared=OLSR.rsquared
   train_r2_tot+=rsquared
   rsquared_mean=train_r2_tot/n
   r2=r2_score(y_test,y_pred)
   print("Kfold# = {} \t Train R2 = {}      \t Train R2 mean ={} \t Test R2 = {}".format(n,rsquared,rsquared_mean,r2))
   test_r2_tot+=r2   
   r2_mean=test_r2_tot/n
   n+=1

print(OLSR.summary())


############################################################################


