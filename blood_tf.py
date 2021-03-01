#!/usr/bin/env python
# coding: utf-8

# In[23]:


#Importing all the necessary libraries for the project

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from operator import itemgetter
from tpot import TPOTClassifier


# In[8]:


#Loading and inspecting the data
transfusion_df = pd.read_csv(r'C:\Users\user\Desktop\datasets\transfusion.data')
transfusion_df.head()


# In[10]:


#Inspecting the info from dataframe and summary statistics
transfusion_df.info()
print('\n')
transfusion_df.describe()


# In[11]:


#Renaming the target column
transfusion_df.rename(columns = {'whether he/she donated blood in March 2007': 'target'}, inplace = True)
transfusion_df.head()


# In[19]:


#Printing target incidence proportions and rounding output to 3 decimal places
transfusion_df['target'].value_counts(normalize = True).round(3)


# In[22]:


# Splitting transfusion_df DataFrame into
# X_train, X_test, y_train and y_test datasets,
# stratifying on the `target` column
X_train, X_test, y_train, y_test = train_test_split(transfusion_df.drop(columns = 'target'), transfusion_df['target'], 
                                                    test_size =.3, stratify = transfusion_df['target'])
X_train.head()


# In[32]:


# Instantiating TPOTClassifier
tpot = TPOTClassifier(generations=5, population_size=25, verbosity=2, scoring='roc_auc',
                      disable_update_check=True,
                      config_dict='TPOT light')

tpot.fit(X_train, y_train)

# AUC score for tpot model
tpot_auc_score = roc_auc_score(y_test, tpot.predict_proba(X_test)[:, 1])
print(f'\nAUC score: {tpot_auc_score:.4f}')

# Printing best pipeline steps
print('\nBest pipeline steps:', end='\n')
for idx, (name, transform) in enumerate(tpot.fitted_pipeline_.steps, start=1):
    # Printing idx and transform
    print(f'{idx}. {transform}')


# In[25]:


# Checking the X_train's variance and rounding the output to 3 decimal places
X_train.var().round(3)


# In[27]:


# Copying X_train and X_test into X_train_norm and X_test_norm
X_train_norm, X_test_norm = X_train.copy(), X_test.copy()

# Specify which column to normalize
logNorm_col = 'Monetary (c.c. blood)'

# Log normalization
for df in [X_train_norm, X_test_norm]:
    df['monetary_logNorm'] = np.log(df[logNorm_col])
    df.drop(columns = logNorm_col, inplace = True)
    
# Checking the variance for X_train_norm
X_train_norm.var().round(3)


# In[33]:


# Instantiate LogisticRegression
lr = LogisticRegression(solver = 'liblinear')

# Train the model
lr.fit(X_train_norm, y_train)
# AUC score for logistic regression model
lr_auc_score = roc_auc_score(y_test, lr.predict_proba(X_test_norm)[:, 1])
print(f'\nAUC score: {lr_auc_score:.3f}')


# TPOT Classifier Score : 0.778
# 
# Logistic Regression Score : 0.789
