#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 00:10:42 2021

@author: leoprimero
"""
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib as mpl
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2, linregress
from scipy.optimize import minimize
from numpy import linalg as LA

import functions
importlib.reload(functions)
import classes
importlib.reload(classes)

nb_decimals = 6  # 2  4  5  6  
notional = 1  #mn USD
print ('-----')
print('impuits;')
print('nb_decimals '+ str(nb_decimals))
print('notional ' + str(notional))

  
rics = ['AAPL',\
        'IBM',\
        'AMZN',\
        'T',\
        'TSM',\
        'RIO',\
        'C',\
        'GE',\
        'GOOGL',\
        'INTC',\
        'JNJ',\
        'JPM',\
        'KO',\
        'MSFT',\
        'PFE',\
        'PG',\
        'WMT',\
        'X',\
        'XOM']

    

#Compute covariance matrix
port_mgr = classes.portfolio_manager(rics ,nb_decimals)
port_mgr.compute_covariance_matrix(bool_print=True)

# #Compute min-variance portfolio
portfolio_min_variance = port_mgr.compute_portfolio('min-variance', notional)
portfolio_min_variance.summary()

#compute PCA or max-variance portfolio
portfolio_pca = port_mgr.compute_portfolio('pca', notional)
portfolio_pca.summary()

#compute Default portfolio with is equi-weigt
portfolio_equiweigth = port_mgr.compute_portfolio('default', notional)
portfolio_equiweigth.summary()

#compute long-only portfolio with minimal variance
portfolio_long_only = port_mgr.compute_portfolio('long-only', notional)
portfolio_long_only.summary()

# compute long-only Markowitz portfolio
portfolio_markovitz = port_mgr.compute_portfolio('markowitz', notional, target_return=0.30)
portfolio_markovitz.summary()        
 
      
# # compute Markowitz-restrict portfolio
portfolio_markovitz_restrict = port_mgr.compute_portfolio('markowitz-rest', notional, target_return=None)
portfolio_markovitz_restrict.summary()  

port_mgr.correlation_matrix

corr = port_mgr.correlation_matrix
sns.heatmap(corr, square = True, cmap = sns.color_palette("Blues"))
plt.show()




