#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 06:31:15 2021

@author: leoprimero
"""

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

nb_decimals = 6
notional = 1  #mn USD
print ('-----')
print('impuits;')
print('nb_decimals '+ str(nb_decimals))
print('notional ' + str(notional))


  
rics = ['AAPL',\
        'RIO',\
        'C',\
        'GE',\
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

# Compute vector of returns and volatilities for Markowitz portfolio
min_returns = np.min(port_mgr.returns)
max_returns = np.max(port_mgr.returns)
returns = min_returns + np.linspace(0.1,0.9,1000)  * (max_returns-min_returns)
volatilities = np.zeros([len(returns),1]) 
counter = 0
for target_return in returns:
    port_markowitz = port_mgr.compute_portfolio('markowitz', notional, target_return)
    volatilities[counter] = port_markowitz.volatility_anual
    counter += 1

#  Compute other portfolios
#Black
label1 = 'markowitz-avg-return' # 'equi-weigth'
port1 = port_mgr.compute_portfolio('markowitz', notional)
x1 = port1.volatility_anual
y1 = port1.return_anual
# red
label2 = 'long-only' # 'long-only' 'min-variance'
port2 = port_mgr.compute_portfolio(label2, notional)
x2 = port2.volatility_anual
y2 = port2.return_anual
# yellow
label3 = 'equi-weight' #  'equi-weigth'
port3 = port_mgr.compute_portfolio(label3, notional)
x3 = port3.volatility_anual
y3 = port3.return_anual

label4 = 'markowitz-target_return' # 'equi-weigth'
port4 = port_mgr.compute_portfolio('markowitz', notional, target_return=None)
x4 = port4.volatility_anual
y4 = port4.return_anual

# label5 = 'My Portfolio' #  'equi-weigth'
# port5 = port_mgr.compute_portfolio(label5, notional, target_return=None)
# x5 = port5.volatility_anual
# y5 = port5.return_anual

# Plot efficient frontier
plt.figure()
plt.title('Efficient frontier for a portfolio including ' + rics[0])
plt.scatter(volatilities, returns)
plt.plot(x1, y1, 'ok', label=label1) #black  
plt.plot(x2, y2, 'or', label=label2) #red
plt.plot(x3, y3, 'oy', label=label3) #yellow
plt.plot(x4, y4, '*y', label=label4) #pink
# plt.plot(x5, y5, '*g', label=label5) #green
plt.ylabel('Portfolio return')
plt.xlabel('Portfolio volatility')
plt.grid()
plt.legend(loc='best')
plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    