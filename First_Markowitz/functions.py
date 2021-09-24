#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 18:33:50 2021

@author: leoprimero
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2, linregress
from scipy.optimize import minimize
from numpy import linalg as LA
import classes






def print_number(n=5):
    print(n)


def load_timeseries (ric, file_extension= 'csv'):

##  Here you must to put your path ok        
    path = '/get_data_from_yahoo/get_yahoo_tickers' + ric + '.' + file_extension
    if file_extension == 'csv':
        table_raw = pd.read_csv(path) #default estension
    else:
        table_raw = pd.read_excel(path)
    
    t = pd.DataFrame()
    t['date'] = pd.to_datetime(table_raw['Date'], dayfirst = True)
    t['close'] = table_raw['Close']
    t['close_previous'] = table_raw['Close'].shift(1)
    t['return_close'] = t['close']/t['close_previous']-1
    t = t.dropna()
    t = t.reset_index(drop= True)
    # imput from Jarque Bera Test
    x = t['return_close'].values  #returns 
    x_size =  len(x)  # size of returns
    x_str = 'Real_returns' + ric     #lebel e.g .ric
    return x, x_str, t
    
def plot_timeseries_price (t, ric ):
    plt.figure()
    plt.plot(t['date'], t['close'])
    plt.title('Time series Real Prieces ' + ric) 
    plt.xlabel ('Time')
    plt.ylabel ('Price')
    plt.show()
def plot_histogram (x,x_str,plot_str,bins=100):
    plt.figure()
    plt.hist(x, bins)
    plt.title('Histogram ' + x_str )
    plt.xlabel(plot_str)
    plt.show()
    
    
def syncronize_timeseries(benchmark,ric, file_extension = 'csv') :
    #loading data from csv or exccel file 
    xi, str1, t1 = load_timeseries(benchmark, file_extension)
    x2, str2, t2 = load_timeseries(ric, file_extension)
    
    #Sincronize Timestamps
    timestamp1 = list(t1['date'].values)
    timestamp2 = list(t2['date'].values)
    timestamps = list(set(timestamp1)& set(timestamp2))
    
    # syncronized timeseries for x1 or ric
    t1_sync = t1[t1['date'].isin(timestamps)]
    t1_sync.sort_values(by='date', ascending= True)
    t1_sync = t1_sync.reset_index(drop=True)
    # syncronized timeseries for x2 or benchmark
    t2_sync = t2[t2['date'].isin(timestamps)]
    t2_sync.sort_values(by='date', ascending= True)
    t2_sync = t2_sync.reset_index(drop=True)
    
    # Table of returns for ric and bechamrk
    t = pd.DataFrame()
    t['date'] = t1_sync['date']
    t['price_1'] = t1_sync['close'] # price benchmark
    t['price_2'] = t2_sync['close'] # price ric
    ### added previous close ###
    t['price_1_previous'] = t1_sync['close_previous'] # previous close benchmark
    t['price_2_previous'] = t2_sync['close_previous'] # previous close ric
    
    t['return_1'] = t1_sync['return_close'] # return benchmark
    t['return_2'] = t2_sync['return_close'] # return ric
    
    # compute vectors of returns
    return_benchmark = t['return_1'].values   # variable x
    return_ric = t['return_2'].values         # variable y
    return return_benchmark, return_ric, t   # x, y, t

def compute_beta (benchmark, ric, bool_print=False,):
    capm = classes.capm_manager(benchmark, ric)
    capm.load_timeseries()
    capm.compute()
    if bool_print:
        print('------')
        print(capm)
    beta = capm.beta
    return beta

def cost_function_beta_delta(x, delta, beta_usd, betas, epsilon=0):
    f_delta = (sum(x).item() + delta)**2
    f_beta = (np.transpose(betas).dot(x).item() + beta_usd)**2
    f_penalty = epsilon * sum(x**2).item()
    f = f_delta + f_beta + f_penalty
    return f     

def compute_portfolio_min_variance(covariance_matrix, notional):
    eigenvalues, eigenvectors = LA.eigh(covariance_matrix)
    variance_explained = eigenvalues[0] / sum(abs(eigenvalues))
    eigenvector = eigenvectors[:,0]
    if max(eigenvector) < 0.0:
        eigenvector = -eigenvector
    port_min_variance = notional * eigenvector / sum(abs(eigenvector))
    return port_min_variance, variance_explained


def compute_portfolio_pca (covariance_matrix, notional):
    eigenvalues, eigenvectors = LA.eigh(covariance_matrix)
    variance_explained = eigenvalues[-1] / sum(abs(eigenvalues))
    eigenvector = eigenvectors[:,-1]
    if max(eigenvector) < 0.0:
        eigenvector = -eigenvector
    port_pca = notional * eigenvector / sum(abs(eigenvector))
    return port_pca, variance_explained

def compute_portfolio_equi_weigth(size, notional):
    port_equi = (notional / size) * np.ones([size])
    return port_equi

def compute_portfolio_long_only(size, notional, covariance_matrix):
    # initialise optimization
    x = np.zeros([size,1])
    # initialise constraints
    cons = [{'type': 'eq', 'fun': lambda x: sum(abs(x)) - 1}]
    bnds = [(0, None) for i in range(size)]
    # compute optimasation
    res = minimize(compute_portfolio_variance, x, args=(covariance_matrix), constraints=cons, bounds=bnds)     
    port_long_only = notional * res.x
    return port_long_only

def compute_portfolio_markowitz(size, notional, covariance_matrix, returns, target_return):
    # initialise optimization
    x = np.zeros([size,1])
    # initialise constraints
    cons = [{'type': 'eq', 'fun': lambda x: np.transpose(returns).dot(x).item() - target_return},\
            {'type': 'eq', 'fun': lambda x: sum(abs(x)) - 1}]
    bnds = [(0, None) for i in range(size)]
    # compute optimasation
    res = minimize(compute_portfolio_variance, x, args=(covariance_matrix), constraints=cons, bounds=bnds)     
    weights = notional * res.x
    return weights

def compute_portfolio_markowitz_restict(size, notional, covariance_matrix, returns, target_return):
    # initialise optimization
    x = np.zeros([size,1])
    # initialise constraints
    cons = [{"type" : "ineq", "fun" : lambda x: np.transpose(returns).dot(x).item() - target_return},\
            {"type" : "ineq", "fun" : lambda x: sum(abs(x)) - 1}]
    bnds = [(0.05, 0.2) for i in range(size)]
    # compute optimasation
    res = minimize(compute_portfolio_variance, x, args=(covariance_matrix), constraints=cons, bounds=bnds)     
    port_equi = (notional / size) * np.ones([size])
    weights = notional * res.x
    return weights 

def compute_My_portfolio(size, notional, covariance_matrix, returns, target_return):
    # initialise optimization
    x = np.zeros([size,1])
    # initialise constraints
    cons = [{"type" : "ineq", "fun" : lambda x: np.transpose(returns).dot(x).item() - target_return},\
            {"type" : "ineq", "fun" : lambda x: sum(abs(x)) - 1}]
    bnds = [(0.0713,0.0713),(0.118,0.118),(0.0847,0.0847),(0.0831,0.0831),(0.0719,0.0719),(0.0937,0.0937),
            (0.0811,0.0811),(0.0793,0.0793),(0.0372,0.0372),(0.0665,0.0665),(0.0182,0.0182),(0.0287,0.0287),
            (0.036,0.036),(0.004,0.004),(0.03,0.03),(0.012,0.012),(0.0843,0.0843)]
    
    res = minimize(compute_portfolio_variance, x, args=(covariance_matrix), constraints=cons, bounds=bnds)     
    port_equi = (notional / size) * np.ones([size])
    weights = notional * res.x
    return weights 
    
def compute_Pruebas_portfolio(size, notional, covariance_matrix, returns, target_return):
    # initialise optimization
    x = np.zeros([size,1])
    # initialise constraints
    cons = [{"type" : "ineq", "fun" : lambda x: np.transpose(returns).dot(x).item() - target_return},\
            {"type" : "ineq", "fun" : lambda x: sum(abs(x)) - 1}]
    bnds = [(0.28317,0.28317),(0.06699,0.06699),(0.0013,0.0013),(0.01774,0.01774),
            (0.20692,0.20692),(0.1174,0.1174),(0.22171,0.22171),(0.08478,0.08478)]
           
    
    # compute optimasation
    res = minimize(compute_portfolio_variance, x, args=(covariance_matrix), constraints=cons, bounds=bnds)     
    port_equi = (notional / size) * np.ones([size])
    weights = notional * res.x
    return weights 

def compute_portfolio_variance (x, covariance_matrix):
    variance = np.dot(x.T, np.dot(covariance_matrix, x)).item()
    return variance

def compute_portfolio_volatility (covariance_matrix, weights):
    notional = sum(abs(weights))  #np.sqrt(sum(weigths**2)) L2 norm
    if notional <= 0.0:
        return 0.0
    weights = weights / notional   # unitary weighhts in the L1 norm 
    variance = np.dot(weights.T, np.dot(covariance_matrix, weights)).item()
    if variance <= 0.0:
        return 0.0
    volatility = np.sqrt(variance)
    return volatility























