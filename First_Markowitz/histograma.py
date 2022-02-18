#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 17:39:34 2021

@author: leoprimero
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import scipy
import importlib
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, chi2

import stream_functions
importlib.reload(stream_functions)
import stream_classes
importlib.reload(stream_classes)

import pandas_datareader.data as web
from pandas_datareader import data as pdr
import datetime as date


# #imput parameters



ric = 'KO'

jb = stream_classes.jarque_bera_test(ric)
jb.load_timeseries()
jb.compute()
jb.plot_timeseries()
jb.plot_histogram()
jb.plot_returns()
jb.plot_returns_cumulatives()
print (jb)
print ('--------------------------')