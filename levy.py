# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 13:48:24 2022

@author: Yibo Wang
"""

import math
import pyodbc
import numpy as np
import pandas as pd
from objective_1 import CarbonEmission
from scipy.stats import levy
from scipy.stats import skewnorm
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.special as sp
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter


class LevyFitting(object):
    
    def __init__(self):
        
        self.car_2017 = 154.37
        self.car_2018 = 153.69
        self.car_2019 = 152.43
        self.car_2020 = 150.78
        self.bus_2017 = 81.91
        self.bus_2018 = 81.57
        self.bus_2019 = 97.03
        self.bus_2020 = 121.97
        
        # Database location
        conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
                    r'DBQ=data\Travel Survey\2018.accdb;')
        
        conn = pyodbc.connect(conn_str)
        
        cursor = conn.cursor()
        for i in cursor.tables(tableType='TABLE'):
            print(i.table_name)
            
        self.df_1 = pd.read_sql(f'select * from 1_QTS_HOUSEHOLDS', conn)
        self.df_3 = pd.read_sql(f'select * from 3_QTS_VEHICLES', conn)
        self.df_5 = pd.read_sql(f'select * from 5_QTS_TRIPS', conn)
        
        self.font = {'size': 10}
        
        self.mode_id = CarbonEmission.ModeChoice(self)
        
        return None
    
    
    def SkewNorm(self, x, sigma, mu, alpha, c, a):
        
        normpdf = (1/(sigma*np.sqrt(2*math.pi)))*np.exp(-(np.power((x-mu),2)/(2*np.power(sigma,2))))
        normcdf = (0.5*(1+sp.erf((alpha*((x-mu)/sigma))/(np.sqrt(2)))))
        
        return 2 * a * normpdf * normcdf + c
    
    
    def Levy(self, x, sigma, mu, a, c):
        
        # levy_fit = np.sqrt(sigma/(2*math.pi)) * (np.exp(-(sigma/2*(x-mu)))) / (np.power((x-mu),2))
        levy_fit = levy.pdf(x, mu, sigma)
        
        return a * levy_fit + c
    
    
    def GaussianKernal(self, x):
        
        sigma = 2500
        mean = 1700
        e = math.e
        pi = math.pi
        
        y = []
        for i in x:
            result = ((1/(sigma*(math.sqrt(2*pi)))) * (e**(-0.5*((x[i]-mean)/sigma)**2)))
            y.append(result)
        
        return y
    
    
    def TripEmission(self):
        
        ''' Compute the carbon emission for each trip. '''
        
        mode_id = self.mode_id
        cum_dist = np.array(self.df_5)[:, 24]
        carbon_emi = []
        car_list = []
        bus_list = []
        zero_cnt = 0
        max_cnt = 0
        min_bound = 0
        max_bound = 10000
        d = 10
                
        for i in range(len(mode_id)):
            if mode_id[i] == 0 or mode_id[i] == 2:
                carbon_emi.append(cum_dist[i] * self.car_2018)
                car_list.append(cum_dist[i] * self.car_2018)
            elif mode_id[i] == 3:
                carbon_emi.append(cum_dist[i] * self.bus_2018)
                bus_list.append(cum_dist[i] * self.bus_2018)
            else:
                carbon_emi.append(0)
                zero_cnt += 1
        
        hist_emi = carbon_emi
        
        for index, value in enumerate(hist_emi):
            if value > max_bound:
                max_cnt += 1
                hist_emi[index] = max_bound
        
        self.non_zero = len(mode_id) - zero_cnt - max_bound
                
        num_bins = (max_bound - min_bound) // d
                
        ''' Levy distribution '''
        x = range(10000)
        y = car_list + bus_list
        para = levy.fit(y)
        
        self.hist_emi = hist_emi
        self.num_bins = num_bins
        
        plt.axis([0, 10000, 0, 320])
        self.n, bin_edges, _ = plt.hist(hist_emi, num_bins)
        self.n[0] = 0
        self.n[-1] = 0
        self.bin_middles = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        plt.title('Carbon emission of each trip by motor vehicle', self.font)
        plt.xlabel('Carbon emissions (g)', self.font)
        plt.ylabel('Number of trips', self.font)
        plt.show()
                
        return None
        # return bin_middles, weight, para
        
        
    def LevyFitting(self):
        
        bin_middles = self.bin_middles
        weight = self.n
        weight = preprocessing.normalize([weight])[0]
        print(sum(weight))
        
        X_train, X_test, y_train, y_test = train_test_split(bin_middles, weight, test_size=0.3, random_state=0)
        popt, pcov = curve_fit(self.Levy, X_train, y_train, p0=( 600, 300, 100, 0))
        print(popt)
        
        y_train_pred = self.Levy(X_train, *popt)
        
        plt.axis([0, 10000, 0, 0.15])
        plt.scatter(X_train, y_train, s=5, c='blue', label='train')
        plt.scatter(X_train, y_train_pred, s=5, c='red', label='model')
        plt.grid()
        plt.legend()
        plt.xlabel('Carbon emissions (g)', self.font)
        plt.ylabel('Weight', self.font)
        plt.show()
        
        x = range(10000)
        y = self.Levy(x, *popt)
        y = y / sum(y_train_pred) * self.non_zero
        # y_train_pred = y_train_pred / sum(y_train_pred) * self.non_zero
        
        plt.axis([0, 10000, 0, 320])
        # plt.scatter(X_train, y_train_pred, s=5, c='red', label='model')
        plt.plot(x, y, 'k', linewidth=2, c='red', label='Levy')
        plt.hist(self.hist_emi, self.num_bins)
        plt.show()
        
        
        return popt

    
    
    
if __name__ == "__main__":
    
    levy_fitting = LevyFitting()
    levy_fitting.TripEmission()
    levy_fitting.LevyFitting()