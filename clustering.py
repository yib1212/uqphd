# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 14:08:05 2022

@author: Yibo Wang
"""

import math
import pyodbc
import numpy as np
import pandas as pd
from objective_1 import CarbonEmission
from scipy.stats import levy
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.special as sp
from sklearn.model_selection import train_test_split


class Clustering(object):
    
    def __init__(self):
        
        self.car_2017 = 154.37
        self.car_2018 = 153.69
        self.car_2019 = 152.43
        self.car_2020 = 150.78
        self.bus_2017 = 81.91
        self.bus_2018 = 81.57
        self.bus_2019 = 97.03
        self.bus_2020 = 121.97
        
        self.car_ave = 152.3
        self.bus_ave = 100.2
        
        # Database location
        conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
                    r'DBQ=data\Travel Survey\2018-21_pooled_seq_qts_erv1.0.accdb;')
        
        conn = pyodbc.connect(conn_str)
                    
        self.df_1 = pd.read_sql(f'select * from 1_QTS_HOUSEHOLDS', conn)
        self.df_3 = pd.read_sql(f'select * from 3_QTS_VEHICLES', conn)
        self.df_5 = pd.read_sql(f'select * from 5_QTS_TRIPS', conn)
        
        self.font = {'size': 10}
        
        self.mode_id = CarbonEmission.ModeChoice(self)
        
        return None
    
    
    def Levy(self, x, sigma, mu, a, c):
        
        # levy_fit = np.sqrt(sigma/(2*math.pi)) * (np.exp(-(sigma/2*(x-mu)))) / (np.power((x-mu),2))
        levy_fit = levy.pdf(x, mu, sigma)
        
        return a * levy_fit + c
    
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
                carbon_emi.append(cum_dist[i] * self.car_ave)
                car_list.append(cum_dist[i] * self.car_ave)
            elif mode_id[i] == 3:
                carbon_emi.append(cum_dist[i] * self.bus_ave)
                bus_list.append(cum_dist[i] * self.bus_ave)
            else:
                carbon_emi.append(0)
                zero_cnt += 1
                
        emission = car_list + bus_list
                
        return emission
    
    
    def Cluster(self, emission):
        
        M = 20
        N = len(emission)
        K = 2
        
        mu = [-3.85422504e+02, -2.20148278e+01]
        a = [2.68429086e+01, 1.74463495e+01]
        c = [-4.98387642e-04-4.98387642e-04, -3.86581062e-04]
        sigma = np.array([3.56895776e+03, 7.34664230e+02])
        tao = np.array([1/4, 3/4])
        
        prob_n = np.zeros((K, N))
        T_n = np.zeros((K, N))
        y = np.zeros((K, 1000))
        
        for m in range(M):
            
            ''' E Step '''
            for n in range(N):
                for k in range(K):
                    prob_n[k, n] = self.Levy(emission[n], sigma[k], mu[k], a[k], c[k])
                    T_n[k, n] = (prob_n[k, n]*tao[k]) / np.dot(prob_n[:, n], tao)
            tao = np.sum(T_n, axis = 1) / N
                
            print(np.sum(T_n, axis = 1))
            
            ''' M Step '''
            sigma = N / np.sum(1/T_n, axis = 1)
            print(sigma)
            for k in range(K):
                x = range(1000)
                y[k] = self.Levy(x, sigma[k], mu[k], a[k], c[k])
            
            plt.plot(x, y[0], 'k', linewidth=2, c='red', label='Commute')
            plt.plot(x, y[1], 'k', linewidth=2, c='blue', label='Shopping')
            plt.show()
            
        
        return None
        
    
    
if __name__ == "__main__":
    
    clustering = Clustering()
    emission = clustering.TripEmission()
    clustering.Cluster(emission)