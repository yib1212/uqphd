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
from scipy.stats import norm
from scipy.stats import skewnorm
import matplotlib.pyplot as plt

class LevyFitting(object):
    
    def __init__(self):
        
        self.car_2018 = 153.69
        self.car_2019 = 152.43
        self.car_2020 = 150.78
        self.bus_2018 = 81.57
        self.bus_2019 = 97.03
        self.bus_2020 = 121.97
        
        self.car_ave = 152.3
        self.bus_ave = 100.2
        
        # Database location
        conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
                    r'DBQ=data\Travel Survey\2018-21_pooled_seq_qts_erv1.0.accdb;')
        
        conn = pyodbc.connect(conn_str)
        
        cursor = conn.cursor()
        for i in cursor.tables(tableType='TABLE'):
            print(i.table_name)
            
        self.df_1 = pd.read_sql(f'select * from 1_QTS_HOUSEHOLDS', conn)
        self.df_3 = pd.read_sql(f'select * from 3_QTS_VEHICLES', conn)
        self.df_5 = pd.read_sql(f'select * from 5_QTS_TRIPS', conn)
        
        self.font = {'size': 10}
        
        self.sa2_array = CarbonEmission.SA2Info(self)
        self.mode_id = CarbonEmission.ModeChoice(self)
        self.sa2_array, _, _ = CarbonEmission.TripNumber(self)
        
        return None
    

    def TripEmission(self):
        
        ''' Compute the carbon emission for each trip. '''
        
        mode_id = self.mode_id
        cum_dist = np.array(self.df_5)[:, 24]
        carbon_emi = []
        car_list = []
        bus_list = []
        zero_cnt = 0
        
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
        
        print(len(mode_id) - zero_cnt)
        #carbon_emi = np.round(np.array(carbon_emi), 2)
        
        hist_emi = carbon_emi
        
        for index, value in enumerate(hist_emi):
            if value > 15000:
                hist_emi[index] = 15000
        
        d = 10
        min_bound = int(min(hist_emi))
        max_bound = int(max(hist_emi))
        num_bins = (max_bound - min_bound) // d
                
        ''' Levy distribution '''
        x = range(15000)
        y = car_list + bus_list
        para = levy.fit(y)
        
        p = levy.pdf(x, *para)
        gau = skewnorm.pdf(x, a=10, loc=120, scale=3200)

        
        for i in range(len(gau)):
            p[i] *= gau[i]
        sum_p = sum(p)
        for i in range(len(gau)):
            p[i] = p[i] / sum_p * 0.8428814301079163
        
        
        plt.axis([0, 15000, 0, 800])
        self.n, bin_edges, _ = plt.hist(hist_emi, num_bins)
        plt.plot(x, 933080*p, 'k', linewidth=2, c='red')
        plt.title('Carbon emission of each trip by motor vehicle', self.font)
        plt.xlabel('Carbon emissions (g)', self.font)
        plt.ylabel('Number of trips', self.font)
        plt.show()
        
        ''' Plot the skew normal weight '''
        gau = skewnorm.pdf(x, a=8, loc=120, scale=3200)
        plt.plot(x, gau, 'k', linewidth=2, c='red')
        plt.show()
        
        ''' Scatter the weight point '''
        weight = []
        bin_middles = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        estimate = levy.pdf(bin_middles, *para)
        for i in range(len(self.n)):
            weight.append(self.n[i] / (estimate[i] * 933080))
        plt.axis([0, 15000, 0, 2.5])
        plt.scatter(bin_middles, weight, s=10, c='red')
        plt.show()
        
        
        ''' Skewnorm fitting '''
        
        bin_middles = bin_middles[ :700]
        weight = weight[ :700]
        para = skewnorm.fit(weight, floc=120, fscale=3200)
        f_skew = skewnorm.pdf(bin_middles, *para)
        
        plt.axis([0, 15000, 0, 2e-4])
        plt.plot(bin_middles, f_skew, 'k', linewidth=2, c='red')
        plt.show()
        
        print(para)
        print(f_skew)
        
        
        
        
        
        return carbon_emi, car_list
    
    
    def GaussianKernal(self, x):
        
        sigma = 2500
        mean = 1700
        e = math.e
        pi = math.pi
        
        # gaussian = (1/(sigma*(math.sqrt(2*pi)))) * (e^(-0.5*((x-mean)/sigma)^2))
        
        y = []
        for i in x:
            result = ((1/(sigma*(math.sqrt(2*pi)))) * (e**(-0.5*((x[i]-mean)/sigma)**2)))
            y.append(result)
        
        return y
    

    
    

if __name__ == "__main__":
    
    levy_fitting = LevyFitting()
    carbon_emi, car_list = levy_fitting.TripEmission()
