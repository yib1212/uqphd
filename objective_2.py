# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 13:24:05 2023

@author: Yibo Wang
"""

import pyodbc
import collections
import pandas as pd
import numpy as np
import objective_1
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import levy


class LevyRegression(object):
    
    def __init__(self):
        
        self.car_2020 = 150.78
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
        
        self.mode_id = objective_1.CarbonEmission.ModeChoice(self)
        self.sa2_array = objective_1.CarbonEmission.SA2Info(self)
        self.sa2_array, _, _ = objective_1.CarbonEmission.TripNumber(self)
        
        return None
    
    
    def SA3Info(self):
        
        ''' Get SA3 information from the database file. '''
        
        sa3_id = np.floor(np.array(self.df_1)[:, 13] / 1e6)
        hhid_1 = np.array(self.df_1)[:, 0].astype(int)
        hhid_5 = np.array(self.df_5)[:, 1].astype(int)
        
        sa3_array = []
        for hhid in hhid_5:
            index = np.argwhere(hhid_1 == hhid)
            sa3_array.append(sa3_id[index[0, 0]])
        
        sa3_main = []
        sa3_cnt = collections.Counter(sa3_id)
        for key in sa3_cnt:
            sa3_main.append(key)
        print(sa3_cnt)
        
        self.sa3_main = np.array(sa3_main).astype(int)
        self.sa3_array = sa3_array
        
        return sa3_main, sa3_array
    
    
    def TripEmission(self):
        
        ''' Compute the carbon emission for each trip. '''
        
        mode_id = self.mode_id
        sa3_main = self.sa3_main
        sa3_array = self.sa3_array
        cum_dist = np.array(self.df_5)[:, 24]
        carbon_emi = []
        
        for i in range(len(mode_id)):
            if mode_id[i] == 0 or mode_id[i] == 2:
                carbon_emi.append(cum_dist[i] * self.car_ave)
            elif mode_id[i] == 3:
                carbon_emi.append(cum_dist[i] * self.bus_ave)
            else:
                carbon_emi.append(0)
                
        emi_sum = np.zeros(len(sa3_main))
        num_cnt = np.zeros(len(sa3_main), dtype = int)
        
        for i in range(len(sa3_array)):
            idx = np.argwhere(sa3_main == sa3_array[i])
            emi_sum[idx] += carbon_emi[i]
            num_cnt[idx] += 1
        
        emi_ave = np.divide(emi_sum, num_cnt, where=num_cnt!=0)
        self.carbon_emi = carbon_emi
        
        # 66 SA2 0, 316-66=250
        return emi_ave, num_cnt
    
    
    def Levy(self, x, sigma, mu, a, c):
        
        # levy_fit = np.sqrt(sigma/(2*math.pi)) * (np.exp(-(sigma/2*(x-mu)))) / (np.power((x-mu),2))
        levy_fit = levy.pdf(x, mu, sigma)
        
        return a * levy_fit
    
            
    def LevyFitting(self, num_cnt):
        
        mode_id = self.mode_id
        sa3_main = self.sa3_main
        sa3_array = self.sa3_array
        carbon_emi = self.carbon_emi
        
        min_bound = 0
        max_bound = 10000
        d = 10
        num_bins = (max_bound - min_bound) // d
        plt.axis([0, max_bound, 0, 180])
        
        for i in range(len(num_cnt)):
        # for i in range(10):
            emissions = []
            if num_cnt[i] > 200: # The SA3 area with more than 200 samples
                for j in range(len(sa3_array)):
                    if sa3_array[j] == sa3_main[i] and carbon_emi[j] != 0:
                        emissions.append(carbon_emi[j])
            if emissions != []:
                n, bin_edges, _ = plt.hist(emissions, num_bins)
                plt.show()
                n[-1] = 0
                bin_middles = 0.5 * (bin_edges[1:] + bin_edges[:-1])
                weight = np.array(n) / sum(n)
                popt, pcov = curve_fit(self.Levy, bin_middles, weight, p0=(600, 300, 15, 0))
                
                y_train_pred = self.Levy(bin_middles, *popt)
        
                plt.axis([0, 10000, 0, 0.08])
                plt.scatter(bin_middles, weight, s=5, c='blue', label='train')
                plt.scatter(bin_middles, y_train_pred, s=5, c='red', label='model')
                plt.grid()
                plt.legend()
                plt.xlabel('Carbon emissions (g)', self.font)
                plt.ylabel('Probability density', self.font)
                plt.show()
                
                print(popt)
                
        return None
    
    
    
if __name__ == "__main__":
    
    reg = LevyRegression()
    
    sa3_main, sa3_array = reg.SA3Info()
    emi_ave, num_cnt = reg.TripEmission()
    reg.LevyFitting(num_cnt)