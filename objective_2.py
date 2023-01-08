# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 13:24:05 2023

@author: Yibo Wang
"""

import pyodbc
import dbfread
import collections
import pandas as pd
import numpy as np
import objective_1


class LevyRegression(object):
    
    def __init__(self):
        
        self.car_2020 = 150.78
        self.bus_2020 = 121.97
        
        # Database location
        conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
                    r'DBQ=data\Travel Survey\2020.accdb;')
        
        conn = pyodbc.connect(conn_str)
                    
        self.df_1 = pd.read_sql(f'select * from 1_QTS_HOUSEHOLDS', conn)
        self.df_3 = pd.read_sql(f'select * from 3_QTS_VEHICLES', conn)
        self.df_5 = pd.read_sql(f'select * from 5_QTS_TRIPS', conn)
        
        self.font = {'size': 10}
        
        self.mode_id = objective_1.CarbonEmission.ModeChoice(self)
        self.sa2_array = objective_1.CarbonEmission.SA2Info(self)
        self.sa2_array, _, _ = objective_1.CarbonEmission.TripNumber(self)
        
        return None
    
    
    def TripEmission(self):
        
        ''' Compute the carbon emission for each trip. '''
        
        mode_id = self.mode_id
        sa2_main = self.sa2_main
        sa2_array = self.sa2_array
        cum_dist = np.array(self.df_5)[:, 24]
        carbon_emi = []
        
        for i in range(len(mode_id)):
            if mode_id[i] == 0 or mode_id[i] == 2:
                carbon_emi.append(cum_dist[i] * self.car_2020)
            elif mode_id[i] == 3:
                carbon_emi.append(cum_dist[i] * self.bus_2020)
            else:
                carbon_emi.append(0)
                
        emi_sum = np.zeros(len(sa2_main))
        num_cnt = np.zeros(len(sa2_main), dtype = int)
        
        for i in range(len(sa2_array)):
            idx = np.argwhere(sa2_main == sa2_array[i])
            emi_sum[idx] += carbon_emi[i]
            num_cnt[idx] += 1
        
        emi_ave = np.divide(emi_sum, num_cnt, where=num_cnt!=0)
        self.carbon_emi = carbon_emi
        
        # 66 SA2 0, 316-66=250
        return emi_ave, num_cnt
    
            
    def LevyFitting(self, num_cnt):
        
        mode_id = self.mode_id
        sa2_main = self.sa2_main
        sa2_array = self.sa2_array
        
        for i in range(len(num_cnt)):
            if num_cnt[i] != 0:
                for j in range
    
    
if __name__ == "__main__":
    
    reg = LevyRegression()
    
    emi_ave, num_cnt = reg.TripEmission()