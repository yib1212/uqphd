# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 22:17:03 2023

@author: Yibo Wang
"""

import math
import pyodbc
import dbfread
import collections
import numpy as np
import pandas as pd
from objective_1 import CarbonEmission
from scipy.stats import levy
from scipy.stats import lognorm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.special as sp
from sklearn.model_selection import train_test_split
from scipy.stats import chi2
from scipy.ndimage import gaussian_filter


class DataBase(object):
    
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
        conn_str_17 = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
                    r'DBQ=data\Travel Survey\2017.accdb;')
        conn = pyodbc.connect(conn_str)
        conn_17 = pyodbc.connect(conn_str_17)        
                    
        self.df_1 = pd.read_sql(f'select * from 1_QTS_HOUSEHOLDS', conn)
        self.df_3 = pd.read_sql(f'select * from 3_QTS_VEHICLES', conn)
        self.df_5 = pd.read_sql(f'select * from 5_QTS_TRIPS', conn)
        
        self.df_1_17 = pd.read_sql(f'select * from 1_QTS_HOUSEHOLDS', conn_17)
        self.df_3_17 = pd.read_sql(f'select * from 3_QTS_VEHICLES', conn_17)
        self.df_5_17 = pd.read_sql(f'select * from 5_QTS_TRIPS', conn_17)
        
        self.font = {'size': 10}
        
        self.mode_id = CarbonEmission.ModeChoice(self)
        self.sa2_main = CarbonEmission.SA2Info(self)
        self.sa2_array, _, _ = CarbonEmission.TripNumber(self)
                
        return None
    
        
    def SA2Info17(self):
        
        ''' Get SA2 information from the database file. '''
        
        sa2 = dbfread.DBF('data\\1270055001_sa2_2016_aust_shape\\SA2_2016_AUST.dbf', encoding='GBK')
        df = pd.DataFrame(iter(sa2))
        
        sa2_main = np.array(df)[:, 1].astype(int)
        self.sa2_main_17 = sa2_main
        
        return sa2_main
    
    
    def TripNumber17(self):
        
        # Household ID, household size, and SA1 ID (length: 15543)
        sa2_main = self.sa2_main_17
        hhid_1 = np.array(self.df_1_17)[:, 0].astype(int)
        sa1_id = np.array(self.df_1_17)[:, 11]
        sa2_id = (sa1_id/1e2).astype(int)
        # Household ID (length: 104024)
        hhid_5 = np.array(self.df_5_17)[:, 1].astype(int)
        
        sa2_array = []
        counter = 0
        trip_pop = np.zeros(len(sa2_main), dtype = int)
        
        # Record the SA2 ID of each trip
        for i in hhid_5:
            index = np.argwhere(hhid_1 == i)
            sa2_array.append(sa2_id[index[0, 0]])
            counter += 1
        
        return sa2_array
    
    
    def SampleRate(self):
        
        sa2_main = self.sa2_main
        sa2_array = self.sa2_array
        
        num_cnt = np.zeros(len(sa2_main), dtype = int)
        
        for i in range(len(sa2_array)):
            idx = np.argwhere(sa2_main == sa2_array[i])
            num_cnt[idx] += 1
            
        sa1_id = np.array(self.df_1)[:, 13]
        sa2_id = (sa1_id/1e2).astype(int)
        hhd_cnt = np.zeros(len(sa2_main), dtype = int)
        
        for j in range(len(sa2_id)):
            idx = np.argwhere(sa2_main == sa2_id[j])
            hhd_cnt[idx] += 1
    
        return sa2_main, num_cnt, hhd_cnt
    
    
    def SampleRate2017(self, sa2_main_17, sa2_array_17):
        
        sa2_main = sa2_main_17
        sa2_array = sa2_array_17
        
        num_cnt = np.zeros(len(sa2_main), dtype = int)
        
        for i in range(len(sa2_array)):
            idx = np.argwhere(sa2_main == sa2_array[i])
            num_cnt[idx] += 1
            
        sa1_id = np.array(self.df_1_17)[:, 11]
        sa2_id = (sa1_id/1e2).astype(int)
        hhd_cnt = np.zeros(len(sa2_main), dtype = int)
        
        for j in range(len(sa2_id)):
            idx = np.argwhere(sa2_main == sa2_id[j])
            hhd_cnt[idx] += 1
        
        return num_cnt, hhd_cnt
    
    
    def WriteData(self):
        
        writer = pd.ExcelWriter('Database.xlsx')

        df1 = pd.DataFrame([1])
        df2 = pd.DataFrame([2])
        
        df1.to_excel(writer, sheet_name='Sheet1')
        df2.to_excel(writer, sheet_name='Sheet2')
        writer.save()
        
        
        return None
    
    
if __name__ == "__main__":
    
    database = DataBase()
    database.WriteData()
    sa2_main, num_cnt, hhd_cnt = database.SampleRate()
    
    sa2_main_17 = database.SA2Info17()
    sa2_array_17 = database.TripNumber17()
    num_cnt_17, hhd_cnt_17 = database.SampleRate2017(sa2_main_17, sa2_array_17)