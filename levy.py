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
        
        # # Unit: MtCO2
        # self.cars = 10.213
        # self.trucks_buses = 4.908
        # self.motorcycles = 0.068
        # self.railways = 0.566
        
        # # Unit: g/km
        # self.petrol = 165
        # self.diesel = 213
        
        # # Unit： g/pkm
        # self.car_2018 = 154.5
        # self.bus_2018 = 79
        
        # # Vehicle: 2020 Unit: Unit： g/km
        # self.cars_2020 = 149.5
        # self.buses_2020 = 216.7
        
        # Table 5.1 Unit: billion pkm
        # 2018: car 284.88 bus 22.08 rail 18.84
        # 2019: car 265.63 bus 16.81 rail 15.05
        # 2020: car 269.48 bus 12.97 rail 09.33
        
        # Table 11.4 Unit: gigagrams CO2
        # 2018: car 43783 bus 1801 rail 3593
        # 2019: car 40490 bus 1631 rail 3545
        # 2020: car 40633 bus 1582 rail 3542
        
        #Carbon Emission pkm
        # 2018: car 153.69 bus 81.57   rail 190.71
        # 2019: car 152.43 bus 97.03   rail 235.55
        # 2020: car 150.78 bus 121.97  rail 379.64*
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
        print(sum(p))
        gau = skewnorm.pdf(x, a=10, loc=120, scale=3200)
        print(sum(gau))
        
        for i in range(len(gau)):
            p[i] *= gau[i]
        sum_p = sum(p)
        
        for i in range(len(gau)):
            p[i] = p[i] / sum_p * 0.8428814301079163
        print(sum(p))
        
        
        plt.axis([0, 15000, 0, 800])
        self.n, _, _ = plt.hist(hist_emi, num_bins)
        plt.plot(x, 933080*p, 'k', linewidth=2, c='red')
        # plt.title('Carbon emission of each trip by motor vehicle', self.font)
        # plt.xlabel('Carbon emissions (g)', self.font)
        # plt.ylabel('Number of trips', self.font)
        plt.show()
        
        gau = skewnorm.pdf(x, a=10, loc=120, scale=3200)
        plt.plot(x, gau, 'k', linewidth=2, c='red')
        plt.show()
        
        
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
