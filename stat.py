# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 17:08:03 2022

@author: Yibo Wang
"""

import math
import pyodbc
import collections
import numpy as np
import pandas as pd
from objective_1 import CarbonEmission
from scipy.stats import levy
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt



class CarbonEachTrip(object):
    
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
        # 2017: car 285.53 bus 21.73 rail 18.02
        # 2018: car 284.88 bus 22.08 rail 18.84
        # 2019: car 265.63 bus 16.81 rail 15.05
        # 2020: car 269.48 bus 12.97 rail 09.33
        
        # Table 11.4 11.5 Unit: gigagrams CO2
        # 2017: car 44078 bus 1780 rail 3655
        # 2018: car 43783 bus 1801 rail 3593
        # 2019: car 40490 bus 1631 rail 3545
        # 2020: car 40633 bus 1582 rail 3542
        
        # Carbon Emission pkm
        # 2017: car 154.37 bus 81.91   rail 202.83
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
        
        plt.axis([0, 15000, 0, 800])
        self.n, self.bins, _ = plt.hist(hist_emi, num_bins)
        plt.title('Carbon emission of each trip by motor vehicle', self.font)
        plt.xlabel('Carbon emissions (g)', self.font)
        plt.ylabel('Number of trips', self.font)
                
        plt.show()
        
        self.hist_emi = hist_emi
        self.num_bins = num_bins
        self.car_list = car_list
        self.car_and_bus = car_list + bus_list
        
        return carbon_emi, car_list
    
    
    def RegionTime(self, carbon_emi):
        
        ''' Explore the relationship between SA2 region, average emission, and travel time. '''
        
        sa2_main = self.sa2_main
        sa2_array = self.sa2_array
        trav_time = np.array(self.df_5)[:, 23]
        
        plt.axis([0, 180, 0, 15000])
        plt.title('Carbon emission and travel time of each trip', self.font)
        plt.xlabel('Travel time (min)', self.font)
        plt.ylabel('Carbon emission (g)', self.font)
        plt.scatter(trav_time, carbon_emi, s=1, c='red')
        plt.show()
        
        time_sum = np.zeros(len(sa2_main))
        emi_sum = np.zeros(len(sa2_main))
        num_cnt = np.zeros(len(sa2_main), dtype = int)
        
        for i in range(len(sa2_array)):
            idx = np.argwhere(sa2_main == sa2_array[i])
            time_sum[idx] += trav_time[i]
            emi_sum[idx] += carbon_emi[i]
            num_cnt[idx] += 1
        
        time_ave = np.divide(time_sum, num_cnt, where=num_cnt!=0)
        emi_ave = np.divide(emi_sum, num_cnt, where=num_cnt!=0)
        
        plt.axis([0, 40, 0, 5000])
        plt.title('Average carbon emission and travel time of different SA2 regions', self.font)
        plt.xlabel('Average travel time (min)', self.font)
        plt.ylabel('Average carbon emission (g)', self.font)
        plt.scatter(time_ave, emi_ave, s=10, c='red')
        plt.show()
        
        # 66 SA2 0, 316-66=250
        self.emi_ave = emi_ave
        self.time_ave = time_ave
        
        return time_ave, emi_ave, num_cnt
    
    
    def ModeChoice(self):
        
        ''' Compute the proportion of main transport mode share of the SEQ. '''
        
        mainmode = np.array(self.df_5)[:, 12]
        
        total = np.size(mainmode)
        data_count = collections.Counter(mainmode)
        
        num_pri = data_count['Car driver'] \
                + data_count['Car passenger'] \
                + data_count['Truck driver'] \
                + data_count['Motorcycle driver'] \
                + data_count['Motorcycle passenger']
        num_act = data_count['Walking'] \
                + data_count['Bicycle']
        num_pub = data_count['Train'] \
                + data_count['Ferry'] \
                + data_count['Light rail'] \
                + data_count['Mobility scooter'] \
                + data_count['Public bus'] \
                + data_count['Public Bus'] \
                + data_count['School bus (with route number)'] \
                + data_count['School bus (private/chartered)'] \
                + data_count['Charter/Courtesy/Other bus'] \
                + data_count['Other method']
        num_shr = data_count['Taxi'] \
                + data_count['Uber / Other Ride Share']
        
        prop_pri = num_pri / total * 100
        prop_act = num_act / total * 100
        prop_pub = num_pub / total * 100
        prop_shr = num_shr / total * 100
                
        print("Mode share of pirvate vehicle: %.2f%%"   % prop_pri)
        print("Mode share of ativate transport: %.2f%%" % prop_act)
        print("Mode share of public transport: %.2f%%"  % prop_pub)
        print("Mode share of ride share: %.2f%%"        % prop_shr)
        
        # Assign four modes to all the trips
        mode_id = [] # Length: 104,024
        for i in mainmode:
            if   i == 'Car driver' or \
                 i == 'Car passenger' or \
                 i == 'Truck driver' or \
                 i == 'Taxi' or \
                 i == 'Uber / Other Ride Share' or \
                 i == 'Motorcycle driver' or \
                 i == 'Motorcycle passenger':
                mode_id.append(0) # Private vehicle
            elif i == 'Walking' or i == 'Bicycle':
                mode_id.append(1) # Active transport
            else:
                mode_id.append(2) # Public transport
        
        return mode_id
    
    
    def FiveTiers(self, c, emi_ave, mode_id):
        
        # 0, 1015, 1255, 1535, 1750
        id_0 = []
        id_1 = []
        id_2 = []
        id_3 = []
        id_4 = []
        id_5 = []
        ave_time = [19.47601958, 20.82147592, 20.58619454, 22.58492163, 24.73033377]
        
        mode_shr = np.zeros((5, 3))
        time = np.zeros((5, 3))
        
        sa2_main = self.sa2_main
        sa2_array = self.sa2_array
        trav_time = np.array(self.df_5)[:, 23]
        
        for i in range(len(emi_ave)):
            if emi_ave[i] == 0:
                id_0.append(sa2_main[i])
            elif emi_ave[i] < 1015:
                id_1.append(sa2_main[i])
            elif emi_ave[i] < 1255:
                id_2.append(sa2_main[i])
            elif emi_ave[i] < 1535:
                id_3.append(sa2_main[i])
            elif emi_ave[i] < 1750:
                id_4.append(sa2_main[i])
            else:
                id_5.append(sa2_main[i])
                
        for j in range(len(sa2_array)):
            if sa2_array[j] in id_1:
                mode_shr[0][mode_id[j]] += 1
                time[0][0] += trav_time[j]
                time[0][1] += 1
                time[0][2] += (trav_time[j] - ave_time[0]) ** 2
            elif sa2_array[j] in id_2:
                mode_shr[1][mode_id[j]] += 1
                time[1][0] += trav_time[j]
                time[1][1] += 1
                time[1][2] += (trav_time[j] - ave_time[1]) ** 2
            elif sa2_array[j] in id_3:
                mode_shr[2][mode_id[j]] += 1
                time[2][0] += trav_time[j]
                time[2][1] += 1
                time[2][2] += (trav_time[j] - ave_time[2]) ** 2
            elif sa2_array[j] in id_4:
                mode_shr[3][mode_id[j]] += 1
                time[3][0] += trav_time[j]
                time[3][1] += 1
                time[3][2] += (trav_time[j] - ave_time[3]) ** 2
            elif sa2_array[j] in id_5:
                mode_shr[4][mode_id[j]] += 1
                time[4][0] += trav_time[j]
                time[4][1] += 1
                time[4][2] += (trav_time[j] - ave_time[4]) ** 2
                
        sum_shr = np.sum(mode_shr, axis=1)
        mode_shr /= sum_shr.reshape((5, 1))
        print(np.divide(time[:, 0], time[:, 1]))
        print(np.sqrt(np.divide(time[:, 2], time[:, 1]-1)))
        print(time)
                
        return id_0, mode_shr
    
    
    def Plot(self, mode_shr):
        
        size = 3
        total_w, n = 0.8, 5
        
        x = np.arange(size)
        width = total_w / n
        tick_label = ['Private vehicle', 'Active transport', 'Public transport']
        
        plt.bar(x + 0 * width, mode_shr[0], width=width, color=(0.00, 0.00, 1.00), label='0-1014 g')
        plt.bar(x + 1 * width, mode_shr[1], width=width, color=(0.25, 0.00, 0.75), label='1015-1254 g')
        plt.bar(x + 2 * width, mode_shr[2], width=width, color=(0.50, 0.00, 0.50), label='1255-1534 g')
        plt.bar(x + 3 * width, mode_shr[3], width=width, color=(0.75, 0.00, 0.25), label='1535-1749 g')
        plt.bar(x + 4 * width, mode_shr[4], width=width, color=(1.00, 0.00, 0.00), label='Over 1750g')
        plt.legend()
        plt.xticks(x+total_w/2,tick_label)
        plt.xlabel('Travel mode', self.font)
        plt.ylabel('Mode share', self.font)
        plt.show()
        
        return None
        

    def FiveTiersEmission(self, data):
        
        ''' Compute the carbon emission for each trip. '''
        
        zero_cnt = 0
        max_cnt = 0
        min_bound = 0
        max_bound = 10000
        d = 10
        
        hist_emi = data
        
        for index, value in enumerate(hist_emi):
            if value > max_bound:
                max_cnt += 1
                hist_emi[index] = max_bound
            elif value == 0:
                zero_cnt += 1
        
        self.non_zero = len(data) - zero_cnt - max_cnt
                
        num_bins = (max_bound - min_bound) // d
        
        self.hist_emi = hist_emi
        self.num_bins = num_bins
        
        plt.axis([0, max_bound, 0, 180])
        self.n, bin_edges, _ = plt.hist(hist_emi, num_bins)
        self.n[0] = 0
        self.n[-1] = 0
        self.bin_middles = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        plt.title('Carbon emission of each trip by motor vehicle', self.font)
        plt.xlabel('Carbon emissions (g)', self.font)
        plt.ylabel('Number of trips', self.font)
        plt.show()
                        
        return self.n, self.non_zero, self.hist_emi
    
    

        
        
if __name__ == "__main__":
    
    emission = CarbonEachTrip()
    carbon_emi, car_list = emission.TripEmission()
    mode_id = emission.ModeChoice()
    time_ave, emi_ave, num_cnt = emission.RegionTime(carbon_emi)
    d0, mode_shr = emission.FiveTiers(carbon_emi, emi_ave, mode_id)
    emission.Plot(mode_shr)
    