# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 12:39:44 2024

@author: Yibo Wang
"""

import pyodbc
import dbfread
import collections
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import levy

class CarbonEmission(object):
    
    def __init__(self):
        
        ''' Read the MS access database and return the dataframe using pandas. '''
        
        # Database location
        conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
                    r'DBQ=data\Travel Survey\2018-21_pooled_seq_qts_erv1.0.accdb;')
        
        conn = pyodbc.connect(conn_str)
        
        cursor = conn.cursor()
        # for i in cursor.tables(tableType='TABLE'):
        #     print(i.table_name)
            
        self.df_1 = pd.read_sql(f'select * from 1_QTS_HOUSEHOLDS', conn)
        self.df_3 = pd.read_sql(f'select * from 3_QTS_VEHICLES', conn)
        self.df_5 = pd.read_sql(f'select * from 5_QTS_TRIPS', conn)
        
        self.car_ave = 152.3
        self.bus_ave = 100.2
        self.font = {'size': 10}
        
        return None
    
    
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
                 i == 'Motorcycle driver' or \
                 i == 'Motorcycle passenger':
                mode_id.append(0) # Private vehicle
            elif i == 'Walking' or i == 'Bicycle':
                mode_id.append(1) # Active transport
            elif i == 'Taxi' or i == 'Uber / Other Ride Share':
                mode_id.append(2) # Taxi or rideshare
            else:
                mode_id.append(3) # Public transport'
        
        return mode_id
    
    
    def TripEmission(self, mode_id):
        
        ''' Compute the carbon emission for each trip. '''
        
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
        
        non_zero_emi = []
        
        for i in range(len(carbon_emi)):
            if carbon_emi[i] != 0:
                non_zero_emi.append(carbon_emi[i])
            
        print(len(non_zero_emi))
        
        hist_emi = carbon_emi
        
        for index, value in enumerate(hist_emi):
            if value > max_bound:
                max_cnt += 1
                hist_emi[index] = max_bound
        
        self.non_zero = len(mode_id) - zero_cnt - max_cnt
                
        num_bins = (max_bound - min_bound) // d
        
        self.hist_emi = hist_emi
        self.num_bins = num_bins
        
        plt.axis([0, max_bound, 0, 900])
        self.n, bin_edges, _ = plt.hist(hist_emi, num_bins, color=(0, 0, 1))
        self.n[0] = 0
        self.n[-1] = 0
        self.bin_middles = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        plt.title('Carbon emission of each trip by motor vehicle', self.font)
        plt.xlabel('Carbon emissions (g)', self.font)
        plt.ylabel('Number of trips', self.font)
        plt.show()
                        
        return self.n, self.non_zero, non_zero_emi
    
    
    def Levy(self, x, sigma, mu, a, c):
        
        # levy_fit = np.sqrt(sigma/(2*math.pi)) * (np.exp(-(sigma/2*(x-mu)))) / (np.power((x-mu),2))
        levy_fit = levy.pdf(x, mu, sigma)
        
        return a * levy_fit
    
    
    def EM_algorithm(self, n, carbon_emi):
        
        SA = 11
        # loc_EM = np.zeros(SA)
        # sigma_EM = np.array([880., 865., 850., 840.])
        # tao = np.array([0.1, 0.2, 0.3, 0.4])
        loc_EM = -np.ones(SA) * 22.86
        sigma_EM = np.array([730., 610., 950., 895., 697., 1080., 480., 760., 850., 960., 830.])
        tao = np.array([0.064, 0.067, 0.104, 0.051, 0.095, 0.184, 0.093, 0.090, 0.078, 0.058, 0.116])
        prob_xi = np.zeros(SA)
        E = []
        M = []
        cnt = []
        ttl = 93046
        
        for iteration in range(n):
            
            print("iteration:", iteration + 1)
            cnt.append(iteration)
        
            sum_of_Tx = np.zeros(SA)
            sum_of_cov = np.zeros(SA)
            T_i = np.zeros((SA, ttl))
            div_T_i = np.zeros((SA, ttl))
            
            log_like_E = 0
            log_like_M = 0
            
            ''' E step. '''
            
            for i in range(ttl):
                for j in range(SA):
                    prob_xi[j] = self.Levy(carbon_emi[i], sigma_EM[j], loc_EM[j], 14.21960069, 0)
                for j in range(SA):
                    T_i[j][i] = (prob_xi[j] * tao[j]) / np.dot(np.array(prob_xi), np.array(tao))
                    div_T_i[j][i] = 1 * (T_i[j][i] / carbon_emi[i])
                    
                log_like_E -= np.log(np.dot(np.array(prob_xi), np.array(tao)))
                
            E.append(log_like_E/ttl)
            print(log_like_E/ttl)
            
            ''' M step '''        
            
            tao = T_i.sum(axis=1) / ttl
            print(tao)
                        
            # print(T_i.sum(axis=1))
            # print(div_T_i.sum(axis=1))
            sigma_EM = T_i.sum(axis=1) / div_T_i.sum(axis=1)
            loc_EM = -0.14756 * sigma_EM + 106.6378
            print(sigma_EM)
            print(loc_EM)
            # [ 9217.1303387  18523.44472832 27927.57088207 37377.85405091]
            # [6.66554190e+06 2.54397058e+04 7.07226861e+02 4.28174069e+02]
            # for i in range(ttl):
            #     print(i, "M step log M")
            #     for j in range(SA):
            #         prob_xi[j] = self.Levy(carbon_emi[i], sigma_EM[j], loc_EM[j], 1, 0)
            #     log_like_M -= np.log(np.dot(np.array(prob_xi), np.array(tao)))
                
            # M.append(log_like_M/ttl)
            
        return(tao, sigma_EM)
            
    

if __name__ == "__main__":
    
    carbon = CarbonEmission()
    mode_id = carbon.ModeChoice()
    n, non_zero, carbon_emi = carbon.TripEmission(mode_id)
    tao, sigma_EM = carbon.EM_algorithm(50, carbon_emi)
    