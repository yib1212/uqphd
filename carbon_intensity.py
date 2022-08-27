# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 17:08:03 2022

@author: Yibo Wang
"""

import pyodbc
import numpy as np
import pandas as pd
import collections

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
        
        return None
    
    
    def ModeChoice(self):
        
        ''' Compute the proportion of main transport mode share of the SEQ. '''
        
        main_mode = np.array(self.df_5)[:, 12]
        
        total = np.size(main_mode)
        data_count = collections.Counter(main_mode)
        
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
        for i in main_mode:
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
        
        for i in range(len(mode_id)):
            if mode_id[i] == 0 or mode_id[i] == 2:
                carbon_emi.append(cum_dist[i] * self.car_ave)
                car_list.append(cum_dist[i] * self.car_ave)
            elif mode_id[i] == 3:
                carbon_emi.append(cum_dist[i] * self.bus_ave)
            else:
                carbon_emi.append(0)
                
        #carbon_emi = np.round(np.array(carbon_emi), 2)
        
        return carbon_emi, car_list
    
    
    def Histogram(self, carbon_emi):
        
        ''' Do some basic statistical analysis. '''
        
        for index, value in enumerate(carbon_emi):
            if value > 15000:
                carbon_emi[index] = 15000
        
        d = 10
        min_bound = int(min(carbon_emi))
        max_bound = int(max(carbon_emi))
        num_bins = (max_bound - min_bound) // d
        
        plt.hist(carbon_emi, num_bins)
        plt.title('Carbon emission of each trip by passenger car', self.font)
        plt.xlabel('Carbon emissions (g)', self.font)
        plt.ylabel('Number of trips', self.font)
                
        plt.show()
        
        return None
    
    
    def TravelPurpose(self, carbon_emi):
        
        ''' Explore the relationship between emissions and travel purpose. '''
        
        for index, value in enumerate(carbon_emi):
            if value > 15000:
                carbon_emi[index] = 15000
        
        d = 10
        min_bound = int(min(carbon_emi))
        max_bound = int(max(carbon_emi))
        num_bins = (max_bound - min_bound) // d
                        
        purpose = np.array(self.df_5)[:, 25]
        # purpose_count = collections.Counter(purpose)
        # print(purpose_count)
        # purpose = list(purpose)
        # print(len(purpose))
        # print(len(carbon_emi))
        
        commute = []
        shopping = []
        pickup = []
        recreation = []
        education = []
        business = []
        accompany = []
        work = []
        social = []
        deliver = []
        other = []
        
        for index, value in enumerate(purpose):
            if value == 'Direct Work Commute':
                commute.append(carbon_emi[index])
            elif value == 'Shopping':
                shopping.append(carbon_emi[index])
            elif value == 'Pickup/Dropoff Someone':
                pickup.append(carbon_emi[index])
            elif value == 'Recreation':
                recreation.append(carbon_emi[index])
            elif value == 'Education':
                education.append(carbon_emi[index])
            elif value == 'Personal Business':
                business.append(carbon_emi[index])
            elif value == 'Accompany Someone':
                accompany.append(carbon_emi[index])
            elif value == 'Work Related':
                work.append(carbon_emi[index])
            elif value == 'Social':
                social.append(carbon_emi[index])
            elif value == 'Pickup/Deliver Something':
                deliver.append(carbon_emi[index])
            else:
                other.append(carbon_emi[index])
        
        plt.figure(figsize=(10, 20))
        
        ax1=plt.subplot(5,2,1)
        plt.axis([0, 15000, 0, 100])
        plt.hist(commute, num_bins)
        plt.title('Carbon emission of each trip: direct work commute', self.font)
        
        ax2=plt.subplot(5,2,2)
        plt.axis([0, 15000, 0, 250])
        plt.hist(shopping, num_bins)
        plt.title('Carbon emission of each trip: shopping', self.font)
        
        ax3=plt.subplot(5,2,3)
        plt.axis([0, 15000, 0, 250])
        plt.hist(pickup, num_bins)
        plt.title('Carbon emission of each trip: pickup', self.font)
                
        ax4=plt.subplot(5,2,4)
        plt.axis([0, 15000, 0, 100])
        plt.hist(recreation, num_bins)
        plt.title('Carbon emission of each trip: recreation', self.font)
        
        ax5=plt.subplot(5,2,5)
        plt.axis([0, 15000, 0, 150])
        plt.hist(education, num_bins)
        plt.title('Carbon emission of each trip: education', self.font)
        plt.ylabel('Number of trips', self.font)
        
        ax6=plt.subplot(5,2,6)
        plt.axis([0, 15000, 0, 100])
        plt.hist(business, num_bins)
        plt.title('Carbon emission of each trip: personal business', self.font)
        
        ax7=plt.subplot(5,2,7)
        plt.axis([0, 15000, 0, 100])
        plt.hist(accompany, num_bins)
        plt.title('Carbon emission of each trip: accompany', self.font)
        
        ax8=plt.subplot(5,2,8)
        plt.axis([0, 15000, 0, 40])
        plt.hist(work, num_bins)
        plt.title('Carbon emission of each trip: work related', self.font)
        
        ax9=plt.subplot(5,2,9)
        plt.axis([0, 15000, 0, 40])
        plt.hist(social, num_bins)
        plt.title('Carbon emission of each trip: social', self.font)
        plt.xlabel('Carbon emissions (g)', self.font)
        
        ax10=plt.subplot(5,2,10)
        plt.axis([0, 15000, 0, 30])
        plt.hist(deliver, num_bins)
        plt.title('Carbon emission of each trip: deliver', self.font)
        plt.xlabel('Carbon emissions (g)', self.font)
        
        
        plt.show()
        
        return None
        
        
if __name__ == "__main__":
    
    emission = CarbonEachTrip()
    mode_id = emission.ModeChoice()
    carbon_emi, car_list = emission.TripEmission(mode_id)
    emission.Histogram(car_list)
    emission.TravelPurpose(carbon_emi)
    