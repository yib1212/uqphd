# -*- coding: utf-8 -*-
"""
Created on Mon Aug 01 14:12:31 2022

@author: Yibo Wang
"""

import numpy as np
import pandas as pd
import pyodbc
import collections


class BusRoute(object):
    
    def __init__(self):
        
        LVE = pd.read_csv('data\\gtfs\\table\\LVE.csv', header=None)
        LineVariants = pd.read_csv('data\\gtfs\\table\\LineVariants.csv', header=None)
        Lines = pd.read_csv('data\\gtfs\\table\\Lines.csv', header=None)
        Trips = pd.read_csv('data\\gtfs\\trips.txt', header=None)
        Calendar = pd.read_csv('data\\gtfs\\calendar.txt', header=None)
        
        self.line_len = np.array(LVE)[1:, 5] # 90174
        self.line_var_id = np.array(LVE)[1:, 1]
        self.id = np.array(LineVariants)[1:, 1] # 3223
        self.line_id = np.array(LineVariants)[1:, 2]
        self.g_route_id = np.array(Lines)[1:, 2] # 1079
        self.route_id = np.array(Trips)[1:, 0] # 128190
        self.service_id = np.array(Trips)[1:, 1]
        self.dir_id = np.array(Trips)[1:, 4]
        self.weekday = np.array(Calendar)[1:, 0:6] # 167
        
        # Database location
        conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
                    r'DBQ=data\Travel Survey\2018-21_pooled_seq_qts_erv1.0.accdb;')
        
        conn = pyodbc.connect(conn_str)
                    
        self.df_1 = pd.read_sql(f'select * from 1_QTS_HOUSEHOLDS', conn)
        self.df_3 = pd.read_sql(f'select * from 3_QTS_VEHICLES', conn)
        self.df_5 = pd.read_sql(f'select * from 5_QTS_TRIPS', conn)
        
        
    def TripsNumber(self):
        
        ''' Compute the number of trips for each route. '''
        
        num_trips = np.zeros((len(self.g_route_id), 2), dtype = int)
        num_days = 0
        
        for i in range(len(self.route_id)):
            idx_trips = np.argwhere(self.g_route_id == self.route_id[i])
            idx_wd = np.argwhere(self.weekday[:, 0] == self.service_id[i])[0][0]
            # If the routes are operated in weekdays
            flag = np.sum(self.weekday[idx_wd, 1:6].astype(int))
            num_days += flag
            if flag > 0:
                # Two directions
                if self.dir_id[i] == '0':
                    num_trips[idx_trips, 0] += flag
                else:
                    num_trips[idx_trips, 1] += flag
        
        df = pd.DataFrame(num_trips)
        # df.to_csv('data\\bus_trip_number.csv')
        
        return num_trips
    
    
    def TripsDistance(self):
        
        ''' Compute the distance of each route. '''
        
        dis_trips = np.zeros(len(self.g_route_id))
        line_len = self.line_len.astype(float)
        
        for i in range(len(self.line_len)):
            idx_lv = np.argwhere(self.id == self.line_var_id[i])
            idx_lines = self.line_id[idx_lv].astype(int)[0][0] - 1 # Notice the 0 and 1
            dis_trips[idx_lines] += line_len[i]
            
        return dis_trips
    
    
    def TotalDistance(self, dis_trips):
        
        ''' Compute the total distance. '''
        
        df = pd.read_csv('data\\bus_trip_number_flag.csv', header=None)
        num_trips = np.array(df)[1:, :].astype(int)
        tot_dist = 0
        tot_num = 0
        for i in range(num_trips.shape[0]):
            if num_trips[i][3] == 1:
                tot_dist += (num_trips[i][1] + num_trips[i][2]) * dis_trips[i]
                tot_num += num_trips[i][1] + num_trips[i][2]
                
        print(tot_num)
        
        return tot_dist
    
    
    def PriVehLoadFactor(self):
        
        ''' Compute the private vehicle loading factor. '''
        
        all_mode = np.array(self.df_5)[:, 13:21]
        driver_cnt = 0
        passenger_cnt = 0
        motordriver_cnt = 0
        motorpass_cnt = 0
        
        for i in range(8):
            role_count = collections.Counter(all_mode[:, i])
            driver_cnt += role_count['Car driver']
            passenger_cnt += role_count['Car passenger']
            motordriver_cnt += role_count['Motorcycle driver']
            motorpass_cnt += role_count['Motorcycle passenger']
                    
        car_lf = (driver_cnt + passenger_cnt) / driver_cnt
        motor_lf = (motordriver_cnt + motorpass_cnt) / motordriver_cnt
        
        car_lf_result = 1.4071246006389777
        motor_lf_result = 1.0505050505050506
        
        return car_lf, motor_lf
            
        
        
if __name__ == "__main__":
    
    b = BusRoute()
    num_trips = b.TripsNumber()
    dis_trips = b.TripsDistance()
    tot_dist = b.TotalDistance(dis_trips)
    car_lf, motor_lf = b.PriVehLoadFactor()
    
    result_bus = 12066198846.531265