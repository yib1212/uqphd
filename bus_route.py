# -*- coding: utf-8 -*-
"""
Created on Mon Aug 01 14:12:31 2022

@author: Yibo Wang
"""

import numpy as np
import pandas as pd


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
            
        
        
if __name__ == "__main__":
    
    b = BusRoute()
    num_trips = b.TripsNumber()
    dis_trips = b.TripsDistance()
    tot_dist = b.TotalDistance(dis_trips)
    print(tot_dist)
    
    result_bus = 12066198846.531265