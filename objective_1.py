# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 03:35:31 2022

@author: Yibo Wang
"""

import pyodbc
import dbfread
import collections
import pandas as pd
import numpy as np


class CarbonEmission(object):
    
    def __init__(self):
        
        ''' Read the MS access database and return the dataframe using pandas. '''
        
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
    
    
    def MultiModeChoice(self):
        
        ''' Count the proportion of detailed transport mode share of the SEQ. '''
        
        all_mode = np.array(self.df_5)[:, 13:21]
        num_pri = 0
        num_act = 0
        num_pub = 0
        num_shr = 0
        
        for i in range(all_mode.shape[1]):
            data_count = collections.Counter(all_mode[:, i])
            num_pri += data_count['Car driver'] \
                    + data_count['Car passenger'] \
                    + data_count['Truck driver'] \
                    + data_count['Motorcycle driver'] \
                    + data_count['Motorcycle passenger']
            num_act += data_count['Walking'] \
                    + data_count['Bicycle']
            num_pub += data_count['Train'] \
                    + data_count['Ferry'] \
                    + data_count['Light rail'] \
                    + data_count['Mobility scooter'] \
                    + data_count['Public bus'] \
                    + data_count['Public Bus'] \
                    + data_count['School bus (with route number)'] \
                    + data_count['School bus (private/chartered)'] \
                    + data_count['Charter/Courtesy/Other bus'] \
                    + data_count['Other method']
            num_shr += data_count['Taxi'] \
                    + data_count['Uber / Other Ride Share']
                    
        total = num_pri + num_act + num_pub + num_shr
        
        prop_pri = num_pri / total * 100
        prop_act = num_act / total * 100
        prop_pub = num_pub / total * 100
        prop_shr = num_shr / total * 100
                
        print("Mode share of pirvate vehicle: %.2f%%"   % prop_pri)
        print("Mode share of ativate transport: %.2f%%" % prop_act)
        print("Mode share of public transport: %.2f%%"  % prop_pub)
        print("Mode share of ride share: %.2f%%"        % prop_shr)
        
        self.num_pub_trip = num_pub
        
        return None
        
        
    def TimePerKilo(self, mode_id):
        
        ''' Compute the time used per kilometer of each trip. '''
        
        # Counter({'petrol': 20407, 'diesel': 5764, 'hybrid': 154, 'lpg': 79, 'electric': 34, None: 31})
        fueltype = np.array(self.df_3)[:, 2]
        # Counter({'car': 22076, 'van': 3470, 'motorcycle': 770, 'truck': 115, 'other': 38})
        vehitype = np.array(self.df_3)[:, 3]
        duration = np.array(self.df_5)[:, 23]
        cumdist = np.array(self.df_5)[:, 24]
        
        fuel_count = collections.Counter(fueltype)
        vehi_count = collections.Counter(vehitype)
        
        time_per_kilo = np.divide(duration, cumdist, out=np.zeros_like(duration), where=cumdist!=0)
        print(time_per_kilo)
        
        #######################################################################
        total_dist = np.zeros((4, 1))
        total_duration = np.zeros((4, 1))
        mode_count = collections.Counter(mode_id)
        print(mode_count)
        
        for i in range(len(mode_id)):
            idx = mode_id[i]
            total_dist[idx][0] += cumdist[i]
            total_duration[idx][0] += duration[i]
            
        ave_act_dist = total_dist[1][0] / 10716 # 1.3441946621873895
        
        print(total_duration / total_dist)
        print(total_dist[1][0] / 10716)
        
        
    def CarbonGenerate(self):
        
        ''' Compute the carbon generated each trip. '''
        
        carbon = None
        
        carb_inten = ['pet', 'die', 'hyb', 'lpg', 'ele']    # Energy intensity of different energy type
        trip_mode = None
        trip_dist = None
        trip_time = None
        load_fctr = None
        
        carbon = trip_mode * trip_dist * trip_time * load_fctr
        
        return carbon
    
    
    def SA2Population(self):
        
        ''' The population of all SA2. '''
        
        df = pd.read_excel('data\TableBuilder\SA2_by_SEXP.xlsx', engine='openpyxl')
        sa2_sex = np.array(df)[9:-8, 4] # length: 317
        
        return sa2_sex
    
    
    def SA2Info(self):
        
        ''' Get SA2 information from the database file. '''
        
        sa2 = dbfread.DBF('data\\1270055001_sa2_2016_aust_shape\\SA2_2016_AUST.dbf', encoding='GBK')
        df = pd.DataFrame(iter(sa2))
        
        self.sa2_main = np.array(df)[:, 0].astype(int)
           
        return self.sa2_main
    
    
    def TripNumber(self):
        
        ''' 
        Number of daily trips in SA2 j. 
        Statistics on the number of people who participated in the traffic survey. 
        
        '''
        
        # Household ID, household size, and SA1 ID (length: 15543)
        sa2_main = self.sa2_main
        hhid_1 = np.array(self.df_1)[:, 0].astype(int)
        hhsize = np.array(self.df_1)[:, 1].astype(int)
        sa1_id = np.array(self.df_1)[:, 13]
        sa2_id = (sa1_id/1e2).astype(int)
        # Household ID (length: 104024)
        hhid_5 = np.array(self.df_5)[:, 1].astype(int)
        
        sa2_array = []
        counter = 0
        trip_pop = np.zeros(len(sa2_main), dtype = int)
        
        # Record the SA2 ID of each trip
        for i in hhid_5:
            index = np.argwhere(hhid_1 == i)
            sa2_array.append(sa2_id[index[0, 0]])
            counter += 1
            
        # The length of trip_num is 275
        trip_num = dict(collections.Counter(sa2_array))
        
        # Rearrange the results in the order of SA2 within the shapefile
        trip_num_sa2 = []
        for j in sa2_main:
            trip_num_sa2.append(trip_num.get(j, 0))
            
        # Comput the number of people who participated in the traffic survey
        for k in range(len(hhsize)):
            idx = np.argwhere(sa2_main == sa2_id[k])
            trip_pop[idx] += hhsize[k]
        
        # Number of trips per people per day
        NT = np.around(np.divide(np.array(trip_num_sa2), trip_pop, where=trip_pop!=0), 3)
            
        return sa2_array, trip_num_sa2, NT
    
    
    def ModeProportion(self, sa2_array, trip_num_sa2, mode_id):
        
        ''' Compute the travel mode proportion of each SA2 region. '''
        
        sa2_main = self.sa2_main
        mode_cnt = np.zeros((len(sa2_main), 4), dtype = int)
        trip_num_sa2 = np.array(trip_num_sa2).reshape((len(trip_num_sa2), 1))
        
        # Rows are SA2 regionm columns are travel model
        # 0: Pri, 1: Act, 2: Shr, 3: Pub
        for i in range(len(mode_id)):
            row_idx = np.argwhere(sa2_main == sa2_array[i])
            col_idx = mode_id[i]
            mode_cnt[row_idx, col_idx] += 1
            
        mode_prop = np.divide(mode_cnt, trip_num_sa2, where=trip_num_sa2!=0)
        
        return mode_cnt, mode_prop
    
    
    def AverDistance(self, sa2_array, mode_id, mode_cnt):
        
        ''' Average distance when using mode i in SA2 region j. '''
        
        sa2_main = self.sa2_main
        all_mode = np.array(self.df_5)[:, 13:21]
        total_dist = np.zeros(mode_cnt.shape)
        num_act_trip = 0
        
        cumdist = np.array(self.df_5)[:, 24]
        for i in range(len(mode_id)):
            row_idx = np.argwhere(sa2_main == sa2_array[i])
            col_idx = mode_id[i]
            total_dist[row_idx, col_idx] += cumdist[i]
            if mode_id[i] == 3:
                num_act_trip += np.sum(all_mode[i] == 'Walking')
                
        ave_dist = np.divide(total_dist, mode_cnt, where=mode_cnt!=0)
        # Average distance by public transport of the entire region: 18.076 km
        self.ave_dist_tot = (np.sum(total_dist[:, 3]) - num_act_trip * 1.344) / self.num_pub_trip
        
        return ave_dist
    
    
    def BusLoadFactor(self):
        
        ''' Compute the LF of public transport. '''
        
        # Translink fast facts
        daytrip_1819 = 519795
        daytrip_1920 = 418054
        daytrip_2021 = 326193
        daytrip_total = daytrip_1819 + daytrip_1920 + daytrip_2021
        yeartrip = daytrip_total * 365
        ppl_day = 421347.3333333333
        tot_dist = 12066198846.531265 / 5 / 1000 #4548033.226433979
        
        total_ppl_dist = ppl_day * self.ave_dist_tot
        print(self.ave_dist_tot)
        print(total_ppl_dist / tot_dist)
        
        result = 3.1560562778093737
        pkm = 68.66
        
        return None
    
    

if __name__ == "__main__":
    
    carbon = CarbonEmission()
    
    
    pop = carbon.SA2Population()
    carbon.SA2Info()
    
    sa2_array, trip_num_sa2, NT = carbon.TripNumber()
    
    mode_id = carbon.ModeChoice()
    carbon.TimePerKilo(mode_id)
    carbon.MultiModeChoice()
    mode_cnt, mode_prop = carbon.ModeProportion(sa2_array, trip_num_sa2, mode_id)
    ave_dist = carbon.AverDistance(sa2_array, mode_id, mode_cnt)
    BLF = carbon.BusLoadFactor()
    
    # GoCard or go ahead for private vehicle LF
    
=======
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 03:35:31 2022

@author: Yibo Wang
"""

import pyodbc
import dbfread
import collections
import pandas as pd
import numpy as np


class CarbonEmission(object):
    
    def __init__(self):
        
        ''' Read the MS access database and return the dataframe using pandas. '''
        
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
    
    
    def MultiModeChoice(self):
        
        ''' Count the proportion of detailed transport mode share of the SEQ. '''
        
        all_mode = np.array(self.df_5)[:, 13:21]
        num_pri = 0
        num_act = 0
        num_pub = 0
        num_shr = 0
        
        for i in range(all_mode.shape[1]):
            data_count = collections.Counter(all_mode[:, i])
            num_pri += data_count['Car driver'] \
                    + data_count['Car passenger'] \
                    + data_count['Truck driver'] \
                    + data_count['Motorcycle driver'] \
                    + data_count['Motorcycle passenger']
            num_act += data_count['Walking'] \
                    + data_count['Bicycle']
            num_pub += data_count['Train'] \
                    + data_count['Ferry'] \
                    + data_count['Light rail'] \
                    + data_count['Mobility scooter'] \
                    + data_count['Public bus'] \
                    + data_count['Public Bus'] \
                    + data_count['School bus (with route number)'] \
                    + data_count['School bus (private/chartered)'] \
                    + data_count['Charter/Courtesy/Other bus'] \
                    + data_count['Other method']
            num_shr += data_count['Taxi'] \
                    + data_count['Uber / Other Ride Share']
                    
        total = num_pri + num_act + num_pub + num_shr
        
        prop_pri = num_pri / total * 100
        prop_act = num_act / total * 100
        prop_pub = num_pub / total * 100
        prop_shr = num_shr / total * 100
                
        print("Mode share of pirvate vehicle: %.2f%%"   % prop_pri)
        print("Mode share of ativate transport: %.2f%%" % prop_act)
        print("Mode share of public transport: %.2f%%"  % prop_pub)
        print("Mode share of ride share: %.2f%%"        % prop_shr)
        
        self.num_pub_trip = num_pub
        
        return None
        
        
    def TimePerKilo(self, mode_id):
        
        ''' Compute the time used per kilometer of each trip. '''
        
        # Counter({'petrol': 20407, 'diesel': 5764, 'hybrid': 154, 'lpg': 79, 'electric': 34, None: 31})
        fueltype = np.array(self.df_3)[:, 2]
        # Counter({'car': 22076, 'van': 3470, 'motorcycle': 770, 'truck': 115, 'other': 38})
        vehitype = np.array(self.df_3)[:, 3]
        duration = np.array(self.df_5)[:, 23]
        cumdist = np.array(self.df_5)[:, 24]
        
        fuel_count = collections.Counter(fueltype)
        vehi_count = collections.Counter(vehitype)
        
        time_per_kilo = np.divide(duration, cumdist, out=np.zeros_like(duration), where=cumdist!=0)
        print(time_per_kilo)
        
        #######################################################################
        total_dist = np.zeros((4, 1))
        total_duration = np.zeros((4, 1))
        mode_count = collections.Counter(mode_id)
        print(mode_count)
        
        for i in range(len(mode_id)):
            idx = mode_id[i]
            total_dist[idx][0] += cumdist[i]
            total_duration[idx][0] += duration[i]
            
        ave_act_dist = total_dist[1][0] / 10716 # 1.3441946621873895
        
        print(total_duration / total_dist)
        print(total_dist[1][0] / 10716)
        
        
    def CarbonGenerate(self):
        
        ''' Compute the carbon generated each trip. '''
        
        carbon = None
        
        carb_inten = ['pet', 'die', 'hyb', 'lpg', 'ele']    # Energy intensity of different energy type
        trip_mode = None
        trip_dist = None
        trip_time = None
        load_fctr = None
        
        carbon = trip_mode * trip_dist * trip_time * load_fctr
        
        return carbon
    
    
    def SA2Population(self):
        
        ''' The population of all SA2. '''
        
        df = pd.read_excel('data\TableBuilder\SA2_by_SEXP.xlsx', engine='openpyxl')
        sa2_sex = np.array(df)[9:-8, 4] # length: 317
        
        return sa2_sex
    
    
    def SA2Info(self):
        
        ''' Get SA2 information from the database file. '''
        
        sa2 = dbfread.DBF('data\\1270055001_sa2_2016_aust_shape\\SA2_2016_AUST.dbf', encoding='GBK')
        df = pd.DataFrame(iter(sa2))
        
        self.sa2_main = np.array(df)[:, 0].astype(int)
           
        return None
    
    
    def TripNumber(self):
        
        ''' 
        Number of daily trips in SA2 j. 
        Statistics on the number of people who participated in the traffic survey. 
        
        '''
        
        # Household ID, household size, and SA1 ID (length: 15543)
        sa2_main = self.sa2_main
        hhid_1 = np.array(self.df_1)[:, 0].astype(int)
        hhsize = np.array(self.df_1)[:, 1].astype(int)
        sa1_id = np.array(self.df_1)[:, 13]
        sa2_id = (sa1_id/1e2).astype(int)
        # Household ID (length: 104024)
        hhid_5 = np.array(self.df_5)[:, 1].astype(int)
        
        sa2_array = []
        counter = 0
        trip_pop = np.zeros(len(sa2_main), dtype = int)
        
        # Record the SA2 ID of each trip
        for i in hhid_5:
            index = np.argwhere(hhid_1 == i)
            sa2_array.append(sa2_id[index[0, 0]])
            counter += 1
            
        # The length of trip_num is 275
        trip_num = dict(collections.Counter(sa2_array))
        
        # Rearrange the results in the order of SA2 within the shapefile
        trip_num_sa2 = []
        for j in sa2_main:
            trip_num_sa2.append(trip_num.get(j, 0))
            
        # Comput the number of people who participated in the traffic survey
        for k in range(len(hhsize)):
            idx = np.argwhere(sa2_main == sa2_id[k])
            trip_pop[idx] += hhsize[k]
        
        # Number of trips per people per day
        NT = np.around(np.divide(np.array(trip_num_sa2), trip_pop, where=trip_pop!=0), 3)
            
        return sa2_array, trip_num_sa2, NT
    
    
    def ModeProportion(self, sa2_array, trip_num_sa2, mode_id):
        
        ''' Compute the travel mode proportion of each SA2 region. '''
        
        sa2_main = self.sa2_main
        mode_cnt = np.zeros((len(sa2_main), 4), dtype = int)
        trip_num_sa2 = np.array(trip_num_sa2).reshape((len(trip_num_sa2), 1))
        
        # Rows are SA2 regionm columns are travel model
        # 0: Pri, 1: Act, 2: Shr, 3: Pub
        for i in range(len(mode_id)):
            row_idx = np.argwhere(sa2_main == sa2_array[i])
            col_idx = mode_id[i]
            mode_cnt[row_idx, col_idx] += 1
            
        mode_prop = np.divide(mode_cnt, trip_num_sa2, where=trip_num_sa2!=0)
        
        return mode_cnt, mode_prop
    
    
    def AverDistance(self, sa2_array, mode_id, mode_cnt):
        
        ''' Average distance when using mode i in SA2 region j. '''
        
        sa2_main = self.sa2_main
        all_mode = np.array(self.df_5)[:, 13:21]
        total_dist = np.zeros(mode_cnt.shape)
        num_act_trip = 0
        
        cumdist = np.array(self.df_5)[:, 24]
        for i in range(len(mode_id)):
            row_idx = np.argwhere(sa2_main == sa2_array[i])
            col_idx = mode_id[i]
            total_dist[row_idx, col_idx] += cumdist[i]
            if mode_id[i] == 3:
                num_act_trip += np.sum(all_mode[i] == 'Walking')
                
        ave_dist = np.divide(total_dist, mode_cnt, where=mode_cnt!=0)
        # Average distance by public transport of the entire region: 18.076 km
        self.ave_dist_tot = (np.sum(total_dist[:, 3]) - num_act_trip * 1.344) / self.num_pub_trip
        
        return ave_dist
    
    
    def BusLoadFactor(self):
        
        ''' Compute the LF of public transport. '''
        
        # Translink fast facts
        daytrip_1819 = 519795
        daytrip_1920 = 418054
        daytrip_2021 = 326193
        daytrip_total = daytrip_1819 + daytrip_1920 + daytrip_2021
        yeartrip = daytrip_total * 365
        ppl_day = 421347.3333333333
        tot_dist = 12066198846.531265 / 5 / 1000 #4548033.226433979
        
        total_ppl_dist = ppl_day * self.ave_dist_tot
        print(self.ave_dist_tot)
        print(total_ppl_dist / tot_dist)
        
        result = 3.1560562778093737
        
        return None
    
    

if __name__ == "__main__":
    
    carbon = CarbonEmission()
    
    
    pop = carbon.SA2Population()
    carbon.SA2Info()
    
    sa2_array, trip_num_sa2, NT = carbon.TripNumber()
    
    mode_id = carbon.ModeChoice()
    carbon.TimePerKilo(mode_id)
    carbon.MultiModeChoice()
    mode_cnt, mode_prop = carbon.ModeProportion(sa2_array, trip_num_sa2, mode_id)
    ave_dist = carbon.AverDistance(sa2_array, mode_id, mode_cnt)
    BLF = carbon.BusLoadFactor()

    # GoCard or go ahead for private vehicle LF
    