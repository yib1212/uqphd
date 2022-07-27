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


def ReadDatabase():
    
    ''' Read the MS access database and return the dataframe using pandas. '''
    
    # Database location
    conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
                r'DBQ=data\Travel Survey\2018-21_pooled_seq_qts_erv1.0.accdb;')
    
    conn = pyodbc.connect(conn_str)
    
    cursor = conn.cursor()
    for i in cursor.tables(tableType='TABLE'):
        print(i.table_name)
        
    return conn


def ReadTable(conn, table_name):
    
    # Table name
    df = pd.read_sql(f'select * from {table_name}', conn)
    
    return df


def ModeChoice(data_frame):
    
    ''' Compute the proportion of transport mode share of the SEQ. '''
    
    mainmode = np.array(data_frame)[:, 12]
    
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
    
    mode_id = []
    for i in mainmode:
        if i == 'Car driver' or \
           i == 'Car passenger' or \
           i == 'Truck driver' or \
           i == 'Motorcycle driver' or \
           i == 'Motorcycle passenger':
            mode_id.append('Private Vehicle')
        elif i == 'Walking' or i == 'Bicycle':
            mode_id.append('Active Transport')
        elif i == 'Taxi' or i == 'Uber / Other Ride Share':
            mode_id.append('Taxi or Rideshare')
        else:
            mode_id.append('Public Transport')
    
    
    return mode_id
    
    
def TimePerKilo(df1, df2):
    
    # Counter({'petrol': 20407, 'diesel': 5764, 'hybrid': 154, 'lpg': 79, 'electric': 34, None: 31})
    fueltype = np.array(df1)[:, 2]
    # Counter({'car': 22076, 'van': 3470, 'motorcycle': 770, 'truck': 115, 'other': 38})
    vehitype = np.array(df1)[:, 3]
    duration = np.array(df2)[:, 23]
    cumdist = np.array(df2)[:, 24]
    
    fuel_count = collections.Counter(fueltype)
    vehi_count = collections.Counter(vehitype)
    
    time_per_kilo = np.divide(duration, cumdist, out=np.zeros_like(duration), where=cumdist!=0)
    print(fuel_count)
    print(vehi_count)
    
    
def CarbonGenerate():
    
    ''' Compute the carbon generated each trip. '''
    
    carbon = None
    
    carb_inten = ['pet', 'die', 'hyb', 'lpg', 'ele']    # Energy intensity of different energy type
    trip_mode = None
    trip_dist = None
    trip_time = None
    load_fctr = None
    
    carbon = trip_mode * trip_dist * trip_time * load_fctr
    
    return carbon


def SA2Population():
    
    ''' The population of all SA2. '''
    
    df = pd.read_excel('data\TableBuilder\SA2_by_SEXP.xlsx', engine='openpyxl')
    sa2_sex = np.array(df)[9:-8, 4] # length: 317
    print(sa2_sex)
    
    return None


def SA2Info():
    
    ''' Get SA2 information from the shapefile. '''
    
    sa2 = dbfread.DBF('data\\1270055001_sa2_2016_aust_shape\\SA2_2016_AUST.dbf', encoding='GBK')
    df = pd.DataFrame(iter(sa2))
    
    sa2_main = np.array(df)[:, 0].astype(int)
       
    return sa2_main    


def TripNumber(df1, df5, sa2_list):
    
    ''' Number of daily trips in SA2 j. '''
    
    # household ID and SA1 ID (length: 15543)
    hhid_1 = np.array(df_1)[:, 0].astype(int)
    sa1_id = np.array(df_1)[:, 13]
    sa2_id = (sa1_id/1e2).astype(int)
    # Household ID (length: 104024)
    hhid_5 = np.array(df_5)[:, 1].astype(int)
    
    sa2_array = []
    counter = 0
    
    # Record the SA2 ID of each trip
    for i in hhid_5:
        index = np.argwhere(hhid_1 == i)
        sa2_array.append(sa2_id[index[0, 0]])
        counter += 1
        print("Matching the %d trip, 104024 in total" % counter)
    
    # The length of trip_num is 275
    trip_num = dict(collections.Counter(sa2_array))
    
    # Rearrange the results in the order of SA2 within the shapefile
    trip_num_sa2 = []
    for j in sa2_list:
        trip_num_sa2.append(trip_num.get(j, 0))
    
    return trip_num_sa2

    

if __name__ == "__main__":
    
    conn = ReadDatabase()
    df_1 = ReadTable(conn, '1_QTS_HOUSEHOLDS')
    df_3 = ReadTable(conn, '3_QTS_VEHICLES')
    df_5 = ReadTable(conn, '5_QTS_TRIPS')
    sa2_list = SA2Info()
    # TimePerKilo(df_3, df_5)
    # num_trip = TripNumber(df_1, df_5, sa2_list)
    # pop = SA2Population()
    
    mode_id = ModeChoice(df_5)