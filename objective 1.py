# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 03:35:31 2022

@author: Yibo Wang
"""

import pyodbc
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
    
    ''' Compute the proportion of transport mode share. '''
    
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
    
    sa2_sex = pd.read_excel("SA2_by_SEXP.xls")
    print(sa2_sex)

if __name__ == "__main__":
    
    # conn = ReadDatabase()
    # df_3 = ReadTable(conn, '3_QTS_VEHICLES')
    # df_5 = ReadTable(conn, '5_QTS_TRIPS')
    # TimePerKilo(df_3, df_5)
    SA2Population()
    #ModeChoice(df_5)