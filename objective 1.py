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
    
    # Table name
    df = pd.read_sql('select * from 5_QTS_TRIPS', conn)
    
    return df


def ModeChoice(data_frame):
    
    mainmode = np.array(data_frame)[:, 12]
    
    
    
    data_count = collections.Counter(mainmode)
    
    print(data_count)
    
    
def TimePerKilo(data_frame):
    
    duration = np.array(data_frame)[:, 23]
    cumdist = np.array(data_frame)[:, 24]
    
    time_per_kilo = np.divide(duration, cumdist, out=np.zeros_like(duration), where=cumdist!=0)
    

if __name__ == "__main__":
    
    df = ReadDatabase()
    ModeChoice(df)
    
