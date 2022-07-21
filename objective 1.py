# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 03:35:31 2022

@author: Yibo Wang
"""

import pyodbc
import pandas as pd
import numpy as np

def read_database():
    
    conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
                r'DBQ=data\Travel Survey\2018-21_pooled_seq_qts_erv1.0.accdb;')
    
    conn = pyodbc.connect(conn_str)
    
    cursor = conn.cursor()
    
    for i in cursor.tables(tableType='TABLE'):
        print(i.table_name)
    
    df = pd.read_sql('select * from 5_QTS_TRIPS', conn)
    
    mainmode = np.array(df)[:, 12]
    duration = np.array(df)[:, 23]
    cumdist = np.array(df)[:, 24]
    
    time_per_kilo = np.divide(duration, cumdist, out=np.zeros_like(duration), where=cumdist!=0)
    data_count = collections.Counter(mainmode)
    
    print(data_count)

