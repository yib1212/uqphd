# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 17:42:27 2022

@author: Yibo Wang
"""

import numpy as np
import pandas as pd
import pyodbc
import collections

class BasicStats(object):
    
    def __init__(self):
        
        ''' Read the MS access database and return the dataframe using pandas. '''
        
        # Database location
        conn_str_18_21 = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
                    r'DBQ=data\Travel Survey\2018-21_pooled_seq_qts_erv1.0.accdb;')
        conn_str_17_18 = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
                    r'DBQ=data\Travel Survey\2017-18qtserv14final.accdb;')
        conn_18_21 = pyodbc.connect(conn_str_18_21)
        conn_17_18 = pyodbc.connect(conn_str_17_18)
        
        cursor_18_21 = conn_18_21.cursor()
        cursor_17_18 = conn_17_18.cursor()
        for i_18_21 in cursor_18_21.tables(tableType='TABLE'):
            print(i_18_21.table_name)
        for i_17_18 in cursor_17_18.tables(tableType='TABLE'):
            print(i_17_18.table_name)
            
        self.df_18_21 = pd.read_sql(f'select * from 1_QTS_HOUSEHOLDS', conn_18_21)
        self.df_17_18 = pd.read_sql(f'select * from 1_QTS_HOUSEHOLDS', conn_17_18)
        
        return None


    def HouseholdNumber(self):
        
        hhid_18_21 = np.array(self.df_18_21)[:, 0]
        hhid_17_18 = np.array(self.df_17_18)[:, 0]
        
        hhid_num = len(hhid_18_21)
        
        for hhid in hhid_17_18:
            if hhid in hhid_17_18:
                hhid_num += 1
        
        return hhid_num




if __name__ == "__main__":
    
    stats = BasicStats()
    hhid_num = stats.HouseholdNumber()