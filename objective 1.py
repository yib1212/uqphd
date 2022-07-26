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
    trip_num = collections.Counter(sa2_array)
    print(trip_num)
    
    #Counter({301011004: 2091, 316081549: 1917, 301011001: 1781, 316051435: 1517, 301011003: 1435, 301021012: 1324, 301011006: 1303, 316051438: 1290, 309071557: 1280, 311061332: 1250, 314011386: 1249, 316031428: 1149, 316071548: 1143, 301021013: 1117, 310041302: 1053, 310031287: 1051, 313041375: 1012, 310031293: 1007, 301021009: 977, 311041570: 941, 316051543: 932, 316021417: 894, 309061249: 892, 310041297: 884, 310031283: 881, 314011383: 853, 316011414: 852, 304021086: 819, 310041300: 804, 313051380: 784, 310041299: 754, 316051434: 751, 316031427: 751, 311051325: 745, 311051327: 743, 314021579: 740, 316031425: 729, 316051437: 727, 310021282: 679, 309011226: 673, 310031292: 661, 311061330: 660, 304011084: 657, 316021423: 654, 309031238: 654, 316031426: 640, 311051326: 637, 301021007: 619, 309091263: 601, 311021307: 595, 310031289: 586, 303011051: 584, 301031021: 584, 316011413: 579, 309071552: 575, 313011363: 560, 302021027: 546, 310041303: 546, 311061329: 542, 314021389: 536, 310041304: 535, 301021008: 534, 316011415: 527, 304041103: 523, 303061078: 518, 309021234: 516, 311061336: 512, 313041373: 512, 310031284: 511, 303021052: 509, 314021577: 508, 314031394: 501, 309071556: 498, 313021366: 492, 311031319: 490, 313041372: 472, 311051328: 472, 311031316: 463, 316061444: 457, 311031311: 453, 310031286: 450, 305041132: 446, 309011225: 446, 309061246: 445, 303041068: 444, 304031094: 443, 316011416: 431, 314011382: 428, 305041137: 425, 309061248: 422, 309031235: 411, 301021011: 402, 313051575: 400, 314031393: 378, 309071251: 377, 311061333: 376, 311051324: 375, 310031288: 369, 309081262: 366, 309031237: 366, 305031126: 363, 309081559: 362, 309021233: 359, 309061247: 357, 316061441: 357, 301011005: 354, 311061334: 352, 316021422: 350, 309071553: 348, 316061442: 345, 303041070: 342, 309021232: 338, 309051243: 336, 313051378: 329, 313021572: 323, 314011384: 322, 303021053: 322, 302021034: 322, 305011107: 320, 309051244: 317, 309021231: 314, 316061439: 314, 311021309: 303, 305011112: 303, 301031015: 303, 316051544: 299, 302011026: 298, 305031129: 297, 303041071: 296, 316071545: 292, 303031065: 291, 309081560: 289, 309031240: 285, 313051377: 284, 309031239: 282, 309061250: 278, 311021310: 275, 304031095: 271, 302041042: 270, 302011024: 270, 305031131: 268, 313021368: 268, 310011564: 265, 303031060: 263, 304011082: 260, 311041567: 259, 309091264: 257, 304011081: 257, 313041376: 255, 301031016: 254, 311021306: 253, 302021032: 249, 314021578: 241, 310041566: 238, 309011228: 235, 305011111: 232, 304031097: 230, 310011275: 229, 316021421: 225, 302031039: 220, 311031312: 219, 316021424: 213, 304021090: 212, 302041045: 211, 302031040: 206, 310011274: 206, 301031020: 199, 303051076: 196, 310041565: 195, 313051379: 193, 303031066: 193, 303021057: 191, 309101561: 191, 309011229: 187, 309091265: 182, 313041374: 181, 304021087: 180, 311031314: 179, 304021088: 178, 304031096: 178, 314031391: 177, 303031064: 177, 311061331: 176, 305021117: 176, 316071547: 171, 302021033: 170, 303051073: 170, 304031092: 170, 304031093: 162, 309091541: 160, 302031038: 159, 309071254: 158, 310031291: 157, 304011085: 156, 305041134: 155, 309071253: 153, 316061440: 150, 302011022: 150, 313021364: 149, 303061077: 147, 313021573: 147, 305011109: 143, 309011224: 142, 309091540: 142, 311041568: 142, 314031392: 138, 305031119: 135, 316021419: 134, 309101267: 133, 303011048: 132, 303021054: 130, 309101268: 126, 303031063: 126, 314011387: 126, 305031121: 124, 309011227: 123, 310031285: 123, 316021581: 120, 302041044: 115, 310011563: 114, 311041569: 113, 310031294: 113, 313021367: 113, 311041571: 113, 311031318: 112, 302021029: 112, 304041101: 110, 303021059: 109, 311031313: 108, 310011272: 108, 303061080: 107, 309071252: 104, 305031124: 103, 305031125: 103, 301031017: 103, 316071546: 102, 302031035: 102, 302021028: 98, 305021115: 98, 303021055: 97, 302041043: 97, 310011276: 97, 303011049: 96, 303021056: 95, 311061335: 92, 303041067: 92, 304021091: 90, 313021369: 90, 303051072: 87, 311051323: 85, 302041041: 84, 316021418: 84, 301021551: 81, 314021576: 81, 311031317: 80, 303031062: 78, 303041069: 78, 302011025: 77, 305011106: 73, 304011083: 71, 302011023: 71, 302041046: 71, 309071256: 70, 313011362: 67, 311021308: 67, 309041241: 61, 304041100: 60, 305021118: 58, 303061079: 58, 304041098: 58, 303031061: 56, 305021113: 55, 305011110: 49, 305031128: 30, 313031371: 26})
    
    return trip_num

    

if __name__ == "__main__":
    
    conn = ReadDatabase()
    df_1 = ReadTable(conn, '1_QTS_HOUSEHOLDS')
    df_3 = ReadTable(conn, '3_QTS_VEHICLES')
    df_5 = ReadTable(conn, '5_QTS_TRIPS')
    sa2_list = SA2Info()
    # TimePerKilo(df_3, df_5)
    num_trip = TripNumber(df_1, df_5, sa2_list)
    # pop = SA2Population()
    
    # ModeChoice(df_5)