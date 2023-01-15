# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 17:08:03 2022

@author: Yibo Wang
"""

import math
import pyodbc
import numpy as np
import pandas as pd
from objective_1 import CarbonEmission
from scipy.stats import levy
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
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
        # 2017: car 285.53 bus 21.73 rail 18.02
        # 2018: car 284.88 bus 22.08 rail 18.84
        # 2019: car 265.63 bus 16.81 rail 15.05
        # 2020: car 269.48 bus 12.97 rail 09.33
        
        # Table 11.4 11.5 Unit: gigagrams CO2
        # 2017: car 44078 bus 1780 rail 3655
        # 2018: car 43783 bus 1801 rail 3593
        # 2019: car 40490 bus 1631 rail 3545
        # 2020: car 40633 bus 1582 rail 3542
        
        # Carbon Emission pkm
        # 2017: car 154.37 bus 81.91   rail 202.83
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
                    
        self.df_1 = pd.read_sql(f'select * from 1_QTS_HOUSEHOLDS', conn)
        self.df_3 = pd.read_sql(f'select * from 3_QTS_VEHICLES', conn)
        self.df_5 = pd.read_sql(f'select * from 5_QTS_TRIPS', conn)
        
        self.font = {'size': 10}
        
        self.sa2_array = CarbonEmission.SA2Info(self)
        self.mode_id = CarbonEmission.ModeChoice(self)
        self.sa2_array, _, _ = CarbonEmission.TripNumber(self)
        
        return None
    
    
    def Norm(self, x, sigma, mu, alpha, c, a):
        
        normpdf = (1/(sigma*np.sqrt(2*math.pi)))*np.exp(-(np.power((x-mu),2)/(2*np.power(sigma,2))))
        
        return a * normpdf + c
    
    
    def Levy(self, x, sigma, mu, a, c):
        
        # levy_fit = np.sqrt(sigma/(2*math.pi)) * (np.exp(-(sigma/2*(x-mu)))) / (np.power((x-mu),2))
        levy_fit = levy.pdf(x, mu, sigma)
        
        return a * levy_fit
    
    
    def GaussianKernal(self, x):
        
        sigma = 2500
        mean = 1700
        e = math.e
        pi = math.pi
        
        y = []
        for i in x:
            result = ((1/(sigma*(math.sqrt(2*pi)))) * (e**(-0.5*((x[i]-mean)/sigma)**2)))
            y.append(result)
        
        return y
    
        
    def TripEmission(self):
        
        ''' Compute the carbon emission for each trip. '''
        
        mode_id = self.mode_id
        cum_dist = np.array(self.df_5)[:, 24]
        carbon_emi = []
        car_list = []
        bus_list = []
        zero_cnt = 0
        
        for i in range(len(mode_id)):
            if mode_id[i] == 0 or mode_id[i] == 2:
                carbon_emi.append(cum_dist[i] * self.car_ave)
                car_list.append(cum_dist[i] * self.car_ave)
            elif mode_id[i] == 3:
                carbon_emi.append(cum_dist[i] * self.bus_ave)
                bus_list.append(cum_dist[i] * self.bus_ave)
            else:
                carbon_emi.append(0)
                zero_cnt += 1
        
        print(len(mode_id) - zero_cnt)
        #carbon_emi = np.round(np.array(carbon_emi), 2)
        
        hist_emi = carbon_emi
        
        for index, value in enumerate(hist_emi):
            if value > 15000:
                hist_emi[index] = 15000
        
        d = 10
        min_bound = int(min(hist_emi))
        max_bound = int(max(hist_emi))
        num_bins = (max_bound - min_bound) // d
        
        plt.axis([0, 15000, 0, 800])
        self.n, self.bins, _ = plt.hist(hist_emi, num_bins)
        plt.title('Carbon emission of each trip by motor vehicle', self.font)
        plt.xlabel('Carbon emissions (g)', self.font)
        plt.ylabel('Number of trips', self.font)
                
        plt.show()
        
        self.hist_emi = hist_emi
        self.num_bins = num_bins
        self.car_list = car_list
        self.car_and_bus = car_list + bus_list
        
        return carbon_emi, car_list
    
    
    def RegionTime(self, carbon_emi):
        
        ''' Explore the relationship between SA2 region, average emission, and travel time. '''
        
        sa2_main = self.sa2_main
        sa2_array = self.sa2_array
        trav_time = np.array(self.df_5)[:, 23]
        
        plt.axis([0, 180, 0, 15000])
        plt.title('Carbon emission and travel time of each trip', self.font)
        plt.xlabel('Travel time (min)', self.font)
        plt.ylabel('Carbon emission (g)', self.font)
        plt.scatter(trav_time, carbon_emi, s=1, c='red')
        plt.show()
        
        time_sum = np.zeros(len(sa2_main))
        emi_sum = np.zeros(len(sa2_main))
        num_cnt = np.zeros(len(sa2_main), dtype = int)
        
        for i in range(len(sa2_array)):
            idx = np.argwhere(sa2_main == sa2_array[i])
            time_sum[idx] += trav_time[i]
            emi_sum[idx] += carbon_emi[i]
            num_cnt[idx] += 1
        
        time_ave = np.divide(time_sum, num_cnt, where=num_cnt!=0)
        emi_ave = np.divide(emi_sum, num_cnt, where=num_cnt!=0)
        
        plt.axis([0, 40, 0, 5000])
        plt.title('Average carbon emission and travel time of different SA2 regions', self.font)
        plt.xlabel('Average travel time (min)', self.font)
        plt.ylabel('Average carbon emission (g)', self.font)
        plt.scatter(time_ave, emi_ave, s=10, c='red')
        plt.show()
        
        # 66 SA2 0, 316-66=250
        self.emi_ave = emi_ave
        self.time_ave = time_ave
        
        return time_ave, emi_ave, num_cnt
    
    
    def FiveTiers(self, c, time_ave, emi_ave, num_cnt):
        
        # 0, 1015, 1255, 1535, 1750
        id_0 = []
        id_1 = []
        id_2 = []
        id_3 = []
        id_4 = []
        id_5 = []
        
        t1 = []
        t2 = []
        t3 = []
        t4 = []
        t5 = []
        
        e1 = []
        e2 = []
        e3 = []
        e4 = []
        e5 = []
        
        sa2_main = self.sa2_main
        sa2_array = self.sa2_array
        
        for i in range(len(emi_ave)):
            if emi_ave[i] < 1:
                id_0.append(sa2_main[i])
            elif emi_ave[i] < 1015:
                id_1.append(sa2_main[i])
                t1.append(time_ave[i])
                # if time_ave[i] == 12.225563909774436: # 12.225563909774436, 29.666666666666668
                if time_ave[i] ==24.046204620462046: # 24.146853146853147
                    print(sa2_main[i])
                e1.append(emi_ave[i])
            elif emi_ave[i] < 1255:
                id_2.append(sa2_main[i])
                t2.append(time_ave[i])
                e2.append(emi_ave[i])
            elif emi_ave[i] < 1535:
                id_3.append(sa2_main[i])
                t3.append(time_ave[i])
                e3.append(emi_ave[i])
            elif emi_ave[i] < 1750:
                id_4.append(sa2_main[i])
                t4.append(time_ave[i])
                e4.append(emi_ave[i])
            else:
                id_5.append(sa2_main[i])
                t5.append(time_ave[i])
                e5.append(emi_ave[i])
        
        data_1 = []
        data_2 = []
        data_3 = []
        data_4 = []
        data_5 = []
        
        for j in range(len(sa2_array)):
            if sa2_array[j] in id_1:
                data_1.append(carbon_emi[j])
            elif sa2_array[j] in id_2:
                data_2.append(carbon_emi[j])
            elif sa2_array[j] in id_3:
                data_3.append(carbon_emi[j])
            elif sa2_array[j] in id_4:
                data_4.append(carbon_emi[j])
            elif sa2_array[j] in id_5:
                data_5.append(carbon_emi[j])
                
        # print(max(t1))    
        # for i in range(len(t1)):
        #     if t1[i] == 29.666666666666668:
        #         print(i)
        
        plt.axis([0, 40, 0, 5000])
        # plt.title('Average carbon emission and travel time of different SA2 regions', self.font)
        plt.xlabel('Average travel time (min)', self.font)
        plt.ylabel('Average carbon emission (g)', self.font)
        plt.scatter(t1, e1, marker='.', s=10, color=(0.00, 0.00, 1.00), label='0-1014 g')
        plt.scatter(t2, e2, marker='.', s=10, color=(0.25, 0.00, 0.75), label='1015-1254 g')
        plt.scatter(t3, e3, marker='.', s=10, color=(0.50, 0.00, 0.50), label='1255-1534 g')
        plt.scatter(t4, e4, marker='.', s=10, color=(0.75, 0.00, 0.25), label='1535-1749 g')
        plt.scatter(t5, e5, marker='.', s=10, color=(1.00, 0.00, 0.00), label='Over 1750g')
        plt.legend()
        plt.show()
        
        print(id_1, id_2, id_3, id_4, id_5)
        
        return id_0, data_1, data_2, data_3, data_4, data_5
    

    def FiveTiersEmission(self, data):
        
        ''' Compute the carbon emission for each trip. '''
        
        zero_cnt = 0
        max_cnt = 0
        min_bound = 0
        max_bound = 10000
        d = 10
        
        hist_emi = data
        
        for index, value in enumerate(hist_emi):
            if value > max_bound:
                max_cnt += 1
                hist_emi[index] = max_bound
            elif value == 0:
                zero_cnt += 1
        
        self.non_zero = len(data) - zero_cnt - max_cnt
                
        num_bins = (max_bound - min_bound) // d
        
        self.hist_emi = hist_emi
        self.num_bins = num_bins
        
        plt.axis([0, max_bound, 0, 180])
        self.n, bin_edges, _ = plt.hist(hist_emi, num_bins)
        self.n[0] = 0
        self.n[-1] = 0
        self.bin_middles = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        plt.title('Carbon emission of each trip by motor vehicle', self.font)
        plt.xlabel('Carbon emissions (g)', self.font)
        plt.ylabel('Number of trips', self.font)
        plt.show()
                        
        return self.n, self.non_zero, self.hist_emi
    
    
    def LevyFitting(self, weight, non_zero, emission):
        
        ''' Levy Fitting '''
        bin_middles = self.bin_middles
        weight = np.array(weight) / sum(weight)
        
        popt, pcov = curve_fit(self.Levy, bin_middles, weight, p0=( 600, 300, 100, 0))
        print('Levy', popt)
        
        y_train_pred = self.Levy(bin_middles, *popt)
        
        plt.axis([0, 10000, 0, 0.01])
        plt.scatter(bin_middles, weight, s=5, c='blue', label='train')
        plt.scatter(bin_middles, y_train_pred, s=5, c='red', label='model')
        plt.grid()
        plt.legend()
        plt.xlabel('Carbon emissions (g)', self.font)
        plt.ylabel('Probability density', self.font)
        plt.show()
        
        ''' Plot the Levy curve '''
        x = range(10000)
        y = self.Levy(x, *popt)
        y_all = self.Levy(bin_middles, *popt)
        y_all[y_all < 0] = 0
        y = y / sum(y_all) * non_zero
        # y_train_pred = y_train_pred / sum(y_train_pred) * self.non_zero
        
        plt.axis([0, 10000, 0, 900])
        plt.plot(x, y, 'k', linewidth=2, c='red', label='Levy')
        plt.hist(emission, self.num_bins, color='blue')
        plt.xlabel('Carbon emissions (g)', self.font)
        plt.ylabel('Number of trips', self.font)
        plt.show()
        
        return popt, y
    
    
    def Weight(self, n, non_zero, popt_levy, weight_max):
        
        bin_middles = self.bin_middles
        estimate = np.array(self.Levy(bin_middles, *popt_levy))
        estimate[estimate < 1e-7] = 0.0
        
        ''' Compute the Weight '''
        weight = n / estimate
        weight[np.isnan(weight)] = 0
        weight[np.isinf(weight)] = 0
        weight /= sum(weight)
        weight[-1] = 0
        weight[0] = weight_max
        weight[weight > weight_max] = weight_max
        
        
        ''' Gaussian filter '''
        for i in range(20):
            weight = gaussian_filter(weight, sigma=1)
        
        ''' Fit and predict '''
        popt_norm, pcov = curve_fit(self.Norm, bin_middles, weight, p0=(4000, 0, 0, 0, 0))
        print('Norm', popt_norm)
        y_train_pred = self.Norm(bin_middles,*popt_norm)
        
        ''' Scatter the Weight and plot the prediction '''
        plt.axis([0, 10000, 0, 1.5*weight_max])
        plt.scatter(bin_middles, weight, s=5, c='blue', label='train')
        plt.scatter(bin_middles, y_train_pred, s=5, c='red', label='model')
        plt.xlabel('Carbon emissions (g)', self.font)
        plt.ylabel('Weight', self.font)
        plt.grid()
        plt.legend()
        plt.show()
        
        return popt_norm
    
    
    def Plot(self, popt_levy, popt_norm):
        
        non_zero = self.non_zero
        bin_middles = self.bin_middles
        hist_emi =self.hist_emi
                
        ''' Plot the Levy and Skew-levy curve '''
        x = range(10000)
        y_levy = np.array(self.Levy(x, *popt_levy))
        y_norm = np.array(self.Norm(x, *popt_norm))
        y = y_levy * y_norm
        y_levy = y_levy / np.sum(y_levy) * non_zero * 10
        y = y / np.sum(y) * np.sum(y_levy)
                
        plt.axis([0, 10000, 0, 180])
        plt.hist(hist_emi, self.num_bins, color='blue')
        plt.plot(x, y, 'k', linewidth=2, c='red', label='Norm-Levy')
        plt.plot(x, y_levy, 'k', linewidth=2, c='black', label='Levy')
        plt.legend()
        plt.xlabel('Carbon emissions (g)', self.font)
        plt.ylabel('Number of trips', self.font)
        plt.show()
        
        return y_levy
    
    
    def LocationResults(self):
        
        y_levy = np.zeros((5, 1000))
        y_norm = np.zeros((5, 1000))
        y = np.zeros((5, 1000))
        x = self.bin_middles
        
        mean = np.zeros(5)
        median = np.zeros(5)
        var = np.zeros(5)
        std = np.zeros(5)
        skew1 = np.zeros(5)
        skew2 = np.zeros(5)
        
        popt_levy = [[602.89515979,   2.2318219,   15.1546452,    0.        ],
                     [749.46859066, -19.58618845,  14.91503241,   0.        ],
                     [745.65785775,  -6.38087173,  14.35999573,   0.        ],
                     [907.92054349,  -8.29285683,  14.20607632,   0.        ],
                     [1288.0559728,  -93.85030177, 14.70759107,   0.        ]]
        popt_norm = [[ 2.63031549e+03, -7.59316270e+02, 0.00000000e+00, 7.90928024e-05, 1.41643429e+01],
                     [ 2.25252221e+03,  4.72157706e+02, 0.00000000e+00, 2.92497623e-04, 1.14909181e+01],
                     [ 4.02165086e+03, -5.43081294e+01, 0.00000000e+00, 1.34036546e-04, 1.32047017e+01],
                     [ 2.85036894e+03,  1.86329002e+03, 0.00000000e+00, 1.17562255e-04, 5.34086622e+00],
                     [ 3.67379162e+03,  1.89542882e+03, 0.00000000e+00, 2.46610418e-04, 1.08381986e+01]]
        non_zero = [7414, 11748, 21972, 24442, 18162]
        mode = [203, 230, 242, 294, 335]
        
        for i in range(5):
            popt_l = popt_levy[i]
            popt_n = popt_norm[i]
            y_levy[i] = np.array(self.Levy(x, *popt_l))
            y_norm[i] = np.array(self.Norm(x, *popt_n))
            y[i] = y_levy[i] * y_norm[i]
            y[i] /= np.sum(y[i])
            mean[i] = self.Mean(y[i])
            median[i] = self.Median(y[i])
            var[i] = self.Variance(y[i], mean[i])
            std[i] = np.sqrt(var[i])
            skew1[i], skew2[i] = self.Skew(mean[i], median[i], std[i], mode[i])
        
        plt.axis([0, 6000, 0, 0.015])

        plt.plot(x, y[0], 'k', linewidth=2, color=(0, 0, 1), label='0-1014 g')
        plt.plot(x, y[1], 'k', linewidth=2, color=(0.25, 0, 0.75), label='1015-1254 g')
        plt.plot(x, y[2], 'k', linewidth=2, color=(0.5, 0, 0.5), label='1255-1534 g')
        plt.plot(x, y[3], 'k', linewidth=2, color=(0.75, 0, 0.25), label='1535-1749 g')
        plt.plot(x, y[4], 'k', linewidth=2, color=(1, 0, 0), label='Over 1750g')
        plt.legend()
        plt.xlabel('Carbon emissions (g)', self.font)
        plt.ylabel('PDF', self.font)
        plt.show()
        
        print(mean, median, var, std, skew1, skew2)
        
        return None
        
    
    def Mean(self, y):
        
        x = self.bin_middles
        sum_x = np.sum(x * y)
        mean_x = sum_x / np.sum(y)
        
        return mean_x
    
    
    def Median(self, y):
        
        x = self.bin_middles
        sum_y_med = np.sum(y) / 2
        median_x = 0
        sum_y = 0
        for i in range(len(x)):
            if sum_y < sum_y_med:
                sum_y += y[i]
                median_x = x[i]
        
        return median_x
        
    def Variance(self, y, mean):
        
        x = self.bin_middles
        sum_y = np.sum(y)
        sum_sqr = 0
        for i in range(len(x)):
            sum_sqr += y[i] * (x[i] - mean) ** 2
        var = sum_sqr / sum_y
        
        return var
    
    # Pearson’s mode/median skewness
    def Skew(self, mean, med, std, mode):
        
        skew1 = (mean - med) / std
        skew2 = 3 * (mean - med) / std
        
        return skew1, skew2
     
        
        
if __name__ == "__main__":
    
    emission = CarbonEachTrip()
    carbon_emi, car_list = emission.TripEmission()
    time_ave, emi_ave, num_cnt = emission.RegionTime(carbon_emi)
    d0, d1, d2, d3, d4, d5 = emission.FiveTiers(carbon_emi, time_ave, emi_ave, num_cnt)
    
    weight, non_zero, emi = emission.FiveTiersEmission(d1)
    popt_levy, _ = emission.LevyFitting(weight, non_zero, emi)
    popt_norm = emission.Weight(weight, non_zero, popt_levy, 0.0025) # 0.0025 0.0030 0.0018
    y_levy = emission.Plot(popt_levy, popt_norm)
    
    emission.LocationResults()
    