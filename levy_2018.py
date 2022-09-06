# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 13:48:24 2022

@author: Yibo Wang
"""
import csv
import math
import pyodbc
import numpy as np
import pandas as pd
from objective_1 import CarbonEmission
from scipy.stats import levy
from scipy.stats import norm
from scipy.stats import skewnorm
from sklearn import preprocessing
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.special as sp
from sklearn.model_selection import train_test_split
from scipy.ndimage import gaussian_filter

class LevyFitting(object):
    
    def __init__(self):
        
        self.car_2017 = 154.37
        self.car_2018 = 153.69
        self.car_2019 = 152.43
        self.car_2020 = 150.78
        self.bus_2017 = 81.91
        self.bus_2018 = 81.57
        self.bus_2019 = 97.03
        self.bus_2020 = 121.97
        
        # Database location
        conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
                    r'DBQ=data\Travel Survey\2018-19.accdb;')
        
        conn = pyodbc.connect(conn_str)
        
        cursor = conn.cursor()
        for i in cursor.tables(tableType='TABLE'):
            print(i.table_name)
            
        self.df_1 = pd.read_sql(f'select * from 1_QTS_HOUSEHOLDS', conn)
        self.df_3 = pd.read_sql(f'select * from 3_QTS_VEHICLES', conn)
        self.df_5 = pd.read_sql(f'select * from 5_QTS_TRIPS', conn)
        
        self.font = {'size': 10}
        
        self.mode_id = CarbonEmission.ModeChoice(self)
        
        return None
    

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
                carbon_emi.append(cum_dist[i] * self.car_2018)
                car_list.append(cum_dist[i] * self.car_2018)
            elif mode_id[i] == 3:
                carbon_emi.append(cum_dist[i] * self.bus_2018)
                bus_list.append(cum_dist[i] * self.bus_2018)
            else:
                carbon_emi.append(0)
                zero_cnt += 1
        
        non_zero = len(mode_id) - zero_cnt
        #carbon_emi = np.round(np.array(carbon_emi), 2)
        
        hist_emi = carbon_emi
        
        for index, value in enumerate(hist_emi):
            if value > 15000:
                hist_emi[index] = 15000
                
        d = 10
        min_bound = int(min(hist_emi))
        max_bound = int(max(hist_emi))
        num_bins = (max_bound - min_bound) // d
                
        ''' Levy distribution '''
        x = range(15000)
        y = car_list + bus_list
        para = levy.fit(y)
        
        gau = skewnorm.pdf(x, a=7, loc=300, scale=4000)

        
        # for i in range(len(gau)):
        #     p[i] *= gau[i]
        # sum_p = sum(p)
        # for i in range(len(gau)):
        #     p[i] = p[i] / sum_p * 0.8428814301079163
        self.hist_emi = hist_emi
        self.num_bins = num_bins
        
        # plt.axis([0, 15000, 0, 800])
        self.n, bin_edges, _ = plt.hist(hist_emi, num_bins)
        self.p = levy.pdf(x, *para) * sum(self.n) * d
        
        # plt.plot(x, 933080*p, 'k', linewidth=2, c='red')
        # plt.title('Carbon emission of each trip by motor vehicle', self.font)
        # plt.xlabel('Carbon emissions (g)', self.font)
        # plt.ylabel('Number of trips', self.font)
        # plt.show()
        
        ''' Plot the skew normal weight '''
        # gau = skewnorm.pdf(x, a=8, loc=120, scale=3200)
        # plt.axis([0, 15000, 0, 3.5e-4])
        # plt.plot(x, gau, 'k', linewidth=2, c='red')
        # plt.show()
        
        ''' Scatter the weight point '''
        weight = []
        bin_middles = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        estimate = levy.pdf(bin_middles, *para)
        self.n[0] = 0
        self.n[-1] = 0
        for i in range(len(self.n)):
            weight.append(self.n[i] / (estimate[i] * non_zero))
        
        weight = preprocessing.normalize([weight])[0]
        
        plt.axis([0, 15000, 0, 0.06])
        plt.scatter(bin_middles, weight, s=10, c='red')
        plt.show()
        
        return bin_middles, weight, para
    
    
    def SkewNorm(self, x, sigma, mu, alpha, c, a):
        
        normpdf = (1/(sigma*np.sqrt(2*math.pi)))*np.exp(-(np.power((x-mu),2)/(2*np.power(sigma,2))))
        normcdf = (0.5*(1+sp.erf((alpha*((x-mu)/sigma))/(np.sqrt(2)))))
        
        return 2 * a * normpdf * normcdf + c
    
    
    def WeightFitting(self, bin_middles, weight):
        
        for i in range(10):
            weight = gaussian_filter(weight, sigma=1)
        
        X_train, X_test, y_train, y_test = train_test_split(bin_middles, weight, test_size=0.2, random_state=0)
        popt, pcov = curve_fit(self.SkewNorm, X_train, y_train, p0=(4000, 300 ,7,0,0))
        print(popt)
        
        y_train_pred = self.SkewNorm(X_train,*popt)
        
        plt.axis([0, 10000, 0, 0.06])
        plt.scatter(X_train, y_train, s=5, c='blue', label='train')
        plt.scatter(X_train, y_train_pred, s=5, c='red', label='model')
        plt.grid()
        plt.legend()
        plt.xlabel('Carbon emissions (g)', self.font)
        plt.ylabel('Weight', self.font)
        plt.show()
        
        return popt
    
    
    def SkewNormalLevy(self, para, popt):
        
        print(para)
        x = range(15000)
        y_levy = levy.pdf(x, *para)
        y_skew = self.SkewNorm(x,*popt)
        y = []
        
        for i in range(15000):
            y.append(y_levy[i] * y_skew[i])
        sum_skew = sum(y_skew)
        sum_levy = sum(y_levy)
        sum_n = sum(self.n)
        
        for i in range(15000):
            y[i] = y[i] * sum_skew / sum_levy * sum_n
        
        
        plt.axis([0, 10000, 0, 320])
        plt.hist(self.hist_emi, self.num_bins, color='blue')
        plt.plot(x, y, 'k', linewidth=2, c='red', label='Skew-Levy')
        plt.plot(x, self.p, 'k', linewidth=2, c='black', label='Levy')
        plt.legend()
        # plt.title('Carbon emission of each trip by motor vehicle', self.font)
        plt.xlabel('Carbon emissions (g)', self.font)
        plt.ylabel('Number of trips', self.font)
        plt.show()
        
        return 
    
    
    def ProfileLikelihood(self, bin_middles, weight):
        
        ''' Fitting skew normal distribution '''
        weight = (np.array(weight) * 10).astype(int)
        weight[0] = 0
        weight[-1] = 0
        data = []
        
        for i in range(len(bin_middles)):
            for j in range(weight[i]):
                data.append(bin_middles[i])
               
        para_sn = skewnorm.fit(weight, f0=10, floc=120)
        f_skew = skewnorm.pdf(bin_middles, *para_sn)
        
        print(para_sn)
        print(f_skew)
        
        plt.axis([0, 15000, 0, 2e-4])
        plt.plot(bin_middles, f_skew, 'k', linewidth=2, c='red')
        plt.show()
        
        # print(data)
        # print(len(data))
        # print(weight)
        
        
        return None
    
    
    def GaussianKernal(self, x):
        
        sigma = 2500
        mean = 1700
        e = math.e
        pi = math.pi
        
        # gaussian = (1/(sigma*(math.sqrt(2*pi)))) * (e^(-0.5*((x-mean)/sigma)^2))
        
        y = []
        for i in x:
            result = ((1/(sigma*(math.sqrt(2*pi)))) * (e**(-0.5*((x[i]-mean)/sigma)**2)))
            y.append(result)
        
        return y
    

    
    

if __name__ == "__main__":
    
    levy_fitting = LevyFitting()
    bin_middles, weight, para = levy_fitting.TripEmission()
    # levy_fitting.ProfileLikelihood(bin_middles, weight)
    popt = levy_fitting.WeightFitting(bin_middles, weight)
    levy_fitting.SkewNormalLevy(para, popt)
    
    # with open('test.csv', 'w') as f:
    #     # create the csv writer
    #     writer = csv.writer(f)
    #     # write a row to the csv file
    #     writer.writerow(weight)
    #     writer.writerow(bin_middles)
    