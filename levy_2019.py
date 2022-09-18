# -*- coding: utf-8 -*-
"""
Created on Wed Aug 31 13:48:24 2022

@author: Yibo Wang
"""

import math
import pyodbc
import collections
import numpy as np
import pandas as pd
from objective_1 import CarbonEmission
from scipy.stats import levy
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.special as sp
from sklearn.model_selection import train_test_split
from scipy.stats import chi2
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
        
        self.car_ave = 152.3
        self.bus_ave = 100.2
        
        # Database location
        conn_str = (r'DRIVER={Microsoft Access Driver (*.mdb, *.accdb)};'
                    r'DBQ=data\Travel Survey\2019.accdb;')
        
        conn = pyodbc.connect(conn_str)
                    
        self.df_1 = pd.read_sql(f'select * from 1_QTS_HOUSEHOLDS', conn)
        self.df_3 = pd.read_sql(f'select * from 3_QTS_VEHICLES', conn)
        self.df_5 = pd.read_sql(f'select * from 5_QTS_TRIPS', conn)
        
        self.font = {'size': 10}
        
        self.mode_id = CarbonEmission.ModeChoice(self)
        
        return None
    
    
    def SkewNorm(self, x, sigma, mu, alpha, c, a):
        
        normpdf = (1/(sigma*np.sqrt(2*math.pi)))*np.exp(-(np.power((x-mu),2)/(2*np.power(sigma,2))))
        normcdf = (0.5*(1+sp.erf((alpha*((x-mu)/sigma))/(np.sqrt(2)))))
        
        return 2 * a * normpdf * normcdf + c
    
    
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
        max_cnt = 0
        min_bound = 0
        max_bound = 10000
        d = 10
                
        for i in range(len(mode_id)):
            if mode_id[i] == 0 or mode_id[i] == 2:
                carbon_emi.append(cum_dist[i] * self.car_2019)
                car_list.append(cum_dist[i] * self.car_2019)
            elif mode_id[i] == 3:
                carbon_emi.append(cum_dist[i] * self.bus_2019)
                bus_list.append(cum_dist[i] * self.bus_2019)
            else:
                carbon_emi.append(0)
                zero_cnt += 1
        
        hist_emi = carbon_emi
        
        for index, value in enumerate(hist_emi):
            if value > max_bound:
                max_cnt += 1
                hist_emi[index] = max_bound
        
        self.non_zero = len(mode_id) - zero_cnt - max_cnt
                
        num_bins = (max_bound - min_bound) // d
        
        self.hist_emi = hist_emi
        self.num_bins = num_bins
        
        plt.axis([0, max_bound, 0, 250])
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
        print(popt)
        
        y_train_pred = self.Levy(bin_middles, *popt)
        
        plt.axis([0, 10000, 0, 0.01])
        plt.scatter(bin_middles, weight, s=5, c='blue', label='train')
        plt.scatter(bin_middles, y_train_pred, s=5, c='red', label='model')
        plt.grid()
        plt.legend()
        plt.xlabel('Carbon emissions (g)', self.font)
        plt.ylabel('Weight', self.font)
        plt.show()
        
        ''' Plot the Levy curve '''
        x = range(10000)
        y = self.Levy(x, *popt)
        y_all = self.Levy(bin_middles, *popt)
        print(sum(y), sum(y_all))
        y = y / sum(y_all) * non_zero
        # y_train_pred = y_train_pred / sum(y_train_pred) * self.non_zero
        
        plt.axis([0, 10000, 0, 250])
        plt.plot(x, y, 'k', linewidth=2, c='red', label='Levy')
        plt.hist(emission, self.num_bins, color='blue')
        plt.xlabel('Carbon emissions (g)', self.font)
        plt.ylabel('Number of trips', self.font)
        plt.show()
        
        return popt
    
    
    def Weight(self, popt_levy):
        
        bin_middles = self.bin_middles
        n = np.array(self.n)
        non_zero = self.non_zero
        estimate = np.array(self.Levy(bin_middles, *popt_levy))
        
        ''' Compute the Weight '''
        weight = n / (estimate * non_zero)
        weight /= sum(weight)
        weight[-1] = 0
        weight[0] = 0.0002
        weight[weight > 0.0002] = 0.0002
        
        
        
        ''' Gaussian filter '''
        for i in range(20):
            weight = gaussian_filter(weight, sigma=1)
        
        ''' Fit and predict '''
        popt_norm, pcov = curve_fit(self.Norm, bin_middles, weight, p0=(400, 100 ,7,0,0))
        print(popt_norm)
        y_train_pred = self.Norm(bin_middles,*popt_norm)
        
        ''' Scatter the Weight and plot the prediction '''
        plt.axis([0, 10000, 0, 0.0002])
        plt.scatter(bin_middles, weight, s=5, c='blue', label='train')
        plt.scatter(bin_middles, y_train_pred, s=5, c='red', label='model')
        plt.grid()
        plt.legend()
        plt.show()
        
        return popt_norm
    
    
    def ChiSquare(self, popt_levy, popt_norm):
        
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
                
        plt.axis([0, 10000, 0, 250])
        plt.hist(hist_emi, self.num_bins, color='blue')
        plt.plot(x, y, 'k', linewidth=2, c='red', label='Skew-Levy')
        plt.plot(x, y_levy, 'k', linewidth=2, c='black', label='Levy')
        plt.legend()
        plt.xlabel('Carbon emissions (g)', self.font)
        plt.ylabel('Number of trips', self.font)
        plt.show()
        
        ''' Compute the Chi-square '''
        A = np.array(self.n)
        E_levy = np.array(self.Levy(bin_middles, *popt_levy))
        E_norm = np.array(self.Norm(bin_middles, *popt_norm))
        E = E_levy * E_norm
        E = E / np.sum(E) * non_zero
        div = np.divide(np.square(A - E), E, out=np.zeros_like(E), where=E!=0)
        X_2 = np.sum(div)
        
        print(X_2)
        
        return E_norm    
    
    
    def TravelPurpose(self, ):
        
        carbon_emi = self.hist_emi
                        
        purpose = np.array(self.df_5)[:, 25]
        print(collections.Counter(purpose))
        
        commute = []
        shopping = []
        pickup = []
        recreation = []
        education = []
        business = []
        accompany = []
        work = []
        social = []
        deliver = []
        other = []
        
        for index, value in enumerate(purpose):
            if value == 'Direct Work Commute':
                commute.append(carbon_emi[index])
            elif value == 'Shopping':
                shopping.append(carbon_emi[index])
            elif value == 'Pickup/Dropoff Someone':
                pickup.append(carbon_emi[index])
            elif value == 'Recreation':
                recreation.append(carbon_emi[index])
            elif value == 'Education':
                education.append(carbon_emi[index])
            elif value == 'Personal Business':
                business.append(carbon_emi[index])
            elif value == 'Accompany Someone':
                accompany.append(carbon_emi[index])
            elif value == 'Work Related':
                work.append(carbon_emi[index])
            elif value == 'Social':
                social.append(carbon_emi[index])
            elif value == 'Pickup/Deliver Something':
                deliver.append(carbon_emi[index])
            else:
                other.append(carbon_emi[index])
        
        dict_purp = {'commute':    commute,
                     'shopping':   shopping,
                     'pickup':     pickup,
                     'recreation': recreation,
                     'education':  education, 
                     'business':   business,
                     'work':       work
                     }
        
        return dict_purp
    
    
    def PurposeAnalysis(self, dict_purp):
        
        num_bins = self.num_bins
        dist_estm = {}
                
        for key in dict_purp:
            print(key)
        
            plt.axis([0, 10000, 0, 250])
            n, _, _ = plt.hist(dict_purp[key], num_bins)
            plt.show()
        
            non_zero = len(dict_purp[key]) - n[0] - n[-1]
            n[0] = 0
            n[-1] = 0
            
            dist_estm[key] = self.LevyFitting(n, non_zero, dict_purp[key])
        
        x = range(10000)
        plt.axis([0, 10000, 0, 200])
        plt.plot(x, dist_estm['commute'], 'k', linewidth=2, c='red', label='Commute')
        plt.plot(x, dist_estm['shopping'], 'k', linewidth=2, c='blue', label='Shopping')
        plt.plot(x, dist_estm['pickup'], 'k', linewidth=2, c='green', label='Pickup')
        plt.plot(x, dist_estm['recreation'], 'k', linewidth=2, c='yellow', label='Recreation')
        plt.plot(x, dist_estm['education'], 'k', linewidth=2, c='black', label='Education')
        plt.plot(x, dist_estm['business'], 'k', linewidth=2, c='orange', label='Personal business')
        plt.plot(x, dist_estm['work'], 'k', linewidth=2, c='purple', label='Work related')
        plt.legend()
        plt.xlabel('Carbon emissions (g)', self.font)
        plt.ylabel('Number of trips', self.font)
        plt.show()
        
        return None
        
    
    
    
if __name__ == "__main__":
    
    levy_fitting = LevyFitting()
    weight, non_zero, emission = levy_fitting.TripEmission()
    
    popt_levy = levy_fitting.LevyFitting(weight, non_zero, emission)
    popt_norm = levy_fitting.Weight(popt_levy)
    y_norm = levy_fitting.ChiSquare(popt_levy, popt_norm)
    
    print(chi2.ppf(0.5,1000))
    
    # for i in range(10):
    #     weight /= y_norm
    #     popt_levy = levy_fitting.LevyFitting(weight, non_zero, emission)
    #     popt_norm = levy_fitting.Weight(popt_levy)
    #     y_norm = levy_fitting.ChiSquare(popt_levy, popt_norm)
    
    # dict_purp = levy_fitting.TravelPurpose()
    # levy_fitting.PurposeAnalysis(dict_purp)