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
from matplotlib import cm
from scipy.optimize import curve_fit
import scipy.special as sp
from sklearn.model_selection import train_test_split
from scipy.stats import chi2
from scipy.ndimage import gaussian_filter
from matplotlib.collections import PolyCollection


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
                    r'DBQ=data\Travel Survey\2018-21_pooled_seq_qts_erv1.0.accdb;')
        
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
                carbon_emi.append(cum_dist[i] * self.car_ave)
                car_list.append(cum_dist[i] * self.car_ave)
            elif mode_id[i] == 3:
                carbon_emi.append(cum_dist[i] * self.bus_ave)
                bus_list.append(cum_dist[i] * self.bus_ave)
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
        
        plt.axis([0, max_bound, 0, 900])
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
    
    
    def Weight(self, n, non_zero, popt_levy, weight_max, p0):
        
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
        popt_norm, pcov = curve_fit(self.Norm, bin_middles, weight, p0)
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
    

    def TravelPurpose(self):
        
        carbon_emi = self.hist_emi
                        
        purpose = np.array(self.df_5)[:, 25]
        
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
        popt_levy = {}
        popt_norm = {}
        non_zero = {}
        levy_res = {}
        
        # w_max = {'commute':    0.002,
        #           'shopping':   0.0014,
        #           'pickup':     0.0015,
        #           'recreation': 0.0003,
        #           'education':  0.002, 
        #           'business':   0.0017,
        #           'work':       0.0011
        #           }
        
        w_max = {'commute':    0.0020,
                  'shopping':   0.0014,
                  'pickup':     0.00012,
                  'recreation': 0.0010,
                  'education':  0.0007, 
                  'business':   0.0011,
                  'work':       0.0011
                  }

        # p0 = {'commute':    (2726, 2738, 0, 0, 8),
        #       'shopping':   (3683, -2790, 0, 0, 10),
        #       'pickup':     (3200, -500, 0, 0, 5),
        #       'recreation': (3042, -320, 0, 0, 9e-5),
        #       'education':  (2000, 500, 0, 0, 5), 
        #       'business':   (4000, -1000, 0, 0, 15),
        #       'work':       (3042, 1400, 0, 0, 6)
        #       }
        
        p0 = {'commute':    (2726, 2738, 0, 0, 8),
              'shopping':   (3683, -2790, 0, 0, 10),
              'pickup':     (3200, -500, 0, 0, 5),
              'recreation': (2797, 243, 0, 0, 9e-5),
              'education':  (2000, 500, 0, 0, 5), 
              'business':   (4000, -1000, 0, 0, 15),
              'work':       (3042, 1400, 0, 0, 6)
              }
                
        for key in dict_purp:
            print(key)
        
            plt.axis([0, 10000, 0, 250])
            n, _, _ = plt.hist(dict_purp[key], num_bins)
            plt.show()
        
            non_zero[key] = len(dict_purp[key]) - n[0] - n[-1]
            n[0] = 0
            n[-1] = 0
            
            popt_levy[key], dist_estm[key] = self.LevyFitting(n, non_zero[key], dict_purp[key])
            popt_norm[key] = self.Weight(n, non_zero[key], popt_levy[key], w_max[key], p0[key])
        
        for key in dict_purp:
            levy_res[key] = np.log10(dist_estm[key])
            
        x = range(10000)
        
        for key in dict_purp:
            
            y_levy = np.array(self.Levy(x, *popt_levy[key]))
            y_norm = np.array(self.Norm(x, *popt_norm[key]))
            
            y = y_levy * y_norm
            dist_estm[key] = y / np.sum(y) * non_zero[key] * 10
        
        
        plt.axis([0, 10000, 0, 240])
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
        
        # x = np.log10(x)
        # for key in dict_purp:
        #     dist_estm[key] = np.log10(dist_estm[key])
                    
        plt.axis([2, 4, 0, 2.4])
        plt.plot(x, dist_estm['commute'], 'k', linewidth=1, c='red', label='Commute')
        plt.plot(x, dist_estm['shopping'], 'k', linewidth=1, c='blue', label='Shopping')
        plt.plot(x, dist_estm['pickup'], 'k', linewidth=1, c='green', label='Pickup')
        plt.plot(x, dist_estm['recreation'], 'k', linewidth=1, c='yellow', label='Recreation')
        plt.plot(x, dist_estm['education'], 'k', linewidth=1, c='black', label='Education')
        plt.plot(x, dist_estm['business'], 'k', linewidth=1, c='orange', label='Personal business')
        plt.plot(x, dist_estm['work'], 'k', linewidth=1, c='purple', label='Work related')
        plt.plot(x, levy_res['commute'], '--', linewidth=1, c='red')
        plt.plot(x, levy_res['shopping'], '--', linewidth=1, c='blue')
        plt.plot(x, levy_res['pickup'], '--', linewidth=1, c='green')
        plt.plot(x, levy_res['recreation'], '--', linewidth=1, c='yellow')
        plt.plot(x, levy_res['education'], '--', linewidth=1, c='black')
        plt.plot(x, levy_res['business'], '--', linewidth=1, c='orange')
        plt.plot(x, levy_res['work'], '--', linewidth=1, c='purple')
        plt.legend()
        plt.xlabel('Carbon emissions (10^g)', self.font)
        plt.ylabel('Number of trips (10^)', self.font)
        plt.show()
        
        return np.sum(dist_estm['recreation'])
    
    
    def polygon_under_graph(self, x, y):
        """
        Construct the vertex list which defines the polygon filling the space under
        the (x, y) line graph. This assumes x is in ascending order.
        """
        return [(x[0], 0.), *zip(x, y), (x[-1], 0.)]
    
    
    def CommuteResults(self):
        
        y_levy = np.zeros((4, 1000))
        y_norm = np.zeros((4, 1000))
        y = np.zeros((4, 1000))
        x = self.bin_middles
        
        popt_levy = [[2484.79562213, -155.90210124, 17.16763672, 0.],
                     [2532.52256689, -172.26129494, 17.12420064, 0.],
                     [2630.65370779, -197.52188536, 17.28904614, 0.],
                     [2352.04643379, -151.52520698, 16.93699735, 0.]]
        popt_norm = [[2703.49331, 2556.60419, 0.00000000e+00, 0.000349698516, 7.58381501],
                     [3150.55305, 2848.96081, 0.00000000e+00, 0.000140834215, 10.3897752],
                     [2749.19574, 2794.73387, 0.00000000e+00, 0.000301396624, 7.76366877],
                     [2784.95686, 2664.82151, 0.00000000e+00, 0.000227880113, 8.93355848]]
        sum_y = [69559.16388171402, 66969.2821200769, 48689.59643118815, 52139.42086463295]
                
        for i in range(4):
            popt_l = popt_levy[i]
            popt_n = popt_norm[i]
            y_levy[i] = np.array(self.Levy(x, *popt_l))
            y_norm[i] = np.array(self.Norm(x, *popt_n))
            y[i] = y_levy[i] * y_norm[i]
            y[i] /= np.sum(y[i]) 
            # y[i] = y[i] * sum_y[i] / 10
            
        plt.axis([0, 6000, 0, 0.004])

        plt.plot(x, y[0], 'k', linewidth=2, color=(0.0, 0, 1.0), label='2017-2018')
        plt.plot(x, y[1], 'k', linewidth=2, color=(0.3, 0, 0.6), label='2018-2019')
        plt.plot(x, y[2], 'k', linewidth=2, color=(0.6, 0, 0.3), label='2019-2020')
        plt.plot(x, y[3], 'k', linewidth=2, color=(1.0, 0, 0.0), label='2020-2021')
        plt.legend()
        plt.xlabel('Carbon emissions (g)', self.font)
        plt.ylabel('PDF', self.font)
        plt.show()
        
        
        # ax = plt.figure(figsize=(10,6)).add_subplot(projection='3d')
        
        # years = range(1, 5)
        
        # # verts[i] is a list of (x, y) pairs defining polygon i.
        # gamma = np.vectorize(math.gamma)
        # verts = [self.polygon_under_graph(x, y[i-1])
        #          for i in years]
        
        # viridis = cm.get_cmap('plasma', 12)
        # facecolors = viridis.colors
        # facecolors = [[0.050383, 0.029803, 0.527975, 1.      ], [0.241396, 0.014979, 0.610259, 1.      ], [0.387183, 0.001434, 0.654177, 1.      ], [0.523633, 0.024532, 0.652901, 1.      ]]
        # facecolors = [[1, 0, 0, 1], [0.9, 0, 0, 1], [0.8, 0, 0, 1], [0.7, 0, 0, 1]]
        # print(facecolors)
        
        # poly = PolyCollection(verts, facecolors=facecolors, alpha=.7)
        # ax.add_collection3d(poly, zs=years, zdir='y')
        
        # ax.set(xlim=(0, 10000), ylim=(1, 4.5), zlim=(0, 25),
               
        #        xlabel='Carbon emissions (g)', ylabel=r'$year$', zlabel='Number of trips')
        
        # plt.show()
        
        return None
    
    
    def ShoppingResults(self):
        
        y_levy = np.zeros((4, 1000))
        y_norm = np.zeros((4, 1000))
        y = np.zeros((4, 1000))
        x = self.bin_middles
        
        popt_levy = [[638.28089978, -9.055735860, 15.05635202, 0.],
                     [651.79002160,  1.216277420, 15.08220682, 0.],
                     [667.85035700, -0.246296216, 15.3365203, 0.],
                     [625.36202313,  6.012844070, 15.44689564, 0.]]
        popt_norm = [[2948.58000, -987.274653, 0.00000000e+00, 0.00011699469, 10.2909404],
                     [3581.71138, -2034.32316, 0.00000000e+00, 0.0000407511668, 10.6283708],
                     [3872.17441, -3118.23603, 0.00000000e+00, 0.0000411965824, 16.0611839],
                     [4676.27599, -6092.29384, 0.00000000e+00, 0.00000928583716, 6.36574477]]
        sum_y = [65350.0, 57670.0, 44230.00000000001, 52139.42086463295]
        
        for i in range(4):
            popt_l = popt_levy[i]
            popt_n = popt_norm[i]
            y_levy[i] = np.array(self.Levy(x, *popt_l))
            y_norm[i] = np.array(self.Norm(x, *popt_n))
            y[i] = y_levy[i] * y_norm[i]
            y[i] /= np.sum(y[i]) 
            # y[i] = y[i] * sum_y[i] / 10
            
        plt.axis([0, 6000, 0, 0.015])

        plt.plot(x, y[0], 'k', linewidth=2, color=(0.0, 0, 1.0), label='2017-2018')
        plt.plot(x, y[1], 'k', linewidth=2, color=(0.3, 0, 0.6), label='2018-2019')
        plt.plot(x, y[2], 'k', linewidth=2, color=(0.6, 0, 0.3), label='2019-2020')
        plt.plot(x, y[3], 'k', linewidth=2, color=(1.0, 0, 0.0), label='2020-2021')
        plt.legend()
        plt.xlabel('Carbon emissions (g)', self.font)
        plt.ylabel('PDF', self.font)
        plt.show()
        
        
        # ax = plt.figure(figsize=(10,6)).add_subplot(projection='3d')
        
        # years = range(1, 5)
        
        # # verts[i] is a list of (x, y) pairs defining polygon i.
        # gamma = np.vectorize(math.gamma)
        # verts = [self.polygon_under_graph(x, y[i-1])
        #          for i in years]
        
        # viridis = cm.get_cmap('plasma', 12)
        # facecolors = viridis.colors
        # facecolors = [[0.050383, 0.029803, 0.527975, 1.      ], [0.241396, 0.014979, 0.610259, 1.      ], [0.387183, 0.001434, 0.654177, 1.      ], [0.523633, 0.024532, 0.652901, 1.      ]]
        # facecolors = [[0, 0, 1, 1], [0, 0, 0.9, 1], [0, 0, 0.8, 1], [0, 0, 0.7, 1]]
        # print(facecolors)
        
        # poly = PolyCollection(verts, facecolors=facecolors, alpha=.7)
        # ax.add_collection3d(poly, zs=years, zdir='y')
        
        # ax.set(xlim=(0, 10000), ylim=(1, 4.5), zlim=(0, 90),
               
        #        xlabel='Carbon emissions (g)', ylabel=r'$year$', zlabel='Number of trips')
        
        # plt.show()
        
        return None
            
       
    def RecreationResults(self):
        
        y_levy = np.zeros((4, 1000))
        y_norm = np.zeros((4, 1000))
        y = np.zeros((4, 1000))
        x = self.bin_middles
        
        popt_levy = [[903.86732213, 7.18356327, 15.05298344, 0.],
                     [908.39271423, 26.39063912, 15.29474426, 0.],
                     [861.76833284, 6.87323499, 15.0391782, 0.],
                     [694.19437525, 53.20996437, 14.27691785, 0.]]
        popt_norm = [[2394.62587, 418.131815, 0.00000000e+00, 0.0000343171486, 1.25236704],
                     [2797.63614, 243.439671, 0.00000000e+00, 0.0000118294796, 1.31524791],
                     [1998.27889, 821.056174, 0.00000000e+00, 0.0000631647039, 1.94677144],
                     [4085.9307, -786.419002, 0.00000000e+00, -0.000000406283983, 0.470568787]]
        sum_y = [29460.0, 28080.0, 20000.0, 21650.0]
        
        for i in range(4):
            popt_l = popt_levy[i]
            popt_n = popt_norm[i]
            y_levy[i] = np.array(self.Levy(x, *popt_l))
            y_norm[i] = np.array(self.Norm(x, *popt_n))
            y[i] = y_levy[i] * y_norm[i]
            y[i] /= np.sum(y[i]) 
            y[i] = y[i] * sum_y[i] / 10
            
        plt.axis([0, 6000, 0, 0.015])

        plt.plot(x, y[0], 'k', linewidth=2, color=(0.0, 0, 1.0), label='2017-2018')
        plt.plot(x, y[1], 'k', linewidth=2, color=(0.3, 0, 0.6), label='2018-2019')
        plt.plot(x, y[2], 'k', linewidth=2, color=(0.6, 0, 0.3), label='2019-2020')
        plt.plot(x, y[3], 'k', linewidth=2, color=(1.0, 0, 0.0), label='2020-2021')
        plt.legend()
        plt.xlabel('Carbon emissions (g)', self.font)
        plt.ylabel('PDF', self.font)
        plt.show()
        
        
        ax = plt.figure(figsize=(10,6)).add_subplot(projection='3d')
        
        years = range(1, 5)
        
        # verts[i] is a list of (x, y) pairs defining polygon i.
        gamma = np.vectorize(math.gamma)
        verts = [self.polygon_under_graph(x, y[i-1])
                  for i in years]
        
        viridis = cm.get_cmap('plasma', 12)
        facecolors = viridis.colors
        facecolors = [[0.050383, 0.029803, 0.527975, 1.      ], [0.241396, 0.014979, 0.610259, 1.      ], [0.387183, 0.001434, 0.654177, 1.      ], [0.523633, 0.024532, 0.652901, 1.      ]]
        facecolors = [[1, 1, 0, 1], [0.9, 0.9, 0, 1], [0.8, 0.8, 0, 1], [0.7, 0.7, 0, 1]]
        print(facecolors)
        
        poly = PolyCollection(verts, facecolors=facecolors, alpha=.7)
        ax.add_collection3d(poly, zs=years, zdir='y')
        
        ax.set(xlim=(0, 10000), ylim=(1, 4.5), zlim=(0, 30),
               
                xlabel='Carbon emissions (g)', ylabel=r'$year$', zlabel='Number of trips')
        
        plt.show()
        
        return None
        
    
    def PurposeModeShare(self):
        
        # 0, 1015, 1255, 1535, 1750
        mode_id = self.mode_id
        purpose = np.array(self.df_5)[:, 25]
        mode_shr = np.zeros((7, 4))
        
        for i in range(len(mode_id)):
            if purpose[i] == 'Direct Work Commute':
                mode_shr[0][mode_id[i]] += 1
            elif purpose[i] == 'Shopping':
                mode_shr[1][mode_id[i]] += 1
            elif purpose[i] == 'Pickup/Dropoff Someone':
                mode_shr[2][mode_id[i]] += 1
            elif purpose[i] == 'Recreation':
                mode_shr[3][mode_id[i]] += 1
            elif purpose[i] == 'Education':
                mode_shr[4][mode_id[i]] += 1
            elif purpose[i] == 'Personal Business':
                mode_shr[5][mode_id[i]] += 1
            elif purpose[i] == 'Work Related':
                mode_shr[6][mode_id[i]] += 1
                
        sum_shr = np.sum(mode_shr, axis=1)
        mode_shr /= sum_shr.reshape((7, 1))
                
        return mode_shr
    
    
    def PlotYear(self):
        
        size = 3
        total_w, n = 0.8, 4
        
        # mode_shr = [[ 0.6577, 	 0.3110, 	 0.0313 ],
        #             [ 0.6546, 	 0.3215, 	 0.0238 ],
        #             [ 0.6476, 	 0.3355, 	 0.0169 ],
        #             [ 0.5594, 	 0.4277, 	 0.0130 ]]
        # mode_shr = [[  0.8886, 	 0.0804, 	 0.0310 ],
        #             [  0.9057, 	 0.0704, 	 0.0239 ],
        #             [  0.8905, 	 0.0835, 	 0.0260 ],
        #             [  0.9043, 	 0.0786, 	 0.0171 ]]
        mode_shr = [[ 0.8750, 	 0.0235, 	 0.1015 ],
                    [ 0.8709, 	 0.0278, 	 0.1013 ],
                    [ 0.8746, 	 0.0295, 	 0.0959 ],
                    [ 0.9078, 	 0.0362, 	 0.0561 ]]
        
        x = np.arange(size)
        width = total_w / n
        tick_label = ['Private vehicle', 'Active transport', 'Public transport']
        
        plt.bar(x + 0 * width, mode_shr[0], width=width, color=(1, 0, 0), label='2017-2018')
        plt.bar(x + 1 * width, mode_shr[1], width=width, color=(.85, 0, 0), label='2018-2019')
        plt.bar(x + 2 * width, mode_shr[2], width=width, color=(.70, 0, 0), label='2019-2020')
        plt.bar(x + 3 * width, mode_shr[3], width=width, color=(.55, 0, 0), label='2020-2021')
        plt.legend()
        plt.xticks(x+total_w/2,tick_label)
        plt.xlabel('Travel mode', self.font)
        plt.ylabel('Mode share', self.font)
        plt.show()
        
        return None
    
    
    def PlotPurpose(self):
        
        size = 3
        total_w, n = 0.8, 7
        
        mode_shr = [[0.88295688, 2.99494165e-02, 8.70937046e-02],
                    [0.89716560, 7.97612005e-02, 2.30731996e-02],
                    [0.95497257, 3.83426878e-02, 6.68474491e-03],
                    [0.61944913, 3.62402634e-01, 1.81482374e-02],
                    [0.68024457, 1.55528763e-01, 1.64226662e-01],
                    [0.91861731, 4.25966249e-02, 3.87860642e-02],
                    [0.89966555, 6.00246435e-02, 4.03098046e-02]]
        
        x = np.arange(size)
        width = total_w / n
        tick_label = ['Private vehicle', 'Active transport', 'Public transport']
        
        plt.bar(x + 0 * width, mode_shr[0], width=width, color='red', label='Commute')
        plt.bar(x + 1 * width, mode_shr[1], width=width, color='blue', label='Shopping')
        plt.bar(x + 2 * width, mode_shr[2], width=width, color='green', label='Pickup')
        plt.bar(x + 3 * width, mode_shr[3], width=width, color='yellow', label='Recreation')
        plt.bar(x + 4 * width, mode_shr[4], width=width, color='black', label='Education')
        plt.bar(x + 5 * width, mode_shr[5], width=width, color='orange', label='Personal business')
        plt.bar(x + 6 * width, mode_shr[6], width=width, color='purple', label='Work related')
        plt.legend()
        plt.xticks(x+total_w/2,tick_label)
        plt.xlabel('Travel mode', self.font)
        plt.ylabel('Mode share', self.font)
        plt.show()
        
        return None
    
    
    
    
if __name__ == "__main__":
    
    levy_fitting = LevyFitting()
    weight, non_zero, emission = levy_fitting.TripEmission()
    
    popt_levy, _ = levy_fitting.LevyFitting(weight, non_zero, emission)
    popt_norm = levy_fitting.Weight(weight, non_zero, popt_levy, 0.002, (4000, 0, 0, 0, 0))
    
    # dict_purp = levy_fitting.TravelPurpose()
    # sum_y = levy_fitting.PurposeAnalysis(dict_purp)
    
    levy_fitting.CommuteResults()
    levy_fitting.ShoppingResults()
    levy_fitting.RecreationResults()
    
    mode_shr = levy_fitting.PurposeModeShare()
    levy_fitting.PlotYear()
    levy_fitting.PlotPurpose()