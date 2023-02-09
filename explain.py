# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:09:42 2023

@author: Yibo Wang
"""

import numpy as np
import matplotlib.pyplot as plt

def Draw():
    
    data = [40, 50, 10]
    x = [50, 150, 250]
    
    # plt.title('Carbon emission of each trip by motor vehicle')
    plt.axis([0, 300, 0, 55])
    plt.ylabel('Number of trips')
    plt.xlabel('Carbon emission (g)')
    plt.bar(x, data, width=30)
    plt.show()
    
    data = [16, 24, 30, 20, 8, 2]
    x = [25, 75, 125, 175, 225, 275]
    
    # plt.title('Carbon emission of each trip by motor vehicle')
    plt.axis([0, 300, 0, 55])
    plt.ylabel('Number of trips')
    plt.xlabel('Carbon emission (g)')
    plt.bar(x, data, width=30)
    plt.show()
    
    data = [0.4, 0.5, 0.1]
    x = [50, 150, 250]
    
    # plt.title('Carbon emission of each trip by motor vehicle')
    plt.axis([0, 300, 0, 1])
    plt.ylabel('Frequency')
    plt.xlabel('Carbon emission (g)')
    plt.bar(x, data, width=30)
    plt.show()
    
    data = [0.16, 0.24, 0.30, 0.20, 0.08, 0.02]
    x = [25, 75, 125, 175, 225, 275]
    
    # plt.title('Carbon emission of each trip by motor vehicle')
    plt.axis([0, 300, 0, 1])
    plt.ylabel('Frequency')
    plt.xlabel('Carbon emission (g)')
    plt.bar(x, data, width=30)
    plt.show()
    
    
Draw()
