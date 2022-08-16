# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 17:08:03 2022

@author: Yibo Wang
"""

import numpy as np
import pandas as pd
import collections


class CarbonIntensity(object):
    
    def __init__(self):
        
        # Unit: MtCO2
        self.cars = 10.213
        self.trucks_buses = 4.908
        self.motorcycles = 0.068
        self.railways = 0.566
        
        # Unit: g/km
        self.petrol = 165
        self.diesel = 213
        
        # Unit： g/pkm
        self.car_2018 = 154.5
        self.bus_2018 = 79