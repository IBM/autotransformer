#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 13:52:47 2023

@author: subhadramokashe
"""

import numpy as np

import matplotlib.pyplot as plt

I=100                         #Set the parameter I.
C=1                          #Set the parameter C.
Vth = 1;                     #Define the voltage threshold.
Vreset = 0;                  #Define the reset voltage.
dt=0.01                      #Set the timestep.
V = np.zeros((1000,1))          #Initialize V.
V[0]=0.2;                    #Set the initial condition.

for k in range(1,999):       #March forward in time,
    V[k+1] = V[k] + dt*(I/C) #Update the voltage,
    if V[k+1] > Vth:         #... and check if the voltage exceeds the threshold.
        V[k+1] = Vreset
        
t = np.arange(0,len(V))*dt      #Define the time axis.

plt.figure()                     #Plot the results.
plt.plot(t,V)
plt.xlabel('Time [s]')
plt.ylabel('Voltage [mV]')
plt.show()