#!/usr/bin/env python3

import numpy as np

def euler(uo, to, tf, delta_t):
    """
    Forward Euler's Method.
    f: Given differential equation
    uo: initial condition
    to: initial time
    tf: end time
    delta_t: change in time
    
    """
    num_points = (tf - to)/float(delta_t)
    t_array = np.linspace(to, tf, num_points)
    u = np.zeros(num_points)
    u[0] = uo
    
    for h in range(0, int(num_points), 1):
        
        u[h+1] = u[h] + ((2 * u[h] - 1) * t_array) * delta_t
        
    return u

def solve():
    uo = 2
    to = 0
    tf = 6
    delta_t = .25
    
    print euler(uo, to, tf, delta_t)