#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import multiprocessing as mp
import traceback
import logging
import gc
import time

import numpy
from matplotlib import pyplot
#from joblib import Parallel, delayed
from scipy.signal import argrelextrema
import multiprocessing
import itertools
from constants import c, G, Msun, rho_0, rho_ns, rho_1, rho_2, rho_3, P_0

def generate_params(n, low_gamma=0.5, high_gamma=6.5, Plow=[33.5, 34.5, 35.], Phigh=[34.8, 36., 37.]):
    """
    Generates an array of combinations of P1, P2 and P3, making sure that P3 > P2 > P1. 
    It also sets a lower and upper limit on the polytropic index.
    
    Args:
        n (int)        : The number of points in logspace for each parameter.
        low_gamma (float): The lower limit on the polytropic index, default is 0.5.
        high_gamma (float): The upper limit on the polytropic index, default is 6.5.
        Plow (array)      : Array of the three lower limits for P1, P2 and P3 in log10, 
                            default is [33.5, 34.5, 35.].
        Phigh (array)     : Array of the three upper limits for P1, P2 and P3 in log10, 
                            default is [34.8, 36., 37.].

    Returns:
        out (ndarray)  : Returns an array with all the possible combinations of P1, P2, P3.
    """
    
    excluded = []
    P_1 = numpy.logspace(Plow[0], Phigh[0], n)
    P_2 = numpy.logspace(Plow[1], Phigh[1], n)
    P_3 = numpy.logspace(Plow[2], Phigh[2], n)
    
    #P_1 = numpy.linspace(10**33.5, 10**34.8, n)
    #P_2 = numpy.linspace(10**34.5, 10**36., n)
    #P_3 = numpy.linspace(10**35., 10**37., n)
    
    iterables = [P_1, P_2, P_3]

    permutations = []
    for t in itertools.product(*iterables):
        if t[0] < t[1] < t[2]:
            permutations.append(t)
        
    permutations = numpy.array(permutations)
    
    for i in range(len(permutations)):
        P_1 = permutations[i][0]
        P_2 = permutations[i][1]
        P_3 = permutations[i][2]
        
        gamma_1 = numpy.log10(P_1/P_0) / numpy.log10(rho_1/rho_0)
        gamma_2 = numpy.log10(P_2/P_1) / numpy.log10(rho_2/rho_1)
        gamma_3 = numpy.log10(P_3/P_2) / numpy.log10(rho_3/rho_2)

        if not low_gamma <= gamma_1 <= high_gamma or not low_gamma <= gamma_2 <= high_gamma or not low_gamma <= gamma_3 <= high_gamma:
            excluded.append(i)
    
    permutations = numpy.delete(permutations, excluded, axis=0)    
    
    return permutations
    

def calc_causal_limit(rho, P_1, P_2, P_3):
    """
    Calculates if an EOS with parameters P1, P2 and P3 is causal at a given density.
    
    Args:
        rho (float): The density at which the function checks causality. 
        P_1 (float): The first parameter of the EOS.
        P_2 (float): The second parameter of the EOS.
        P_3 (float): The third parameter of the EOS.
        
    Returns:
        rho (float): If causality is not violated, the density is returned. 
    """
   
    gamma_1 = numpy.log10(P_1/P_0) / numpy.log10(rho_1/rho_0)
    gamma_2 = numpy.log10(P_2/P_1) / numpy.log10(rho_2/rho_1)
    gamma_3 = numpy.log10(P_3/P_2) / numpy.log10(rho_3/rho_2)
    
    
    epsilon_0 = rho_0 + P_0/c**2.0  * 1./1.7
    a_1 = epsilon_0/(rho_0) - 1. - P_1/((gamma_1 -1.)*rho_0*c**2.0) * (rho_0/rho_1)**gamma_1
    a_2 = a_1 + P_1/((gamma_1 -1.)*rho_1*c**2.0) - P_1/((gamma_2 -1.)*rho_1*c**2.0)
    a_3 = a_2 + P_1/((gamma_2 -1.)*rho_2*c**2.0) * (rho_2/rho_1)**gamma_2  - P_2/((gamma_3 -1.) * rho_2*c**2.0)

    causality = 0
    
    if rho_0 < rho <= rho_1:
        pres = P_1 * (rho/rho_1)**gamma_1
        epsilon = (1. + a_1) * rho + P_1/((gamma_1 - 1.)*c**2.) *(rho/rho_1)**gamma_1
        cs = gamma_1*pres/(epsilon*c**2. + pres)

        if gamma_1*pres/(epsilon*c**2. + pres) > 1.12:
            causality = 1
        
    if rho_1 < rho <= rho_2:
        pres = P_1 * (rho/rho_1)**gamma_2
        epsilon = (1. + a_2) * rho  + P_1/((gamma_2 - 1.)*c**2.) *(rho/rho_1)**gamma_2
        cs = gamma_2*pres/(epsilon*c**2. + pres)
     
        if gamma_2*pres/(epsilon*c**2. + pres) > 1.12:
            causality = 1
        
    if rho > rho_2:
        pres = P_2 * (rho/rho_2)**gamma_3
        epsilon = (1. + a_3) * rho  + P_2/((gamma_3 - 1.)*c**2.) *(rho/rho_2)**gamma_3  
        cs = gamma_3*pres/(epsilon*c**2. + pres)    

        if gamma_3*pres/(epsilon*c**2. + pres) > 1.12:
            causality = 1
            
    if causality==0:
        return rho 
    else:
        return 0.0
                
def calc_maxrho(parameters, n=1000, low_lim=14.31, up_lim=16.5):
    """
    Generates an array of len(parameters) containing the maximum central density for each EOSs.
    
    Args:
        parameters (ndarray): An array of EOS parameters, dimensions (q,3) for q EOSs.
        n (int)             : The number of points in logspace for each parameter. Default is 10^3.
        low_lim (float)     : The log10 of the lower limit of central densities to test causality for, 
                              must be greater than 14.3, default is 14.31.
        up_lim (float)      : The log10 of the lower limit of central densities to test causality for, 
                              default is 15.5.

    Returns:
        out (ndarray)       : Returns an array with all maximum central densities, length (q).
    """
    rhocents = numpy.logspace(low_lim, up_lim, n)        
    max_rho = numpy.zeros(len(parameters))

    for i,e in enumerate(parameters):
        causal_rho = numpy.zeros(len(rhocents))
    
        P_1 = e[0]
        P_2 = e[1]
        P_3 = e[2]
            
        for j, k in enumerate(rhocents):
            causal_rho[j] = calc_causal_limit(k, P_1, P_2, P_3)
        
        if causal_rho[0]==0.0:
            max_rho[i] = 0.0
            continue
        
        locmax = numpy.where(causal_rho==0.0)[0]
        if len(locmax) < 1:
            max_rho[i] = 10**(up_lim)
        else:
            max_rho[i] = causal_rho[locmax[0]-1]
    max_rho = numpy.array(max_rho) 
    return max_rho        


info = mp.get_logger().info
def main(n, low_lim, up_lim):
    logger = mp.log_to_stderr()
    logger.setLevel(logging.INFO)

    parameters = generate_params(n, low_lim, up_lim)
    print len(parameters)

    nproc = mp.cpu_count()# - 1
    nproc = max(1, nproc)
       
    div_par = numpy.array_split(parameters, nproc)
    
    ntasks = nproc
    inputs = [[div_par[t], t] for t in xrange(ntasks)]
    
    input_q = mp.Queue()
    output_q = mp.Queue()

        
    procs = [ mp.Process(target=worker, args=(input_q,output_q)) for i in xrange(nproc)]
    
    for i in xrange(ntasks):
        input_q.put(inputs[i])
    
    for i in xrange(nproc):
        input_q.put('STOP')
    
    for p in procs:
        p.start()
    
    result = []
    while ntasks > 0:
        result.append(output_q.get())
        ntasks -= 1 

    for p in procs:
        p.join()

    result = numpy.array(sorted(result, key=lambda x: x[1]))
    max_rho_array = numpy.concatenate(result[:,0]).ravel()

    numpy.save('max_rho', max_rho_array)
    numpy.save('input_parameters', parameters)
    
def worker(input_q, output_q):
    
    start = time.clock()  
    while True:
        try:
            tmp = input_q.get()
            if 'STOP' == tmp :
                break
            
            parameters, task = tmp
            
            max_rho = calc_maxrho(parameters)
            
            output_q.put([max_rho, task])
            
        except Exception as exception:
            trace = str(traceback.format_exc())
            info(trace)
    end = (time.clock() - start)
    info(end)
    
    return    
        
  
    

if __name__ == '__main__':
  
    main(15, 1., 5.5)
    























