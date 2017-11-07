#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import multiprocessing as mp
import traceback
import logging
import gc
import numpy 
from matplotlib import pyplot
from argparse import ArgumentParser
from scipy.interpolate import UnivariateSpline, RegularGridInterpolator
from scipy.integrate import simps, dblquad
import time
from constants import G, Msun, c

def calc_determinant(JacPart):
    indices = numpy.where(numpy.invert(numpy.isnan(JacPart).any(axis=1)))[0]
    deter = numpy.zeros((len(JacPart[indices]), len(JacPart[indices]), len(JacPart[indices])))
    derivs = numpy.zeros((6,6))
    if indices.size==0:
        return numpy.nan, numpy.nan, numpy.nan
    else:
        for p, i in enumerate(indices):
            for r, j in enumerate(indices):
                for w, k in enumerate(indices): 

                    derivs[:,0][0:3] = JacPart[i][3:6]
                    derivs[:,1][0:3] = JacPart[j][3:6]
                    derivs[:,2][0:3] = JacPart[k][3:6]

                    derivs[:,3][0:3] = JacPart[i][6::]
                    derivs[:,4][0:3] = JacPart[j][6::]
                    derivs[:,5][0:3] = JacPart[k][6::]

                    derivs[3][[0,3]] = JacPart[i][1:3]
                    derivs[4][[1,4]] = JacPart[j][1:3]
                    derivs[5][[2,5]] = JacPart[k][1:3]

                    deter[p, r, w] = abs(numpy.linalg.det(derivs))
        return min(JacPart[indices][:,0]), max(JacPart[indices][:,0]), deter

def calculate_norm(distribution):
    
    def Multivariate_notNorm(R, M, distribution):
        Mobs = distribution[0]
        Robs = distribution[1]
        sigmaM = distribution[2]
        sigmaR = distribution[3]
        rho = distribution[4]      
        
        return numpy.exp(-1./(2.*(1.-rho**2.)) * ((R-Robs)**2. / sigmaR**2.0 + (M-Mobs)**2. / sigmaM**2.0 - \
                                                     2.*rho*(R-Robs)*(M-Mobs)/(sigmaM*sigmaR)))
    
    norm = numpy.zeros(3)
    for i in range(3):
        norm[i] = dblquad(Multivariate_notNorm, 0.5, 3.3, lambda M: 2.94*G*M*Msun/(c**2. * 100000), 
        lambda M: 14.3, args=([distribution[i]]))[0]
    return norm
    

def Pobs(rho1, rho2, rho3, Jac_func, curveM, curveR, obs, norm):
    
    meanM = numpy.array([x[0] for x in obs])
    meanR = numpy.array([x[1] for x in obs])
    sigmaM = numpy.array([x[2] for x in obs])
    sigmaR = numpy.array([x[3] for x in obs])
    rho = numpy.array([x[4] for x in obs])
    Rho = numpy.array([rho1, rho2, rho3])
    
    obs = numpy.zeros(3, dtype=object)
    for i in range(3):  
        obs[i] = 1./norm[i] * numpy.exp(-1./(2.*(1.-rho[i]**2.)) * ((((curveM(Rho[i])-meanM[i])**2.0)/sigmaM[i]**2.0) +\
                    (((curveR(Rho[i])-meanR[i])**2.0)/sigmaR[i]**2.0) - (2.*rho[i]*(curveM(Rho[i])-meanM[i])*\
                                                                  (curveR(Rho[i])-meanR[i])/(sigmaM[i]*sigmaR[i]))))
    
    return obs[0] * obs[1] * obs[2] * abs(Jac_func((rho1, rho2, rho3)))
    
def simps_integration(points, low_lim, up_lim, Jac_func, curveM, curveR, obs, norm):
    
    n = points
    rho1s = numpy.linspace(low_lim, up_lim, n)
    rho2s = numpy.linspace(low_lim, up_lim, n)
    rho3s = numpy.linspace(low_lim, up_lim, n)
    
    integral1 = numpy.zeros(len(rho1s))
    integral2 = numpy.zeros(len(rho2s))

    for i in range(len(rho3s)):
        for j in range(len(rho2s)):
            integral1[j] = simps(Pobs(rho1s, numpy.full(n, rho2s[j]), numpy.full(n, rho3s[i]), Jac_func, curveM, curveR, obs, norm), rho1s) 
        integral2[i] = simps(integral1, rho2s)

    return simps(integral2, rho3s)



info = mp.get_logger().info
def main(MRIcurves, Jacobian, Observables, outputfile):
    logger = mp.log_to_stderr()
    logger.setLevel(logging.INFO)

    nproc = mp.cpu_count()# - 1
    nproc = max(1, nproc)
     
    #norm = numpy.zeros(len(Observables), dtype=object)
    #for i, e in enumerate(Observables):
    norm = calculate_norm(Observables)
  
    div_MR = numpy.array_split(MRIcurves, nproc)   
    div_jac = numpy.array_split(Jacobian, nproc)
    
    ntasks = nproc
    inputs = [[div_MR[t], div_jac[t], Observables, norm, t] for t in xrange(ntasks)]
    
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
    
        
    result = numpy.array(sorted(result, key=lambda x: x[1]))
    result = numpy.delete(result, 1, axis=1)    
    result = numpy.concatenate(result).ravel()
    result = numpy.concatenate(result).ravel()
    
    #Prob = numpy.zeros(len(Observables), dtype=object)
    #for z in range(len(Observables)):
    #    Prob[z] = numpy.array([e[z] for e in result]) 
    
    numpy.save(outputfile, result)
        
    for p in procs:
        p.join()

def worker(input_q, output_q):
    
    start = time.clock()  
    while True:
        try:
            tmp = input_q.get()
            if 'STOP' == tmp :
                break
            
            MRIcurves, Jacobian, Observables, norm, task = tmp
            info(len(Jacobian))
            Prob = numpy.zeros(len(Jacobian))
            for h, e in enumerate(Jacobian):

                Det = calc_determinant(e)
                
                M, R, I, rhoc = MRIcurves[h]
                x = numpy.log10(rhoc)

                curveM = UnivariateSpline(x, M, k=3, s=0)
                curveR = UnivariateSpline(x, R, k=3, s=0)

                if numpy.isnan(Det[0]) or len(Det[2])<3:
                    continue

                rhos = numpy.linspace(Det[0], Det[1], len(Det[2]))
                Jac_func = RegularGridInterpolator((rhos, rhos, rhos), Det[2])
               
                Prob[h] = simps_integration(25, min(rhos), max(rhos), Jac_func, curveM, curveR, Observables, norm)
    
                # Hack to avoid memory leak. Explicitly delete the instance of Jac_func and collect garbage.
                del Jac_func, curveM, curveR
                gc.collect()
            
            output_q.put([Prob, task])
            
        except Exception as exception:
            trace = str(traceback.format_exc())
            info(trace)
    end = (time.clock() - start)
    info(end)
    
    return    
        
  
    

if __name__ == '__main__':


    Observables = numpy.array([[1.5, 10.7, 0.05*1.5, 0.05*10.7, 0.0], [1.61, 10.5, 0.05*1.61, 0.05*10.5, 0.0], [1.7, 10.2, 0.05*1.7, 0.05*10.2, 0.0]])

    parser = ArgumentParser()

    parser.add_argument("-f", dest="outputFile", help="write probability to FILE", metavar="FILE", required=True)
    parser.add_argument("-i1", dest="inputMRIcurves", help="use as input MRIcurves", required=True)
    parser.add_argument("-i2", dest="inputJacobian", help="use as input Jacobian", required=True)
    parser.add_argument("-i3", dest="inputObservables", help="use as input observables", required=True)     
    
    args = parser.parse_args()

    MRIcurves = numpy.load(args.inputMRIcurves)
    Jacobian = numpy.load(args.inputJacobian)
    Observables = numpy.load(args.inputObservables)
    
    
    main(MRIcurves, Jacobian, Observables, args.outputFile)










