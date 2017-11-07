#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import os
import multiprocessing as mp
import traceback
import logging
import gc
import time

from constants import c, G, Msun, rho_0, rho_ns, rho_1, rho_2, rho_3, P_0, gamma_0, dyncm2_to_MeVfm3, gcm3_to_MeVfm3, oneoverfm_MeV
import numpy
from tqdm import tqdm, trange
import itertools
from scipy.constants import pi
from scipy.interpolate import interp1d, UnivariateSpline
from scipy.signal import argrelextrema
from scipy.integrate import odeint
from matplotlib import pyplot

def print_progressbar(i, N):
    pbwidth = 42

    progress = float(i)/N
    block = int(round(pbwidth*progress))
    text = "\rProgress: [{0}] {1:.1f}%".format( "#"*block + "-"*(pbwidth-block), progress*100)
    sys.stdout.write(text)
    sys.stdout.flush()

    if i == (N-1):
        print " .. done"

def crust_EOS():
    """ 
    Interpolates the SLy EOS to use for the crust and calculates the minimum pressure. 
    
    Returns:
        EOS (function): A function of density representing the EOS
        Inverse EOS (function): A function of pressure representing the EOS
        P_min (float): The minimum pressure that is tabulated.
    """
    Pmin = 1e2
    Pmax = SLYfit(14.3)
    rhotest = numpy.logspace(6, 16, 300)
    prestest = 10**SLYfit(numpy.log10(rhotest))
    ptest = numpy.logspace(numpy.log10(Pmin), Pmax, 500)

    eos = UnivariateSpline(rhotest, prestest, k=3, s=0)
    inveos = UnivariateSpline(prestest, rhotest, k=3, s=0)
  
    
    return eos, inveos, Pmin

def f0(x):
    return 1./(numpy.exp(x) + 1.)

def SLYfit(rho):
    a = numpy.array([6.22, 6.121, 0.005925, 0.16326, 6.48, 11.4971, 19.105, 0.8938, 
                         6.54, 11.4950, -22.775, 1.5707, 4.3, 14.08, 27.80, -1.653, 1.50,
                         14.67])
    part1 = (a[0] + a[1]*rho + a[2]*rho**3.)/(1. + a[3]*rho) * f0(a[4]*(rho-a[5]))
    part2 = (a[6] + a[7]*rho)*f0(a[8]*(a[9]-rho))
    part3 = (a[10] + a[11]*rho)*f0(a[12]*(a[13]-rho))
    part4 = (a[14] + a[15]*rho)*f0(a[16]*(a[17] - rho))
    return part1+part2+part3+part4

def eos(rho, P_1, P_2, P_3, eos_crust):
    """
    The parameterized EOS.
    
    Args:
        rho (float): The density at which to evaluate the EOS in g/cm^3.
        P_1 (float): The first pressure parameter of the parameterization.
        P_2 (float): The second pressure parameter of the parameterization.
        P_3 (float): The third pressure parameter of the parameterization.
        eos_crust (function): The EOS for the low-density part, which inputs a mass density and returns a pressure.
    Returns:
        rho (float): The rest-mass density in g/cm^3.
        epsilon (float): The energy density in g/cm^3
    """
    rho_ns = 2.7e14
    rho_1 = 1.85 * rho_ns 
    rho_2 = 2. * rho_1 
    rho_3 = 2. * rho_2 
    
    gamma_1 = numpy.log10(P_1/P_0) / numpy.log10(rho_1/rho_0)
    gamma_2 = numpy.log10(P_2/P_1) / numpy.log10(rho_2/rho_1)
    gamma_3 = numpy.log10(P_3/P_2) / numpy.log10(rho_3/rho_2)
    
    k1 = P_0/(rho_0**gamma_1)
    
    pres1 = k1*rho_1**gamma_1
    k2 = pres1/(rho_1**gamma_2)
    
    pres2 = k2*rho_2**gamma_2
    k3 = pres2/(rho_2**gamma_3)
    
    #gamma0 = deriv_hpd(rho0) * rho0/P0
    gamma0 = 2.7
    e0 = rho_0 + P_0/c**2.0  * 1./(gamma0 - 1.)
    
    a1 = e0/rho_0 - 1. - k1/((gamma_1-1.)*c**2.0) * rho_0**(gamma_1-1.)
    e1 = (1. + a1)*rho_1 + pres1/(c**2.0 * (gamma_1 -1.))
    a2 = e1/rho_1 - 1. - k2/((gamma_2-1.)*c**2.0) * rho_1**(gamma_2-1.)
    e2 = (1. + a2)*rho_2 + pres2/(c**2.0 * (gamma_2 -1.))
    a3 = e2/rho_2 - 1. - k3/((gamma_3-1.)*c**2.0) * rho_2**(gamma_3-1.)

    if rho <= rho_0:
        pres = eos_crust(rho)
        gamma_05 = 1.7
        epsilon = rho + pres/c**2. * 1./(gamma_05 - 1)
    
    if rho_0 < rho <= rho_1:
        pres = k1 * rho**gamma_1
        epsilon = (1. + a1)*rho + pres/(c**2.0 * (gamma_1 -1.))
        
    if rho_1 < rho <= rho_2:
        pres = k2 * rho**gamma_2
        epsilon = (1. + a2)*rho + pres/(c**2.0 * (gamma_2 -1.))
        
    if rho > rho_2:
        pres = k3 * rho**gamma_3
        epsilon = (1. + a3)*rho + pres/(c**2.0 * (gamma_3 -1.))
        
    
    return pres, epsilon
    
def inveos(pres, P_1, P_2, P_3, eos_crust, inveos_crust, P_min):
    """
    The inverse of the parameterized EOS.
    
    Args:
        pres (float)            : The pressure at which to evaluate the inverse EOS in dyn/cm^2.
        P_1 (float)             : The first pressure parameter of the parameterization.
        P_2 (float)             : The second pressure parameter of the parameterization.
        P_3 (float)             : The third pressure parameter of the parameterization.
        eos_crust (function)    : The EOS for the low-density part, which inputs a mass density and returns a pressure.
        inveos_crust (function) : The inverse EOS for the low density part, which inputs a pressure and returns a mass density.
        P_min (float)           : The minimum pressure for which the low-density EOS function is defined.
    Returns:
        rho (float)     : The rest-mass density in g/cm^3.
        epsilon (float) : The energy density in g/cm^3
    """
      
    gamma_1 = numpy.log10(P_1/P_0) / numpy.log10(rho_1/rho_0)
    gamma_2 = numpy.log10(P_2/P_1) / numpy.log10(rho_2/rho_1)
    gamma_3 = numpy.log10(P_3/P_2) / numpy.log10(rho_3/rho_2)
    
    k1 = P_0/(rho_0**gamma_1)
    
    pres1 = k1*rho_1**gamma_1
    k2 = pres1/(rho_1**gamma_2)
    
    pres2 = k2*rho_2**gamma_2
    k3 = pres2/(rho_2**gamma_3)
    
    #gamma0 = deriv_hpd(rho0) * rho0/P0
    gamma0 = 2.7
    e0 = rho_0 + P_0/c**2.0  * 1./(gamma0 - 1.)
    
    a1 = e0/rho_0 - 1. - k1/((gamma_1-1.)*c**2.0) * rho_0**(gamma_1-1.)
    e1 = (1. + a1)*rho_1 + pres1/(c**2.0 * (gamma_1 -1.))
    a2 = e1/rho_1 - 1. - k2/((gamma_2-1.)*c**2.0) * rho_1**(gamma_2-1.)
    e2 = (1. + a2)*rho_2 + pres2/(c**2.0 * (gamma_2 -1.))
    a3 = e2/rho_2 - 1. - k3/((gamma_3-1.)*c**2.0) * rho_2**(gamma_3-1.)
    
    if pres <= P_0:
        rho = inveos_crust(pres)
        #rho = 10**inveos_crust(numpy.log10(pres))
        gamma_05 = 1.7
        epsilon = rho + pres/c**2. * 1./(gamma_05 - 1)
    
    if P_0 < pres <= P_1:
        rho = (pres/k1)**(1./gamma_1)
        epsilon = (1. + a1)*rho + pres/(c**2.0 * (gamma_1 -1.))
        
    if P_1 < pres <= P_2:
        rho = (pres/k2)**(1./gamma_2)

        epsilon = (1. + a2)*rho + pres/(c**2.0 * (gamma_2 -1.))
        
    if pres > P_2:

        rho = (pres/k3)**(1./gamma_3)
        epsilon = (1. + a3)*rho + pres/(c**2.0 * (gamma_3 -1.))
        
    
    return rho, epsilon


### Define function to integrate

def f(initial, r, P_1, P_2, P_3, eos_crust, inveos_crust, P_min):
    """
    The TOV-equations to pass on to scipy's 'odeint'. 
    
    Args:
        initial (array)        : Array of two values, the inital pressure and the initial mass.
        r (float)              : The radial coordinate of the neutron star (r = 0 is the center).
        P_1 (float)            : The first pressure parameter of the parameterization.
        P_2 (float)            : The second pressure parameter of the parameterization.
        P_3 (float)            : The third pressure parameter of the parameterization.
        eos_crust (function)   : The EOS for the low-density part, which inputs a mass density and returns a pressure.
        inveos_crust (function): The inverse EOS for the low density part, which inputs a pressure and returns a mass density.
        P_min (float)          : The minimum pressure for which the low-density EOS function is defined.
    Returns:
        dpdr (float)           : The derivative of the pressure with respect to the radial coordinate.
        dmdr (float)           : The derivative of the mass with respect to the radial coordinate. 
    """
       
    pres, m = initial

    if pres < P_min:
        pres = P_min 

    rho, eps = inveos(pres, P_1, P_2, P_3, eos_crust, inveos_crust, P_min)
   
    dmdr = 4.*pi*r**2.0 * eps
    if r==0.0:
        dpdr = 0.0
    else:
        dpdr = -G * (eps + pres/c**2.) * (m + 4.*pi*r**3. * pres/c**2.)
        dpdr = dpdr/(r*(r - 2.*G*m/c**2.))
   
    return dpdr, dmdr


### Function to solve the TOV-equations
def tovsolve(rhocent, P_1, P_2, P_3, eos_crust, inveos_crust, P_min):
    """
    Solves the TOV-equations using scipy's 'odeint' package. 
    
    Args:
        rhocent (float)        : The central density of the neutron star. This is the starting
                                 value of the differential integration.
        P_1 (float)            : The first pressure parameter of the parameterization.
        P_2 (float)            : The second pressure parameter of the parameterization.
        P_3 (float)            : The third pressure parameter of the parameterization.
        eos_crust (function)   : The EOS for the low-density part, which inputs a mass density and returns a pressure.
        inveos_crust (function): The inverse EOS for the low density part, which inputs a pressure and returns a mass density.
        P_min (float)          : The minimum pressure for which the low-density EOS function is defined.

    Returns:
        M (float)      : The mass of the neutron star. 
        R (float)      : The radius of the neutron star.
    """
    
    
    dr = 800.
    r = numpy.arange(0.0, 2500000., dr)

    pcent = eos(rhocent, P_1, P_2, P_3, eos_crust)[0]
    m0 =  0.0
    P0 = pcent
    y = P0, m0
    
    psol = odeint(f, y, r, args=(P_1, P_2, P_3, eos_crust, inveos_crust, P_min))

    indices = numpy.where(psol[:,0]>P_min)
    index = indices[-1][-1]
    M_max = psol[index][1]/Msun
    R_max = r[index]/100000
    I = MomentInertia(psol[:,0][indices[0]], psol[:,1][indices[0]], r[indices[0]], P_min)    
       
    return M_max, R_max, I


### Calculate the Masses and Radii for different central pressures
def calculate_MR(logrhomin, logrhomax, n, P_1, P_2, P_3, eos_crust, inveos_crust, P_min):
    """
    Calculate a mass-radius curve by solving the TOV-equations for different central densities.

    Args:
        logrhomin (float)      : The lower limit of the central density in log(g/cm^3).
        logrhomax (float)      : The upper limit of the central density in log(g/cm^3), based on causality. 
        n (int)                : The number of points used in logspace to create the MR-curve.
        P_1 (float)            : The first pressure parameter of the parameterization.
        P_2 (float)            : The second pressure parameter of the parameterization.
        P_3 (float)            : The third pressure parameter of the parameterization.
        eos_crust (function)   : The EOS for the low-density part, which inputs a mass density and returns a pressure.
        inveos_crust (function): The inverse EOS for the low density part, which inputs a pressure and returns a mass density.
        P_min (float)          : The minimum pressure for which the low-density EOS function is defined.

    Returns:
        Masses (array)   : An array of length 'n' with the masses of the neutron stars. 
        Radii (array)    : An array of length 'n' with the radii of the neutron stars. 
        Inert (array)    : An array of length 'n' with the moments of inertia of the stars.
        rhocent (array)  : An array of length 'n' with the central densities of each star.
    """
    
    rhocent = numpy.logspace(logrhomin, logrhomax, n)
    Masses = numpy.zeros(len(rhocent))
    Radii = numpy.zeros(len(rhocent))
    Inert = numpy.zeros(len(rhocent))
    for j, k in enumerate(rhocent):
        Masses[j], Radii[j], Inert[j] = tovsolve(k, P_1, P_2, P_3, eos_crust, inveos_crust, P_min)

    Masses = numpy.array(Masses)
    Radii = numpy.array(Radii)
    Inert = numpy.array(Inert)
    
    #Masses = Masses[Masses>0.]
    #Radii = Radii[Radii>0.]
    #Inert = Inert[Inert>0.]

    return Masses, Radii, Inert, rhocent
    
    
def MomentInertia(P, M, r, P_min):
    """
    Calculate the moment of inertia of a star given a central density and an EOS.
    Args:
        P (array)    : The pressure profile throughout the star as a function of the radial coordinate.
        M (array)    : The mass profile throughout the star as a function of the radial coordinate. 
        r (array)    : The radial coordinates.
        P_min (float): The minimum pressure for which the EOS is defined.
    Returns:
        I (float)    : The moment of inertia of the star in (Msun km^2) 
    """
    curveM = UnivariateSpline(r, M, k=3, s=0)
    curveP = UnivariateSpline(r, P, k=3, s=0)
    
    Mns = M[-1]
    Rns = r[-1]
        
    dr = 100
    rx = numpy.arange(min(r)+.1, max(r), dr)
    nustart = numpy.log(1. - 2.*G*Mns/(c**2. * Rns))
    
    curvenu = UnivariateSpline(rx, nu(rx, curveM, curveP), k=3, s=0)
    nufunc = curvenu.antiderivative(1)
    
    js = numpy.exp(-.5*(nufunc(rx)-nufunc(Rns)+nustart))*numpy.sqrt((1. - 2.*G*curveM(rx)/(c**2. *rx)))
    curvej = UnivariateSpline(rx, js, k=3, s=0)
    derivj = curvej.derivative(1)
    
    sol = odeint(omega, [1., 0.], rx, args=(curvej, derivj))
    w = sol[:,0]
    dw = sol[:,1]
    
    curvew = UnivariateSpline(rx, w, k=3, s=0)
    curvedw = UnivariateSpline(rx, dw, k=3, s=0)
    
    J = 1./6. * Rns**4. * curvedw(Rns) 
    W = curvew(Rns) + 2.*J/(Rns**3.)

    #I = quad(Inertia, 0.1, Rns, args=(derivj, curvew, W))[0] *2.*c**2. /(3.*G)
    
    I = (1. - curvew(Rns)/W)*Rns**3. * c**2./(2.*G)
    
    return I*10**(-10)/Msun

def nu(r, curveM, curveP):
    
    dvdr = 2*G/c**2. * (curveM(r) + 4.*numpy.pi*r**3. * curveP(r)/c**2.)/(r**2. * (1. - 2.*G*curveM(r)/(r*c**2.)))

    return dvdr

def omega(initial, r, curvej, derivj):
    
    x1, x2 = initial
    
    if r==0.0:
       
        dx1 = 0.
        dx2 = 0.
        
    else:
        dx1 = x2
        dx2 = - 4./(r*curvej(r)) * derivj(r) * x1 - 4./r * x2 -1./curvej(r) *derivj(r) *x2
        
    return dx1, dx2
    
def Inertia(r, derivj, curvew, W):
    return -r**3. * derivj(r)*curvew(r)/W

def calculate_MR_all(parameters, eos_crust, inveos_crust, task=0, logrhomin=14.4, logrhomax=16.5, n=100, P_min=0.0):
    """
    Calculate the masses, radii and moments of inertia for all input parameters by solving the TOV-equations 
    for different central densities.

    Args:
        parameters                : The array of parameters for which to solve the TOV-equations.
        eos_crust (function)      : The EOS for the low-density part, which inputs a pressure and returns a mass density.
        inveos_crust (function)   : The inverse EOS for the low density part, which inputs a mass density and returns a pressure.
        task                      : At which line to print the progressbar, default is 0.
        logrhomin (float)         : The lower limit of the central density in log(g/cm^3), default is 14.4.
        logrhomax (float or array): The upper limits of the central density in log(g/cm^3), based on causality. 
                                    If float, the same value for all EoSs is used, if array, every entry should
                                    correspond to the maximum central density of an EoS.  
        n (int)                   : The number of points used in logspace to create the MR-curve.
        eos_crust (function)      : The EoS for the low density part of the star. The function should take density 
                                    as input and outputs pressure.
        inveos_crust (function)   : The inverse of the EoS for the low density part of the star. 
                                    The function should take pressure as input and output density.
        P_min (float)             : The lowest density for which eos_crust and inveos_crust is defined, 
                                    default is 0.0.

    Returns:
        MRIcurves (ndarray)       : An array of length (EoS) with for each EoS an array of masses, 
                                    radii, moments of inertia and the corresponding central densities. 
        Parameters (ndarray)      : An array of length (EoS) with the parameters for which the TOV 
                                    equations generated stable mass-radius curves. 
    """
           
    MR_curves = []
    Error_params = []
    with tqdm(total=len(parameters), position=task, desc='Process %d' %(task), leave=False) as pbar:
        for j in range(len(parameters)):
            P_1 = parameters[j,0]
            P_2 = parameters[j,1]
            P_3 = parameters[j,2]

            try:
                if isinstance(logrhomax, float):
                    masses, radii, inert, rhocent = calculate_MR(logrhomin, logrhomax, n, P_1, P_2, P_3, eos_crust, inveos_crust, P_min)
                else:
                    masses, radii, inert, rhocent = calculate_MR(logrhomin, logrhomax[j], n, P_1, P_2, P_3, eos_crust, inveos_crust, P_min)
                pbar.update(1)
            except UnboundLocalError:
                Error_params.append(j)
                continue
            
            else:     
                locmin = argrelextrema(masses, numpy.less)[0]  #Check for EOS with local minima         
                if not len(locmin)==0:
                    Error_params.append(j)
                    continue
            
                locmax = argrelextrema(masses, numpy.greater)[0]
                
                if not len(locmax)==0:
                
                    if locmax[0] < len(masses)-1:      #check to see if there is a sharp kink in the MR-curve
                        right = locmax[0] + 1
                        left = locmax[0] - 1
                        if abs(radii[right]-radii[left]) < 0.12:
                            Error_params.append(j)
                            continue
                    
                    MR_curves.append([masses[0:locmax[0]+1], radii[0:locmax[0]+1], inert[0:locmax[0]+1], rhocent[0:locmax[0]+1]])
                   
                else:
                    MR_curves.append([masses, radii, inert, rhocent])
                    
    MR_curves = numpy.array(MR_curves)
    Error_params = numpy.array(Error_params)
    Parameters = numpy.delete(parameters, Error_params, axis=0)
    return MR_curves, Parameters

#info = mp.get_logger().info
def main(parameters, maxrho):
    #logger = mp.log_to_stderr()
    #logger.setLevel(logging.INFO)

    nproc = mp.cpu_count() - 1
    nproc = max(1, nproc)
    
    print len(parameters)
       
    div_par = numpy.array_split(parameters, nproc)
    div_maxrho = numpy.array_split(maxrho, nproc)
    
    ntasks = nproc
    inputs = [[div_par[t], div_maxrho[t], t] for t in xrange(ntasks)]
    
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
        
    result = numpy.array(sorted(result, key=lambda x: x[2]))
    result = numpy.delete(result, 2, axis=1)

    MRcurves = []
    Parameters = []

    for i in xrange(nproc):
        for j in range(len(result[i][0])):
            Parameters.append(result[i][1][j])
            MRcurves.append([numpy.array(result[i][0][j][0]), numpy.array(result[i][0][j][1]), 
                             numpy.array(result[i][0][j][2]), numpy.array(result[i][0][j][3])])
    
    Parameters = numpy.array(Parameters)
    MRcurves = numpy.array(MRcurves)

    numpy.save('Parameters4', Parameters)
    numpy.save('MRIRhocurves4', MRcurves)


def worker(input_q, output_q):
    
    start = time.clock()  
    while True:
        try:
            tmp = input_q.get()
            if 'STOP' == tmp :
                break
            
            parameters, maxrho, task = tmp
            
            eos_crust, inveos_crust, P_min = crust_EOS()

            MR_curves, Parameters = calculate_MR_all(parameters, eos_crust, inveos_crust, task, logrhomax=maxrho, n=30, P_min=P_min)
                    
            output_q.put([MR_curves, Parameters, task])
            
        except Exception as exception:
            trace = str(traceback.format_exc())
            #info(trace)
    #end = (time.clock() - start)
    #info(end)
    
    return    
 

if __name__ == '__main__':
  
  
    parameters = numpy.load('input_parameters.npy')
    maxrho = numpy.load('max_rho.npy')
    maxrho = numpy.log10(maxrho)

    main(parameters, maxrho)
    























