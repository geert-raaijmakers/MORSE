import numpy

def Create_Input_Posterior(percent, masses, radii, rho):
    
    Robs = radii
    Mobs = masses
    sigmaM = numpy.zeros(3)
    sigmaR = numpy.zeros(3) 

    sigmaM[0] = percent[0]*Mobs[0]
    sigmaR[0] = percent[0]*Robs[0]
    sigmaM[1] = percent[1]*Mobs[1]
    sigmaR[1] = percent[1]*Robs[1]
    sigmaM[2] = percent[2]*Mobs[2]
    sigmaR[2] = percent[2]*Robs[2]

    distribution = []
    
    for i in range(3):
        
        distribution.append([Mobs[i], Robs[i], sigmaM[i], sigmaR[i], rho[i]])
    
    distribution = numpy.array(distribution)
    
    return distribution

def find_CI_level(array):
    
    NaN_index = numpy.isnan(array)
    array[NaN_index] = 0.0
       
    index_68 = numpy.where(numpy.cumsum(numpy.sort(array)[::-1]) < sum(array)*0.6827)[0]
    index_68 = numpy.argsort(array)[::-1][index_68]

    index_95 = numpy.where(numpy.cumsum(numpy.sort(array)[::-1]) < sum(array)*0.9545)[0]
    index_95 = numpy.argsort(array)[::-1][index_95]
    
    
    return min(array[index_95]), min(array[index_68])

def Pobs(M, R, distribution):
     
    obs = numpy.zeros(3, dtype=object)

    for l in range(3):
        Mobs = distribution[l][0]
        Robs = distribution[l][1]
        sigmaM = distribution[l][2]
        sigmaR = distribution[l][3]
        rho = distribution[l][4]
        
        obs[l] = numpy.exp(-1./(2.*(1.-rho**2.)) *\
	                                  ((R-Robs)**2. / sigmaR**2.0 + (M-Mobs)**2. / sigmaM**2.0 - \
	                                                 2.*rho*(R-Robs)*(M-Mobs)/(sigmaM*sigmaR)))   
    return obs
