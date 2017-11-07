import sys
import os
import multiprocessing as mp
import traceback
import logging
import gc
import time

from argparse import ArgumentParser
import numpy
import pandas
from matplotlib import pyplot
from scipy.interpolate import UnivariateSpline

def UniqueParams(Parameters):

    df = pandas.DataFrame({'P1':Parameters[:,0], 'P2':Parameters[:,1], 'P3':Parameters[:,2]})
    df1 = df.drop_duplicates(['P1', 'P2'])
    df2 = df.drop_duplicates(['P1', 'P3'])
    df3 = df.drop_duplicates(['P2', 'P3'])
    df4 = pandas.concat([df1, df2, df3])
    ParametersOut = numpy.array(df4.drop_duplicates())
    
    return ParametersOut

def calculate_deriv(t, MRcurves, changeParam, Params):
    
    radii = numpy.zeros(len(changeParam))
    masses = numpy.zeros(len(changeParam))
    rhocM = numpy.zeros(len(changeParam))
    rhocR = numpy.zeros(len(changeParam))
    #inert = numpy.zeros(len(changeParam))
    wrong = []
    right = []
    back = numpy.empty((len(changeParam), 4))
    back.fill(numpy.nan)

    for i, e in enumerate(changeParam):
        M, R, I, rc = MRcurves[e]
        rc = numpy.log10(rc)
        curveR = UnivariateSpline(rc, R, k=3, s=1e-4)
        curveM = UnivariateSpline(rc, M, k=3, s=1e-4)
        derivR = curveR.derivative(1)
        derivM = curveM.derivative(1)

        if min(rc) <= t <= max(rc):
            radii[i] = curveR(t)
            masses[i] = curveM(t)
            rhocM[i] = derivM(t)
            rhocR[i] = derivR(t)
            right.append(i)

        else:
            wrong.append(i)
    Params2 = numpy.delete(Params, wrong, axis=0) 
    radii = numpy.delete(radii, wrong, axis=0)
    masses = numpy.delete(masses, wrong, axis=0)
    rhocM = numpy.delete(rhocM, wrong, axis=0)
    rhocR = numpy.delete(rhocR, wrong, axis=0)    
    
    if Params2.size==0 or len(Params2)<4:
        return back[:,0:2], back[:,2], back[:,3]
    
    else:
        curvePR = UnivariateSpline(Params2, radii, k=3, s=1e-3)
        curvePM = UnivariateSpline(Params2, masses, k=3, s=1e-3)
        derivPR = curvePR.derivative(1)
        derivPM = curvePM.derivative(1)
        back[right] = numpy.dstack([rhocM, rhocR, derivPM(Params2), derivPR(Params2)])
        return back[:,0:2], back[:,2], back[:,3]

        
def calculate_jac(Parameters, ParamUnique, MRIcurves, rhoc):
    
    Jac = numpy.zeros((len(Parameters), 9))
    Jac.fill(numpy.nan)
    for i in range(len(ParamUnique)):
        P1 = ParamUnique[i][0]
        P2 = ParamUnique[i][1]
        P3 = ParamUnique[i][2]
        Pvalues = [P1, P2, P3]
        combis = [[1, 2], [0, 2], [0, 1]]

        for j in range(3):
            h1, h2 = combis[j]
            change = numpy.where((Parameters[:,h1]==Pvalues[h1]) & (Parameters[:,h2]==Pvalues[h2]))[0]

            if change.size==0:
                continue
            else:
                Ps = numpy.log10(Parameters[change][:,j])
                Jac[:,1:3][change], Jac[:,j+3][change], Jac[:,j+6][change] = calculate_deriv(rhoc, MRIcurves, change, Ps)

    return Jac

info = mp.get_logger().info
def main(Parameters, MRIcurves, rhoc, outputfile):
    logger = mp.log_to_stderr()
    logger.setLevel(logging.INFO)
  
    nproc = mp.cpu_count() - 1
    nproc = max(1, nproc)
       
    div_rhoc = numpy.array_split(rhoc, nproc)
    ParamUnique = UniqueParams(Parameters)
    
    ntasks = nproc
    inputs = [[Parameters, ParamUnique, MRIcurves, div_rhoc[t], t] for t in xrange(ntasks)]
    
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
    result = numpy.delete(result, 1, axis=1)
       
    new_jac = []
    for i in xrange(nproc):
        for j in range(len(result[i][0])):
            new_jac.append(result[i][0][j])
    new_jac = numpy.array(new_jac)
    
    Jacobian = numpy.zeros((len(Parameters), len(rhoc), 9))
    for i in range(len(Parameters)):
        for j in range(len(rhoc)):
            Jacobian[i][j] = new_jac[j][i]
        indices = numpy.invert(numpy.any(numpy.isnan(Jacobian[i][:,1:]), axis=1))
        Jacobian[i][:,0][indices] = rhoc[indices]

    numpy.save(outputfile, Jacobian)


def worker(input_q, output_q):
    
    start = time.clock()  
    while True:
        try:
            tmp = input_q.get()
            if 'STOP' == tmp :
                break
            
            Parameters, ParamUnique, MRIcurves, rhoc, task = tmp
            
            Jacobian = []

            for i, e in enumerate(rhoc):
                #info(e)
                jacpart = calculate_jac(Parameters, ParamUnique, MRIcurves, e)
                Jacobian.append(jacpart)           
            
            output_q.put([Jacobian, task])
            
        except Exception as exception:
            trace = str(traceback.format_exc())
            info(trace)
    end = (time.clock() - start)
    info(end)
    
    return    
 
    

if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument("-f", dest="outputFile", help="write jacobian to FILE", metavar="FILE", required=True)
    parser.add_argument("-i1", dest="inputMRIcurves", help="use as input MRIcurves", required=True)                        
    parser.add_argument("-i2", dest="inputParams", help="use as input Parameters", required=True)                     
    args = parser.parse_args()

    Parameters = numpy.load(args.inputParams)
    MRIcurves = numpy.load(args.inputMRIcurves)
    rhoc = numpy.linspace(14.3, 16., 40)

    print len(Parameters), len(MRIcurves)
        
    main(Parameters, MRIcurves, rhoc, args.outputFile)













    
    
