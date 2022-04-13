from src.FDMsim import NLOE2D_sim
from src.FDManalysis import NLOE2D_analysis
import numpy as np
import multiprocessing
import itertools,os,pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
      
    
parameters = {'tf':1., #total time
              'pt':0.01,#print interval
              'dt':0.1e-4,#integration step
              'alpha':600,#model parameter
              'Lx':1,#boxsize
              'Ly':1,#boxsize
              'Nx':20,#spatial discretization
              'Ny':20,#spatial discretization
              'nx':4,#initial condition mode (sine wave)
              'ny':4,#initial condition mode (sine wave)
              'amp':1e-3,#initial field amplitude
              'BC':[False,False], #Boundary conditions in x and y directions, True = periodical 
              'BCtype': 'auto_periodic_neumann',
              'NL':'passive_cubic',#NL type: choose 'active_bilinear' or 'passive_cubic'
              'IC':'ran',#initial conditions: 'ran' for random, 'sin', for sinusoidal
              'savefolder':'datasets/run1/',#folder for 
              'subfolder': 'alpha=600/',#subfolder parameter value
              'savefile':'test', #filename of pickle data file
              'data_output':'all',#'all' for full timeseries, 'defects' for defect data only
              'basis': 'strain', #'dislacement' or 'strain'
              'Fields to Plot': ['argu','defectfield','absu'],
              'Colormaps': ['hsv','RdBu','viridis']
              }


def SingleRun(parameters):
    sim = NLOE2D_sim(parameters)
    sim.runsim()

def RepeatRun(paramlist,cores=1):    
    def main():
        pool = multiprocessing.Pool(cores)
        pool.map(SingleRun, paramlist)
        pool.close()
    if __name__ ==  '__main__':
        main()
        
def GetParamList(pname,prange,n):#make a list of parameter dicts for parameter sweep    
    paramlist = []
    for val in prange:
        for run in range(n):
            p = parameters.copy()
            p['subfolder'] = f'{pname} = {val}'
            p[pname] = val
            p['savefile'] = f'run {run}'
            paramlist.append(p)
    return paramlist


def animate(parameters):
    filename = parameters['savefolder']+parameters['subfolder']+parameters['basis'] + ' - ' + parameters['savefile']
    A  = NLOE2D_analysis(filename)
    A.animate_fields('output/',parameters['savefile'])

# Run a single simulation based on parameters:
SingleRun(parameters) 


# Run multiple simulations over a range of parameters
# pname = 'alpha' #parameter to sweep over
# prange = [600]  #range to sweep over
# n=4             #number of runs
# paramlist = GetParamList(pname,prange,n) #get list of dicts with parameters
# RepeatRun(paramlist,cores=2)


animate(parameters)



