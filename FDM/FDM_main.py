
from src.FDMsim import NLOE2D_sim
from src.FDManalysis import NLOE2D_analysis
import numpy as np
import multiprocessing
import itertools,os,pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
      
    
parameters = {#time
              'tf':50, #total time
              'pt':0.1, #print interval
              'dt':1e-5,#integration step
              
              #system parameters
              'alpha':600,#activity
              'B': 0, #model parameter for displacement formulation: bulk modulus
              'Lx':14,#boxsize
              'Ly':10,#boxsize
              'basis': 'strain', #'dislacement' or 'strain'
              'NL':'passive_cubic',#NL type: choose 'passive_cubic' or 'active_bilinear'(strain only)
              
              #Domain
              'Nx':560,#spatial discretization
              'Ny':400,#spatial discretization
              'BC':[False,False], #Boundary conditions in x and y directions, True = periodical 
              'BCtype': 'auto_periodic_neumann',
              
              #IC
              'IC':'ran',#initial conditions: 'ran' for random, 'sin', for sinusoidal
              'nx':4,#initial condition mode (sine wave)
              'ny':4,#initial condition mode (sine wave)
              'amp':1e-3,#initial field amplitude
              
              #saving/loading/plotting
              'datafolder': 'datasets/',
              'subfolder': 'experiment2/',
              'subsubfolder': 'alpha = 300/',
              
            #   'savefolder':'datasets/run1/',#folder for 
            #   'subfolder': 'alpha = 600/',#subfolder parameter value
              'savefile':'run 1', #filename of pickle data file
              'outputfolder': 'output/',#folder for plots/animations
              'output_data':'all',#'all' for full timeseries, 'defects' for defect data only
}
plotparams=  {'Fields to Plot': ['argu','defectfield','absu','abslapu','abslapv','abs(phi|phi|^2)','inertia'],
              'Colormaps': ['hsv','RdBu','viridis','viridis','viridis','viridis','viridis'],            #colormaps for those fields
              'Timeseries to Plot':['energy','N_t','Q_t','sumabslapu','sumabslapv','sumabs(phi|phi|^2)','suminertia'], #fields to animat
              'Timeseries scale':[['linear','log'],['log','log'],['log','log'],['linear','log'],['linear','log'],['linear','log'],['linear','log']]
              }

class Jobs():
    def __init__(self):
        pass
    def SingleRun(self,parameters):
        sim = NLOE2D_sim(parameters)
        sim.runsim()
    def RepeatRun(self,pname,prange,n=1,cores=1):
        paramlist = self.GetParamList(pname,prange,n) #get list of dicts with parameters
        def main():
            pool = multiprocessing.Pool(cores)
            pool.map(self.SingleRun, paramlist)
            pool.close()
        if __name__ ==  '__main__':
            main()   
    def GetParamList(self,pname,prange,n):#make a list of parameter dicts for parameter sweep    
        paramlist = []
        for val in prange:
            for run in range(n):
                p = parameters.copy()
                p['subfolder'] = f'{pname} = {val}/'
                p[pname] = val
                p['savefile'] = f'run {run}'
                paramlist.append(p)
        return paramlist

# Run a single simulation based on parameters:
J = Jobs()
J.SingleRun(parameters)

# Run multiple simulations over a range of parameters:
# pname = 'alpha' #parameter to sweep over
# prange = [600]  #range to sweep over
# n=1             #number of runs
# J.RepeatRun(pname,prange,n=n,cores=1)




#### analysis/plotting
loadfolder = parameters['datafolder'] + parameters['subfolder']+ parameters['subsubfolder']+parameters['basis'] + ' - ' + parameters['savefile']
savefolder = parameters['outputfolder']+ parameters['subfolder']
savefile = parameters['savefile']


# ana  = NLOE2D_analysis(loadfolder,plotparams=plotparams)
# ana.AnimateFields(savefolder,savefile)




# fig,ax= plt.subplots(1,2)
# ax[0].scatter(t[:-100],Q[:-100])
# ax[1].scatter(t[:-100],N[:-100])
# plt.show()


# ana.PlotTimeseries(savefolder,savefile)
# ana.PlotCorrelation(savefolder,savefile)


# loadfolder = parameters['datafolder'] + parameters['subfolder']
# ana  = NLOE2D_analysis(loadfolder)
# ana.PlotDefectStatistics(loadfolder,savefolder,savefile)



### defect testing
# ana.compute_qties()
# N=ana.timeseries['N_t']
# Q=ana.timeseries['Q_t']
# t=ana.timeseries['times']
# defects= ana.timeseries['defects'] 
# q = ana.timeseries['charges'] 
# print(np.shape(q))
# print(np.shape(defects))


