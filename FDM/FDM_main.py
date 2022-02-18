from src.FDMsim import NLOE2D_sim
from src.FDManalysis import NLOE2D_animate
import numpy as np
import multiprocessing
import itertools

class Job():
    'methods for different simulations to run'
    def __init__(self,p):
        self.p=p
    def SingleRun(self,basis ='strain'):
        'run a single simulation'
        sim = NLOE2D_sim(self.p)
        if basis=='strain':
            sim.StrainPDE()
        elif basis=='displacement':
            sim.DisplacementPDE()
        anim  = NLOE2D_animate(self.p['savefolder']+self.p['savefile'])
        anim.animate_fields('output/',self.p['savefile'])
        anim.get_final_state()

    def Sweep(self,prange,pname,cores=1,subfolder='test',basis ='strain'):
        'Sweep over a n-D parameter range with values prange over parameters pname.'
        'plist is a list of dictionaries containing all permutations of the different parameter ranges'
        def main():
            self.p['savefolder'] += subfolder
            sim = NLOE2D_sim(self.p)
            if basis=='strain':
                eq = sim.Strain()
            elif basis=='displacement':
                eq = sim.Displacement()
            
            pval = list(itertools.product(*prange))
            plist = sim.get_plist(self.p,pname,pval,self.p['savefolder'])
            pool = multiprocessing.Pool(cores)
            pool.map(eq.runsim, plist)
            pool.close()
        if __name__ ==  '__main__':
            main()

parameters = {'tf':50, #total time
              'pt':0.5,#print interval
              'dt':0.5e-4,#integration step
              'alpha':10,#model parameter
              'Lx':1,#boxsize
              'Ly':1.25,#boxsize
              'Nx':40,#spatial discretization
              'Ny':50,#spatial discretization
              'nx':4,#initial condition mode (sine wave)
              'ny':4,#initial condition mode (sine wave)
              'amp':1e-3,#initial field amplitude
              'BC':[True,True], #Boundary conditions in x and y directions, True = periodical 
              'BCtype': 'auto_periodic_neumann',
              'NL':'passive_cubic',#NL type: choose 'active_bilinear' or 'passive_cubic'
              'IC':'ran',#initial conditions: 'ran' for random, 'sin', for sinusoidal
              'savefolder':'data/',#folder where data is stored
              'savefile':'test' #filename of data
              }



#single run
# parameters['savefile'] = 'disp'
# J = Job(parameters)
# J.SingleRun(basis ='displacement')


for i in np.arange(4):
    parameters['savefile'] = f'strain {i}' 
    J = Job(parameters)
    J.SingleRun(basis ='strain')

# 

#parameter sweep
sweep=False
if sweep:
    range1=np.linspace(0,1,2)
    range2=np.linspace(0,1,2)
    prange = [range1,range2]
    pname = ['alpha','amp']
    J.Sweep(prange=prange,pname=pname,cores=1,subfolder = 'test/',basis ='strain')