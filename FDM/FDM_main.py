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
        sim = NLOE2D_sim()
        if basis=='strain':
            eq = sim.Strain(self.p)
        elif basis=='displacement':
            eq = sim.Displacement(self.p)
        eq = sim.Strain(self.p)
        eq = sim.Displacement(self.p)
        eq.runsim()
        anim  = NLOE2D_animate(self.p['savefolder']+self.p['savefile'])
        anim.animate_fields('output/',self.p['savefile'])

    def Sweep(self,prange,pname,cores=1,subfolder='test',basis ='strain'):
        'Sweep over a n-D parameter range with values prange over parameters pname.'
        'plist is a list of dictionaries containing all permutations of the different parameter ranges'
        def main():
            self.p['savefolder'] += subfolder
            sim = NLOE2D_sim()
            if basis=='strain':
                eq = sim.Strain(self.p)
            elif basis=='displacement':
                eq = sim.Displacement(self.p)
            
            pval = list(itertools.product(*prange))
            plist = sim.get_plist(self.p,pname,pval,self.p['savefolder'])
            pool = multiprocessing.Pool(cores)
            pool.map(eq.runsim, plist)
            pool.close()
        if __name__ ==  '__main__':
            main()

parameters = {'tf':1, #total time
              'pt':0.4,#print interval
              'dt':2e-4,#integration step
              'alpha':10,#model parameter
              'Lx':1,#boxsize
              'Ly':1,#boxsize
              'Nx':5,#spatial discretization
              'Ny':5,#spatial discretization
              'nx':4,#initial condition mode (sine wave)
              'ny':4,#initial condition mode (sine wave)
              'amp':1e-2,#initial field amplitude
              'BC':[False,False], #Boundary conditions in x and y directions, True = periodical, False = NEUMANN BC. 
              'BCtype': 'auto_periodic_neumann',
              'NL':'active_bilinear',#NL type: choose 'active_bilinear' or 'passive_cubic'
              'IC':'ran',#initial conditions: 'ran' for random, 'sin', for sinusoidal
              'savefolder':'data/',#folder where data is stored
              'savefile':'test' #filename of data
              }

J = Job(parameters)

#single run
J.SingleRun(parameters,basis ='displacement')


#parameter sweep
sweep=False
if sweep:
    range1=np.linspace(0,1,2)
    range2=np.linspace(0,1,2)
    prange = [range1,range2]
    pname = ['alpha','amp']
    J.Sweep(prange=prange,pname=pname,cores=1,subfolder = 'test/',basis ='strain')