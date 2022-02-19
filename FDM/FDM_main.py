from src.FDMsim import NLOE2D_sim
from src.FDManalysis import NLOE2D_analysis
import numpy as np
import multiprocessing
import itertools,os,pickle

class SimJob():
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


    
        
class AnalyzeJob():
    def __init__(self,p):
        self.p=p
        pass
    def animate(self):
        A  = NLOE2D_analysis(self.p['savefolder']+self.p['savefile'])
        A.animate_fields('output/',self.p['savefile'])
        A.get_final_state()
        
    def track_defects(self):
        A  = NLOE2D_analysis(self.p['savefolder']+self.p['subfolder']+self.p['savefile'])
        A.compute_qties()

    def analyze_sweep(self):
        folder = self.p['savefolder'] + self.p['subfolder']
        directory = os.fsencode(folder)
        print(directory)
        lst =[ os.fsdecode(i) for i in os.listdir(directory)]
        if '.DS_Store' in lst: lst.remove('.DS_Store')
        defects=[]
        alpha = []
        for nf,f in enumerate(lst):
            self.p['savefile'] = f
            self.track_defects()
            defects.append(A.defects)
            alpha.append(A.p['alpha'])
        defects = np.mean(np.asarray(defects),axis=0)



parameters = {'tf':0.5, #total time
              'pt':0.0005,#print interval
              'dt':0.1e-4,#integration step
              'alpha':300,#model parameter
              'Lx':1,#boxsize
              'Ly':1,#boxsize
              'Nx':100,#spatial discretization
              'Ny':100,#spatial discretization
              'nx':4,#initial condition mode (sine wave)
              'ny':4,#initial condition mode (sine wave)
              'amp':1e-3,#initial field amplitude
              'BC':[False,False], #Boundary conditions in x and y directions, True = periodical 
              'BCtype': 'auto_periodic_neumann',
              'NL':'passive_cubic',#NL type: choose 'active_bilinear' or 'passive_cubic'
              'IC':'ran',#initial conditions: 'ran' for random, 'sin', for sinusoidal
              'savefolder':'data/',#folder where data is stored
              'savefile':'test', #filename of 
              'subfolder': 'defects/'
              
              }



#single run
# parameters['savefile'] = 'disp'
# J = Job(parameters)
# J.SingleRun(basis ='displacement')


for i in np.arange(10):
    parameters['savefile'] = f'strain {i} alpha = {parameters["alpha"]}' 
    SJ = SimJob(parameters)
    
    SJ.SingleRun(basis ='strain')
    # J.animate()
    # J.track_defects()
    # AJ = AnalyzeJob(parameters)
    # AJ.analyze_sweep()



# 

#parameter sweep
sweep=False
if sweep:
    range1=np.linspace(0,1,2)
    range2=np.linspace(0,1,2)
    prange = [range1,range2]
    pname = ['alpha','amp']
    SJ.Sweep(prange=prange,pname=pname,cores=1,subfolder = 'test/',basis ='strain')