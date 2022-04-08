from src.FDMsim import NLOE2D_sim
from src.FDManalysis import NLOE2D_analysis
import numpy as np
import multiprocessing
import itertools,os,pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm

class SimJob():
    'methods for different simulations to run'
    def __init__(self,p):
        self.p=p
    def SingleRun(self,basis ='strain',n=1,cores=1):
        'run a single simulation'
        sim = NLOE2D_sim(self.p)
        self.p['savefolder'] += self.p['subfolder']
        if basis=='strain':
            sim.StrainPDE(self.p)    
        elif basis=='displacement':
            sim.DisplacementPDE(self.p)    
        

    def Sweep(self,prange,pname,cores=1,basis ='strain'):
        'Sweep over a n-D parameter range with values prange over parameters pname.'
        'plist is a list of dictionaries containing all permutations of the different parameter ranges'
        def main():
            sim = NLOE2D_sim(self.p)
            plist = sim.get_plist(self.p,pname,prange)
            pool = multiprocessing.Pool(cores)
            if basis=='strain':
                pool.map(sim.StrainPDE, plist)
            elif basis=='displacement':
                pool.map(sim.DisplacementPDE, plist)
            pool.close()
        if __name__ ==  '__main__':
            main()


    
        
class AnalyzeJob():
    def __init__(self,p):
        self.p=p
        pass
    def animate(self):
        A  = NLOE2D_analysis(self.p['savefolder']+self.p['subfolder']+self.p['savefile'])
        A.animate_fields('output/',self.p['savefile'])
        
    def track_defects(self):
        self.A  = NLOE2D_analysis(self.p['savefolder']+self.p['subfolder']+self.p['savefile'])
        self.A.compute_qties()

    def analyze_defects(self):
        folder = self.p['savefolder'] + self.p['subfolder']
        directory = os.fsencode(folder)
        lst =[ os.fsdecode(i) for i in os.listdir(directory)]
        if '.DS_Store' in lst: lst.remove('.DS_Store')
        
        NdMean=[]        
        for nf,f in enumerate(lst):    
            with open(self.p['savefolder'] + self.p['subfolder']+f, 'rb') as output:
                p=pickle.load(output)
                if p['times']!=101:
                    pass
                else:
                    t,x,y=p['defects']
                    Ndt= np.zeros(p['times'])
                    Nd = np.bincount(t)
                    Nd =Nd[Nd!=0]
                    times = np.unique(t)
                    Ndt[times] = Nd 
                    NdMean.append(Ndt)
                
                
        NdMean=np.mean(np.asarray(NdMean),axis=0)
        return NdMean
        
        


        
    
parameters = {'tf':1., #total time
              'pt':0.01,#print interval
              'dt':0.05e-4,#integration step
              'alpha':600,#model parameter
              'Lx':1,#boxsize
              'Ly':1,#boxsize
              'Nx':200,#spatial discretization
              'Ny':200,#spatial discretization
              'nx':4,#initial condition mode (sine wave)
              'ny':4,#initial condition mode (sine wave)
              'amp':1e-3,#initial field amplitude
              'BC':[False,False], #Boundary conditions in x and y directions, True = periodical 
              'BCtype': 'auto_periodic_neumann',
              'NL':'passive_cubic',#NL type: choose 'active_bilinear' or 'passive_cubic'
              'IC':'ran',#initial conditions: 'ran' for random, 'sin', for sinusoidal
              'savefolder':'datasets/strain dataset 1/',#folder where data is stored
              'savefile':'strain 0', #filename of 
              'subfolder': 'alpha=600/',
              'data_output':'all'
              }




def animate():
    AJ = AnalyzeJob(parameters)
    AJ.animate()
    



def plot_defect_statistics():
    folder = parameters['savefolder']
    directory = os.fsencode(folder)
    print(directory)
    lst =[ os.fsdecode(i) for i in os.listdir(directory)]
    
    if '.DS_Store' in lst: lst.remove('.DS_Store')
    print(lst)
    lst = sorted(lst, key=lambda x: float(x.split('=')[1]))
    
    Nd_alpha=[]
    fig,ax0= plt.subplots(1,figsize=(6,6),dpi=100)
    axes =np.asarray([ax0])
    colors = cm.rainbow(np.linspace(0, 1, len(lst)))
    for n,l in enumerate(lst):
        N_NESS=0
        parameters['subfolder'] = l+'/'
        AJ = AnalyzeJob(parameters)

        NdMean = AJ.analyze_defects()
        times = np.arange(len(NdMean))
        alpha =float(l.split('=')[1])
        times=times*AJ.p['pt'] * alpha
        NdMean = (NdMean-N_NESS)/alpha
        
        ax0.scatter(times,NdMean,s=1,label='$\\'+l+'$',color=colors[n])
        plt.suptitle(f'20 runs,$L=1$')
        ax0.set_ylabel('$\\frac{ N(t)}{\\alpha}$')
        for i in axes.flatten():
            i.set_xscale('log')
            i.set_yscale('log')
            # i.set_xlim([5e0,1e3])
            i.set_xlabel('$\\alpha t$')
    tline = np.logspace(1.5,2.5,100)
    ax0.plot(tline,5e3*tline**(-5/2),label='$\propto t^{-5/2}$',lw=2,c='k')
    ax0.legend(markerscale=5)
    plt.tight_layout()
    plt.savefig(parameters['savefolder'] + 'test.pdf')
    plt.show()

        
def sim(basis):
    SJ = SimJob(parameters)
    SJ.SingleRun(basis=basis, n=2, cores=1)

def sweeper(basis,n,cores=1):
    SJ = SimJob(parameters)
    prange = [parameters['savefile']+f' {i}' for i in np.arange(n)]
    pname=['savefile']
    SJ.Sweep(prange,pname,cores=cores,basis =basis)
    


SJ = SimJob(parameters)
# sim(basis='displacement')
# sweeper(basis='strain',n=2,cores=1)

plot_defect_statistics()
# animate()