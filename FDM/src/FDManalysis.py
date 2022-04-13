from re import S
import numpy as np
import os,sys
import pickle as pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

class NLOE2D_analysis():
    def __init__(self,path):
        self.results={}
        if type(path) is dict:
            self.p = path
            self.compute_qties()
        else:
            self.load(path)
            
    def load(self,path):
        
        with open(path, 'rb') as output:
            print(output)
            self.p = pickle.load(output)

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
                    t,x,y=self.results['defects']
                    Ndt= np.zeros(self.p['times'])
                    Nd = np.bincount(t)
                    Nd =Nd[Nd!=0]
                    times = np.unique(t)
                    Ndt[times] = Nd 
                    NdMean.append(Ndt)
        self.NdMean=np.mean(np.asarray(NdMean),axis=0)


    def track_defects(self,th):
        nbMat = np.zeros((np.shape(th)[0],8,self.p['Nx'],self.p['Ny'])) 
        nb = np.asarray([[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]]) # 3x3 window
        
        for n,i in enumerate(nb):
            nbMat[:,n] = np.roll(np.roll(th,i[0],axis=1),i[1],axis=2)
        nbMat = nbMat[:,:,1:-1,1:-1]
        difMat = (nbMat - np.roll(nbMat,1,axis=1))%(2*np.pi)
        defectMat = np.sum(difMat,axis=1)
        defectMat += - 8*np.pi
        defectMat[np.abs(defectMat)<5*np.pi]=0        
        defects = np.nonzero(defectMat)
        charges= defectMat[defectMat != 0]
        self.results['defectfield'] = defectMat/(2*np.pi)
        self.results['defects'] = defects
        self.results['charges'] = charges

    def compute_qties(self):
        
        if self.p['basis'] == 'displacement':
            print('displacement')
            ux,uy=self.p['u']
            vx,vy=self.p['v']
            u=ux+1j*uy
            v=vx+1j*vy
        if self.p['basis'] == 'strain':
            print('strain')
            u = self.p['u']
            v = self.p['v']
            
        self.results['frames'] = len(u)
        self.results['argu'] = np.angle(u)
        self.results['argv'] = np.angle(v)
        self.results['absu'] = np.abs(u)
        self.results['absv'] = np.abs(v)
        self.results['Re(u)'] = np.real(u)
        self.results['Im(u)'] = np.imag(u)
        self.results['Re(u)'] = np.real(v)
        self.results['Re(u)'] = np.imag(v)
        self.results['|u|^2'] = self.results['absu']**2
        self.results['momentum'] = np.abs(np.sum(u,axis=(1,2)))**2
        self.results['energy'] = np.sum(self.results['|u|^2'],axis=(1,2))
        self.results['error'] = self.results['momentum']/self.results['energy']
        self.results['defectfield'] = self.results['argu']
        
        if self.p['basis'] == 'strain':
            self.results['defectfield'] = np.angle(np.sqrt((self.results['Re(u)']+1j*self.results['Im(u)'])/self.results['absu']))%np.pi
        elif self.p['basis'] == 'strain':
            self.results['defectfield'] = self.results['argu']
            
        self.track_defects(th=self.results['defectfield'])                
        self.times = np.arange(self.results['frames'])*self.p['pt']
        self.FieldsToPlot = [self.results[f] for f in self.p['Fields to Plot']]   #   [self.u_angle,self.defectMat,self.u_mag]
        

            
    def get_frame(self,f=-1,save=False):
        self.compute_qties()
        plt.rcParams.update({'font.size': 8})
        self.fig,axes = plt.subplots(1,3,figsize=(12,4),dpi=100)  
        self.ims = [ax.imshow(self.FieldsToPlot[n][f],cmap = cmap,aspect='equal') for n,[ax,cmap] in enumerate(zip(axes.flatten(),self.p['Colormaps']))]
        plt.suptitle(f'{self.p["NL"]} NL \n'+f'($L_x={self.p ["Lx"]},L_y={self.p["Ly"]},N_x={self.p["Nx"]},N_y={self.p["Ny"]}, \\alpha={self.p["alpha"]}$)')    
        self.ims = []
        for n,[cmap,field,ax,label] in enumerate(zip(self.p['Colormaps'],self.FieldsToPlot,axes.flatten(),['$\\phi$', 'defects','mag'])):    
            if n==0:
                self.ims.append(ax.imshow(field[f],cmap=cmap,aspect='equal',vmin=-np.pi,vmax=np.pi))
            # if n==0:
            #     self.ims.append(ax.imshow(field[f],cmap=cmap,aspect='equal',vmin=0,vmax=np.pi))
            elif cmap == 'viridis':
                self.ims.append(ax.imshow(field[f],cmap=cmap,aspect='equal',vmin=np.min(field[f]),vmax=np.max(field[f])))
            elif cmap == 'RdBu':
                self.ims.append(ax.imshow(field[f],cmap=cmap,aspect='equal',vmin=np.min(field),vmax=np.max(field)))
            ax.set_title(label)
            plt.colorbar(self.ims[n],ax=ax)
        if save:
            plt.savefig('output/final_state.pdf')
            umag = self.FieldsToPlot[2]
            vmag = self.FieldsToPlot[3]
            fig,[ax,bx] = plt.subplots(2,2,figsize=(12,12),dpi=100)
            
            ax[0].scatter(np.arange(len(umag[-1,int(self.p['Ny']/2)])), umag[-1,int(self.p['Ny']/2)])
            ax[0].scatter(np.arange(len(vmag[-1,int(self.p['Ny']/2)])), vmag[-1,int(self.p['Ny']/2)])
            ax[1].scatter(np.arange(len(umag)), np.sum(umag,axis=(1,2))+np.sum(vmag,axis=(1,2)))
            phi = self.FieldsToPlot[2] * np.exp(1j*self.FieldsToPlot[0])
            dphi = self.FieldsToPlot[3] * np.exp(1j*self.FieldsToPlot[1])
            self.f={}
            self.f['Po'] =  np.real(dphi)*np.imag(phi) - np.imag(dphi) *np.real(phi) 
            self.f['Pd'] = np.abs(dphi)**2
            self.f['sumPo'] = np.sum(self.f['Po'],axis=(1,2))
            self.f['sumPd'] = np.sum(self.f['Pd'],axis=(1,2))
            bx[0].scatter(np.arange(len(umag)), self.f['sumPo'])
            bx[1].scatter(np.arange(len(umag)), self.f['sumPd'])
            
            plt.savefig('output/cut'+self.p['savefile']+'.pdf')
        plt.tight_layout()
        self.axes=axes
    
    def get_final_state(self):
        self.get_frame(save=True)    
        
    
    def animate_fields(self,folder,filename):
        self.get_frame(f=0)
        def animate(i):
            for n,[ax,im] in enumerate(zip(self.axes.flatten(),self.ims)):
                f = self.FieldsToPlot[n][i]
                im.set_array(f)
                if n>1:
                    vmin = np.min(f)
                    vmax = np.max(f)
                    im.set_clim(vmin, vmax)
        anim = animation.FuncAnimation(self.fig, animate, 
                        frames=self.results['frames'], interval=40, blit=False)
        if not os.path.exists(folder):
            os.makedirs(folder)
        anim.save(folder+filename+'.mp4')    
        
    

    def plot_defect_statistics(self):
        folder = self.p['savefolder']
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
            self.p['subfolder'] = l + '/'
            self.load(self.p['savefolder']+self.p['subfolder']+self.p['savefile'])

            NdMean = self.analyze_defects()
            times = np.arange(len(NdMean))
            alpha =float(l.split('=')[1])
            times=times*self.p['pt'] * alpha
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
        plt.savefig(self.p['savefolder'] + 'test.pdf')
        plt.show()

            