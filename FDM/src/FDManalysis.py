from re import S
import numpy as np
import os,sys
import pickle as pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from scipy import signal
from pde import ScalarField,VectorField
class NLOE2D_analysis():
    def __init__(self,path,plotparams={}):
        self.timeseries={}
        self.fielddata={}
        if type(path) is dict: # calculate timeseries to save with raw data
            self.p = path
            self.compute_qties()
        else:                  # calculate field data for immediate plotting
            try:
                self.load(path) #single file
            except IsADirectoryError:
                self.p={}       #folder?
            self.p = {**self.p, **plotparams}
            
    def load(self,path):
        with open(path, 'rb') as output:
            self.p = pickle.load(output)

    def get_defect_statistics(self,folder):
        directory = os.fsencode(folder)
        lst =[ os.fsdecode(i) for i in os.listdir(directory)]
        if '.DS_Store' in lst: lst.remove('.DS_Store')
        NdMean=[]
        
        for nf,f in enumerate(lst):    
            with open(folder+f, 'rb') as output:
                self.p=pickle.load(output)
                t,x,y=self.p['defects']
                Ndt= np.zeros(self.p['frames'])
                Nd = np.bincount(t)
                Nd =Nd[Nd!=0]
                times = np.unique(t)
                Ndt[times] = Nd 

                NdMean.append(Ndt)
        self.NdMean=np.mean(np.asarray(NdMean),axis=0)

        return self.NdMean

    def track_defects(self,th):
        nbMat = np.zeros((np.shape(th)[0],8,self.p['Nx'],self.p['Ny'])) 
        nb = np.asarray([[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]]) # 3x3 window
        for n,i in enumerate(nb):
            nbMat[:,n] = np.roll(np.roll(th,i[0],axis=1),i[1],axis=2)
        nbMat = nbMat[:,:,1:-1,1:-1]
        difMat = (nbMat - np.roll(nbMat,1,axis=1))%(2*np.pi)
        defectMat = np.sum(difMat,axis=1)
        defectMat += - 8*np.pi
        defectMat[np.abs(defectMat)<6*np.pi]=0        
        defects = np.nonzero(defectMat)
        charges= defectMat[defectMat != 0]
        self.fielddata['defectfield'] = defectMat/(2*np.pi)
        self.timeseries['defects'] = defects
        self.timeseries['charges'] = charges

        t,x,y=defects
        N_t= np.zeros(self.timeseries['frames'])
        Q_t= np.zeros(self.timeseries['frames'])
        Nd = np.bincount(t)
        Nd =Nd[Nd!=0]
        times,idx = np.unique(t,return_index=True)
        N_t[times] = Nd 
        Q_t = np.sum(defectMat , axis=(1,2))


        self.timeseries['N_t'] = N_t
        self.timeseries['Q_t'] = Q_t


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

            Length = len(u)
            lapu=np.empty([Length,self.p["Nx"],self.p["Ny"]],dtype=complex) # scalar field (time, y, x), as np is row/column!
            lapv=np.empty([Length,self.p["Nx"],self.p["Ny"]],dtype=complex) # scalar field (time, y, x), as np is row/column!
            lapu3 = np.empty([Length,self.p["Nx"],self.p["Ny"]],dtype=complex) # scalar field (time, y, x), as np is row/column!
            u3 = np.abs(u)**2*u
            for i in range(len(u)):
                u_scalar=ScalarField(self.p['grid'],u[i,:,:])
                v_scalar=ScalarField(self.p['grid'],v[i,:,:])
                u3_scalar = ScalarField(self.p['grid'],np.abs(u[i,:,:])**2*u[i,:,:])
                lapu[i,:,:] = u_scalar.laplace(self.p['BCtype']).data
                lapv[i,:,:] = v_scalar.laplace(self.p['BCtype']).data
                lapu3[i,:,:] = u3_scalar.laplace(self.p['BCtype']).data
            inertia = np.abs((1+1j*self.p['alpha']) * lapu + lapv + lapu3)

            self.fielddata['abslapu'] = np.abs(lapu)
            self.fielddata['abslapv'] = np.abs(lapv)
            self.fielddata['abslap(phi|phi|^2)'] = np.abs(lapu3)
            self.fielddata['inertia'] = np.abs(inertia)
        
        
        self.fielddata['argu'] = np.angle(u)
        self.fielddata['argv'] = np.angle(v)
        self.fielddata['absu'] = np.abs(u)
        self.fielddata['absv'] = np.abs(v)
        self.fielddata['Re(u)'] = np.real(u)
        self.fielddata['Im(u)'] = np.imag(u)
        self.fielddata['Re(u)'] = np.real(v)
        self.fielddata['Re(u)'] = np.imag(v)
        self.fielddata['|u|^2'] = self.fielddata['absu']**2
        
        
        self.timeseries['sumabslapu'] = np.sum(np.abs(lapu),axis=(1,2))
        self.timeseries['sumabslapv'] = np.sum(np.abs(lapv),axis=(1,2))
        self.timeseries['sumabslap(phi|phi|^2)'] = np.sum(np.abs(np.abs(u)**2*u),axis=(1,2))
        self.timeseries['suminertia'] = np.sum(np.abs(inertia),axis=(1,2))
        self.timeseries['sumoddterm']= np.sum(np.abs(1j*self.p['alpha'] *lapu),axis=(1,2))


#         corrtime = np.linspace(0,len(u)-1,6).astype(int)
#         corr = np.zeros([len(corrtime),np.shape(u)[1],np.shape(u)[2]],dtype=complex)
#         for n,t in enumerate(corrtime):
#             for x in range(self.p['Nx']):
#                 for y in range(self.p['Ny']):
#                     corr[n,x,y] = np.average(signal.correlate(u[t,:,:],np.roll(u[t,:,:],(x,y))),axis=(0,1))
# 
#         self.timeseries['correlation'] = corr        


        self.timeseries['frames'] = len(u)
        self.timeseries['momentum'] = np.abs(np.sum(u,axis=(1,2)))**2
        self.timeseries['energy'] = np.sum(self.fielddata['|u|^2'],axis=(1,2))
        self.timeseries['error'] = self.timeseries['momentum']/self.timeseries['energy']
        self.timeseries['times'] = np.arange(self.timeseries['frames'])*self.p['pt']

        if self.p['basis'] == 'strain':
            self.fielddata['defectfield'] = np.angle(np.sqrt((self.fielddata['Re(u)']+1j*self.fielddata['Im(u)'])/self.fielddata['absu']))%np.pi
        elif self.p['basis'] == 'displacement':
            self.fielddata['defectfield'] = self.fielddata['argu']
        print('tracking defects')
        self.track_defects(th=self.fielddata['defectfield'])                
        
        
        
    def get_frame(self,f=-1,save=False):
        self.compute_qties()
        self.FieldsToPlot = [self.fielddata[f] for f in self.p['Fields to Plot']]
        self.SeriesToPlot = [self.timeseries[i] for i in self.p['Timeseries to Plot']]        
            
        plt.rcParams.update({'font.size': 8})
        self.fig,[fax1,fax2,sax1,sax2] = plt.subplots(4,4,figsize=(14,14),dpi=100) 
        fax = np.asarray([fax1,fax2]).flatten()
        sax = np.asarray([sax1,sax2]).flatten() 
        # self.ims = [ax.imshow(self.FieldsToPlot[n][f],cmap = cmap,aspect='equal') for n,[ax,cmap] in enumerate(zip(axes[0].flatten(),self.p['Colormaps']))]
        self.ims = []
        
        for n,[cmap,field,ax,label] in enumerate(zip(self.p['Colormaps'],self.FieldsToPlot,fax,self.p['Fields to Plot'])):
            if n==0:
                self.ims.append(ax.imshow(field[f],cmap=cmap,aspect='equal',vmin=-np.pi,vmax=np.pi))
            elif cmap == 'viridis':
                self.ims.append(ax.imshow(field[f],cmap=cmap,aspect='equal',vmin=np.min(field[f]),vmax=np.max(field[f])))
            elif cmap == 'RdBu':
                self.ims.append(ax.imshow(field[f],cmap=cmap,aspect='equal',vmin=np.min(field),vmax=np.max(field)))
            ax.set_title(label)
            plt.colorbar(self.ims[n],ax=ax)
        
        t=self.timeseries['times']
        self.scats=[]
        self.scattot=[]
        sax[-1].set_yscale('log')
        for n,[a,tit,series,scale] in enumerate(zip(sax,self.p['Timeseries to Plot'],self.SeriesToPlot,self.p['Timeseries scale'])):
            self.scats.append(a.scatter(t[:3],series[:3],s=5))
            a.set_xlim([self.p['pt'],np.max(t[:5])])
            a.set_title(tit)
            a.set_xscale(scale[0])
            a.set_yscale(scale[1]) 
            if n>1:
                self.scattot.append(sax[-1].scatter(t[:3],series[:3],s=5,label=tit))
        sax[-1].legend()
        
        
            
            

        self.fig.tight_layout()
        self.fig.suptitle(f'{self.p["NL"]} NL \n'+f'($L_x={self.p ["Lx"]},L_y={self.p["Ly"]},N_x={self.p["Nx"]},N_y={self.p["Ny"]}, \\alpha={self.p["alpha"]}$)')    
        self.axes=[fax,sax]
    
    def AnimateFields(self,savefolder,savefile):
        
        
        if self.p['output_data'] == 'defects':
            print('dataset without field data, cannot generate animation. Set data_output to "all" ')
        else:
            self.get_frame(f=0)
            fax,sax=self.axes
            
            def update(i):
                print(i)
                j=0
                for n,[ax,im,field] in enumerate(zip(fax,self.ims,self.FieldsToPlot)):
                    f = field[i]
                    im.set_array(f)
                    if n>1:
                        vmin = np.min(f)
                        vmax = np.max(f)
                        im.set_clim(vmin, vmax)
                if i>5:
                    t = self.timeseries['times'][:i]
                    lims=[self.p['pt'],self.p['pt'],0,0]
                    for n,serie in enumerate(self.SeriesToPlot):
                        verts = np.transpose([t,serie[:i]])
                        self.scats[n].set_offsets(verts)
                        sax[n].set_xlim([self.p['pt'],np.max(t)])

                        sax[n].set_ylim([np.min(serie[:i]),np.max(serie[:i])])

                        if n>1:    
                            lims=np.maximum([np.min(t),np.max(t),np.min(serie[:i]),np.max(serie[:i])],lims)
                            verts = np.transpose([t,serie[:i]])
                            self.scattot[j].set_offsets(verts)
                            j+=1
                    
                    sax[-1].set_xlim(lims[0],lims[1])
                    sax[-1].set_ylim(lims[2],lims[3])
                            
                    
                    
            anim = animation.FuncAnimation(self.fig, update, 
                            frames=self.timeseries['frames'], interval=40, blit=False)
            if not os.path.exists(savefolder):
                os.makedirs(savefolder)
            anim.save(savefolder+savefile+' - animation.mp4')    
        
    def PlotTimeseries(self,savefolder,savefile):

        fig,axes = plt.subplots(2,3,figsize=(18,6))
        ax = axes.flatten()
        t = self.timeseries['times']
        ax[0].scatter(t,self.timeseries['momentum'])
        ax[1].scatter(t,self.timeseries['energy'])
        ax[2].scatter(t,self.timeseries['error'])
        ax[3].scatter(t,self.timeseries['N_t'])
        ax[4].scatter(t,self.timeseries['Q_t'])
        ax[3].set_xscale('log')
        for i in ax[:4]:
            i.set_yscale('log')
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)

        plt.savefig(savefolder+savefile+' - timeseries.pdf')
    
    
    def PlotCorrelation(self,savefolder,savefile):
        fig,axes = plt.subplots(2,3,figsize=(18,6))
        ax = axes.flatten()
        t = self.timeseries['times']
        cmap = 'viridis'
        for n,i in enumerate(ax):
            i.imshow(np.abs(self.timeseries['correlation'][n]),cmap=cmap,aspect='equal',origin='lower')
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
        plt.savefig(savefolder+savefile+' - correlations.pdf')
        
    def PlotDefectStatistics(self,loadfolder,savefolder,savefile):
        directory = os.fsencode(loadfolder)

        lst =[ os.fsdecode(i) for i in os.listdir(directory)]
        if '.DS_Store' in lst: lst.remove('.DS_Store')
        fig,ax0= plt.subplots(1,figsize=(6,6),dpi=100)
        axes =np.asarray([ax0])
        colors = cm.rainbow(np.linspace(0, 1, len(lst)))
        for n,l in enumerate(lst):
            N_NESS=0
            subfolder = loadfolder + l +'/'
            
            NdMean = self.get_defect_statistics(subfolder)

            alpha =float(l.split('=')[1])
            timestamps=np.arange(self.p['frames'])*self.p['pt'] * alpha
            NdMean = (NdMean-N_NESS)/alpha       
            ax0.scatter(timestamps,NdMean,s=1,label='$\\'+l+'$',color=colors[n])
            plt.suptitle(f'{len(lst)} runs,$Lx={self.p["Lx"]}$, $Ly={self.p["Ly"]}$')
            ax0.set_ylabel('$\\frac{ N(t)}{\\alpha}$')
            for i in axes.flatten():
                i.set_xscale('log')
                i.set_yscale('log')
                # i.set_xlim([5e0,1e3])
                i.set_xlabel('$\\alpha t$')
        # plot powerlaw:
        # tline = np.logspace(1.5,2.5,100)
        # ax0.plot(tline,5e3*tline**(-5/2),label='$\propto t^{-5/2}$',lw=2,c='k')
        # ax0.legend(markerscale=5)
        plt.tight_layout()
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
        plt.savefig(savefolder+savefile+' - defects.pdf')

