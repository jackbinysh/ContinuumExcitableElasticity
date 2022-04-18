from re import S
import numpy as np
import os,sys
import pickle as pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm

class NLOE2D_analysis():
    def __init__(self,path):
        self.timeseries={}
        self.fielddata={}
        if type(path) is dict:
            self.p = path
            self.compute_qties()
        else:
            try:
                self.load(path)
            except IsADirectoryError:
                self.p={}

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
                # if self.p['output_data'] == 'all':
                #     self.compute_qties
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
            
        
        self.fielddata['argu'] = np.angle(u)
        self.fielddata['argv'] = np.angle(v)
        self.fielddata['absu'] = np.abs(u)
        self.fielddata['absv'] = np.abs(v)
        self.fielddata['Re(u)'] = np.real(u)
        self.fielddata['Im(u)'] = np.imag(u)
        self.fielddata['Re(u)'] = np.real(v)
        self.fielddata['Re(u)'] = np.imag(v)
        self.fielddata['|u|^2'] = self.fielddata['absu']**2
        
        self.timeseries['frames'] = len(u)
        self.timeseries['momentum'] = np.abs(np.sum(u,axis=(1,2)))**2
        self.timeseries['energy'] = np.sum(self.fielddata['|u|^2'],axis=(1,2))
        self.timeseries['error'] = self.timeseries['momentum']/self.timeseries['energy']
        self.timeseries['times'] = np.arange(self.timeseries['frames'])*self.p['pt']
        
        if self.p['basis'] == 'strain':
            self.fielddata['defectfield'] = np.angle(np.sqrt((self.fielddata['Re(u)']+1j*self.fielddata['Im(u)'])/self.fielddata['absu']))%np.pi
        elif self.p['basis'] == 'displacement':
            self.fielddata['defectfield'] = self.fielddata['argu']
            
        self.track_defects(th=self.fielddata['defectfield'])                
        self.FieldsToPlot = [self.fielddata[f] for f in self.p['Fields to Plot']]   #   [self.u_angle,self.defectMat,self.u_mag]
        
    def get_frame(self,f=-1,save=False):
        self.compute_qties()
        plt.rcParams.update({'font.size': 8})
        self.fig,axes = plt.subplots(1,3,figsize=(12,4),dpi=100)  
        self.ims = [ax.imshow(self.FieldsToPlot[n][f],cmap = cmap,aspect='equal') for n,[ax,cmap] in enumerate(zip(axes.flatten(),self.p['Colormaps']))]
        plt.suptitle(f'{self.p["NL"]} NL \n'+f'($L_x={self.p ["Lx"]},L_y={self.p["Ly"]},N_x={self.p["Nx"]},N_y={self.p["Ny"]}, \\alpha={self.p["alpha"]}$)')    
        self.ims = []
        for n,[cmap,field,ax,label] in enumerate(zip(self.p['Colormaps'],self.FieldsToPlot,axes.flatten(),self.p['Fields to Plot'])):    
            if n==0:
                self.ims.append(ax.imshow(field[f],cmap=cmap,aspect='equal',vmin=-np.pi,vmax=np.pi))
            elif cmap == 'viridis':
                self.ims.append(ax.imshow(field[f],cmap=cmap,aspect='equal',vmin=np.min(field[f]),vmax=np.max(field[f])))
            elif cmap == 'RdBu':
                self.ims.append(ax.imshow(field[f],cmap=cmap,aspect='equal',vmin=np.min(field),vmax=np.max(field)))
            ax.set_title(label)
            plt.colorbar(self.ims[n],ax=ax)
        plt.tight_layout()
        self.axes=axes
    
    def AnimateFields(self,savefolder,savefile):
        if self.p['output_data'] == 'defects':
            print('dataset without field data, cannot generate animation. Set data_output to "all" ')
        else:
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
                            frames=self.timeseries['frames'], interval=40, blit=False)
            if not os.path.exists(savefolder):
                os.makedirs(savefolder)
            anim.save(savefolder+savefile+' - animation.mp4')    
        
    def PlotTimeseries(self,savefolder,savefile):
        # self.compute_qties()
        fig,ax = plt.subplots(1,3,figsize=(18,6))
        t = self.p['times']
        ax[0].scatter(t,self.p['momentum'])
        ax[1].scatter(t,self.p['energy'])
        ax[2].scatter(t,self.p['error'])
        for i in ax:
            i.set_yscale('log')
        if not os.path.exists(savefolder):
            os.makedirs(savefolder)
        plt.savefig(savefolder+savefile+' - timeseries.pdf')
    
    
        
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

