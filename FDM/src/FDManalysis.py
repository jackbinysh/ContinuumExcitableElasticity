from re import S
import numpy as np
import os,sys
import pickle as pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class NLOE2D_analysis():
    def __init__(self,path):
        if type(path) is dict:
            self.p = path
        else:
            with open(path, 'rb') as output:
                self.p = pickle.load(output)

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
        self.defectMat = defectMat/(2*np.pi)
        self.defects = np.nonzero(defectMat)
        self.charges= defectMat[defectMat != 0]


    def compute_qties(self):
        if 'u' in self.p.keys():
            print('displacement')
            ux,uy=self.p['u']
            vx,vy=self.p['v']
            self.frames = len(ux)
            self.u_angle = np.angle(ux+1j*uy)
            self.v_angle = np.angle(vx+1j*vy)
            self.u_mag = np.abs(ux+1j*uy)
            self.v_mag = np.abs(vx+1j*vy)
            self.track_defects(th=self.u_angle)
            self.FieldsToPlot = [self.u_angle,self.defectMat,self.u_mag]
        elif 'phi' in self.p.keys():
            print('strain')
            phi = self.p['phi']
            self.frames = len(phi)
            phidot = self.p['phidot']
            self.S1=np.real(phi)
            self.S2=np.imag(phi)
            self.argphi = np.angle(phi)
            self.modphi = np.abs(phi)
            self.argphidot = np.angle(phidot)
            self.modphidot = np.abs(phidot)
            self.density = self.modphi**2
            self.momentum = np.abs(np.sum(phi,axis=(1,2)))**2
            self.energy = np.sum(self.density,axis=(1,2))
            self.error = self.momentum/self.energy
            self.dispbasis = np.angle(np.sqrt((self.S1+1j*self.S2)/self.modphi))%np.pi
            self.track_defects(th=self.dispbasis)
            self.FieldsToPlot = [self.argphi,self.defectMat,self.modphi]
        self.times = np.arange(self.frames)*self.p['pt']
        
        

            
    def get_frame(self,f=-1,save=False):
        self.compute_qties()
        plt.rcParams.update({'font.size': 8})
        self.fig,axes = plt.subplots(1,3,figsize=(12,4),dpi=100)  
        self.cmaps = ['hsv','RdBu','viridis']
        self.ims = [ax.imshow(self.FieldsToPlot[n][f],cmap = cmap,aspect='equal') for n,[ax,cmap] in enumerate(zip(axes.flatten(),self.cmaps))]
        plt.suptitle(f'{self.p["NL"]} NL \n'+f'($L_x={self.p["Lx"]},L_y={self.p["Ly"]},N_x={self.p["Nx"]},N_y={self.p["Ny"]}, \\alpha={self.p["alpha"]}$)')    
        self.ims = []
        for n,[cmap,field,ax,label] in enumerate(zip(self.cmaps,self.FieldsToPlot,axes.flatten(),['$\\phi$', 'defects','mag'])):    
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
                        frames=self.frames, interval=40, blit=False)
        if not os.path.exists(folder):
            os.makedirs(folder)
        anim.save(folder+filename+'.mp4')    