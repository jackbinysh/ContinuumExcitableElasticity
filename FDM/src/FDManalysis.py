import numpy as np
from pde import CartesianGrid, MemoryStorage, ScalarField, FieldCollection, PDEBase, VectorField
from pde.tools.numba import jit
import os
import numba as nb
import pickle as pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize

class NLOE2D_analysis():
    def __init__(self,path):
        with open(path, 'rb') as output:
            self.p = pickle.load(output)


        
    def track_defects(self,th):
        nbMat = np.zeros((np.shape(th)[0],8,self.p['Nx'],self.p['Ny']))
        nb = np.asarray([[1,0],[1,1],[0,1],[-1,1],[-1,0],[-1,-1],[0,-1],[1,-1]])        
        for n,i in enumerate(nb):
            nbMat[:,n] = np.roll(np.roll(th,i[0],axis=1),i[1],axis=2)
        nbMat = nbMat[:,:,1:-1,1:-1]
        difMat = (nbMat - np.roll(nbMat,1,axis=1))%np.pi/2
        defectMat = np.sum(difMat,axis=1)
        defectMat += - 2*np.pi
        defectMat[np.abs(defectMat)<np.pi+0.1]=0
        self.defectMat =defectMat

        plus=[]
        minus=[]
        for t in defectMat:
            plus.append(np.count_nonzero(t>0))
            minus.append(np.count_nonzero(t<0))
        self.plus=np.asarray(plus)
        self.minus=np.asarray(minus)
        self.defects = self.plus+self.minus
        self.defects = self.defects/np.max(self.defects)
        
        
        plot=False
        if plot:
            fig,ax = plt.subplots(1,3,figsize=(12,6))
            ax[0].set_yscale('log')
            ax[2].set_yscale('log')
            ax[0].scatter(np.arange(len(plus)),plus,s=1)
            ax[0].scatter(np.arange(len(plus)),minus,s=1)
            ax[1].scatter(np.arange(len(plus)),self.minus-self.plus,s=1)
            ax[2].scatter(np.arange(len(plus)),self.defects,s=1)
            plt.show()
            
        
        




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
            self.FieldsToPlot = [self.u_angle,self.v_angle,self.u_mag,self.v_mag]
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
            
            # self.track_defects(th=np.angle(self.S1-np.sqrt(self.S1**2+self.S2**2 + 1j*self.S2))%np.pi)
            # theta = np.angle(S1 - np.sqrt(S1**2 + S2**2) +1j*S2)%np.pi
            self.track_defects(th=np.angle(np.sqrt((self.S1+1j*self.S2)/self.modphi))%np.pi)
            self.FieldsToPlot = [self.argphi,self.defectMat,self.modphi,self.modphidot]
        self.times = np.arange(self.frames)*self.p['pt']
        
        
        
        
        
            
    def get_frame(self,f=-1,save=False):
        self.compute_qties()
        plt.rcParams.update({'font.size': 8})
        self.fig,axes = plt.subplots(2,2,figsize=(6,6),dpi=100)  
        self.cmaps = ['hsv','hsv','viridis','viridis']
        self.ims = [ax.imshow(self.FieldsToPlot[n][f],cmap = cmap,aspect='equal') for n,[ax,cmap] in enumerate(zip(axes.flatten(),self.cmaps))]
        plt.suptitle(f'{self.p["NL"]} NL \n'+f'($L_x={self.p["Lx"]},L_y={self.p["Ly"]},N_x={self.p["Nx"]},N_y={self.p["Ny"]}, \\alpha={self.p["alpha"]}$)')    
        self.ims = []
        
        for n,[cmap,field,ax,label] in enumerate(zip(self.cmaps,self.FieldsToPlot,axes.flatten(),['angle', 'angle dot','mag','mag dot'])):    
            if cmap == 'viridis':
                self.ims.append(ax.imshow(field[f],cmap=cmap,aspect='equal',vmin=np.min(field[f]),vmax=np.max(field[f])))
            elif cmap == 'hsv':
                self.ims.append(ax.imshow(field[f],cmap=cmap,aspect='equal',vmin=-np.pi,vmax=np.pi))
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
                if n>-1:
                    vmin = np.min(f)
                    vmax = np.max(f)
                    im.set_clim(vmin, vmax)
        anim = animation.FuncAnimation(self.fig, animate, 
                        frames=self.frames, interval=40, blit=False)
        if not os.path.exists(folder):
            os.makedirs(folder)
        anim.save(folder+filename+'.mp4')    