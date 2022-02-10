import numpy as np
from pde import CartesianGrid, MemoryStorage, ScalarField, FieldCollection, PDEBase, VectorField
from pde.tools.numba import jit
import os
import numba as nb
import pickle as pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize

class NLOE2D_animate():
    def __init__(self,path):
        with open(path, 'rb') as output:
            self.p = pickle.load(output)

    def compute_qties(self):
        if 'u' in self.p.keys():
            ux,uy=self.p['u']
            vx,vy=self.p['v']
            self.frames = len(ux)
            self.u_angle = np.arctan(uy,ux)
            self.v_angle = np.arctan(vy,vx)
            self.u_mag = np.abs(ux+1j*uy)
            self.v_mag = np.abs(vx+1j*vy)
            self.FieldsToPlot = [self.u_angle,self.v_angle,self.u_mag,self.v_mag]
        elif 'phi' in self.p.keys():
            phi = self.p['phi']
            self.frames = len(phi)
            phidot = self.p['phidot']
            self.argphi = np.angle(phi)
            self.modphi = np.abs(phi)
            self.argphidot = np.angle(phidot)
            self.modphidot = np.abs(phidot)
            self.density = self.modphi**2
            self.momentum = np.abs(np.sum(phi,axis=(1,2)))**2
            self.energy = np.sum(self.density,axis=(1,2))
            self.error = self.momentum/self.energy
            self.FieldsToPlot = [self.argphi,self.modphi,self.argphidot,self.modphidot]
        self.times = np.arange(self.frames)*self.p['pt']
            
    def get_frame(self,f=-1,save=False):
        self.compute_qties()
        plt.rcParams.update({'font.size': 8})
        self.fig,axes = plt.subplots(2,2,figsize=(6,6),dpi=100)  
        self.cmaps = ['hsv','hsv','viridis','viridis']
        self.ims = [ax.imshow(self.FieldsToPlot[n][f],cmap = cmap,aspect='equal') for n,[ax,cmap] in enumerate(zip(axes.flatten(),self.cmaps))]
        for n,[ax,field,label] in enumerate(zip(axes.flatten(),['u','\\phi'],['angle', 'angle dot','mag','mag dot'])):
            plt.suptitle(f'$|{field}'+'_{\\mathrm{NESS}}|$ for '+f'{self.p["NL"]} NL \n'+f'($L_x={self.p["Lx"]},L_y={self.p["Ly"]},N_x={self.p["Nx"]},N_y={self.p["Ny"]}, \\alpha={self.p["alpha"]}$)')    
            ax.set_title(label)
            plt.colorbar(self.ims[n],ax=ax)
        if save:
            plt.savefig('ness.pdf')
        plt.tight_layout()
        self.axes=axes
        
    def animate_fields(self,folder,filename):
        self.get_frame(f=0)
        def animate(i):
            for n,[ax,im] in enumerate(zip(self.axes.flatten(),self.ims)):
                f = self.FieldsToPlot[n][i]
                im.set_array(f)
                vmin = np.min(f)
                vmax = np.max(f)
                im.set_clim(vmin, vmax)
        anim = animation.FuncAnimation(self.fig, animate, 
                        frames=self.frames, interval=40, blit=False)
        if not os.path.exists(folder):
            os.makedirs(folder)
        anim.save(folder+filename+'.mp4')    