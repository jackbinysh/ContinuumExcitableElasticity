# Define the Elasticity PDE's
import pde
import numpy as np
from pde import PDE,CartesianGrid, ScalarField, FieldCollection, PDEBase, VectorField
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class pde(PDEBase):
    def __init__(self,B, mu,eta,ko,mu_tilde,gamma,rho,N,bc):
        #initialize parameters
        self.B,self.mu,self.eta,self.ko,self.mu_tilde,self.gamma,self.rho,self.N,self.bc = B,mu,eta,ko,mu_tilde,gamma,rho,N,bc
        
    def evolution_rate(self, state, t=0):
        """"pure python"""
        
        u,p=state
        
        # vector laplacian of the displacement
        apply_laplace = u.grid.make_operator('vector_laplace', self.bc)
        lapu= apply_laplace(u.data) 
        # grad(div) of the displacement
        apply_divergence = u.grid.make_operator('divergence', self.bc)
        divu= apply_divergence(u.data)
        apply_gradient=u.grid.make_operator('gradient',self.bc)
        graddivu=apply_gradient(divu)
        # vector laplacian of the velocity
        apply_laplace = p.grid.make_operator('vector_laplace', self.bc)
        lapp=apply_laplace(p.data)   
        
        ### linear parts ###
        # shear force
        ShearForce= self.mu*lapu
        #bulk force
        BulkForce=(self.B)*graddivu 
        # Odd force
        OddForce=self.ko*np.array([lapu[1],-lapu[0]])       
        #viscous force
        ViscousForce=self.eta*lapp
        
        ### nonlinear parts### 
        # Adding the divergence of the nonlinear shear stress, d_j ( (S_kl S_kl)S_ij )
        diuj= u.gradient(self.bc)       
        Sij=diuj.symmetrize(make_traceless=True) # the traceless part of the strain tensor
        NonLinearShearForce=self.mu_tilde*( ((Sij.dot(Sij).trace())*Sij).divergence(self.bc) ).data
        
        udot=p
        pdot=(1/self.rho)*(BulkForce+ShearForce+OddForce+ViscousForce+NonLinearShearForce) 
        return FieldCollection([VectorField(u.grid,udot),VectorField(p.grid,pdot)])

# a helper function to make animations, code stolen from:
# https://stackoverflow.com/questions/42386372/increase-the-speed-of-redrawing-contour-plot-in-matplotlib
def ContourPlotAnimator(data
,Times
,Title=None                        
,cmap=plt.get_cmap("viridis")
,cmaplevels=20
,repeat=False
,FrameMin=20):
    
    L=data.shape[0]
    # get the colorbar range
    # don't use the first few frames of data to get the max/min bounds.
    vmax=np.max(data[FrameMin:,:,:])
    vmin=np.min(data[FrameMin:,:,:])
    levels = np.linspace(vmin, vmax,cmaplevels)

    # make the fig
    fig, ax=plt.subplots()
    ax.set_title(Title)

    # make the initial drawing
    p = [ax.contourf(np.transpose(data[0,:,:]), levels, cmap=cmap,vmin=vmin,vmax=vmax)]

    # make the fixed colorbar, and time label
    cbar=fig.colorbar(p[0])
    props = dict(boxstyle='round', facecolor='wheat')
    timelabel = ax.text(0.9,0.9, "", transform=ax.transAxes, ha="right",bbox=props)


    def update(i):
        # remove the old drawing
        for tp in p[0].collections:
            tp.remove()
        p[0] = ax.contourf(np.transpose(data[i,:,:]), levels,cmap=cmap,vmin=vmin,vmax=vmax)  
        label="T="+'{:.2f}'.format(Times[i])   
        timelabel.set_text(label)  
        return p[0].collections

    ani = animation.FuncAnimation(fig, update, frames=L, 
                                             interval=1, blit=True, repeat=repeat)
    
    return ani

# a few random helper functions
def x(i):
    return (i-((Npts-1)/2))*h

# remember, i is 0 indexed!
def i(x):
    return (x/h)+((Npts-1)/2)
