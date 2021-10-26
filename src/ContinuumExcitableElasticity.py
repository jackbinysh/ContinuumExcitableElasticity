# Define the Elasticity PDE's
import pde
import numpy as np
from pde import PDE,CartesianGrid, ScalarField, FieldCollection, PDEBase, VectorField
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

class pde(PDEBase):
    def __init__(self,B, mu,eta,ko,mu_tilde,gamma,rho,N,bc):
        #initialize parameters
        self.B,self.mu,self.eta,self.ko,self.mu_tilde,self.gamma,self.rho,self.N,self.bc = B,mu,eta,ko,mu_tilde,gamma,rho,N,bc

    def evolution_rate(self, state, t=0):
        """"pure python"""

        u,p=state

        # grad(div) of the displacement
        apply_divergence = u.grid.get_operator('divergence', self.bc)
        divu= apply_divergence(u.data)
        apply_gradient=u.grid.get_operator('gradient',self.bc)
        graddivu=apply_gradient(divu)
        # vector laplacian of the displacement
        apply_laplace = u.grid.get_operator('vector_laplace', self.bc)
        lapu= apply_laplace(u.data)

        # vector laplacian of the velocity
        apply_laplace = p.grid.get_operator('vector_laplace', self.bc)
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
,FrameMin=20
,title=''):


    Ndata = data.shape[0]
    L=data.shape[1]
    # make the fig

    fig, axes = plt.subplots(2,2)
    p=[[],[],[],[]]
    fig.suptitle(title, fontsize="medium")
    for n,ax in enumerate(axes.flatten()):
        ax.set_title(Title[n],fontsize=8)
        # get the colorbar range
        # don't use the first few frames of data to get the max/min bounds.
        vmax=np.max(data[n,FrameMin:,:,:])
        vmin=np.min(data[n,FrameMin:,:,:])
        levels = np.linspace(vmin, vmax,cmaplevels)

        # make the initial drawing
        p[n] = ax.imshow(np.transpose(data[n,0,:,:]),cmap=cmap[n]) #, levels, cmap=cmap,vmin=vmin,vmax=vmax)]
        # make the fixed colorbar, and time label
        divider = make_axes_locatable(ax)

        cax = divider.append_axes('right', size='5%', pad=0.05)
        cbar = fig.colorbar(p[n], cax=cax, orientation='vertical')

        props = dict(boxstyle='round', facecolor='wheat')
        timelabel = ax.text(1.4,2.3, "", transform=ax.transAxes, ha="right",bbox=props)

    def update(i):
        # remove the old drawing
        #p[0] = ax.imshow(np.transpose(data[i,:,:]))#, levels,cmap=cmap,vmin=vmin,vmax=vmax)
        for n,j in enumerate(p):
            p[n].set_array(data[n,i,:,:])
            label="T="+'{:.2f}'.format(Times[i])
            timelabel.set_text(label)


    ani = animation.FuncAnimation(fig, update, frames=L,
                                             interval=1, blit=True, repeat=repeat)

    return ani

