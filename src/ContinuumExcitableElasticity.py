# Define the Elasticity PDE's
import pde
import numpy as np
from pde import PDE,CartesianGrid, ScalarField, FieldCollection, PDEBase, VectorField
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class pde(PDEBase):
    def __init__(self,B, mu,eta,ko,mu_tilde,gamma,rho,bc):
        #initialize parameters
        self.B,self.mu,self.eta,self.ko,self.mu_tilde,self.gamma,self.rho,self.bc = B,mu,eta,ko,mu_tilde,gamma,rho,bc
        
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

