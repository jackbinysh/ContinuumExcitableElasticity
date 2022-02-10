
import numpy as np
from pde import CartesianGrid, MemoryStorage, ScalarField, FieldCollection, PDEBase, VectorField
from pde.tools.numba import jit
import os
import numba as nb
import pickle as pickle
import matplotlib.pyplot as plt
import sys

class NLOE2D_sim():
    def __init__(self,parameters):
        self.p=parameters
        pass
             
    def get_plist(self,p0,pname,pval,path):#make a list of parameter dicts for parameter sweeps
        dictlist=[]
        for v in pval:
            p = p0.copy()
            filename = []
            for n,i in enumerate(pname):
                p[i] = v[n]
                filename.append(str(v[n]))
            p['dpath'] = path
            p['savefile']=','.join(filename)
            dictlist.append(p)
        return dictlist
    
    def get_initial_state(self):
        Nx = self.p['Nx']
        Ny = self.p['Ny']
        Lx = self.p['Lx']
        Ly = self.p['Ly']
        amp = self.p['amp']
        bounds=[(-Lx/2,Lx/2),(-Ly/2,Ly/2)]
        shape=[Nx,Ny]
        self.grid = CartesianGrid(bounds,shape, periodic=self.p['BC'])
        self.init = amp*(np.random.rand(4,Nx,Ny)-0.5)
        # initialstate = FieldCollection([VectorField(grid,u_init),VectorField(grid,p_init)])

    
    def DisplacementPDE(self):
        self.get_initial_state()
        self.init=FieldCollection([VectorField(self.grid,self.init[:2]),VectorField(self.grid,self.init[2:])])
        self.runsim(self.p,'displacement')
    
    def StrainPDE(self):
        self.get_initial_state()
        self.init=FieldCollection([ScalarField(self.grid,np.vectorize(complex)(*self.init[:2]),dtype=complex),ScalarField(self.grid,np.vectorize(complex)(*self.init[2:]),dtype=complex)])
        self.runsim(self.p,'strain')
        
        
    def runsim(self,p,basis):
        storage = MemoryStorage()
        trackers = ['progress'    , 'consistency'  ,     storage.tracker(interval=self.p['pt']) ]
        if basis=='displacement':
            A=self.Displacement(self.p,self.init)
        elif basis=='strain':
            A=self.Strain(self.p,self.init)
        sol = A.solve(self.init, t_range=self.p['tf'],tracker=trackers,dt=self.p['dt'])
        filename = self.p['savefolder']+self.p['savefile']
        if not os.path.exists(self.p['savefolder']):
            os.makedirs(self.p['savefolder'])
        field =[]    
        for j,i in storage.items():
            field.append(np.array(i.data))
        with open(filename, 'wb') as output:
            p_ = p.copy()
            if basis=='displacement':
                field=np.moveaxis(np.asarray(field),0,1)
                u = field[:2]
                v = field[2:]
                p_['u'] = u
                p_['v'] = v
            elif basis=='strain':
                phi,phidot=np.moveaxis(np.asarray(field),0,1)
                p_['phi']  = phi
                p_['phidot'] = phidot
            print('saved as:   ', filename)
            pickle.dump(p_,output,pickle.HIGHEST_PROTOCOL)
        

    class Displacement(PDEBase):
        def __init__(self,p,state):
            self.p=p
            self.init=state
        
        def evolution_rate(self, state, t=0):
            """"pure python"""
            u,p=state
            # vector laplacian of the displacement
            apply_laplace = u.grid.make_operator('vector_laplace', self.p['BCtype'])
            lapu= apply_laplace(u.data) 
            # vector laplacian of the velocity
            apply_laplace = p.grid.make_operator('vector_laplace', self.p['BCtype'])
            lapp=apply_laplace(p.data)
            ### linear parts ###
            # shear force
            ShearForce=lapu
            # Odd force
            OddForce=self.p['alpha']*np.array([lapu[1],-lapu[0]])       
            #viscous force
            ViscousForce=lapp
            ### nonlinear parts### 
            # Adding the divergence of the nonlinear shear stress, d_j ( (S_kl S_kl)S_ij )
            diuj= u.gradient(self.p['BCtype'])       
            Sij=diuj.symmetrize(make_traceless=True) # the traceless part of the strain tensor
            NonLinearShearForce= ((Sij.dot(Sij).trace())*Sij).divergence(self.p['BCtype']).data
            udot=p
            pdot=(ShearForce+OddForce+ViscousForce+NonLinearShearForce)
            return FieldCollection([VectorField(u.grid,udot),VectorField(p.grid,pdot)])
        def _make_pde_rhs_numba(self, state):
            #### This part is to speed up things ####
            """ numba-compiled implementation of the PDE """
            alpha,NL,Nx,Ny = self.p['alpha'],self.p['NL'],self.p['Nx'],self.p['Ny']
            laplace = state.grid.get_operator("laplace", bc =self.p['BCtype'])
            @nb.jit
            def pde_rhs(state_data, t):
                [ux,uy],[px,py] = state_data
                rate = np.empty_like(state_data)
                rate[0] = [px,py]
                if NL == 'passive_cubic':
                    
                    rate[1] = laplace(ux+1j*alpha*uy + px + np.abs(ux)**2*ux)
                return rate
            return pde_rhs
        
        
        

    class Strain(PDEBase):
        def __init__(self,p,state):
            self.p=p
            self.init=state

        def evolution_rate(self, state, t=0):
            alpha,NL = self.p['alpha'],self.p['NL']
            ph, phdot = state
            rhs = state.copy()
            ##### Nonlinear odd elasticity, strain formulation 
            rhs[0] = phdot
            if NL == 'passive_cubic':
                rhs[1] = (ph+1j*alpha*ph + phdot + np.abs(ph)**2*ph).laplace(bc=self.p['BCtype'])
            if NL == 'active_bilinear':
                thr=1
                phmod_ = np.abs(ph.data)
                ph_ = np.copy(phmod_)
                ph_[ph_>thr]=thr
                ph_unit = ph.data/phmod_
                ph_NL = ph_*ph_unit    
                ph_NL = ScalarField(self.grid,ph_NL,dtype=complex)
                rhs[1] = (ph+phdot+alpha*1j*ph_NL).laplace(bc=self.p['BCtype'])
            return rhs

        def _make_pde_rhs_numba(self, state):
            #### This part is to speed up things ####
            """ numba-compiled implementation of the PDE """
            alpha,NL,Nx,Ny = self.p['alpha'],self.p['NL'],self.p['Nx'],self.p['Ny']
            laplace = state.grid.get_operator("laplace", bc =self.p['BCtype'])
            @nb.jit
            def pde_rhs(state_data, t):
                ph = state_data[0]
                phdot = state_data[1]
                rate = np.empty_like(state_data)
                rate[0] = phdot
                if NL == 'passive_cubic':
                    rate[1] = laplace(ph+1j*alpha*ph + phdot + np.abs(ph)**2*ph)
                if NL == 'active_bilinear':
                    thr=1
                    phmod_ = np.abs(ph.flatten())
                    ph_ = np.copy(phmod_)
                    ph_[ph_>thr]=thr
                    phi_unit = ph.flatten()/phmod_
                    ph_NL = ph_*phi_unit
                    ph_NL = np.reshape(ph_NL,(Nx,Ny))
                    rate[1] = laplace(ph+phdot+alpha*1j*ph_NL)
                return rate
            return pde_rhs
        
