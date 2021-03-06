
import numpy as np
from pde import CartesianGrid, MemoryStorage, ScalarField, FieldCollection, PDEBase, VectorField
from pde.tools.numba import jit
import os,time
import numba as nb
import pickle as pickle
import matplotlib.pyplot as plt
import sys
from src.FDManalysis import NLOE2D_analysis
class NLOE2D_sim():
    def __init__(self,parameters):
        self.p=parameters
        pass
             
    
    def get_initial_state(self):
        Nx = self.p['Nx']
        Ny = self.p['Ny']
        Lx = self.p['Lx']
        Ly = self.p['Ly']
        amp = self.p['amp']
        alpha = self.p['alpha']
        bounds=[(-Lx/2,Lx/2),(-Ly/2,Ly/2)]
        shape=[Nx,Ny]
        self.grid = CartesianGrid(bounds,shape, periodic=self.p['BC'])
        self.p['grid'] = self.grid
        self.init = amp*(np.random.rand(4,Nx,Ny)-0.5)
        for n,i in enumerate(self.init):
            self.init[n] = i - np.average(i)
        
        
        
    def runsim(self):
        storage = MemoryStorage()
        trackers = ['progress'    , 'consistency'  ,     storage.tracker(interval=self.p['pt']) ]
        self.get_initial_state()
        if self.p['basis']=='displacement':
            print('running displacement simulation')
            self.init=FieldCollection([VectorField(self.grid,self.init[:2]),VectorField(self.grid,self.init[2:])])
            eq=self.Displacement(self.p,self.init)
        elif self.p['basis']=='strain':
            print('running strain simulation')
            self.init=FieldCollection([ScalarField(self.grid,np.vectorize(complex)(*self.init[:2]),dtype=complex),ScalarField(self.grid,np.vectorize(complex)(*self.init[2:]),dtype=complex)])            
            eq=self.Strain(self.p,self.init)
        elif self.p['basis']=='first order strain':
            print('running first order strain simulation')
            self.init=ScalarField(self.grid,np.vectorize(complex)(*self.init[:2]),dtype=complex)
            eq=self.FirstOrderStrain(self.p,self.init)
        sol = eq.solve(self.init, t_range=self.p['tf'],tracker=trackers,dt=self.p['dt'])        
        filename = self.p['datafolder'] + self.p['subfolder']+self.p['subsubfolder']+self.p['basis'] + ' - ' + self.p['savefile']
        if not os.path.exists(self.p['datafolder'] + self.p['subfolder']+self.p['subsubfolder']):
            os.makedirs(self.p['datafolder'] + self.p['subfolder']+self.p['subsubfolder'])
        field =[]    
        for j,i in storage.items():
            field.append(np.array(i.data))
        with open(filename, 'wb') as output:
            data = {}
            if self.p['basis']=='displacement':
                field=np.moveaxis(np.asarray(field),0,1)
                u = field[:2]
                v = field[2:]
            elif self.p['basis']=='strain':
                u,v=np.moveaxis(np.asarray(field),0,1)
            elif self.p['basis']=='first order strain':
                u=np.asarray(field)
            data['u'] = u
            if self.p['basis']!='first order strain':
                data['v'] = v
            else:
                data['v'] = np.zeros_like(u)
            data = {**self.p, **data}
            
            if self.p['output_data']=='defects':
                ana = NLOE2D_analysis(data)
                data = {**data, **ana.timeseries}
                data.pop('u', None)
                data.pop('v', None)
            pickle.dump(data,output,pickle.HIGHEST_PROTOCOL)
            print('saved as:   ', filename)
            
            
            
    class Displacement(PDEBase):
        def __init__(self,p,state):
            self.st = time.time()
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
            
            
            apply_div =  u.grid.make_operator('divergence', self.p['BCtype'])
            apply_grad =  u.grid.make_operator('gradient', self.p['BCtype'])
            graddivu = apply_grad(apply_div(u.data))

            
            ### linear parts ###
            # shear force
            ShearForce=lapu
            #bulk force
            BulkForce=self.p['B'] * graddivu 
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
            pdot=(ShearForce+BulkForce+OddForce+ViscousForce+NonLinearShearForce)

            return FieldCollection([VectorField(u.grid,udot),VectorField(p.grid,pdot)])
            
            
        def _make_pde_rhs_numba(self, state):
            #numba-compiled implementation of the PDE
            mu=1
            B=self.p['B']
            ko=self.p['alpha']
            eta=1
            mu_tilde=1
            rho=1
            
            # make the operators
            apply_laplace = state.grid.make_operator('vector_laplace',self.p['BCtype'] )
            apply_divergence = state.grid.make_operator('divergence',self.p['BCtype'] )
            apply_tensor_divergence = state.grid.make_operator('tensor_divergence',self.p['BCtype'] )
            apply_gradient = state.grid.make_operator('gradient',self.p['BCtype'] )
            apply_vector_gradient =state.grid.make_operator('vector_gradient',self.p['BCtype'] )
            
            # some things for the jit
            d=len(state.grid.shape)
            #https://py-pde.readthedocs.io/en/latest/_modules/pde/fields/tensorial.html?highlight=transpose#
            axes = (1, 0) + tuple(range(2, 2 + state.grid.num_axes)) # for tranposing

            @nb.jit
            def pde_rhs(state_data, t):
                
                u = state_data[0:2,:,:]
                p = state_data[2:4,:,:]
                
                # apply all the operators
                lapu= apply_laplace(u) # vector laplacian of the displacement
                
                divu= apply_divergence(u)
                graddivu=apply_gradient(divu)  # grad(div) of the displacement
                lapp=apply_laplace(p)   # vector laplacian of the velocity

                ### linear parts ###
                # shear force
                ShearForce= mu*lapu
                #bulk force
                BulkForce=B*graddivu 
                # Odd force 
                lapustar=np.copy(lapu)
                lapustar[0,:,:]=lapu[1,:,:]
                lapustar[1,:,:]=-lapu[0,:,:]
                OddForce=ko*lapustar
                #viscous force
                ViscousForce=eta*lapp

                # Adding the divergence of the nonlinear shear stress, d_j ( (S_kl S_kl)S_ij )
                diuj= apply_vector_gradient(u) 
                S_ij=0.5*(diuj+np.transpose(diuj,axes)) # symmetrize

                # time to get low level
                # Make the strain traceless
                for k in range(S_ij.shape[2]):
                    for l in range(S_ij.shape[3]):
                        trace= S_ij[0,0,k,l] + S_ij[1,1,k,l]
                        S_ij[0,0,k,l]= S_ij[0,0,k,l]-(1/d)*trace
                        S_ij[1,1,k,l]= S_ij[1,1,k,l]-(1/d)*trace

                # Now compute the nonlinear term  
                ModSsqS=np.copy(S_ij)
                for k in range(S_ij.shape[2]):
                    for l in range(S_ij.shape[3]):
                        # we are sitting at a point (k,l) in space. Now get |S|^2, and multiply it into S
                        # at every data point, get |S^2|S
                        ssq=0
                        for i in range(S_ij.shape[0]):
                            for j in range(S_ij.shape[1]):
                                ssq+=S_ij[i,j,k,l]*S_ij[i,j,k,l] # this guy is symmetric so the order doesnt matter

                        # now multiply it in, to get our final answer
                        for i in range(S_ij.shape[0]):
                            for j in range(S_ij.shape[1]):
                                ModSsqS[i,j,k,l]= ssq*S_ij[i,j,k,l]

                NonLinearShearForce=mu_tilde*apply_tensor_divergence(ModSsqS)
                
                rate = np.empty_like(state_data)
                rate[0:2]=p
                rate[2:4]=(1/rho)*(BulkForce+ShearForce+OddForce+ViscousForce+NonLinearShearForce) 
                return rate

            return pde_rhs

    class Strain(PDEBase):
        def __init__(self,p,state):
            self.p=p
            self.init=state

        def evolution_rate(self, state, t=0):
            alpha,NL = self.p['alpha'],self.p['NL']
            u, v = state
            rhs = state.copy()
            ##### Nonlinear odd elasticity, strain formulation 
            rhs[0] = v
            if NL == 'passive_cubic':
                rhs[1] = (u+1j*alpha*u + v + np.abs(u)**2*u).laplace(bc=self.p['BCtype'])
            if NL == 'active_bilinear':
                thr=1
                absu_ = np.abs(u.data)
                u_ = np.copy(absu_)
                u_[u_>thr]=thr
                u_unit = u.data/absu_
                u_NL = u_*u_unit    
                u_NL = ScalarField(self.p['grid'],u_NL,dtype=complex)
                rhs[1] = (u+v+alpha*1j*u_NL).laplace(bc=self.p['BCtype'])
            return rhs

        def _make_pde_rhs_numba(self, state):
            #### This part is to speed up things ####
            """ numba-compiled implementation of the PDE """
            alpha,NL,Nx,Ny = self.p['alpha'],self.p['NL'],self.p['Nx'],self.p['Ny']
            laplace = state.grid.get_operator("laplace", bc =self.p['BCtype'])
            @nb.jit
            def pde_rhs(state_data, t):
                u = state_data[0]
                v = state_data[1]
                rate = np.empty_like(state_data)
                rate[0] = v
                if NL == 'passive_cubic':
                    rate[1] = laplace(u+1j*alpha*u + v + np.abs(u)**2*u)
                if NL == 'active_bilinear':
                    thr=1
                    absu_ = np.abs(u.flatten())
                    u_ = np.copy(absu_)
                    u_[u_>thr]=thr
                    ui_unit = u.flatten()/absu_
                    u_NL = u_*ui_unit
                    u_NL = np.reshape(u_NL,(Nx,Ny))
                    rate[1] = laplace(u+v+alpha*1j*u_NL)
                return rate
            return pde_rhs

    class FirstOrderStrain(PDEBase):
        def __init__(self,p,state):
            self.p=p
            self.init=state

        def evolution_rate(self, state, t=0):
            alpha,NL = self.p['alpha'],self.p['NL']
            u = state
            rhs = state.copy()
            ##### Nonlinear odd elasticity, strain formulation 
            rhs = (u + np.abs(u)**2*u).laplace(bc=self.p['BCtype'])/(1j*alpha)
            return rhs

        def _make_pde_rhs_numba(self, state):
            #### This part is to speed up things ####
            """ numba-compiled implementation of the PDE """
            alpha,NL,Nx,Ny = self.p['alpha'],self.p['NL'],self.p['Nx'],self.p['Ny']
            laplace = state.grid.get_operator("laplace", bc =self.p['BCtype'])
            @nb.jit
            def pde_rhs(state_data, t):
                u = state_data
                rate = np.empty_like(state_data)
                rate = laplace(u + np.abs(u)**2*u)/(1j*alpha)
                return rate
            return pde_rhs
        





















     