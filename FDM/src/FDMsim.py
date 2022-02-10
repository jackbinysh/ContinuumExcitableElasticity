
import numpy as np
from pde import CartesianGrid, MemoryStorage, ScalarField, FieldCollection, PDEBase, VectorField
from pde.tools.numba import jit
import os
import numba as nb
import pickle as pickle


class NLOE2D_sim():
    def __init__(self):
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


    class Displacement(PDEBase):#
        'displacement formulation'
        def __init__(self,parameters):
            self.p=parameters
            pass
        def get_initial_state(self):
            'define your initial states here'
            Nx = self.p['Nx']
            Ny =self.p['Ny'] 
            Lx =  self.p['Lx'] 
            Ly = self.p['Ly']
            amp = self.p['amp']
            if self.p['IC']=='ran':
                u_init=amp*np.random.rand(2,Nx,Ny)-0.5
                p_init=amp*np.random.rand(2,Nx,Ny)-0.5
            if self.p['IC'] == 'fuzz':
                u_init=np.full((2,Nx,Ny),amp)+ amp**2 * np.random.rand(2,Nx,Ny)-0.5
                p_init=np.zeros_like(u_init)
            bounds=[(-Lx/2,Lx/2),(-Ly/2,Ly/2)]
            shape=[Nx,Ny]
            grid = CartesianGrid(bounds,shape, periodic=self.p['BC'])
            initialstate = FieldCollection([VectorField(grid,u_init),VectorField(grid,p_init)])
            return initialstate
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
        def runsim(self,p=[]):
            if p!=[]:
                self.p=p
            
            storage = MemoryStorage()
            trackers = ['progress'    , 'consistency'  ,     storage.tracker(interval=self.p['pt']) ]
            state = self.get_initial_state() #initial field for u,p
            sol = self.solve(state, t_range=self.p['tf'],tracker=trackers,dt=self.p['dt'])
            field =[]
            for j,i in storage.items():
                field.append(np.array(i.data))
            field=np.moveaxis(np.asarray(field),0,1)
            u = field[:2]
            v = field[2:]
            print(np.shape(u))
            filename = self.p['savefolder']+self.p['savefile']
            if not os.path.exists(self.p['savefolder']):
                os.makedirs(self.p['savefolder'])
            with open(filename, 'wb') as output:
                p_ = p.copy()
                p_['u'] = u
                p_['v'] = v
                print('saved as:   ', filename)
                pickle.dump(p_,output,pickle.HIGHEST_PROTOCOL)
       
    class Strain(PDEBase):
        def __init__(self,parameters):
            self.p=parameters

        def get_initial_state(self):
            Nx = self.p['Nx']
            Ny =self.p['Ny'] 
            Lx =  self.p['Lx'] 
            Ly = self.p['Ly']        
            amp = self.p['amp']
            alpha = self.p['alpha']
            nx = self.p['nx']
            ny =self.p['ny'] 
            if self.p['IC']=='ran':
                ph0 = amp*(np.random.rand(Nx,Ny)-0.5 + 1j* (np.random.rand(Nx,Ny)-0.5))
                phdot0 = amp*(np.random.rand(Nx,Ny)-0.5 + 1j* (np.random.rand(Nx,Ny)-0.5))
                #constrain initial energy and momentum to be zero:
            elif self.p['IC']=='sin':
                mesh = np.meshgrid(np.arange(Ny),np.arange(Nx))
                ampx = np.sqrt((alpha*Lx/(2*nx*np.pi))**2-1)
                ampy = np.sqrt((alpha*Ly/(2*ny*np.pi))**2-1)
                ph =  ampx * np.cos(2*np.pi *nx*mesh[0]/Nx) +ampy* 1j*np.cos(2*np.pi*ny*mesh[1]/Ny)
                phdot0 = -1j*alpha*ph
                #constrain initial energy and momentum to be zero:
            ph0 = ph0 - np.average(ph0)
            phdot0 = phdot0 - np.average(phdot0)    
            bounds=[(-Lx/2,Lx/2),(-Ly/2,Ly/2)]
            shape=[Nx,Ny]
            grid = CartesianGrid(bounds,shape, periodic=self.p['BC'])
            self.grid = grid
            return FieldCollection([ScalarField(grid,ph0,dtype=complex),ScalarField(grid,phdot0,dtype=complex)])

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
        
        def runsim(self,p):
            storage = MemoryStorage()
            trackers = ['progress'    , 'consistency'  ,     storage.tracker(interval=self.p['pt']) ]
            state = self.get_initial_state() #initial field for phi,phidot
            sol = self.solve(state, t_range=self.p['tf'],tracker=trackers,dt=self.p['dt'])
            field =[]
            for j,i in storage.items():
                field.append(np.array(i.data))
            phi,phidot=np.moveaxis(np.asarray(field),0,1)
            filename = self.p['savefolder']+self.p['savefile']
            if not os.path.exists(self.p['savefolder']):
                os.makedirs(self.p['savefolder'])
            with open(filename, 'wb') as output:
                p_ = p.copy()
                p_['phi']  = phi
                p_['phidot'] = phidot
                print('saved as:   ', filename)
                pickle.dump(p_,output,pickle.HIGHEST_PROTOCOL)
