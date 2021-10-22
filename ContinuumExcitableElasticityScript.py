from ContinuumExcitableElasticity import pde
from pde import CartesianGrid, Controller, MemoryStorage, FileStorage, ScalarField, FieldCollection, PlotTracker, ExplicitSolver,VectorField
import numpy as np
import os
import glob
import json
import sys 

### DATA READ IN ###

# which line of input file defines me?
line=int(sys.argv[1])
# read in arguments from file
reader=open("Parameters.txt","r")
parameters=reader.readlines()[line].split()

# Simulation parameters
L=int(parameters[0]) # Physical Box Size
T=int(parameters[1]) # Run time
Npts=int(parameters[2]) # number of points, its easier to do the math if this is odd

h= L/(Npts-1) #derived grid spacing
dt=5e-3 #timestep
pt = 0.5 # Time interval for printing

# Material Parameters
B=float(parameters[3])
mu=float(parameters[4])
eta=float(parameters[5])
ko=float(parameters[6])
mu_tilde=float(parameters[7])
rho=float(parameters[8])
gamma=0 # no friction currently

bc='auto_periodic_dirichlet' #periodic or dirichlet depending on periodic=...

### DATA WRITE OUT ###

# root folder for data
#DataFolder='/mnt/jacb23-XDrive/Physics/ResearchProjects/ASouslov/RC-PH1229/ActiveElastocapillarity/2020-10-23-EnergyMinimization/'+"kc_"+"{0:0.1f}".format(kc)+"_alpha_"+"{0:0.2f}".format(MatNon)+"/"
DataFolder="/Users/jackbinysh/Code/ContinuumExcitableElasticity/Data/"+str(ko)+"/"
# filepath for the data output
# see https://github.com/zwicker-group/py-pde/discussions/39
filepath=DataFolder+"Output.hdf5"

# Name of the current file
#ScriptName="EnergyMinimizationScript3D.py"
# Name of the file of functions used for this run
#FunctionFileName="EnergyMinimization.py"
# Dump an exact copy of this code into the data file
#shutil.copyfile(ScriptName,DataFolder+ScriptName)
#shutil.copyfile(FunctionFileName,DataFolder+FunctionFileName)

try:
    os.mkdir(DataFolder)
except OSError:
    print ("Creation of the directory %s failed" % DataFolder)
else:
    print ("Successfully created the directory %s " % DataFolder)
    
# try and clear out the folder of vtk files and log files, if there was a previous run in it
for filename in glob.glob(DataFolder+'*.hdf5')+glob.glob(DataFolder+'*.log'):
    file_path = os.path.join(DataFolder, filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print('Failed to delete %s. Reason: %s' % (file_path, e))
                
#Dump all the parameters to a file in the run folder        
f=open(DataFolder+"Parameters.log","w+")
datadict= { 
        "T":T,
        "L":L,
        "Npts":Npts,
        "B":B,
        "mu":mu, 
        "eta":eta,
        "ko":ko,
        "mu_tilde": mu_tilde, 
        "rho": rho,
        "bc":bc
}
json.dump(datadict,f)
f.close()

### SOME NUMBERS IT IS GOOD TO KNOW ###
J=ko**2
ViscThresh=J-(B/2)**2
qc=np.sqrt(( rho/(eta**2) )*( (J-((B/2)**2))/(mu+(B/2)) ))
lamc=(2*np.pi)/qc

# initialise the PDE
eq = pde(B,mu,eta,ko,mu_tilde,gamma,rho,L, bc)

# Initialise the Grid
bounds=[(-L/2,L/2),(-L/2,L/2)]
shape=[Npts,Npts]
grid = CartesianGrid(bounds,shape, periodic=[True,True])

### INITIAL STATE ###
# Random Fuzz
A=0.1
ux_init=np.random.rand(Npts,Npts)-0.5
uy_init=np.random.rand(Npts,Npts)-0.5
u_init= A*np.array([ux_init,uy_init])

px_init=np.random.rand(Npts,Npts)-0.5
py_init=np.random.rand(Npts,Npts)-0.5
p_init= A*np.array([px_init,py_init])

initialstate = FieldCollection([VectorField(grid,u_init),VectorField(grid,p_init)])
sol1=initialstate # allow the possibility we will restart the simulation a few times.

### GOGO! ###

# initialise a file storage class
info={'grid':grid}
storage=FileStorage(filepath,info)
storagetracker=storage.tracker(interval=pt) 
### OUTPUT ###
trackers = ['progress', 'consistency', storagetracker]
solver1 = ExplicitSolver(eq)
controller1 = Controller(solver1, t_range=T, tracker=trackers)

sol1 = controller1.run(sol1,dt=dt)
print("Diagnostic information:")
print(controller1.diagnostics)
storage.end_writing()
