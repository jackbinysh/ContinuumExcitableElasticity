from ContinuumExcitableElasticity import pde
from pde import CartesianGrid, Controller, MemoryStorage, FileStorage, ScalarField, FieldCollection, CallbackTracker, PlotTracker, ExplicitSolver,VectorField
import numpy as np
import os
import glob
import json
import sys 
from datetime import datetime

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
dt=1e-3 #timestep
pt=10 # Time interval for printing

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
DataFolder="/Users/jackbinysh/Code/ContinuumExcitableElasticity/Data/Test/"
# filepath for the data output
# see https://github.com/zwicker-group/py-pde/discussions/39
filepath=DataFolder+"Output.hdf5"
# filepath for the log output
logfilepath=DataFolder+"Output.log"

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
print(ViscThresh)
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

### INITIALISE TRACKERS ###

# initialise a file storage class
storage=FileStorage(filepath)
storagetracker=storage.tracker(interval=pt) 
# define a callback tracker to print some data to a file
def mycallback(state,time):

    logFile = open(logfilepath, 'a')
    now = datetime.now()
    current_time = now.strftime("%D:%H:%M")
    print("T {T}".format(T=int(time))+" Current Time: "+current_time,file=logFile)
    logFile.close()

trackers = ['progress'
            ,'consistency'
            , storagetracker
            , CallbackTracker(mycallback,interval=pt)]

### GOGO! ###
solver1 = ExplicitSolver(eq)
controller1 = Controller(solver1, t_range=T, tracker=trackers)
sol1 = controller1.run(sol1,dt=dt)

### SOME FINAL OUTPUT ###
storage.end_writing()
logFile = open(logfilepath, 'a')
print("Diagnostic information:", file=logFile)
print(controller1.diagnostics, file=logFile)
logFile.close()
