# [depends] %LIB%/hvacFuncs.py %LIB%/sysidtools.py
# [makes] pickle
import sys
sys.path.append('lib/')
import numpy as np
from hvacFuncs import getHVACModel, getHVACXs
from sysidtools import PickleTool
from sysidtools import sample_prbs_like, generateData
from numpy.random import multivariate_normal as mvnrnd

def get_sfeed_calcparameters(Delta, us, ps):
    """ Get calculation parameters for the state feedback 
        simulation experiments. """

    # Set the numpy seed for state feedback calculations. 
    np.random.seed(5)

    # Get the linear, discrete-time, HVAC system model. 
    linModel = getHVACModel(Delta)

    # Get the steady-state xs of the HVAC system (used in regulator 
    # tuning). 
    xs = getHVACXs(linModel['A'], linModel['B'], linModel['Bp'], us, ps)
    
    # Sizes. 
    Nx, Nu = linModel['B'].shape
    Ny = linModel['C'].shape[0]

    # Generate the control input sequence for the training data. 
    Nt = int(2.5*24*3600/Delta) # Number of time steps in the training data.
    useq_sigma = 400*np.diag([1., 1.]) # Variance of the input signal. 
    useq = mvnrnd(np.zeros((Nu, )), useq_sigma, Nt)

    # Get the list of noise covariance values for the simulation 
    # experiments. The noise covariances are specified as tuple
    # entries (Qw, Rv) in the list.
    noiseCovs_list = [(0*np.eye(Nx), 0*np.eye(Ny)),
                      (np.diag([8e-2, 1e-2, 9e-2, 1e-2]), 0*np.eye(Ny))]

    # Initial state for the simulation. 
    x0 = np.zeros((Nx, 1))

    # Empty list to store all the training data. 
    trainingData_list = []

    # Loop over the noise covariance values, and generate 
    # different simulation data sets for each case.
    for noiseCovs in noiseCovs_list:

        # Extract the noise covariances. 
        Qw, Rv = noiseCovs

        # Run a simulation for the current value of the noise 
        # covariances in the iteration. 
        simData = generateData(linModel['A'], linModel['B'], 
                               linModel['C'], useq, x0, Qw, Rv, Delta)

        # Add the generated simulation data to the list.
        trainingData_list += [simData]

    # Specify the LQR parameters (specifying these parameters here, 
    # so that these don't get reset in every subsequent calculations). 
    xs = xs.squeeze()
    us = us.squeeze()
    regulatorPars = dict(gamma=0.98,
                         Q=1e+3*np.eye(Nx) @ np.diag(1/xs**2), 
                         R=1e+2*np.eye(Nu) @ np.diag(1/us**2), 
                         SRoc=1e+2*np.eye(Nu) @ np.diag(1/us**2))

    # Specify some closed-loop simulation parameters (number of trajectories, 
    # number of time steps in each trajectory).
    # (Same reason that these don't get set in every script)
    clSimPars = dict(Nsim=int(4*3600/Delta), 
                     Ntraj=500,
                     x0Lb=-10*np.ones((Nx, 1)),
                     x0Ub=10*np.ones((Nx, 1)))

    # Save all the data.
    sfeed_parameters = dict(linModel=linModel, 
                            trainingData_list=trainingData_list,
                            noiseCovs_list=noiseCovs_list, 
                            regulatorPars=regulatorPars, 
                            clSimPars=clSimPars)

    # Return. 
    return sfeed_parameters

def get_ofeed_calcparameters(Delta, us, ps):
    """ Get calculation parameters for the output feedback
        simulation experiments. """ 

    # Set the numpy seed for the output feedback calculations. 
    np.random.seed(35)

    # Get the linear model dictionary. 
    linModel = getHVACModel(Delta, sfeed=False)

    # Get the steady-state xs of the HVAC system (used in regulator 
    # tuning). 
    xs = getHVACXs(linModel['A'], linModel['B'], linModel['Bp'], us, ps)

    # Sizes. 
    Nx, Nu = linModel['B'].shape
    Ny = linModel['C'].shape[0]

    # Generate the control input sequence for the training data. 
    Nt = int(24*3600/Delta) # Number of time steps in the training data.
    useq_sigma = 625*np.diag([1., 1.]) # Variance of the input signal. 
    useq = mvnrnd(np.zeros((Nu, )), useq_sigma, Nt)

    # Get the list of noise covariance values for the simulation 
    # experiments. The noise covariances are specified as tuple
    # entries (Qw, Rv) in the list.
    noiseCovs_list = [(0*np.eye(Nx), 0*np.eye(Ny)),
                      (np.diag([8e-2, 1e-2, 9e-2, 1e-2]), 
                       np.diag([3e-1, 2e-1]))]

    # Initial state for the simulation. 
    x0 = np.zeros((Nx, 1))

    # Empty list to store all the training data. 
    trainingData_list = []

    # Loop over the noise covariance values, and generate 
    # different simulation data sets for each case.
    for noiseCovs in noiseCovs_list:

        # Extract the noise covariances. 
        Qw, Rv = noiseCovs

        # Run a simulation for the current value of the noise 
        # covariances in the iteration. 
        simData = generateData(linModel['A'], linModel['B'], 
                               linModel['C'], useq, x0, Qw, Rv, Delta)

        # Add the generated simulation data to the list. 
        trainingData_list += [simData]

    # Specify the LQR parameters (specifying these parameters here, 
    # so that these don't get reset in every subsequent calculations). 
    ys = (linModel['C'] @ xs).squeeze()
    us = us.squeeze()
    regulatorPars = dict(gamma=0.98,
                         Qy=1e+3*np.eye(Ny) @ np.diag(1/ys**2), 
                         R=1e+2*np.eye(Nu) @ np.diag(1/us**2), 
                         SRoc=1e+2*np.eye(Nu) @ np.diag(1/us**2))
    
    # Specify some closed-loop simulation parameters (number of trajectories, 
    # number of time steps in each trajectory). 
    # (Same reason that these don't get set in every script)
    clSimPars = dict(Nsim=int(4*3600/Delta), 
                     Ntraj=500, 
                     x0Lb=-10*np.ones((Nx, 1)),
                     x0Ub=10*np.ones((Nx, 1)))

    # Construct a dictionary with all the data.
    ofeed_parameters = dict(linModel=linModel, 
                            trainingData_list=trainingData_list,
                            noiseCovs_list=noiseCovs_list, 
                            regulatorPars=regulatorPars, 
                            clSimPars=clSimPars)

    # Return. 
    return ofeed_parameters

def main():

    # Sample time for measurements from the HVAC system (in seconds). 
    Delta = 60.

    # Specify a steady-state control input us and disturbance. 
    us = np.array([[50.], [40.]])
    ps = np.array([[60.], [40.], [22.]])
    
    # Get parameters for the state feedback calculations. 
    sfeed_parameters = get_sfeed_calcparameters(Delta, us, ps)

    # Get parameters for the output feedback calculations. 
    ofeed_parameters = get_ofeed_calcparameters(Delta, us, ps)

    # Save data.
    PickleTool.save([sfeed_parameters, 
                     ofeed_parameters], 'hvac_parameters.pickle')

main()