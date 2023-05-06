# [depends] %LIB%/linsystools.py
# [makes] pickle
import sys
sys.path.append('lib/')
import numpy as np
from sysidtools import PickleTool
from linsystools import LQGController, do_clsims_controllerevals

# Set the numpy random number generator seed. 
np.random.seed(10)

def main():

    # Load the HVAC parameters data file.
    hvac_parameters, _ = PickleTool.load('hvac_parameters.pickle')

    # Extract the linear model matrices.
    linModel = hvac_parameters['linModel']
    A, B, C = linModel['A'], linModel['B'], linModel['C']
    
    # Sizes.
    Nx, Nu = B.shape

    # Linear quadratic regulator parameters.
    gamma = hvac_parameters['regulatorPars']['gamma']
    Q = hvac_parameters['regulatorPars']['Q']
    R = hvac_parameters['regulatorPars']['R']
    SRoc = hvac_parameters['regulatorPars']['SRoc']

    # Specify the simulation index of which to use the noise covariances 
    # for the closed-loop simulation. 
    simIdx = 1

    # Extract the true noise covariances. 
    Qw, Rv = hvac_parameters['noiseCovs_list'][simIdx]

    # Setup the LQR controller. 
    controller = LQGController(A=A, B=B, C=C, gamma=gamma, Qy=Q, 
                               R=R, SRoc=SRoc, Qw=Qw, Rv=Rv, 
                               Swv=np.zeros((Nx, Nx)), 
                               xPrior=np.zeros((Nx, 1)), 
                               yprev=np.zeros((Nx, 1)), 
                               uprev=np.zeros((Nu, 1)))

    # Extract the dictionary of closed-loop simulation parameters. 
    clSimPars = hvac_parameters['clSimPars']

    # Run the closed-loop simulations.
    clSimsData = do_clsims_controllerevals(clSimPars['Nsim'], 
                                           clSimPars['Ntraj'],
                                           A, B, C, Qw, Rv, Q,
                                           R, SRoc, controller, 
                                           clSimPars['x0Lb'], clSimPars['x0Ub'])
    
    # Get the mean of all the performance metrics (to print). 
    lamT_mean = np.mean(clSimsData['lamT'])

    # Print the mean of the performance metric. 
    print("Mean of the performance metrics: " + str(lamT_mean))

    # Save data.
    PickleTool.save(clSimsData, 'hvac_sfeed_optlqg.pickle')

main()