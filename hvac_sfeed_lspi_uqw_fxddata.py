# [depends] %LIB%/sysidtools.py %LIB%/linsystools.py %LIB%/qlearningtools.py
# [depends] hvac_parameters.pickle
# [makes] pickle
import sys
sys.path.append('lib/')
import numpy as np
from sysidtools import PickleTool, get_simData_forliftedsys
from linsystools import get_optSPKLam_sfeed, get_augmats_rocpenalty_sfeed
from linsystools import ModelFreeController, do_clsims_controllerevals
from qlearningtools import doLSPI_fixedDataSize_sfeed

# Set the numpy random number generator seed. 
np.random.seed(10)

def main():

    # Load data for learning.
    hvac_parameters, _ = PickleTool.load('hvac_parameters.pickle')

    # Extract model and noise covariances.
    linModel = hvac_parameters['linModel']
    A, B, C = linModel['A'], linModel['B'], linModel['C']

    # Sizes.
    Nx, Nu = B.shape
    Ny = C.shape[0]

    # Extract the LQR regulator tuning parameters. 
    gamma = hvac_parameters['regulatorPars']['gamma']
    Q = hvac_parameters['regulatorPars']['Q']
    R = hvac_parameters['regulatorPars']['R']
    SRoc = hvac_parameters['regulatorPars']['SRoc']

    # Simulation index on which to perform the Q-learning calculation. 
    simIdx = 1

    # Extract the training data.
    simData = hvac_parameters['trainingData_list'][simIdx]
    # Get a lifted data set.
    simData = get_simData_forliftedsys(simData)
    # Extract the state and control input sequence.
    zSeq, uSeq = simData.x, simData.u

    # Extract the noise covariances. 
    Qw, Rv = hvac_parameters['noiseCovs_list'][simIdx]

    # Get the actual S, K, and lamda using the plant model 
    # and true noise covariance.
    S, _, K, lam = get_optSPKLam_sfeed(A, B, gamma, Q, R, SRoc, Qw)

    # Contruct the augmented matrices so a Lyapunov equation corresponding 
    # to a standard cross term can be solved to get the P matrix. 
    (Aaug, Baug, Qaug, 
     Raug, Maug, _) = get_augmats_rocpenalty_sfeed(A, B, Q, R, SRoc, Qw)

    # Specify the initial LQR gain for the LSPI algorithm. 
    K0 = np.zeros((Nu, Nx + Nu))

    # Run the LSPI algorithm. 
    NumIter = 15
    Shat, Khat, lamHat = doLSPI_fixedDataSize_sfeed(zSeq, uSeq, K0, NumIter, 
                                                    gamma, Qaug, Raug, Maug)

    # Create the controller object. 
    controller = ModelFreeController(K=Khat, alpha=0.4, Np=1, Nu=Nu, Ny=Ny, 
                                     uprev=np.zeros((Nu, 1)))

    # Extract the dictionary of closed-loop simulation parameters. 
    clSimPars = hvac_parameters['clSimPars']

    # Run the closed-loop simulation and get a data object. 
    clSimsData = do_clsims_controllerevals(clSimPars['Nsim'], 
                                           clSimPars['Ntraj'],
                                           A, B, C, Qw, Rv, Q,
                                           R, SRoc, controller, 
                                           clSimPars['x0Lb'], clSimPars['x0Ub'])
    
    # Get the mean of all the performance metrics (to print). 
    lamT_mean = np.mean(clSimsData['lamT'])

    # Get the eigenvalues of the closed-loop system. 
    AK = Aaug + Baug @ Khat
    AKEigvals = np.abs(np.linalg.eigvals(AK))

    # Print the error in the estimates. 
    errS = np.linalg.norm(Shat - S)/np.linalg.norm(S)
    errK = np.linalg.norm(Khat - K)/np.linalg.norm(K)
    errLam = np.linalg.norm(lamHat - lam)/np.linalg.norm(lam)
    print("Error in the S matrix estimate: " + str(errS))
    print("Error in the feedback gain K estimate: " + str(errK))
    print("Error in the noise contribution estimate: " + str(errLam))
    print("Mean of the performance metrics: " + str(lamT_mean))
    print("Closed-loop eigenvalues: " + str(AKEigvals))

    # Save the closed-loop trajectories, and histogram of the cost plots. 
    PickleTool.save(clSimsData, 'hvac_sfeed_lspi_uqw_fxddata.pickle')

main()