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

    # Extract the linear model.
    linModel = hvac_parameters['linModel']
    A, B, C = linModel['A'], linModel['B'], linModel['C']

    # Sizes.
    Nx, Nu = B.shape
    Ny = C.shape[1]

    # Choose regulator tuning parameters. 
    gamma = hvac_parameters['regulatorPars']['gamma']
    Q = hvac_parameters['regulatorPars']['Q']
    R = hvac_parameters['regulatorPars']['R']
    SRoc = hvac_parameters['regulatorPars']['SRoc']

    # Index of the simulation for which to do the Q-learining 
    # calculation. 
    simIdx = 1

    # Extract the training data.
    simData = hvac_parameters['trainingData_list'][simIdx]
    # Get a lifted data set.
    simData = get_simData_forliftedsys(simData)
    # Extract the state and control input sequence.
    zSeq, uSeq = simData.x, simData.u

    # Get the noise covariances.
    Qw, Rv = hvac_parameters['noiseCovs_list'][simIdx]

    # Contruct the augmented matrices so a Lyapunov equation corresponding 
    # to a standard cross term can be solved to get the P matrix. 
    (Aaug, Baug, Qaug, 
     Raug, Maug, Qwaug) = get_augmats_rocpenalty_sfeed(A, B, Q, R, SRoc, Qw)

    # Initial gain. 
    K0 = np.zeros((Nu, Nx + Nu))

    # Compute the optimal S, K, and lambda.
    S, _, K, lam = get_optSPKLam_sfeed(A, B, gamma, Q, R, SRoc, Qw)

    # Eun the LSPI algorith to determine the S, K, and contribution 
    # of the noise.
    NumIter = 15
    Shat, Khat, lamHat = doLSPI_fixedDataSize_sfeed(zSeq, uSeq, K0, NumIter, 
                                                    gamma, Qaug, Raug, Maug, 
                                                    Qwaug)

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
    print("Error in the S matrix estimate: " + str(errS))
    print("Error in the feedback gain K estimate: " + str(errK))
    print("Mean of the performance metrics: " + str(lamT_mean))
    print("Closed-loop eigenvalues: " + str(AKEigvals))

    # Save the closed-loop trajectories, and histogram of the cost plots. 
    PickleTool.save(clSimsData, 'hvac_sfeed_lspi_kqw_fxddata.pickle')

main()