# [depends] %LIB%/sysidtools.py %LIB%/linsystools.py %LIB%/qlearningtools.py
# [depends] hvac_parameters.pickle
# [makes] pickle
import sys
sys.path.append('lib/')
import numpy as np
from sysidtools import PickleTool, get_simData_forliftedsys
from linsystools import get_optSPKLam_sfeed, do_clsims_controllerevals
from linsystools import get_augmats_rocpenalty_sfeed, ModelFreeController
from qlearningtools import doLSPI_varyingDataSize_sfeed

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
    Sopt, _, Kopt, lamOpt = get_optSPKLam_sfeed(A, B, gamma, Q, R, SRoc, Qw)

    # Number of iterations in the LSPI algorithm and number of data set 
    # partitions. 
    NumIter = 15
    NumDataPartition = 20

    # Eun the LSPI algorith to determine the S, K, and contribution 
    # of the noise.
    (errS, errK, errLam, 
     estTimes, estK_list) = doLSPI_varyingDataSize_sfeed(zSeq, uSeq, K0, NumIter,
                                            gamma, Qaug, Raug, Maug, 
                                            NumDataPartition, Sopt, Kopt, 
                                            lamOpt, Qwaug, algorithm='knownQw')

    # Extract the dictionary of closed-loop simulation parameters. 
    clSimPars = hvac_parameters['clSimPars']

    # Create a list to store the simulation data, largest closed-loop 
    # eigenvalue, and the mean of the performance metric. 
    clSimsData_list = []
    maxClEigVal_list = []
    lamTmean_list = []

    # Loop over all the estimated feedback laws.
    for Khat in estK_list:

        # Specify the random number generator at the beginning of each 
        # set of closed-loop simulation. 
        np.random.seed(10)

        # Create the controller object. 
        controller = ModelFreeController(K=Khat, alpha=0.2, Np=1, Nu=Nu, Ny=Ny, 
                                        uprev=np.zeros((Nu, 1)))

        # Run the closed-loop simulation and get a data object. 
        clSimsData = do_clsims_controllerevals(clSimPars['Nsim'], 
                                            clSimPars['Ntraj'],
                                            A, B, C, Qw, Rv, Q,
                                            R, SRoc, controller, 
                                            clSimPars['x0Lb'], 
                                            clSimPars['x0Ub'])    

        # Save the closed-loop simulation data to the list. 
        clSimsData_list += [clSimsData]

        # Get and save the maximum closed-loop eigenvalue. 
        AK = Aaug + Baug @ controller.K
        maxClEigVal = np.max(np.abs(np.linalg.eigvals(AK)))
        maxClEigVal_list += [maxClEigVal]

        # Get and save the mean of the performance metric.
        lamTmean = np.mean(clSimsData['lamT'])
        lamTmean_list += [lamTmean]

    # Get a time array to plot the errors in the quantities.
    NtEachPartition =  len(simData.t)/NumDataPartition
    t = np.arange(1, NumDataPartition + 1)*NtEachPartition*linModel['Delta']

    # Get the maximum closed-loop eigenvalues and mean of the performance 
    # metric as arrays. 
    maxClEigVals = np.array(maxClEigVal_list)
    lamTmeans = np.array(lamTmean_list)

    # Create a dictionary of the results to save. 
    hvac_sfeed_lspi_kqw = dict(t=t, errS=errS, errK=errK, errLam=errLam, 
                               estTimes=estTimes, 
                               clSimsData_list=clSimsData_list, 
                               maxClEigVals=maxClEigVals, 
                               lamTmeans=lamTmeans)

    # Save the closed-loop trajectories, and histogram of the cost plots. 
    PickleTool.save(hvac_sfeed_lspi_kqw, 'hvac_sfeed_lspi_kqw.pickle')

main()