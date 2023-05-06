# [depends] %LIB%/linsystools.py %LIB%/sysidtools.py %LIB%/qlearningtools.py
# [depends] hvac_parameters.pickle
# [makes] pickle
import sys
sys.path.append('lib/')
import numpy as np
from sysidtools import PickleTool, get_simData_forliftedsys
from qlearningtools import doLSPI_varyingDataSize_ofeed
from linsystools import get_optSK_ofeed, get_liftedsysmodelmats
from linsystools import ModelFreeController, do_clsims_controllerevals

# Set the numpy random number generator seed. 
np.random.seed(10)

def main():

    # Load data for learning.
    _, hvac_parameters = PickleTool.load('hvac_parameters.pickle')

    # Extract the linear model matrices of the plant model. 
    linModel = hvac_parameters['linModel']
    A, B, C = linModel['A'], linModel['B'], linModel['C']

    # Number of past inputs and outputs for the augmented linear model.
    Np = 2

    # Sizes.
    Nx, Nu = B.shape
    Ny = C.shape[0]
    Nz = Np*(Ny + Nu)

    # Extract the LQR regulator tuning parameters. 
    gamma = hvac_parameters['regulatorPars']['gamma']
    Qy = hvac_parameters['regulatorPars']['Qy']
    R = hvac_parameters['regulatorPars']['R']
    SRoc = hvac_parameters['regulatorPars']['SRoc']

    # Get linear dynamic model matrices for the lifted system and the 
    # augmented version to check if the estimated feedback law is stabilizing. 
    Az, Bz, Cz = get_liftedsysmodelmats(A=A, B=B, C=C, Np=Np)

    # Compute the optimal, S, K, and lam.
    SOpt, KOpt = get_optSK_ofeed(Az, Bz, Cz, gamma, Qy, R, SRoc)

    # Simulation index at which to run the Q-learning algorithm. 
    simIdx = 1

    # Extract and transform the the simulation data for augmented system. 
    simData = hvac_parameters['trainingData_list'][simIdx]
    # Get data for the lifted state.
    simData = get_simData_forliftedsys(simData, sfeed=False, Np=Np)
    zSeq, uSeq, ySeq = simData.x, simData.u, simData.y

    # Initial gain.
    K0 = np.zeros((Nu, Nz))
    
    # Number of iterations in the LSPI algorithm and number of data set 
    # partitions. 
    NumIter = 15
    NumDataPartition = 12

    # Run the LSPI algorithm to determine the S, K, and lam estimates. 
    (errS, errK, 
     estTimes, estK_list) = doLSPI_varyingDataSize_ofeed(zSeq, uSeq, ySeq, K0, 
                                                         NumIter, gamma, Qy, R, 
                                                         SRoc, NumDataPartition, 
                                                         SOpt, KOpt)

    # Get the noise covariances for the plant simulation in the case studies. 
    Qw, Rv = hvac_parameters['noiseCovs_list'][simIdx]

    # Extract the dictionary of closed-loop simulation parameters. 
    clSimPars = hvac_parameters['clSimPars']

    # Create empty lists to store the closed-loop simulation dataset, 
    # closed-loop eigenvalues of the system, and mean of the performance 
    # metrics across the trajectories. 
    clSimsData_list = []
    maxClEigVal_list = []
    lamTmean_list = []

    # Go over all the estimated feedback laws and run closed-loop simulations.
    for Khat in estK_list:

        # Set the numpy random number generator seed. 
        np.random.seed(10)

        # Create a controller class.
        controller = ModelFreeController(K=Khat, alpha=0.2, Np=Np, 
                                         Nu=Nu, Ny=Ny, 
                                         uprev=np.zeros((Nu, 1)))
 
        # Run the closed-loop simulation and get a data object. 
        clSimsData = do_clsims_controllerevals(clSimPars['Nsim'], 
                                            clSimPars['Ntraj'],
                                            A, B, C, Qw, Rv, Qy, 
                                            R, SRoc, controller, 
                                            clSimPars['x0Lb'], clSimPars['x0Ub'])

        # Add the closed-loop simulation data set to the list. 
        clSimsData_list += [clSimsData]

        # Get and save the maximum eigenvalue. 
        AKz = Az + Bz @ controller.K
        maxClEigVal = np.max(np.abs(np.linalg.eigvals(AKz)))
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
    hvac_ofeed_lspi = dict(t=t, estTimes=estTimes, 
                           clSimsData_list=clSimsData_list, 
                           maxClEigVals=maxClEigVals, 
                           lamTmeans=lamTmeans)
    
    # Save the closed-loop trajectories, and histogram of the cost plots. 
    PickleTool.save(hvac_ofeed_lspi, 'hvac_ofeed_lspi.pickle')

main()