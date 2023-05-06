# [depends] %LIB%/linsystools.py %LIB%/sysidtools.py
# [depends] hvac_parameters.pickle
# [makes] pickle
import sys
sys.path.append('lib/')
import numpy as np
from sysidtools import PickleTool, get_simData_forliftedsys
from sysidtools import doMLE_varyingDataSize_ofeed
from linsystools import get_optSK_ofeed, get_liftedsysmodelmats
from linsystools import LQGController, do_clsims_controllerevals

def main():

    # Load the output feedback data.
    _, hvac_parameters = PickleTool.load('hvac_parameters.pickle')
    
    # Extract the linear dynamic model matrices.
    linModel = hvac_parameters['linModel']
    A, B, C = linModel['A'], linModel['B'], linModel['C']

    # Number of past inputs and outputs for the lifted state.
    Np = 2

    # Number of states, control inputs, and measurements.
    Nu = B.shape[1]
    Ny = C.shape[0]
    Nz = Np*(Ny + Nu)

    # Extract the LQR tuning parameters. 
    gamma = hvac_parameters['regulatorPars']['gamma']
    Qy = hvac_parameters['regulatorPars']['Qy']
    R = hvac_parameters['regulatorPars']['R']
    SRoc = hvac_parameters['regulatorPars']['SRoc']

    # Get linear dynamic model matrices for the lifted system and the 
    # augmented version to check if the estimated feedback law is stabilizing. 
    Az, Bz, Cz = get_liftedsysmodelmats(A=A, B=B, C=C, Np=Np)

    # Compute the optimal, S, K, and lam.
    Sopt, Kopt = get_optSK_ofeed(Az, Bz, Cz, gamma, Qy, R, SRoc)

    # Specify the simulation index for which to run the system identification 
    # algorithm.
    simIdx = 1

    # Extract the training data.
    simData = hvac_parameters['trainingData_list'][simIdx]
    # Transform the data for the lifted state. 
    simData = get_simData_forliftedsys(simData, sfeed=False, Np=Np)
    # Get the lifted state, control input, and measurement sequence.
    zSeq, uSeq, ySeq = simData.x, simData.u, simData.y

    # Specify the number of partitions of the dataset.
    NumDataPartition = 12

    # Run the maximum likelihood algorithm and get the linear model matrices 
    # and the noise covariances. 
    (errS, errK, estTimes,
     estA_list, estB_list, estC_list, estD_list, 
     estQw_list, estRv_list, 
     estSwv_list, estK_list) = doMLE_varyingDataSize_ofeed(zSeq, uSeq, ySeq, 
                                            gamma, Qy, R, SRoc, 
                                            NumDataPartition, Sopt, Kopt)

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

    # Loop over the estimated models and noise covariances. 
    for (Ahat, Bhat, Chat, 
         QwHat, RvHat, SwvHat) in zip(estA_list, estB_list, estC_list, 
                                      estQw_list, estRv_list, estSwv_list):

        # Set the numpy random number generator seed. 
        np.random.seed(10)

        # Create a LQR controller object. 
        controller = LQGController(A=Ahat, B=Bhat, C=Chat, gamma=gamma, Qy=Qy, 
                                    R=R, SRoc=SRoc, Qw=QwHat, Rv=RvHat, 
                                    Swv=SwvHat, 
                                    xPrior=np.zeros((Nz, 1)), 
                                    yprev=np.zeros((Ny, 1)), 
                                    uprev=np.zeros((Nu, 1)), 
                                    liftedsys=True)

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
    hvac_ofeed_sysid = dict(t=t, estTimes=estTimes, 
                            clSimsData_list=clSimsData_list, 
                            maxClEigVals=maxClEigVals, 
                            lamTmeans=lamTmeans)        

    # Save the closed-loop trajectories, and data to make the 
    # histogram of the cost plots.
    PickleTool.save(hvac_ofeed_sysid, 'hvac_ofeed_sysid.pickle')

main()