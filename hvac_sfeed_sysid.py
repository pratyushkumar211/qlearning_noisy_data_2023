# [depends] %LIB%/linsystools.py %LIB%/sysidtools.py
# [depends] hvac_parameters.pickle
# [makes] pickle
import sys
sys.path.append('lib/')
import numpy as np
from sysidtools import PickleTool, doMLE_varyingDataSize_sfeed
from linsystools import get_augmats_rocpenalty_sfeed
from linsystools import get_optSPKLam_sfeed, LQGController
from linsystools import do_clsims_controllerevals

def main():

    # Load data for learning.
    hvac_parameters, _ = PickleTool.load('hvac_parameters.pickle')
    
    # Extract the true linear HVAC model.
    linModel = hvac_parameters['linModel']
    A, B, C = linModel['A'], linModel['B'], linModel['C']
    
    # Number of states and control inputs. 
    Nx, Nu = B.shape

    # Extract the LQR regulator tuning parameters. 
    gamma = hvac_parameters['regulatorPars']['gamma']
    Q = hvac_parameters['regulatorPars']['Q']
    R = hvac_parameters['regulatorPars']['R']
    SRoc = hvac_parameters['regulatorPars']['SRoc']

    # Specify the simulation index on which to perform the maximum 
    # likelihood (sys ID) calculation. 
    simIdx = 1

    # Extract the training data.
    simData = hvac_parameters['trainingData_list'][simIdx]
    xSeq, uSeq = simData.x, simData.u

    # Extract the true noise covariance used to generate the data. 
    Qw, Rv = hvac_parameters['noiseCovs_list'][simIdx]

    # Get the true S, K, and lamda based on the true HVAC model and 
    # noise covariance.
    Sopt, _, Kopt, lamOpt = get_optSPKLam_sfeed(A, B, gamma, Q, R, SRoc, Qw)    

    # Get the augmented model for the rate of change penalty. 
    # (used to determine the closed-loop eigenvalue). 
    (Aaug, Baug, _, 
     _, _, _) = get_augmats_rocpenalty_sfeed(A, B, Q, R, SRoc, Qw)

    # Specify the number of partitions of the dataset.
    NumDataPartition = 20

    # Get the linear model and noise covariances estimates for the varying 
    # amounts of training data. 
    (errS, errK, errLam, estTimes, 
     estA_list, estB_list, 
     estQw_list, estK_list) = doMLE_varyingDataSize_sfeed(xSeq, uSeq, gamma, 
                                                Q, R, SRoc, NumDataPartition,
                                                Sopt, Kopt, lamOpt)

    # Extract the dictionary of closed-loop simulation parameters. 
    clSimPars = hvac_parameters['clSimPars']

    # Create a list to store the simulation data, largest closed-loop 
    # eigenvalue, and the mean of the performance metric. 
    clSimsData_list = []
    maxClEigVal_list = []
    lamTmean_list = []

    # Loop over all the estimated models and noise covariances to perform
    # closed-loop simulations. 
    for Ahat, Bhat, QwHat in zip(estA_list, estB_list, estQw_list):

        # Specify the random number generator at the beginning of each 
        # set of closed-loop simulation. 
        np.random.seed(10)

        # Perform closed-loop simulations using the estimated model, noise 
        # covariance, and the feedback law.
        # Get a LQG controller object. 
        controller = LQGController(A=Ahat, B=Bhat, C=np.eye(Nx), 
                                   gamma=gamma, Qy=Q, R=R, SRoc=SRoc, Qw=QwHat, 
                                   Rv=0*np.eye(Nx), Swv=np.zeros((Nx, Nx)), 
                                   xPrior=np.zeros((Nx, 1)), 
                                   yprev=np.zeros((Nx, 1)), 
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
    hvac_sfeed_sysid = dict(t=t, errS=errS, errK=errK, errLam=errLam, 
                            estTimes=estTimes, clSimsData_list=clSimsData_list, 
                            maxClEigVals=maxClEigVals, 
                            lamTmeans=lamTmeans)

    # Save the closed-loop trajectories, and histogram of the cost plots. 
    PickleTool.save(hvac_sfeed_sysid, 'hvac_sfeed_sysid.pickle')

main()