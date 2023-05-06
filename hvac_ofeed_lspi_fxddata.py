# [depends] %LIB%/linsystools.py %LIB%/sysidtools.py %LIB%/qlearningtools.py
# [depends] hvac_parameters.pickle
# [makes] pickle
import sys
sys.path.append('lib/')
import numpy as np
from sysidtools import PickleTool, get_simData_forliftedsys
from qlearningtools import doLSPI_fixedDataSize_ofeed
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

    # Extract the LQR regulator tuning parameters. 
    gamma = hvac_parameters['regulatorPars']['gamma']
    Qy = hvac_parameters['regulatorPars']['Qy']
    R = hvac_parameters['regulatorPars']['R']
    SRoc = hvac_parameters['regulatorPars']['SRoc']

    # Simulation index of the data on which to run the Q-learning algorithm. 
    simIdx = 1

    # Lists to store the mean performance metric and feedback laws. 
    Khat_list, cond_numbers = [], []

    # List of Nps to iterate over.
    Np_vals = [i for i in range(1, 9)]

    # Number of past inputs and outputs for the augmented linear model.
    for Np in Np_vals:
        
        # Informative print statement.
        print("Running algorithm for Np = " + str(Np))

        # Sizes.
        Nx, Nu = B.shape
        Ny = C.shape[0]

        # Number of state in the lifted state. 
        Nz = Np*(Ny + Nu)

        # Get linear dynamic model matrices for the lifted system and the 
        # augmented version to check if the estimated feedback law is stabilizing. 
        Az, Bz, Cz = get_liftedsysmodelmats(A=A, B=B, C=C, Np=Np)

        # Compute the optimal, S, K, and lam.
        S, K = get_optSK_ofeed(Az, Bz, Cz, gamma, Qy, R, SRoc)

        # Extract and transform the the simulation data for augmented system. 
        simData = hvac_parameters['trainingData_list'][simIdx]
        # Get data for the lifted state.
        simData = get_simData_forliftedsys(simData, sfeed=False, Np=Np)
        zSeq, uSeq, ySeq = simData.x, simData.u, simData.y
        
        # Compute and print the condition number of data matrix. 
        cond = np.linalg.cond(zSeq.T)
        print("Condition number of the data matrix: " + str(cond))

        # Initial gain.
        K0 = np.zeros((Nu, Nz))
        
        # Number of iterations in the LSPI algorithm. 
        NumIter = 15

        # Run the LSPI algorithm to determine the S, K, and lam estimates. 
        (Shat, Khat, 
         lamHat) = doLSPI_fixedDataSize_ofeed(zSeq, uSeq, ySeq, K0, 
                                            NumIter, gamma, Qy, R, SRoc)
        
        # Print the eigenvalues of the closed-loop system.
        AKz = Az + Bz @ Khat
        print("Eigenvalues of the closed-loop system: " + 
            str(np.abs(np.linalg.eigvals(AKz))))

        # Store the estimated feedback law and condition number for the 
        # current value of Np. 
        Khat_list += [Khat]
        cond_numbers += [cond]

    # Get the noise covariances for the plant simulation in the case studies. 
    Qw, Rv = hvac_parameters['noiseCovs_list'][simIdx]

    # Create a controller class.
    Khat = Khat_list[1] # First extract the feedback controller of interest.
    Np = Np_vals[1]
    controller = ModelFreeController(K=Khat, alpha=0.2, Np=Np, Nu=Nu, Ny=Ny, 
                                     uprev=np.zeros((Nu, 1)))

    # Extract the dictionary of closed-loop simulation parameters. 
    clSimPars = hvac_parameters['clSimPars']

    # Run the closed-loop simulation and get a data object. 
    clSimsData = do_clsims_controllerevals(clSimPars['Nsim'], 
                                           clSimPars['Ntraj'],
                                           A, B, C, Qw, Rv, Qy, 
                                           R, SRoc, controller, 
                                           clSimPars['x0Lb'], clSimPars['x0Ub'])
    
    # Get the mean of all the performance metrics and print. 
    lamT_mean = np.mean(clSimsData['lamT'])
    print("Mean of the performance metrics: " + str(lamT_mean))

    # Create another dictionary to select the performance metric. 
    order_selec_data = dict(Np_vals=np.array(Np_vals), 
                        cond_numbers=np.array(cond_numbers))

    # Save the closed-loop trajectories, and histogram of the cost plots. 
    PickleTool.save([clSimsData, order_selec_data], 
                     'hvac_ofeed_lspi_fxddata.pickle')

main()