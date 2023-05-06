# [depends] %LIB%/linsystools.py %LIB%/sysidtools.py
# [depends] hvac_parameters.pickle
# [makes] pickle
import sys
sys.path.append('lib/')
import numpy as np
from sysidtools import PickleTool, get_simData_forliftedsys
from sysidtools import doMLE_fixedDataSize_ofeed
from linsystools import get_optSK_ofeed, get_liftedsysmodelmats
from linsystools import LQGController, do_clsims_controllerevals

# Set the numpy random number generator seed. 
np.random.seed(10)

def main():

    # Load the output feedback data.
    _, hvac_parameters = PickleTool.load('hvac_parameters.pickle')
    
    # Extract the linear dynamic model matrices.
    linModel = hvac_parameters['linModel']
    A, B, C = linModel['A'], linModel['B'], linModel['C']

    # Number of past inputs and outputs for the lifted state.
    Np = 2

    # Number of states, control inputs, and measurements.
    Nx, Nu = B.shape
    Ny = C.shape[0]

    # Extract the LQR tuning parameters. 
    gamma = hvac_parameters['regulatorPars']['gamma']
    Qy = hvac_parameters['regulatorPars']['Qy']
    R = hvac_parameters['regulatorPars']['R']
    SRoc = hvac_parameters['regulatorPars']['SRoc']

    # Get linear dynamic model matrices for the lifted system and the 
    # augmented version to check if the estimated feedback law is stabilizing. 
    Az, Bz, Cz = get_liftedsysmodelmats(A=A, B=B, C=C, Np=Np)

    # Compute the optimal, S, K, and lam.
    S, K = get_optSK_ofeed(Az, Bz, Cz, gamma, Qy, R, SRoc)

    # Specify the simulation index for which to run the system identification 
    # algorithm.
    simIdx = 1

    # Extract the training data.
    simData = hvac_parameters['trainingData_list'][simIdx]
    # Transform the data for the lifted state. 
    simData = get_simData_forliftedsys(simData, sfeed=False, Np=Np)
    # Get the lifted state, control input, and measurement sequence.
    zSeq, uSeq, ySeq = simData.x, simData.u, simData.y

    # Run the maximum likelihood algorithm and get the linear model matrices 
    # and the noise covariances. 
    (Ahat, Bhat, Chat, 
     Dhat, QwHat, RvHat, SwvHat) = doMLE_fixedDataSize_ofeed(zSeq, uSeq, ySeq)

    # State size of the lifted system. 
    Nz = Np*Ny + Np*Nu

    # Determine the estimated S, K, and lamda using the 
    # estimated model and the noise covariances.
    # First get the LQR Regulator penalty.
    # Compute the S, K, and lambda.
    Shat, Khat = get_optSK_ofeed(Ahat, Bhat, Chat, gamma, Qy, R, SRoc)

    # Check if the estimated controller is stabilizing by examining the 
    # closed-loop eigenvalues. 
    AKz = Az + Bz @ Khat
    print("Eigenvalues of the closed-loop system: " + 
          str(np.abs(np.linalg.eigvals(AKz))))

    # Get the noise covariances for the plant simulation in the case studies. 
    Qw, Rv = hvac_parameters['noiseCovs_list'][simIdx]

    # Create a LQR controller object. 
    controller = LQGController(A=Ahat, B=Bhat, C=Chat, gamma=gamma, Qy=Qy, 
                               R=R, SRoc=SRoc, Qw=QwHat, Rv=RvHat, Swv=SwvHat, 
                               xPrior=np.zeros((Nz, 1)), 
                               yprev=np.zeros((Ny, 1)), 
                               uprev=np.zeros((Nu, 1)), liftedsys=True)

    # Extract the dictionary of closed-loop simulation parameters. 
    clSimPars = hvac_parameters['clSimPars']

    # Run the closed-loop simulation and get a data object. 
    clSimsData = do_clsims_controllerevals(clSimPars['Nsim'], 
                                           clSimPars['Ntraj'],
                                           A, B, C, Qw, Rv, Qy, 
                                           R, SRoc, controller, 
                                           clSimPars['x0Lb'], clSimPars['x0Ub'])
    
    # Get the mean of all the performance metrics (to print). 
    lamT_mean = np.mean(clSimsData['lamT'])

    # Print the error in the estimates. 
    errS = np.linalg.norm(Shat - S)/np.linalg.norm(S)
    errK = np.linalg.norm(Khat - K)/np.linalg.norm(K)
    print("Error in the S matrix estimate: " + str(errS))
    print("Error in the feedback gain K estimate: " + str(errK))
    print("Mean of the performance metrics: " + str(lamT_mean))

    # Save the closed-loop trajectories, and data to make the 
    # histogram of the cost plots.
    PickleTool.save(clSimsData, 'hvac_ofeed_sysid_fxddata.pickle')

main()