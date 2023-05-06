# [depends] %LIB%/linsystools.py %LIB%/sysidtools.py
# [depends] hvac_parameters.pickle
# [makes] pickle
import sys
sys.path.append('lib/')
import numpy as np
from sysidtools import PickleTool, doMLE_fixedDataSize_sfeed
from linsystools import get_optSPKLam_sfeed, get_augmats_rocpenalty_sfeed
from linsystools import LQGController, do_clsims_controllerevals

# Set the numpy random number generator seed. 
np.random.seed(10)

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

    # Contruct the augmented matrices so a Lyapunov equation corresponding 
    # to a standard cross term can be solved to get the P matrix. 
    (Aaug, Baug, Qaug, 
     Raug, Maug, Qwaug) = get_augmats_rocpenalty_sfeed(A, B, Q, R, SRoc, Qw)

    # Get the true S, K, and lamda based on the true 
    # HVAC model and noise covariance.
    S, _, K, lam = get_optSPKLam_sfeed(A, B, gamma, Q, R, SRoc, Qw)

    # Get the linear model and noise covariances estimates. 
    (Ahat, Bhat, QwHat) = doMLE_fixedDataSize_sfeed(xSeq, uSeq)

    # Get the estimated S, K, and noise contribution 
    # based on the estimated linear model and noise covariance.  
    Shat, _, Khat, lamHat = get_optSPKLam_sfeed(Ahat, Bhat, 
                                                gamma, Q, R, SRoc, QwHat)
    
    # Perform closed-loop simulations using the estimated model, noise 
    # covariance, and the feedback law.
    # Get a LQG controller object. 
    controller = LQGController(A=Ahat, B=Bhat, C=np.eye(Nx), 
                               gamma=gamma, Qy=Q, R=R, SRoc=SRoc, Qw=QwHat, 
                               Rv=0*np.eye(Nx), Swv=np.zeros((Nx, Nx)), 
                               xPrior=np.zeros((Nx, 1)), 
                               yprev=np.zeros((Nx, 1)), 
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
    PickleTool.save(clSimsData, 'hvac_sfeed_sysid_fxddata.pickle')

main()