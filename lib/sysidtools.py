# [depends] linsystools.py
import pickle
import time
import collections
import numpy as np
from linsystools import get_optSK_ofeed, get_optSPKLam_sfeed
from numpy.random import multivariate_normal as mvnrnd

# Custom class to store simulation datasets.
SimData = collections.namedtuple('SimData',
                                ['t', 'x', 'u', 'y', 'p'])

class PickleTool:
    """ Class which contains a few static methods for saving and
        loading pickle data files conveniently. 
    """

    @staticmethod
    def load(filename):
        """ Wrapper to load a pickle file. """
        with open(filename, "rb") as stream:
            return pickle.load(stream)

    @staticmethod
    def save(data_object, filename):
        """ Wrapper to save a data_object in a pickle file."""
        with open(filename, "wb") as stream:
            pickle.dump(data_object, stream)

def sample_prbs_like(*, num_change, num_steps,
                     lb, ub, mean_change, sigma_change):
    """ Sample a PRBS like sequence.
        num_change: Number of changes in the signal.
        num_steps: Total number of time steps in the signal.
        mean_change: The mean number of times each value in the 
                     signal should be repeated.
        sigma_change: standard deviation of the number of times each value
                      in the signal is repeated.
    """

    def sample_numRepeats(num_change, num_simulation_steps,
                          mean_change, sigma_change):
        """ Sample the number of times each value in the PRBS signal 
            should be repeated. 
        """

        # Sample the number of repeats using a Gaussian distribution.
        # Then round off the sampled values to the nearest lowest integer and
        # change all the negative values to zero.
        numRepeat = sigma_change*np.random.randn(num_change - 1) + mean_change
        numRepeat = np.rint(numRepeat).astype(int)
        numRepeat = np.where(numRepeat <= 0, 0, numRepeat)

        # Assign the number of repeats for the last value as the number
        # of the timesteps remaining in the signal.
        numRepeat = np.append(numRepeat,
                              num_simulation_steps - np.sum(numRepeat))

        # Return.
        return numRepeat

    # Signal dimension.
    signal_dimension = lb.shape[0]

    # Reduce any unnecessary dimension if present in the
    # provided lower and upper bound values.
    lb = lb.squeeze()
    ub = ub.squeeze()

    # Sample values of the PRBS signal.
    values = (ub - lb)*np.random.rand(num_change, signal_dimension) + lb

    # Sample how many times each of the above sampled values
    # in the signal should be repeated.
    numRepeat = sample_numRepeats(num_change, num_steps,
                                  mean_change, sigma_change)

    # Repeat the values for the numRepeat times to get the final
    # PRBS signal.
    signal = np.repeat(values, numRepeat, axis=0)

    # Return.
    return signal

def generateData(A, B, C, useq, x0, Qw, Rv, Delta, Bp=None, pseq=None):
    """ Simulate a linear system with the provided control input sequence, 
        inital state, and noise sampled from a Gaussian distribution with the 
        given covariances. 

        This function is used to generate data for either Q-learning or 
        system identification algorithms.
    """

    # Output and state sizes.
    Ny, Nx = C.shape

    # Number of timesteps in the control input sequence. 
    Nt = useq.shape[0]
    
    # Initial state and measurement.
    xt = x0
    yt = C @ xt + mvnrnd(np.zeros((Ny, )), Rv)[:, np.newaxis]
    
    # Lists to store the state, measurement, and control input sequences.
    x = [xt]
    y = [yt]

    # Loop over the number of time steps.
    for t in range(Nt):
        
        # Extract the control input for the current time step.
        ut = useq[t:t+1, :].T

        # Sample process and measurement noises at the current time step.
        wt = mvnrnd(np.zeros((Nx, )), Qw)[:, np.newaxis]
        vt = mvnrnd(np.zeros((Ny, )), Rv)[:, np.newaxis]

        # Propagate the linear plant.
        if pseq is None and Bp is None:
            xt = A @ xt + B @ ut + wt
        else:
            pt = pseq[t:t+1, :].T
            xt = A @ xt + B @ ut + Bp @ pt + wt


        # Get measurement.
        yt = C @ xt + vt

        # Save the state and measurement to the list.
        x += [xt]
        y += [yt]

    # Create arrays for the time, state, and measurement sequences.
    t = np.arange(0, Nt + 1, 1)*Delta
    x = np.array(x).squeeze(axis=-1)
    y = np.array(y).squeeze(axis=-1)

    # Create a simData object to return.
    simData = SimData(t=t, x=x, u=useq, y=y, p=pseq)

    # Return.
    return simData

def doMLE_fixedDataSize_sfeed(xSeq, uSeq):
    """ Get the linear model matrices (A, B) and the noise covariance (Qw)
        using the provided data set with the Maximum Likelihood Esimation 
        method.
    """

    # Number of time steps. 
    Nt = uSeq.shape[0]

    # State and control input sizes.
    Nx = xSeq.shape[1]
    Nu = uSeq.shape[1]

    # Create empty lists to store the s and t vectors in the MLE algorithm.
    sSeq = []
    tSeq = []

    # Loop over the number of time steps. 
    for i in range(Nt):

        # Get the current state, control, measurement, and state 
        # at the next time step. 
        xi = xSeq[i:i+1, :].T
        ui = uSeq[i:i+1, :].T
        xiplus = xSeq[i+1:i+2, :].T

        # Get the s and t vectors at the current time step. 
        si = xiplus
        ti = np.concatenate((xi, ui), axis=0)

        # Append the current s and t vectors to their lists. 
        sSeq += [si]
        tSeq += [ti]

    # Get the required matrices to compute the estimates of the linear 
    # model matrices.
    ST = np.zeros((Nx, Nx + Nu))
    TT = np.zeros((Nx + Nu, Nx + Nu))
    for si, ti in zip(sSeq, tSeq):
        ST += si @ ti.T 
        TT += ti @ ti.T

    # Get the linear model estimate.
    theta = ST @ np.linalg.pinv(TT)
    A, B = theta[:Nx, :Nx], theta[:Nx, Nx:]

    # Get the unbiased covariance estimate. 
    Qw = np.zeros((Nx, Nx))
    for si, ti in zip(sSeq, tSeq):
        Qw += (si - theta @ ti) @ (si - theta @ ti).T
    Qw /= (Nt - Nx - Nu)

    # Return. 
    return A, B, Qw

def doMLE_varyingDataSize_sfeed(xSeq, uSeq, gamma, Q, R, SRoc, 
                                NumDataPartition, Sopt, Kopt, lamOpt):
    """ Do maximum likelihood estimate for the state feedback case 
        with varying amounts of training data. 
    """
    
    # Lists to store the errors in the estimated S, K, lambda, and the 
    # estimation times of the feedback laws.
    errS = []
    errK = []
    errLam = []
    estTimes = []

    # Size of the state and control input. 
    Nx = xSeq.shape[1]
    Nu = uSeq.shape[1]

    # Lists of the estimated linear model matrices, noise covariance, and the 
    # feedback gain K. 
    estA_list = []
    estB_list = []
    estQw_list = []
    estK_list = []

    # Total number of time steps in the data set. 
    Nt = uSeq.shape[0]

    # Size of each partion of the data set.
    NtEachPartition = Nt//NumDataPartition

    # Loop over the number of partitions of the data set.
    for i in range(1, NumDataPartition + 1):
        
        # Print the current data set partition index 
        # at which maximum likelihood is being run.
        print("Running Maximum Likelihood on Data Set: " + str(i))

        # Start time of the calculations. 
        tStart = time.time()

        # Get the data set for the current partition. 
        xSeqi = xSeq[:i*NtEachPartition + 1, :]
        uSeqi = uSeq[:i*NtEachPartition, :]

        # Do maximum likelihood estimation to get the linear models. 
        # The noise covariance is actually not used 
        # anywhere, so don't extract it. 
        Ai, Bi, Qwi = doMLE_fixedDataSize_sfeed(xSeqi, uSeqi)

        # Get the optimal S, K, and lam for this estimated model/covariance. 
        Si, _, Ki, lami = get_optSPKLam_sfeed(Ai, Bi, gamma, Q, R, SRoc, Qwi)

        # End time of the calculations. 
        tEnd = time.time()

        # Compute and store the errors in the estimated S, K, and Lam 
        # compared to the optimal quantities. 
        errS += [np.linalg.norm(Sopt - Si)/np.linalg.norm(Sopt)]
        errK += [np.linalg.norm(Kopt - Ki)/np.linalg.norm(Kopt)]
        errLam += [np.linalg.norm(lamOpt - lami)/np.linalg.norm(lamOpt)]

        # Store the total model estimation and gain calculation time. 
        estTimes += [tEnd - tStart]

        # Store the estimated linear model, and noise covariances to the list. 
        estA_list += [Ai]
        estB_list += [Bi]
        estQw_list += [Qwi]
        estK_list += [Ki]

    # Get the errors as arrays. 
    errS = np.array(errS)
    errK = np.array(errK)
    errLam = np.array(errLam)
    estTimes = np.array(estTimes)

    # Return. 
    return (errS, errK, errLam, estTimes, 
            estA_list, estB_list, estQw_list, estK_list)

def doMLE_fixedDataSize_ofeed(xSeq, uSeq, ySeq):
    """ Run the maximum likelihood estimation algorithm with the provided 
        data sets to estimate the linear model matrices (A, B, C, D) and 
        the noise covariances (Qw, Rv, Swv).
    """

    # Number of time steps. 
    Nt = uSeq.shape[0]

    # State, control input, and measurement sizes.
    Nx = xSeq.shape[1]
    Nu = uSeq.shape[1]
    Ny = ySeq.shape[1]

    # Lists to store the s and t vectors for each time step.
    sSeq = []
    tSeq = []

    # Loop over the number of time steps. 
    for i in range(Nt):

        # Get the current state, control, measurement, and state 
        # at the next time step. 
        xi = xSeq[i:i+1, :].T
        ui = uSeq[i:i+1, :].T
        yi = ySeq[i:i+1, :].T
        xiplus = xSeq[i+1:i+2, :].T

        # Get the s and t vectors at the current time step. 
        si = np.concatenate((xiplus, yi))
        ti = np.concatenate((xi, ui))

        # Append the current s and t vectors to the list. 
        sSeq += [si]
        tSeq += [ti]

    # Get the incumbent matrices to compute the estimates of the linear 
    # model matrices.
    ST = np.zeros((Nx + Ny, Nx + Nu))
    TT = np.zeros((Nx + Nu, Nx + Nu))
    for si, ti in zip(sSeq, tSeq):
        ST += si @ ti.T
        TT += ti @ ti.T

    # Compute the estimates of the linear model matrices.
    theta = ST @ np.linalg.pinv(TT)
    A, B = theta[:Nx, :Nx], theta[:Nx, Nx:]
    C, D = theta[Nx:, :Nx], theta[Nx:, Nx:]

    # First get the full estimate of the noise covariance. 
    SQwRv = np.zeros((Nx + Ny, Nx + Ny))
    for si, ti in zip(sSeq, tSeq):
        SQwRv += (si - theta @ ti) @ (si - theta @ ti).T
    SQwRv /= (Nt - Nx - Nu)

    # Extract the estimates of the covariances of the processs noise, 
    # measurement noise, and the cross correlation between the two. 
    Qw = SQwRv[:Nx, :Nx]
    Rv = SQwRv[-Ny:, -Ny:]
    Swv = SQwRv[:Nx, -Ny:]

    # Return. 
    return A, B, C, D, Qw, Rv, Swv

def doMLE_varyingDataSize_ofeed(xSeq, uSeq, ySeq, gamma, Qy, R, SRoc, 
                                NumDataPartition, Sopt, Kopt):
    """ Run the maximum likelihood estimation algorithm with incremental 
        amounts of data for the output feedback case. 
    """
    
    # Create lists to store the error in the S and K computations, and 
    # the estimation time for the feedback law.
    errS = []
    errK = []
    estTimes = []

    # Create lists to store the estimated linear models, noise covariances, 
    # and the feedback laws. 
    estA_list = []
    estB_list = []
    estC_list = []
    estD_list = []
    estQw_list = []
    estRv_list = []
    estSwv_list = []
    estK_list = []

    # Total number of time steps in the data set. 
    Nt = uSeq.shape[0]

    # Number of states. 
    Nx = xSeq.shape[1]

    # Size of each partion of the data set.
    NtEachPartition = Nt//NumDataPartition

    # Loop over the number of partitions of the data set.
    for i in range(1, NumDataPartition + 1):
        
        # Print the current data set partition index 
        # at which maximum likelihood is being run.
        print("Running Maximum Likelihood on Data Set: " + str(i))

        # Start of calculations. 
        tStart = time.time()

        # Get the data set for the current partition. 
        xSeqi = xSeq[:i*NtEachPartition + 1, :]
        uSeqi = uSeq[:i*NtEachPartition, :]
        ySeqi = ySeq[:i*NtEachPartition, :]

        # Run the maximum likelihood estimation algorithm and get the model
        # and noise covariance estimates. 
        (Ai, Bi, Ci, Di, 
         Qwi, Rvi, Swvi) = doMLE_fixedDataSize_ofeed(xSeqi, uSeqi, ySeqi)

        # Get the optimal S, K, and lam for this estimated model/covariance. 
        # Transform the output penalty matrix to state penalty matrix. 
        Si, Ki = get_optSK_ofeed(Ai, Bi, Ci, gamma, Qy, R, SRoc)

        # End of calculations. 
        tEnd = time.time()

        # Compute and store errors from the optimal ones. 
        errS += [np.linalg.norm(Sopt - Si)/np.linalg.norm(Sopt)]
        errK += [np.linalg.norm(Kopt - Ki)/np.linalg.norm(Kopt)]
        
        # Store the computation time. 
        estTimes += [tEnd - tStart]

        # Store the estimated model, noise covariance, and feedback law.
        estA_list += [Ai]
        estB_list += [Bi]
        estC_list += [Ci]
        estD_list += [Di]
        estQw_list += [Qwi]
        estRv_list += [Rvi]
        estSwv_list += [Swvi]
        estK_list += [Ki]

    # Get the errors and the computation times as numpy arrays. 
    errS = np.array(errS)
    errK = np.array(errK)
    estTimes = np.array(estTimes)

    # Return.
    return (errS, errK, estTimes,
            estA_list, estB_list, estC_list, estD_list, 
            estQw_list, estRv_list, estSwv_list, estK_list)

def get_simData_forliftedsys(simData, sfeed=True, Np=None):
    """ Create a data matrix for the lifted state for the Q-learning
        and system identification calculations. 
    """

    # Number of time steps.
    Nt = simData.u.shape[0]

    # Measurement and control input sizes. 
    Ny = simData.y.shape[1]
    Nu = simData.u.shape[1]

    # Time index of the simulation from when to start grabbing the data set. 
    if sfeed:
        NtStart = 1
    else:
        NtStart = Np

    # Create an empty list to store the lifted state. 
    zSeq = []
    
    # Loop over all the time steps in simulation.
    for t in range(NtStart, Nt + 1):
        
        # Get the sequences of the past ouputs and inputs. 
        if sfeed:
            
            # For the state feedback case, the current measurement (state)
            # and one previous control input is the full state.
            ypseqt = simData.y[t:t+1, :]
            upseqt = simData.u[t-1:t, :]
            
            # Get the lifted state at the current time step.
            zt = np.concatenate((np.reshape(ypseqt, (Ny, )), 
                                 np.reshape(upseqt, (Nu, ))))
        else:

            # For the output feedback case use the specified number of 
            # past measurements and control inputs as the state. 
            ypseqt = simData.y[t-Np:t, :]
            upseqt = simData.u[t-Np:t, :]
            
            # Get the lifted state at the current time step.
            zt = np.concatenate((np.reshape(ypseqt, (Np*Ny, )), 
                                 np.reshape(upseqt, (Np*Nu, ))))

        # Save the lifted state to the list. 
        zSeq += [zt] 

    # Get the state sequence as an array.
    zSeq = np.array(zSeq)

    # Check the rank of the "data matrix" containing the lifted state. 
    # rankZSeq = np.linalg.matrix_rank(zSeq.T)
    # assert rankZSeq == zSeq.shape[1], "The data matrix is not full rank."
    
    # Create the simdata object for the augmented system.
    simDataAugSystem = SimData(t=simData.t[NtStart:], 
                               x=zSeq, 
                               u=simData.u[NtStart:, :], 
                               y=simData.y[NtStart:, :], 
                               p=None)

    # Return.
    return simDataAugSystem