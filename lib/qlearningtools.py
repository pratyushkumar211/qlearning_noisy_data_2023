import time
import numpy as np

def basisFromKron(z):
    """ Get a basis vector from the kronecker product z. 
        z is of dimension : n^2, 1
        basisZ is of dimension: n*(n+1)/2, 1
    """
    
    # Size of the input vector.
    n = int(np.sqrt(z.shape[0]))
    
    # Re-transform the Z vector into a matrix for easier 
    # extraction. 
    Zmat = z.T.reshape(n, n)

    # Create a list to store the elements of the 
    # symKronBasis.
    basisZ = []

    # Nested loop over elements of the matrix Zmat. 
    for i in range(n):
        for j in range(i, n):
            basisZ += [Zmat[i, j]]

    # Get the basis as a vector.
    basisZ = np.asarray(basisZ)[:, np.newaxis]

    # Return. 
    return basisZ

def smat(z):
    """ Transform the vector z into a symmetric matrix. """

    # Sizes. 
    Nz = z.shape[0]
    NsmatZ = int((-1+np.sqrt(8*Nz+1))/2)

    # Create the empty symmetric matrix. 
    SmatZ = np.zeros((NsmatZ, NsmatZ))

    # Counter to go through the elements of z. 
    zElemCounter = 0

    # Loop over all the elements in the empty matrix 
    # and corresponding elements in the z vector. 
    for i in range(NsmatZ):
        for j in range(i, NsmatZ):
            SmatZ[i, j] = z[zElemCounter, :]
            SmatZ[j, i] = z[zElemCounter, :]
            if i != j:
                SmatZ[i, j] = SmatZ[i, j]/2
                SmatZ[j, i] = SmatZ[j, i]/2
            zElemCounter +=1

    # Return.
    return SmatZ

def svec(S):
    """ Return the transformation of a symmetric matrix S to a vector
        according to the svec operator. """

    # Number of rows/column in the S matrix. 
    Ns = S.shape[0]

    # Empty list to store the elements of svecS.
    svecS = []

    # Loop through all the elements of the S matrix. 
    for i in range(Ns):
        for j in range(i, Ns):
            
            if i == j:
                svecS += [S[i, j]]
            else:
                svecS += [2*S[i, j]]

    # Get the svecS as an array. 
    svecS = np.asarray(svecS)[:, np.newaxis]

    # Return.
    return svecS

def partitionSymS(S, N2):
    """ Return the three partitions of a symmetric S matrix. 
        Give S = [S11, S12, 
                  S21, S22]
        S22 block is of size
    """

    # Size.
    N1 = S.shape[0] - N2

    # Get the partitiones.
    S11 = S[:N1, :N1]
    S22 = S[-N2:, -N2:]
    S21 = S[-N2:, :N1]

    # Return.
    return (S11, S22, S21)

def basisKwnNoiseCov(z, Qw, gamma, K):
    """ Basis function for the case when the noise 
        covariance is known. """

    # Sizes.
    Nu, Nx = K.shape

    # Transpose of the kronecker product of z.
    zKronT = np.kron(z, z).T
    
    # Get the noise term.
    eta = gamma/(1 - gamma)
    IK = np.concatenate((np.eye(Nx), K))
    vecQw = Qw.reshape(Nx**2, 1)
    noiseTerm = eta*(vecQw.T @ (np.kron(IK.T, IK.T)))

    # Get the basis. 
    basis = basisFromKron((zKronT + noiseTerm).T)

    # Return.
    return basis

def basisUnkwnNoiseCov(z):
    """ Basis function for the case when the noise 
        covariance is unknown. """

    # Get the kronecker product of z. 
    zKron = np.kron(z, z)

    # Get the basis. 
    basis = np.concatenate((basisFromKron(zKron), np.eye(1))) 

    # Return.
    return basis

def get_qLearningLSMats_sfeed(zSeq, uSeq, K, gamma, Q, R, M, Qw=None):
    """ Function to get the matrices for the Q-learning 
        least squares problem with process noise. 
    """

    # Sizes. 
    Nz = zSeq.shape[1]
    Nu = uSeq.shape[1]

    # Number of time steps. 
    Nt = uSeq.shape[0]

    # Select the basis function for approximating the 
    # Q-function based on the Qw flag and get the 
    # number of parameters in the Least squares problem. 
    if Qw is not None:
        basis = lambda z: basisKwnNoiseCov(z, Qw, gamma, K)
        numQPars = int((Nz + Nu)*(Nz + Nu + 1)/2)
    else:
        basis = basisUnkwnNoiseCov
        numQPars = int((Nz + Nu)*(Nz + Nu + 1)/2) + 1

    # Create empty A and b matrices. 
    A = np.zeros((numQPars, numQPars))
    b = np.zeros((numQPars, 1))

    # Loop over all the data points.
    for t in range(Nt):
        
        # Get the x, u, x^+ tuple. 
        zt = zSeq[t:t+1, :].T
        ut = uSeq[t:t+1, :].T
        ztplus = zSeq[t+1:t+2, :].T

        # Get z and z plus vectors. 
        basisZ = basis(np.concatenate((zt, ut)))
        basisZplus = basis(np.concatenate((ztplus, K @ ztplus)))

        # Get the stage cost.
        lzu = zt.T @ (Q @ zt) + ut.T @ (R @ ut) 
        lzu += zt.T @ (M @ ut) + ut.T @ (M.T @ zt) 

        # Get contributions to the A and B matrix. 
        A += basisZ @ (basisZ - gamma*basisZplus).T
        b += basisZ*lzu

    # Divide by the number of data points. 
    A = A/Nt
    b = b/Nt

    # Return the matrices. 
    return A, b

def doLSPI_fixedDataSize_sfeed(zSeq, uSeq, K0, NumIter, gamma, Q, R, M,
                               Qw=None, rankCheckTol=1e-6, 
                               *, algorithm, lambdaReg):
    """ Do least squares policy iteration. """

    # Control input size.
    Nu = uSeq.shape[1]

    # Initial LQR gain to iterate over.
    Ki = K0
    
    # Do Q-learning iterations.
    for i in range(NumIter):

        # Print the iteration number in the LSPI algorithm.
        print("Running the LSPI algorithm, Iteration Number: " + str(i + 1))

        # Use zero noise covariance to construct the matrices for the 
        # regularized LSPI case.
        if algorithm == 'unknownQw_regularizedLSPI':
            Qw = 0*np.eye(zSeq.shape[1])

        # Get matrices for Q-learning.
        A, b = get_qLearningLSMats_sfeed(zSeq, uSeq, Ki, gamma, Q, R, M, Qw=Qw)

        # Solve the least squares problem.
        if algorithm == 'unknownQw_modifiedLSPI' or algorithm == 'knownQw':
            thetaHat = np.linalg.solve(A, b)
        else:
            numQPars = A.shape[0] + 1
            Areg = np.concatenate((A, np.ones((A.shape[0], 1))), axis=1)
            thetaHat = Areg.T @ Areg + lambdaReg*np.eye(numQPars)
            thetaHat = np.linalg.pinv(thetaHat) @ (Areg.T @ b)

        # Print the rank of the regressor matrix.
        print("Rank deficiency in the regressor matrix: " + 
              str(A.shape[0] - np.linalg.matrix_rank(A, tol=rankCheckTol)))

        # S matrix estimate.
        if algorithm == 'unknownQw_modifiedLSPI' or algorithm == 'unknownQw_regularizedLSPI':
            Shat = smat(thetaHat[:-1])
            lamHat = thetaHat[-1:]
        else:
            Shat = smat(thetaHat)
            lamHat = np.array([np.inf])

        # Partition the S matrix estimate and get 
        # the next gain in the iteration.
        _, Suu, Sux = partitionSymS(Shat, Nu)
        Ki = -np.linalg.inv(Suu) @ Sux

    # Return.
    return Shat, Ki, lamHat

def doLSPI_varyingDataSize_sfeed(zSeq, uSeq, K0, NumIter, gamma, Q, R, M,
                                 NumDataPartition, Sopt, Kopt, lamOpt, 
                                 Qw=None, *, algorithm, lambdaReg=1e-3):
    """ Do Least Squares Policy Iteration in increments of data. """
    
    # Create lists to store errors during iterations in the S, K, lambda 
    # parameters, and the feedback law estimation time.
    errS = []
    errK = []
    errLam = []
    estTimes = []

    # List to store the estimated feedback law. 
    estK_list = []

    # Total number of time steps in the data set. 
    Nt = uSeq.shape[0]

    # State and control input size. 
    Nz = zSeq.shape[1]
    Nu = uSeq.shape[1]

    # Make sure that the provided initial feedback gain is of the right 
    # dimensions. 
    gainKSizeErrorMessage = """ Provide the initial gain K of an appropriate 
                                size. """
    assert K0.shape[0] == Nu and K0.shape[1] == Nz, gainKSizeErrorMessage

    # Size of each partion of the data set.
    NtEachPartition = Nt//NumDataPartition

    # Loop over the number of partitions of the data set.
    for i in range(1, NumDataPartition + 1):
        
        # Print the partition index of the data 
        # set for which LSPI is being run.
        print("Running LSPI on Data Set: " + str(i))

        # Start time of the calculations. 
        tStart = time.time()

        # Get the data set for the current size. 
        zSeqi = zSeq[:i*NtEachPartition + 1, :]
        uSeqi = uSeq[:i*NtEachPartition, :]

        # Do Policy iteration on this data set. 
        Si, Ki, lami = doLSPI_fixedDataSize_sfeed(zSeqi, uSeqi, K0, NumIter, 
                                                  gamma, Q, R, M, Qw=Qw, 
                                                  algorithm=algorithm, 
                                                  lambdaReg=lambdaReg)

        # End time of the calculations. 
        tEnd = time.time()

        # Compute the errors in the S, K, and lambda, compared to the optimal
        # quantities. 
        errS += [np.linalg.norm(Sopt - Si)/np.linalg.norm(Sopt)]
        errK += [np.linalg.norm(Kopt - Ki)/np.linalg.norm(Kopt)]
        errLam += [np.linalg.norm(lamOpt - lami)/np.linalg.norm(lamOpt)]
        
        # Store the total time taken for the feedback law estimation.
        estTimes += [tEnd - tStart]

        # Save the estimated gain to the list. 
        estK_list += [Ki]

    # Get the errors in the estimated quantities and the computation times 
    # as numpy arrays. 
    errS = np.array(errS)
    errK = np.array(errK)
    errLam = np.array(errLam)
    estTimes = np.array(estTimes)

    # Return. 
    return (errS, errK, errLam, estTimes, estK_list)

def get_qLearningLSMats_ofeed(zSeq, uSeq, ySeq, K, gamma, Qy, R, SRoc):
    """ Function to get the matrices for the Q-learning least squares problem 
        for the output feedback problem setup. 
    """

    # State and control input sizes. 
    Nz = zSeq.shape[1]
    Nu = uSeq.shape[1]

    # Number of time steps. 
    Nt = uSeq.shape[0]

    # Number of parameters in the Q-learning least squares problem.
    # The one extra parameter corresponds to all the contribution of 
    # the noise sources.
    numQPars = int((Nz + Nu)*(Nz + Nu + 1)/2) + 1

    # Create empty A and b matrices.
    A = np.zeros((numQPars, numQPars))
    b = np.zeros((numQPars, 1))

    # Loop over all the data points.
    for t in range(Nt):
        
        # Get the z, u, y, z^+ tuple. 
        zt = zSeq[t:t+1, :].T
        ut = uSeq[t:t+1, :].T
        yt = ySeq[t:t+1, :].T
        ztplus = zSeq[t+1:t+2, :].T

        # Get z and z plus vectors. 
        basisZU = basisUnkwnNoiseCov(np.concatenate((zt, ut)))
        basisZplus = basisUnkwnNoiseCov(np.concatenate((ztplus, K @ ztplus)))

        # Get the stage cost.
        Du = ut - zt[-Nu:, :]
        lyu = yt.T @ (Qy @ yt) + ut.T @ (R @ ut) + Du.T @ (SRoc @ Du)

        # Get contributions to the A and B matrix. 
        A += basisZU @ (basisZU - gamma*basisZplus).T
        b += basisZU*lyu

    # Divide by the number of data points. 
    A = A/Nt
    b = b/Nt

    # Return the matrices. 
    return A, b

def doLSPI_fixedDataSize_ofeed(zSeq, uSeq, ySeq, K0, NumIter, 
                               gamma, Qy, R, SRoc, rankCheckTol=1e-6):
    """ Do least squares policy iteration for the output feedback case. """

    # Initial LQR gain to iterate over.
    Ki = K0

    # Number of control inputs.
    Nu = uSeq.shape[1]

    # Do Q-learning iterations.
    for i in range(NumIter):
        
        # Print the iteration number in the LSPI algorithm. 
        print("Running the LSPI algorithm, Iteration Number: " + str(i))
        
        # Get matrices for Q-learning.
        A, b = get_qLearningLSMats_ofeed(zSeq, uSeq, ySeq, Ki, 
                                         gamma, Qy, R, SRoc)

        # Solve the least squares problem.
        thetaHat = np.linalg.solve(A, b)

        # Print the rank of the regressor matrix. 
        print("Rank deficiency in the regressor matrix: " + 
              str(A.shape[0] - np.linalg.matrix_rank(A, tol=rankCheckTol)))

        # S matrix and noise contribution estimate.
        Shat = smat(thetaHat[:-1])
        lamHat = thetaHat[-1:]

        # Partition the S matrix estimate and get the next gain 
        # in the iteration. 
        _, Suu, Sux = partitionSymS(Shat, Nu)
        Ki = -np.linalg.pinv(Suu) @ Sux

    # Return.
    return Shat, Ki, lamHat

def doLSPI_varyingDataSize_ofeed(zSeq, uSeq, ySeq, K0, NumIter, gamma, Qy, R, 
                                 SRoc, NumDataPartition, Sopt, Kopt):
    """ Do Least Squares Policy Iteration in increments of data. """
    
    # Lists to store the errors in the S and K estimates, and feedback law 
    # estimation times.
    errS = []
    errK = []
    estTimes = []

    # List to store the estimated feedback laws. 
    estK_list = []

    # State and control input size. 
    Nz = zSeq.shape[1]
    Nu = uSeq.shape[1]

    # Make sure that the provided initial feedback gain is of the right 
    # dimensions. 
    gainKSizeErrorMessage = """ Provide the initial gain K of an appropriate 
                                size. """
    assert K0.shape[0] == Nu and K0.shape[1] == Nz, gainKSizeErrorMessage

    # Total number of time steps in the data set. 
    Nt = uSeq.shape[0]

    # Size of each partion of the data set.
    NtEachPartition = Nt//NumDataPartition

    # Loop over the number of partitions of the data set.
    for i in range(1, NumDataPartition + 1):
        
        # Print the partition index of the data 
        # set for which LSPI is being run.
        print("Running LSPI on Data Set: " + str(i))

        # Start time of the calculations. 
        tStart = time.time()

        # Get the data set for the current size. 
        zSeqi = zSeq[:i*NtEachPartition + 1, :]
        uSeqi = uSeq[:i*NtEachPartition, :]
        ySeqi = ySeq[:i*NtEachPartition, :]

        # Do Policy iteration on this data set. 
        Si, Ki, lami = doLSPI_fixedDataSize_ofeed(zSeqi, uSeqi, ySeqi, K0, 
                                                  NumIter, gamma, Qy, R, SRoc)

        # End time of the calculations. 
        tEnd = time.time()

        # Store the errors in the estimated S and K compared to the 
        # optimal quantities. 
        errS += [np.linalg.norm(Sopt - Si)/np.linalg.norm(Sopt)]
        errK += [np.linalg.norm(Kopt - Ki)/np.linalg.norm(Kopt)]

        # Store the feedback law estimation times. 
        estTimes += [tEnd - tStart]

        # Store the estimated feedabck law to the list. 
        estK_list += [Ki]

    # Get the errors and feedback law estimation times as arrays. 
    errS = np.array(errS)
    errK = np.array(errK)
    estTimes = np.array(estTimes)

    # Return. 
    return (errS, errK, estTimes, estK_list)