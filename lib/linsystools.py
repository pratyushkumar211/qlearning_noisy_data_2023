import numpy as np
import scipy.linalg
from numpy.random import multivariate_normal as mvnrnd

def c2d(A, B, Delta):
    """ Custom function to convert continuous-time 
        linear models to discrete-time. 
    """

    # Sizes.
    Nx, Nu = B.shape

    # Construct the incumbent matrix to take the exponential.
    MRow1 = np.concatenate((A, B), axis=1)
    MRow2 = np.zeros((Nu, Nx + Nu))
    M = np.concatenate((MRow1, MRow2), axis=0)
    Mexp = scipy.linalg.expm(M*Delta)

    # Extract the discrete-time matrices.
    Ad = Mexp[:Nx, :Nx]
    Bd = Mexp[:Nx, -Nu:]

    # Return. 
    return (Ad, Bd)

def get_augmats_rocpenalty_sfeed(A, B, Q, R, SRoc, Qw):
    """ Get the augmented dynamic model, stage costs, and noise covariance
        matrices to setup the Q-function S matrix when using a rate of 
        change penalty.
    """

    # State and control sizes.
    Nx, Nu = B.shape 

    # Augmented A.
    Aaug1 = np.concatenate((A, np.zeros((Nx, Nu))), axis=1)
    Aaug2 = np.zeros((Nu, Nx + Nu))
    Aaug = np.concatenate((Aaug1, Aaug2), axis=0)

    # Augmented B.
    Baug = np.concatenate((B, np.eye((Nu))), axis=0)

    # Augmented Q.
    Qaug = scipy.linalg.block_diag(Q, SRoc)

    # Augmented R.
    Raug = R + SRoc

    # Augmented M.
    Maug = np.concatenate((np.zeros((Nx, Nu)), -SRoc), axis=0)

    # Augmented Qw. 
    Qwaug = scipy.linalg.block_diag(Qw, np.zeros((Nu, Nu)))

    # Return the augmented matrices. 
    return (Aaug, Baug, Qaug, Raug, Maug, Qwaug)

def get_SPLamforagainK_sfeed(A, B, gamma, Q, R, SRoc, K, Qw):
    """ Get the S and P matrices and the noise term lamda for a particular 
        feedback gain K.
    """

    # State and control input size. 
    Nx, Nu = B.shape

    # Assert that the gain K is of dimensions suitable for the augmented 
    # state with one past control input. 
    gainKSizeErrorMessage = """ Provide the gain K in appropriate size for a 
                                system with the current state and one past 
                                control input as the full state. """
    assert K.shape[0] == Nu and K.shape[1] == Nx + Nu, gainKSizeErrorMessage

    # Contruct the augmented matrices so a Lyapunov equation corresponding 
    # to a standard cross term can be solved to get the P matrix. 
    (Aaug, Baug, Qaug, 
     Raug, Maug, Qwaug) = get_augmats_rocpenalty_sfeed(A, B, Q, R, SRoc, Qw)

    # Solve the Lyapunov equation to get the P matrix. 
    Adlyap = np.sqrt(gamma)*(Aaug + Baug @ K).T
    Qdlyap = Qaug + K.T @ (Raug @ K) + (Maug @ K) + (K.T @ Maug.T)
    P = scipy.linalg.solve_discrete_lyapunov(Adlyap, Qdlyap)

    # Setup the S matrix.
    S1 = np.concatenate((Qaug + gamma*(Aaug.T @ (P @ Aaug)), 
                         gamma*(Aaug.T @ (P @ Baug)) + Maug), axis=1)
    S2 = np.concatenate((gamma*(Baug.T @ (P @ Aaug)) + Maug.T, 
                         Raug + gamma*(Baug.T @ (P @ Baug))), axis=1)
    S = np.concatenate((S1, S2), axis=0)

    # Get the noise contribution.
    lam = (gamma/(1-gamma))*np.trace(P @ Qwaug)

    # Return.
    return S, P, lam

def get_optSPKLam_sfeed(A, B, gamma, Q, R, SRoc, Qw):
    """ Function to get the optimal S, P, and feedback law matrices 
        and the noise contribution term to the Q-function in the simulations. 
    """
    
    # Contruct the augmented matrices so a Lyapunov equation corresponding 
    # to a standard cross term can be solved to get the P matrix. 
    (Aaug, Baug, Qaug, 
     Raug, Maug, _) = get_augmats_rocpenalty_sfeed(A, B, Q, R, SRoc, Qw)

    # Solve the Lyapunov equation to get the P matrix. 
    Atilde = np.sqrt(gamma)*Aaug
    Btilde = np.sqrt(gamma)*Baug

    # Solve the discrete algebriac riccati equation.
    P = scipy.linalg.solve_discrete_are(Atilde, Btilde, Qaug, Raug, s=Maug)

    # Get feedback law.
    K = -np.linalg.pinv(Raug + gamma*(Baug.T @ (P @ Baug)))  
    K = K @ (gamma*(Baug.T @ (P @ Aaug)) + Maug.T)

    # Get the S matrix and noise contribution. 
    S, Pdummy, lam = get_SPLamforagainK_sfeed(A, B, gamma, Q, R, SRoc, K, Qw)

    # Make sure that the Pdummy calculated from the function call is equal to 
    # the P calculated in the current function.
    assert np.linalg.norm(P - Pdummy) < 1e-6, """ Double check the calculation 
                                        of the optimal S and P matrices. """
    
    # Return.
    return S, P, K, lam

def get_tByseq_ithrow(i, Np, A, B, C):
    """ Function to get a row of the matrix used to predict the measurement 
        sequence from the control input sequence. 
    """

    # Sizes.
    Nu = B.shape[1]
    Ny = C.shape[0]

    # Create a list to store the column wise elements of the ith row.
    tByseq_ithrow = []

    # Collect all the column wise elements. 
    for j in range(Np):

        # Get the row element depending on the column index.             
        if j < i:
            tByseq_ithrow += [C @ (np.linalg.matrix_power(A, i-j-1) @ B)]
        else:
            tByseq_ithrow += [np.zeros((Ny, Nu))]

    # Concatenate each element of the row column wise. 
    tByseq_ithrow = np.concatenate(tByseq_ithrow, axis=1)

    # Return.
    return tByseq_ithrow

def get_liftedsysmodelmats(*, A, B, C, Np):
    """ For a given number of past inputs and outputs  = Np

        Compute the linear model matrices for an augmented system with 
        the state z containing the past y's and u's
        
        Such that
        z^+ = Az \times z  + Bz \times u
        y = Cz \times z
    """

    # Sizes.
    Nx, Nu = B.shape
    Ny = C.shape[0]

    # Construct matrices to predict the measurement sequences.
    tAyseq = np.concatenate([C @ np.linalg.matrix_power(A, i)
                             for i in range(Np)], axis=0)
    tByseq = np.concatenate([get_tByseq_ithrow(i, Np, A, B, C)
                             for i in range(Np)], axis=0)
    tBy = np.concatenate([C @ (np.linalg.matrix_power(A, j) @ B)
                              for j in range(Np - 1, -1, -1)], axis=1)

    # Construct the rows of the Az matrix.
    # Row 1.
    AzRow1 = np.concatenate((np.zeros(((Np-1)*Ny, Ny)), np.eye((Np-1)*Ny),
                             np.zeros(((Np-1)*Ny, Np*Nu))), axis=1)
    # Row 2.
    AzRow2C1 = C @ (np.linalg.matrix_power(A, Np) @ np.linalg.pinv(tAyseq))
    AzRow2C2 =  -AzRow2C1 @ tByseq + tBy
    AzRow2 = np.concatenate((AzRow2C1, AzRow2C2), axis=1)
    # Row 3.
    AzRow3 = np.concatenate((np.zeros(((Np-1)*Nu, Np*Ny +Nu)), 
                             np.eye((Np-1)*Nu)), axis=1)
    # Row 4.
    AzRow4 = np.zeros((Nu, Np*Ny + Np*Nu))
    # Get the full Az batrix.
    Az = np.concatenate((AzRow1, AzRow2, AzRow3, AzRow4), axis=0)

    # Get the full Bz matrix. 
    Bz= np.concatenate((np.zeros((Np*Ny + (Np-1)*Nu, Nu)), np.eye(Nu)), axis=0)

    # Get the Cz matrix. 
    Cz = AzRow2

    # Return. 
    return Az, Bz, Cz

def get_ztox_Hxzmatrix(A, B, C, Np):
    """ Construct the matrix to map the vector of past inputs and 
        measurements (z) to the current state (x). 
    """ 

    # Construct matrices to predict the measurement sequence, 
    # and obtain the current state as a function of the z.
    tAyseq = np.concatenate([C @ np.linalg.matrix_power(A, i)
                             for i in range(Np)], axis=0)
    tByseq = np.concatenate([get_tByseq_ithrow(i, Np, A, B, C)
                             for i in range(Np)], axis=0)
    tBx = np.concatenate([np.linalg.matrix_power(A, j) @ B
                          for j in range(Np - 1, -1, -1)], axis=1)

    # Now construct the matrix to map z to x. 
    HxzCol1 = np.linalg.matrix_power(A, Np) @ np.linalg.pinv(tAyseq)
    HxzCol2 = -HxzCol1 @ tByseq + tBx
    Hxz = np.concatenate((HxzCol1, HxzCol2), axis=1)

    # Return. 
    return Hxz

def get_augmats_rocpenalty_ofeed(Bz, Cz, Qy, R, SRoc):
    """ Get the augmented dynamic model, stage costs, and noise covariance
        matrices to setup the Q-function S matrix when using a rate of 
        change penalty.
    """

    # State and control sizes.
    Nz, Nu = Bz.shape 

    # Augmented Q.
    Qaug = Cz.T @ (Qy @ Cz) + scipy.linalg.block_diag(0*np.eye(Nz-Nu), SRoc)

    # Augmented R.
    Raug = R + SRoc

    # Augmented M.
    Maug = np.concatenate((np.zeros((Nz - Nu, Nu)), -SRoc), axis=0)

    # Return the augmented matrices. 
    return (Qaug, Raug, Maug)

def get_SforagainK_ofeed(Az, Bz, Cz, gamma, Qy, R, SRoc, K):
    """ Get the S matrix for the output feedback problem. 

        The dynamic model matrices (A, B, C) are in the lifted state space.

        The gain matrix K is in the state space with the state defined as the 
        current state and one vector of past inputs and outputs.  
    """

    # State and control input size. 
    Nz, Nu = Bz.shape

    # Assert that the gain K is of dimensions suitable for the augmented 
    # state with one past control input. 
    gainKSizeErrorMessage = """ Provide the gain K in appropriate size as the 
                                size of the lifted state.
                            """
    assert K.shape[0] == Nu and K.shape[1] == Nz, gainKSizeErrorMessage

    # Contruct the augmented matrices so a Lyapunov equation corresponding 
    # to a standard cross term can be solved to get the P matrix. 
    (Qaug, Raug, Maug) = get_augmats_rocpenalty_ofeed(Bz, Cz, Qy, R, SRoc)

    # Solve the Lyapunov equation to get the P matrix. 
    Adlyap = np.sqrt(gamma)*(Az + Bz @ K).T
    Qdlyap = Qaug + K.T @ (Raug @ K) + Maug @ K + K.T @ Maug.T
    P = scipy.linalg.solve_discrete_lyapunov(Adlyap, Qdlyap)

    # Setup the S matrix.
    S1 = np.concatenate((Qaug + gamma*(Az.T @ (P @ Az)), 
                         gamma*(Az.T @ (P @ Bz)) + Maug), axis=1)
    S2 = np.concatenate((gamma*(Bz.T @ (P @ Az)) + Maug.T, 
                         Raug + gamma*(Bz.T @ (P @ Bz))), axis=1)
    S = np.concatenate((S1, S2), axis=0)

    # Return. 
    return S

def get_optSK_ofeed(Az, Bz, Cz, gamma, Qy, R, SRoc):
    """ Function to get the optimal S, and P matrices and feedback law for 
        the lifted state.

        The model matrices (A, B, C) are all in the original state space.
    """

    # Contruct the augmented matrices so a Lyapunov equation corresponding 
    # to a standard cross term can be solved to get the P matrix. 
    (Qaug, Raug, 
     Maug) = get_augmats_rocpenalty_ofeed(Bz, Cz, Qy, R, SRoc)

    # Solve the Lyapunov equation to get the P matrix. 
    Atilde = np.sqrt(gamma)*Az
    Btilde = np.sqrt(gamma)*Bz

    # Solve the discrete algebriac riccati equation.
    P = scipy.linalg.solve_discrete_are(Atilde, Btilde, Qaug, Raug, s=Maug)

    # Get feedback law.
    # First get the feedback law in the state-space of the current state 
    # and one past control input.
    K = -np.linalg.pinv(Raug + gamma*(Bz.T @ (P @ Bz)))
    K = K @ (gamma*(Bz.T @ (P @ Az)) + Maug.T)

    # Get the S matrix and noise contribution.
    S = get_SforagainK_ofeed(Az, Bz, Cz, gamma, Qy, R, SRoc, K)

    # Return.
    return S, K

# def get_SforagainK_ofeed(A, B, C, gamma, Qy, R, SRoc, K, Np):
#     """ Get the S matrix for the output feedback problem. 

#         The dynamic model matrices (A, B, C) are in the original state space.

#         The gain matrix K is in the state space with the state defined as the 
#         current state and one vector of past inputs and outputs.  
#     """

#     # State and control input size. 
#     Nx, Nu = B.shape
#     Ny = C.shape[0]

#     # Assert that the gain K is of dimensions suitable for the augmented 
#     # state with one past control input. 
#     gainKSizeErrorMessage = """ Provide the gain K in appropriate size for a 
#                                 system with the current state and one past 
#                                 control input as the full state. """
#     assert K.shape[0] == Nu and K.shape[1] == Nx + Nu, gainKSizeErrorMessage

#     # Transform the output penalty matrix to a state penalty matrix. 
#     Q = C.T @ (Qy @ C)

#     # Contruct the augmented matrices so a Lyapunov equation corresponding 
#     # to a standard cross term can be solved to get the P matrix. 
#     (Aaug, Baug, Qaug, 
#      Raug, Maug, _) = get_augmats_rocpenalty(A, B, Q, R, SRoc, 0*np.eye(Nx))

#     # Solve the Lyapunov equation to get the P matrix. 
#     Adlyap = np.sqrt(gamma)*(Aaug + Baug @ K).T
#     Qdlyap = Qaug + K.T @ (Raug @ K) + Maug @ K + K.T @ Maug.T
#     P = scipy.linalg.solve_discrete_lyapunov(Adlyap, Qdlyap)

#     # Setup the S matrix.
#     S1 = np.concatenate((Qaug + gamma*(Aaug.T @ (P @ Aaug)), 
#                          gamma*(Aaug.T @ (P @ Baug)) + Maug), axis=1)
#     S2 = np.concatenate((gamma*(Baug.T @ (P @ Aaug)) + Maug.T, 
#                          Raug + gamma*(Baug.T @ (P @ Baug))), axis=1)
#     S = np.concatenate((S1, S2), axis=0)

#     # Get the transformation matrix to transform the lifted state to a 
#     # state containing the current state and one past control input. 
#     Hxz = get_ztox_Hxzmatrix(A, B, C, Np)
#     Huprevz = np.concatenate((np.zeros((Nu, Np*(Ny + Nu - 1))), 
#                               np.eye(Nu)), axis=1)
#     Hxuprevz = np.concatenate((Hxz, Huprevz), axis=0)

#     # Transform the S matrix for the lifted state.
#     StransMatrix = scipy.linalg.block_diag(Hxuprevz, np.eye(Nu))
#     Sz = StransMatrix.T @ (S @ StransMatrix)

#     # Return. 
#     return Sz

# def get_optSK_ofeed(A, B, C, gamma, Qy, R, SRoc, Np):
#     """ Function to get the optimal S, and P matrices and feedback law for 
#         the lifted state.

#         The model matrices (A, B, C) are all in the original state space.
#     """
    
#     # Get the sizes. 
#     Nx, Nu = B.shape
#     Ny = C.shape[0]

#     # Transform the output penalty matrix to a state penalty matrix. 
#     Q = C.T @ (Qy @ C)

#     # Contruct the augmented matrices so a Lyapunov equation corresponding 
#     # to a standard cross term can be solved to get the P matrix. 
#     (Aaug, Baug, Qaug, 
#      Raug, Maug, _) = get_augmats_rocpenalty(A, B, Q, R, SRoc, 0*np.eye(Nx))

#     # Solve the Lyapunov equation to get the P matrix. 
#     Atilde = np.sqrt(gamma)*Aaug
#     Btilde = np.sqrt(gamma)*Baug

#     # Solve the discrete algebriac riccati equation.
#     P = scipy.linalg.solve_discrete_are(Atilde, Btilde, Qaug, Raug, s=Maug)

#     # Get feedback law.
#     # First get the feedback law in the state-space of the current state 
#     # and one past control input. 
#     K = -np.linalg.pinv(Raug + gamma*(Baug.T @ (P @ Baug)))  
#     K = K @ (gamma*(Baug.T @ (P @ Aaug)) + Maug.T)

#     # Transform the feedback law for the lifed state.
#     # First get the transformation matrix to transform the lifted state to a 
#     # state containing the current state and one past control input. 
#     Hxz = get_ztox_Hxzmatrix(A, B, C, Np)
#     Huprevz = np.concatenate((np.zeros((Nu, Np*(Ny + Nu - 1))), 
#                               np.eye(Nu)), axis=1)
#     Hxuprevz = np.concatenate((Hxz, Huprevz), axis=0)
#     # Second, transform the gain for the lifted state. 
#     Kz = K @ Hxuprevz

#     # Get the S matrix and noise contribution.
#     S = get_SforagainK_ofeed(A, B, C, gamma, Qy, R, SRoc, K, Np)

#     # Return.
#     return S, Kz

def dlqe(A, C, Qw, Rv, Swv=None):
    """ Get the discrete-time Kalman filter for the given system.
    """

    # Two different cases based on the cross correlation between 
    # the noise sources. 
    if Swv is None:
        
        # Solve the DARE to get the state estimation error noise covariance.
        P = scipy.linalg.solve_discrete_are(A.T, C.T, Qw, Rv)

        # Get the Kalman filter gain.
        L = scipy.linalg.solve(C @ (P @ C.T) + Rv, C @ (P @ A.T)).T

    else:

        # Solve the DARE to get the state estimation error noise covariance.
        P = scipy.linalg.solve_discrete_are(A.T, C.T, Qw, Rv, s=Swv)
        
        # Get the Kalman filter gain.
        L = scipy.linalg.solve(C @ (P @ C.T) + Rv, C @ (P @ A.T) + Swv.T).T

    # Return.
    return (L, P)

class KalmanFilter:
    """ Class to construct and perform state estimation
        using Kalman Filtering.
    """

    def __init__(self, *, A, B, C, xPrior, yprev, Qw, Rv, Swv=None):

        # Model and noise covariances.
        self.A = A
        self.B = B
        self.C = C
        self.Qw = Qw
        self.Rv = Rv
        self.Swv = Swv
        
        # Compute and save the Kalman filter gain as an attribute.
        self.L, _ = dlqe(A, C, Qw, Rv, Swv)
        
        # Create lists for saving data. The state estimates are stored using 
        # the Kalman predictor. 
        self.xhat = [xPrior]
        self.y = [yprev]
        self.u = []

    def solve(self, y, uprev):
        """ Run the Kalman filter with the predictor form.
        """

        # Get predict state estimate and measurement at the previous 
        # time step.
        xhatprev = self.xhat[-1]
        yprev = self.y[-1]

        # Get the predicted estimate. 
        xhat = self.A @ xhatprev + self.B @ uprev 
        xhat += self.L @ (yprev - self.C @ xhatprev) 
        
        # Save the state estimates, measurements, and control inputs 
        # to their lists (might be useful for plotting later).
        self.xhat += [xhat]
        self.y += [y]
        self.u += [uprev]

        # Return. 
        return xhat

class LQGController:
    """ Class to form a linear quadratic regulator and Kalman filter using 
        the provided model matrices and noise covariances. 

        Then, during closed-loop operation, do state estimation and 
        use the feedback law to get the control input.
    """

    def __init__(self, *, A, B, C, gamma, Qy, R, SRoc, Qw, Rv, Swv, 
                          xPrior, yprev, uprev, liftedsys=False):

        # Save linear model matrices. 
        self.A = A
        self.B = B
        self.C = C

        # Save the lifted sys attribute. 
        self.liftedsys = liftedsys

        # Construct the linear quadratic regulator. 
        if liftedsys:
            _, self.K = get_optSK_ofeed(A, B, C, gamma, Qy, R, SRoc)
        else:
            # Transform the output penalty matrix to a state penalty matrix. 
            Q = C.T @ (Qy @ C)
            _, _, self.K, _ = get_optSPKLam_sfeed(A, B, gamma, Q, R, SRoc, Qw)

        # Construct a Kalman Filter.
        self.filter = KalmanFilter(A=A, B=B, C=C, xPrior=xPrior, 
                                   yprev=yprev, Qw=Qw, Rv=Rv, Swv=Swv)
        
        # Create a variable to keep track of the uprev used at the first 
        # time step of the controller implementation. 
        # (Used when reinitializing the controller for a new trajectory).
        self.uprev0 = uprev

        # Create a variable to keep track of the previous control input.
        # (Used for state estimation and returning the control input).
        self.uprev = uprev

        # Create an empty list to store the measurement estimates. 
        # (Used to compute the stage cost). 
        self.yhat = []

    def control_law(self, y):
        """ Compute the control input to inject to the plant. """

        # Get the current state estimate using the Kalman filter.
        xhat = self.filter.solve(y, self.uprev)

        # Save the measurement estimate to the list. 
        self.yhat += [self.C @ xhat]
        
        # Get the augmented state comprising the current state and one previous
        # control input. 
        if self.liftedsys:
            zt = xhat
        else:
            zt = np.concatenate((xhat, self.uprev), axis=0)

        # Get the control input using the state estimate and the feedback law.
        self.uprev = self.K @ zt

        # Return. 
        return self.uprev

    def clear_data(self):
        """ Reset the data sets in the controller to the initial state. """

        # Reset the data set lists in the Kalman filter class. 
        self.filter.xhat = [self.filter.xhat[0]]
        self.filter.y = [self.filter.y[0]]
        self.filter.u = []

        # Create an empty list for the measurement estimates.        
        self.yhat = []

        # Reset the previous control input. 
        self.uprev = self.uprev0

class ModelFreeController:
    """ Class to implement a model-free feedback controller estimated using 
        Q-learning. A simple first order filter is used for noise filtering.
    """

    def __init__(self, *, K, alpha, Np, Nu, Ny, uprev):

        # Save the feedback law and noise filtering parameter. 
        self.K = K
        self.alpha = alpha

        # Sizes.
        self.Np = Np
        self.Nu = Nu
        self.Ny = Ny

        # Variable to save the previous control input used at the first 
        # time step of the closed-loop simulation. 
        # (Used to reset the uprev variable when reinitializing the controller).
        self.uprev0 = uprev

        # Control input to use for some initial few time steps.
        self.uprev = uprev

        # Create empty lists to store the measurements, filtered measurements,
        # and control inputs. 
        self.y = []
        self.u = []
        self.yhat = []
        
    def control_law(self, y):
        """ Compute the control input to inject to the plant. """

        # If enough "filtered" measurements are available, then use the 
        # feedback law to get the control input. 
        if len(self.yhat) >= self.Np:
            
            # Get the list of past measurements and control inputs. 
            ypseqt = np.asarray(self.yhat[-self.Np:]).squeeze(axis=-1)
            upseqt = np.asarray(self.u[-self.Np:]).squeeze(axis=-1)

            # Get the state.
            z = np.concatenate((np.reshape(ypseqt, (self.Np*self.Ny, 1)), 
                                np.reshape(upseqt, (self.Np*self.Nu, 1))))

            # Compute the control. 
            self.uprev = self.K @ z

        # Get the filtered measurement and save to the list. 
        if len(self.yhat) > 0:
            self.yhat += [self.alpha*self.yhat[-1] + (1 - self.alpha)*y]
        else:
            self.yhat += [y]

        # Save the current measurement and control input to their 
        # respective lists.
        self.y += [y]
        self.u += [self.uprev]

        # Return.
        return self.uprev

    def clear_data(self):
        """ Reset the data sets in the controller to the initial state. """

        # Reset the lists of data sets for the controller. 
        self.y = []
        self.u = []
        self.yhat = []

        # Reset the previous control input. 
        self.uprev = self.uprev0

def do_clsims_controllerevals(Nsim, Ntraj, A, B, C, Qw, Rv, Qy, R, SRoc,
                              controller, x0Lb, x0Ub):
    """ Run multiple closed-loop simulations with the provided model 
        matrices and the plant, the provided controller, and for the number 
        of the simulations and time steps per trajectory. Sample the initial 
        states from a Gaussian distribution with the provided statistics.
    """

    # Get state and output sizes.
    Ny, Nx = C.shape

    # Empty lists to store the "sequences" of the states, control inputs, 
    # measurements, and stage costs for each closed-loop trajectory.
    xseq_list = []
    useq_list = []
    yseq_list = []
    lyuseq_list = []

    # Loop over the number of trajectories.
    for _ in range(Ntraj):

        # Empty lists to store the sequence of the states, control inputs
        # measurement, and stage cost for the current closed-loop trajectory.
        xseq = []
        useq = []
        yseq = []
        lyuseq = []

        # Sample the initial state and measurement for the 
        # current trajectory from a normal distribution. 
        xt = (x0Ub - x0Lb)*np.random.rand(Nx, 1) + x0Lb

        # Get the measurement. 
        yt = C @ xt + mvnrnd(np.zeros((Ny, )), Rv)[:, np.newaxis]

        # Save the initial state and measurement to their lists. 
        xseq += [xt]
        yseq += [yt]

        # Clear the controller data to start the current trajectory. 
        controller.clear_data()

        # Get the uprev (to compute the rate-of-change penalty in 
        # the stage cost). 
        uprev = controller.uprev

        # Run the simulation for the current trajectory. 
        for _ in range(Nsim):

            # Get the control input using the controller. 
            ut = controller.control_law(yt)

            # Compute the stage cost.
            # (Use estimated measurement rather than the noisy measurement). 
            yhat = controller.yhat[-1]
            Du = ut - uprev
            lyut = yt.T @ (Qy @ yt) + ut.T @ (R @ ut) + Du.T @ (SRoc @ Du)

            # Sample the noise values for plant propagation.
            wt = mvnrnd(np.zeros((Nx, )), Qw)[:, np.newaxis]
            vt = mvnrnd(np.zeros((Ny, )), Rv)[:, np.newaxis]

            # Propagate the plant.
            xt = A @ xt + B @ ut + wt
            yt = C @ xt + vt

            # Update the uprev variable to the current control input.
            uprev = ut

            # Save data to lists.
            useq += [ut]
            lyuseq += [lyut]
            xseq += [xt]
            yseq += [yt]

        # Save all the data for the current trajectory to the lists. 
        xseq_list += [np.asarray(xseq[:-1]).squeeze(axis=-1)]
        useq_list += [np.asarray(useq).squeeze(axis=-1)]
        yseq_list += [np.asarray(yseq[:-1]).squeeze(axis=-1)]
        lyuseq_list += [np.asarray(lyuseq).squeeze(axis=-1)]

    # Compute the mean of the performance metric for each trajectory.
    lyuseqs = np.asarray(lyuseq_list).squeeze(axis=-1)
    lamT = np.mean(lyuseqs, axis=1)

    # Create a dictionary of all the data. 
    clSimsData = dict(xseq_list=xseq_list, 
                      useq_list=useq_list, 
                      yseq_list=yseq_list, 
                      lyuseq_list=lyuseq_list,
                      lamT=lamT)

    # Return.
    return clSimsData