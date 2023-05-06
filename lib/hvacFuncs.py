# [depends] linsystools.py
import numpy as np
from linsystools import c2d

def getHVACModel(Delta, sfeed=True):
    """ Return a dictionary containing the HVAC model. """
    
    # List of model parameters. 
    H1byCz1 = 6e-4
    beta12byCz1 = 2e-4
    beta12byCm1 = 3e-4
    beta24byCm1 = 6e-4
    H2byCz2 = 1e-3
    beta34byCz2 = 5e-4
    beta24byCm2 = 2e-4
    beta34byCm2 = 4e-4
    onebyCz1 = 2.5e-4
    onebyCz2 = 2e-4

    # Write down the continuous time model matrices.
    # A Matrix.
    A = np.array([[-(H1byCz1 + beta12byCz1), beta12byCz1, 0., 0.], 
                  [beta12byCm1, -(beta12byCm1 + beta24byCm1), 0., beta24byCm1], 
                  [0., 0., -(H2byCz2 + beta34byCz2), beta34byCz2], 
                  [0., beta24byCm2, beta34byCm2, -(beta24byCm2 + beta34byCm2)]])
    # B Matrix. 
    B  = np.array([[-onebyCz1, 0.], 
                   [0., 0.], 
                   [0., -onebyCz2], 
                   [0., 0.]])
    # Bp matrix. 
    Bp = np.array([[onebyCz1, 0., H1byCz1], 
                   [0., 0., 0.], 
                   [0., onebyCz2, H2byCz2], 
                   [0., 0., 0.]])

    # Convert the model to discrete time. 
    BBp = np.concatenate((B, Bp), axis=1)
    Ad, BBpd = c2d(A, BBp, Delta)

    # Extract the input and disturbance models. 
    Bd, Bpd = BBpd[:, :2], BBpd[:, 2:]

    # Construct the measurement matrix based on the sfeed flag. 
    if sfeed:
        Nx = A.shape[0]
        C = np.eye(Nx)
    else:
        C = np.array([[1., 0., 0., 0.], 
                      [0., 0., 1., 0.]])

    # Construct a dictionary that contains the discrete-time 
    # matrices and the sample time. 
    linModel = dict(A=Ad, B=Bd, Bp=Bpd, C=C, Delta=Delta)

    # Return. 
    return linModel

def getHVACXs(A, B, Bp, us, ps):
    """ Get the steady state xs for the HVAC model
        corresponding to control input and disturbance 
        steady states us and ps. 
    """

    # State size. 
    Nx, _ = B.shape

    # Concatenate the steady-state control input, disturbance, 
    # and incumbent matrices to obtain the steady-state xs. 
    usps = np.concatenate((us, ps), axis=0)
    BBp = np.concatenate((B, Bp), axis=1)

    # Determine the steady-state xs.
    xs = np.linalg.pinv(np.eye(Nx) - A) @ (BBp @ usps)

    # Return.
    return xs
