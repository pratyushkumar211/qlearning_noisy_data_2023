# [depends] %LIB%/sysidtools.py %LIB%/hvacFuncs.py
import sys
sys.path.append('lib/')
import itertools
import numpy as np
from hvacFuncs import getHVACModel, getHVACXs
from sysidtools import generateData
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_xudata(t, x, u, xs, us, *, figureSize,
                ylabelXcoordinate, left_frac=0.15, right_frac=0.95, 
                wspace=0.4, hspace=0.3):
    """ Plot measured states and control inputs. """

    # Y axes labels.
    ylabels = [[r'$T_1$', r'$T_2$'], 
               [r'$\dot{Q}_{cz1}$', r'$\dot{Q}_{cz2}$'], 
               [r'$\dot{Q}_{az1}$', r'$\dot{Q}_{az2}$'], 
               [r'$T_a$', None]]
    
    # Number of rows.
    nrow = 4
    ncol = 2

    # Create figure/axes.
    (figure, axes) = plt.subplots(nrows=nrow, ncols=ncol,
                                  sharex=True, figsize=figureSize, 
                                  gridspec_kw=dict(left=left_frac, 
                                                 right=right_frac,
                                                 wspace=wspace, 
                                                 hspace=hspace))


    # Indices to loop over the u array. 
    uidx = 0

    # Scale to absolute physical units. 
    x = x + xs.squeeze()
    u = u + us.squeeze()

    # Go through each row and column.
    for row, col in itertools.product(range(nrow), range(ncol)):
        
        # Plot control inputs.
        if row > 0:
            try:
                axes[row, col].plot(t[:-1], u[:, uidx])
                uidx += 1
            except:
                pass

        # Plot the states for the first zone.
        if row == 0  and col == 0:

            # Plot the states and make the legend.
            axes[row, col].plot(t, x[:, 0], label=r'$T_{Z1}$')
            axes[row, col].plot(t, x[:, 1], label=r'$T_{M1}$')
            axes[row, col].legend()

        # Plot the states for the second zone.
        if row == 0  and col == 1:

            # Plot the states and make the legend.
            axes[row, col].plot(t, x[:, 2], label=r'$T_{Z2}$')
            axes[row, col].plot(t, x[:, 3], label=r'$T_{M2}$')
            axes[row, col].legend()

        # Axes labels.
        axes[row, col].set_ylabel(ylabels[row][col], rotation=False)
        axes[row, col].get_yaxis().set_label_coords(ylabelXcoordinate, 0.5)

        if row == nrow - 1:
            # X axis label and limits.
            axes[row, col].set_xlabel('Time (Hour)')
            axes[row, col].set_xlim([np.min(t), np.max(t)])

    # Return.
    return [figure]

def getStepTestData(A, BBp, C, Delta, upslb, upsub, upsind):
    """ Function to get the plant steady-state profile. 
        usind is the index of the input (either control input
        or the disturbance) to vary.
    """

    # Number of simulation time steps (seconds). 
    Nt = 12*60
    
    # Sizes.
    Nx, Nup = BBp.shape

    # Initial state. 
    x0 = np.zeros((Nx, 1))

    # Get the control input sequence.
    # Approximately 3 hours of inputs at both the upper and lower bounds.
    upseq = np.tile(np.zeros((1, Nup)), (Nt, 1))
    upseq[10:6*60, upsind] = upsub[upsind]
    upseq[6*60:, upsind] = upslb[upsind]

    # Noise covariance values. 
    Qw = 0*np.eye(Nx)
    Rv = 0*np.eye(Nx)

    # Run the simulation. 
    simData = generateData(A, BBp, C, upseq, x0, Qw, Rv, Delta)

    # Extract the time (converted to hours) and state sequence. 
    t = simData.t/3600
    xseq = simData.x

    # Return.
    return t, xseq, upseq

def doStepTests():
    """ Vary each of the 5 inputs and simulate the HVAC model. """

    # Get the HVAC model. 
    Delta = 60. # seconds.
    linModel = getHVACModel(Delta)
    BBp = np.concatenate((linModel['B'], linModel['Bp']), axis=1)

    # Get a steady-state.
    us = np.array([[50.], [40.]])
    ps = np.array([[60.], [40.], [22.]])
    xs = getHVACXs(linModel['A'], linModel['B'], linModel['Bp'], us, ps)
    
    # Lower and upper limits of control input and disturbances. 
    ups = np.concatenate((us, ps))
    upslb = np.array([[30.], [20.], [30.], [10.], [16.]]) - ups
    upsub = np.array([[70.], [60.], [90.], [70.], [28.]]) - ups

    # Empty lists to store the state and control input/disturbances. 
    xlist = []
    uplist = []

    # Vary each input and get the step test data. 
    for upsind in range(5):

        # Get the step test data. 
        t, xseq, upseq = getStepTestData(linModel['A'], BBp, linModel['C'], 
                                         Delta, upslb, upsub, upsind)

        # Save the state and control input/disturbance sequence to the list.
        xlist += [xseq]
        uplist += [upseq]

    # Return.
    return t, xlist, uplist, xs, ups

def main():

    # Do the step tests. 
    t, xlist, uplist, xs, ups = doStepTests()

    # Empty list to store the figures.
    figures = []

    # Loop over all the step tests. 
    for x, up in zip(xlist, uplist):

        # Make the figure. 
        figures += plot_xudata(t, x, up, xs, ups, figureSize=(6, 8), 
                               ylabelXcoordinate=-0.28)
    
    # Save the step test.
    with PdfPages('hvac_steptest.pdf', 'w') as pdf_file:
        for figure in figures:
            pdf_file.savefig(figure)

main()