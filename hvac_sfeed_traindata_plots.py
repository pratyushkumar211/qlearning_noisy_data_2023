# [depends] hvac_parameters.pickle
import sys
sys.path.append('lib/')
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_xmudata(*, t, x, u, figureSize,
                    ylabelXcoordinate, markersize=0.4, 
                    left_frac=0.15, right_frac=0.95, 
                    wspace=0.4, hspace=0.5):
    """ Plot measured states and control inputs. """

    # Y axes labels.
    ylabels = [[r'$T_{Z1}$', r'$T_{M1}$'], 
               [r'$T_{Z2}$', r'$T_{M2}$'],
               [r'$\dot{Q}_{cz1}$', r'$\dot{Q}_{cz2}$']]

    # Number of rows and columns.
    nrow = 3
    ncol = 2

    # Create figure/axes.
    (figure, axes) = plt.subplots(nrows=nrow, ncols=ncol,
                                  sharex=True, figsize=figureSize, 
                                  gridspec_kw=dict(left=left_frac, 
                                                   right=right_frac,
                                                   wspace=wspace, 
                                                   hspace=hspace))

    # Create the data list.
    data = [[x[:, 0], x[:, 1]], 
            [x[:, 2], x[:, 3]],
            [u[:, 0], u[:, 1]]]

    # Go through each row.
    for row, col in itertools.product(range(nrow), range(ncol)):

        # Plot the data.   
        axes[row, col].plot(t, data[row][col], color='b')

        # Axes labels.
        axes[row, col].set_ylabel(ylabels[row][col], rotation=False)
        axes[row, col].get_yaxis().set_label_coords(ylabelXcoordinate, 0.5)
    
        # X axis label and limits.
        if row == 2:
            axes[row, col].set_xlabel('Time (min)')
            axes[row, col].set_xlim([np.min(t), np.max(t)])

    # Return.
    return [figure]

def main():
    """ Load the pickle files and plot. """

    # Load data.
    with open("hvac_parameters.pickle", "rb") as stream:
        hvac_parameters, _ = pickle.load(stream)

    # Extract the dictionary containing the linear model matrices. 
    linModel = hvac_parameters['linModel']

    # Number of time steps to plot. 
    Nplot = int(24*3600/linModel['Delta']) # 24 hours of data.

    # Create an empty list to store all the figures. 
    figures = []

    # Get the training data list.
    trainingData_list = hvac_parameters['trainingData_list']

    # Plot data for all the cases.
    for simData in trainingData_list:
        
        # Get the time, state, and control input arrays.
        t = simData.t[:Nplot]/60 # convert to minutes.
        x = simData.x[:Nplot]
        u = simData.u[:Nplot]

        # Save the plot. 
        figures += plot_xmudata(t=t, x=x, u=u, 
                                figureSize=(6, 6), 
                                ylabelXcoordinate=-0.25)

    # Save the figures.
    with PdfPages('hvac_sfeed_traindata_plots.pdf', 'w') as pdf_file:
        for figure in figures:
            pdf_file.savefig(figure)

# Execute main.
main()