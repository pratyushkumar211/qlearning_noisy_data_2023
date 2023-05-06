# [depends] hvac_ofeed_optlqg.pickle hvac_ofeed_sysid.pickle
# [depends] hvac_ofeed_lspi.pickle
import sys
sys.path.append('lib/')
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_xmudata(*, t, xlist, ulist, ylist, legendNames, 
                    legendColors, figureSize,
                    ylabelXcoordinate, titleLoc, 
                    left_frac=0.16, right_frac=0.95, top_frac=0.9, 
                    wspace=0.5, hspace=0.1):
    """ Plot the measurements, states, and control inputs. """

    # Y axes labels.
    ylabels = [[r'$T_{z1} (^\circ$ C)', r'$T_{m1} (^\circ$ C)'], 
               [r'$T_{z2} (^\circ$ C)', r'$T_{m2} (^\circ$ C)'],
               [r'$\dot{Q}_{cz1}$ (kW)', r'$\dot{Q}_{cz2}$ (kW)']]

    # Number of rows and columns.
    nrow = 3
    ncol = 2

    # Create figure/axes.
    (figure, axes) = plt.subplots(nrows=nrow, ncols=ncol,
                                  sharex=True, figsize=figureSize, 
                                  gridspec_kw=dict(left=left_frac, 
                                                   right=right_frac,
                                                   top=top_frac,
                                                   wspace=wspace, 
                                                   hspace=hspace))

    # Create an empty list to store the legend handles. 
    legendHandles = []

    # Loop over all the closed-loop trajectories. 
    for i, x, u, y, color in zip(range(len(xlist)), xlist, ulist, ylist, legendColors):

        # Create the data list.
        data = [[y[:, 0], x[:, 1]], 
                [y[:, 1], x[:, 3]],
                [u[:, 0], u[:, 1]]]

        # Go through each row.
        for row, col in itertools.product(range(nrow), range(ncol)):
            
            # Make the plot.
            legendHandle = axes[row, col].plot(t, data[row][col], color=color)

            # Y-axis labels.
            axes[row, col].set_ylabel(ylabels[row][col], rotation=False)
            axes[row, col].get_yaxis().set_label_coords(ylabelXcoordinate, 0.5)
        
            # Add dashed black lines for the setpiont.
            if i == 0:
                if row in [0, 1] and col == 0: 
                    if row == 0: 
                        ysp_label = r'$T_{z1sp}$'
                    else: 
                        ysp_label = r'$T_{z2sp}$'
                    axes[row, col].plot(t, np.zeros((len(t))), 
                                        color='k', linestyle='dashed', label=ysp_label)
                    axes[row, col].legend()

            # X axis label and limits.
            if row == 2:
                axes[row, col].set_xlabel('Time (hours)')
                axes[row, col].set_xlim([np.min(t), np.max(t)])

        # Add the legend handle to the list. 
        legendHandles += legendHandle

    # Figure legends. 
    if legendNames is not None:
        figure.legend(handles=legendHandles,
                      labels=legendNames, 
                      loc=titleLoc, ncol=len(legendNames))

    # Return.
    return [figure]

def plot_costHistogram(*, cost_list, legendColors, legendNames,
                          xlabel, ylabel, nBins, xlims, ylims, figureSize, 
                          left_frac=0.1, right_frac=0.95, top_frac=0.95, 
                          ylabelXcoordinate=-0.08):
    """ Make a histogram of the costs computer over different 
        closed-loop trajectories. """
    
    # Create figures.
    figure, axes = plt.subplots(figsize=figureSize, 
                                gridspec_kw=dict(left=left_frac, 
                                                   right=right_frac,
                                                   top=top_frac))

    # Loop over the cost list.
    for cost, color in zip(cost_list, legendColors):

        # Make the histogram.
        axes.hist(cost, bins=nBins, range=xlims, color=color, histtype='step')

    # Legend.
    if legendNames is not None:
        axes.legend(legendNames)

    # X and Y labels.
    axes.set_ylabel(ylabel)
    axes.get_yaxis().set_label_coords(ylabelXcoordinate, 0.5)
    axes.set_xlabel(xlabel)

    # X limits.
    axes.set_xlim(xlims)
    axes.set_ylim(ylims)

    # Return the figure.
    return [figure]

def print_metrics(controllerNames, lamTmean_optlqg, lamTmeans, estTimes):
    """ Print performance metrics (mean-of-the closed-loop cost, and controller 
        estimation time) for the three controllers (Optimal LQG, SYSID, LSPI). 
    """

    # Loop through the provided controller names, mean of the performance 
    # metric, and the estimation times. 
    for (controllerName, 
         lamTmean, estTime) in zip(controllerNames, lamTmeans, estTimes):

        # Compute the perfomance metric compared to the optimal controller. 
        perf_loss = 100*(lamTmean - lamTmean_optlqg)/lamTmean_optlqg

        # Round the perfomance metric and controller estimation time. 
        perf_loss = np.round(perf_loss, decimals=2)
        estTime = np.round(estTime, decimals=2)

        # Print the mean of the performance metric. 
        print("Percent performance loss (" + controllerName + "): " + 
               str(perf_loss))

        # Print the controller estimation time. 
        print("Controller estimation time (" + controllerName + 
                "): " + str(estTime) + " secs")

def main():
    """ Load the pickle files and plot. """

    # Load the optimal lqg calculations.
    with open("hvac_ofeed_optlqg.pickle", "rb") as stream:
        hvac_ofeed_optlqg = pickle.load(stream)

    # Load the maximum likelihood calculations.
    with open("hvac_ofeed_sysid.pickle", "rb") as stream:
        hvac_ofeed_sysid = pickle.load(stream)

    # Load the Q-learning calculations.
    with open("hvac_ofeed_lspi.pickle", "rb") as stream:
        hvac_ofeed_lspi = pickle.load(stream)

    # Load computations for the oracle lqg controller. 
    xseq_list_optlqg = hvac_ofeed_optlqg['xseq_list']
    useq_list_optlqg = hvac_ofeed_optlqg['useq_list']
    yseq_list_optlqg = hvac_ofeed_optlqg['yseq_list']
    lamT_optlqg = hvac_ofeed_optlqg['lamT']

    # Specify the data calculation ID and plot.
    dataCalcID = 11

    # Get the closed-loop simulation dictionaries for the three types of 
    # calculations. 
    clSimData_sysid = hvac_ofeed_sysid['clSimsData_list'][dataCalcID]
    clSimData_lspi = hvac_ofeed_lspi['clSimsData_list'][dataCalcID]

    # Load computations for the maximum likelihood algorithm. 
    xseq_list_sysid = clSimData_sysid['xseq_list']
    useq_list_sysid = clSimData_sysid['useq_list']
    yseq_list_sysid = clSimData_sysid['yseq_list']
    lamT_sysid = clSimData_sysid['lamT']

    # Load computations for the LSPI Qlearning algorithm. 
    xseq_list_lspi = clSimData_lspi['xseq_list']
    useq_list_lspi = clSimData_lspi['useq_list']
    yseq_list_lspi = clSimData_lspi['yseq_list']
    lamT_lspi = clSimData_lspi['lamT']

    # Create an empty list to store all the figures. 
    figures = []

    # Construct lists to plot the closed-loop trajectories. 
    simIdx = 102
    Nt = useq_list_sysid[simIdx].shape[0]
    xlist = [xseq_list_optlqg[simIdx], xseq_list_sysid[simIdx], 
             xseq_list_lspi[simIdx]]
    ulist = [useq_list_optlqg[simIdx], useq_list_sysid[simIdx], 
             useq_list_lspi[simIdx]]
    ylist = [yseq_list_optlqg[simIdx], yseq_list_sysid[simIdx], 
             yseq_list_lspi[simIdx]]
    t = np.arange(0, Nt, 1)/60 # (Convert the time array to hours).

    # Make a plot of the closed-loop trajectories. 
    figures += plot_xmudata(t=t, xlist=xlist, ulist=ulist, ylist=ylist, 
                            legendNames=['PLQG', 'SYSID', 'LSPI'], 
                            legendColors=['r', 'b', 'g'], 
                            figureSize=(5, 4), 
                            ylabelXcoordinate=-0.28,
                            titleLoc=(0.25, 0.92))

    # Make a plot of the histogram of the performance metric obtained over different 
    # trajectories. 
    cost_list = [lamT_optlqg, lamT_sysid, lamT_lspi]
    figures += plot_costHistogram(cost_list=cost_list, 
                                   legendColors=['r', 'b', 'g'], 
                                   legendNames=['PLQG', 'SYSID', 'LSPI'], 
                                   figureSize=(4.5, 4.5), 
                                   xlabel=r'$\Lambda_{Tj}$', 
                                   ylabel='Frequency', nBins=100, 
                                   xlims=[0., 20], ylims=[0, 25])

    # Specify the controller names, list of the performanc metrics, and 
    # controller estimation times. 
    controllerNames = ['SYSID', 'LSPI']
    lamTmean_optlqg = np.mean(lamT_optlqg)
    lamTmeans = [np.mean(lamT_sysid), np.mean(lamT_lspi)]
    estTimes = [hvac_ofeed_sysid['estTimes'][dataCalcID], 
                hvac_ofeed_lspi['estTimes'][dataCalcID]]
    # Call the print function. 
    print_metrics(controllerNames, lamTmean_optlqg, lamTmeans, estTimes)

    # Save the figures.
    with PdfPages('hvac_ofeed_clanalysis_plots.pdf', 'w') as pdf_file:
        for figure in figures:
            pdf_file.savefig(figure)

# Execute main.
main()