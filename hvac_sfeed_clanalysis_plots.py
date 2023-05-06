# [depends] hvac_sfeed_optlqg.pickle hvac_sfeed_sysid.pickle  
# [depends] hvac_sfeed_lspi_kqw.pickle hvac_sfeed_lspi_uqw.pickle
import sys
sys.path.append('lib/')
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_xmudata(*, t, xlist, ulist, legendNames, 
                    legendColors, figureSize,
                    ylabelXcoordinate, titleLoc, 
                    left_frac=0.12, right_frac=0.95, 
                    wspace=0.3, hspace=0.4):
    """ Plot the measurements, states, and control inputs. """

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

    # Create an empty list to store the legend handles. 
    legendHandles = []

    # Loop over all the closed-loop trajectories. 
    for x, u, color in zip(xlist, ulist, legendColors):

        # Create the data list.
        data = [[x[:, 0], x[:, 1]], 
                [x[:, 2], x[:, 3]],
                [u[:, 0], u[:, 1]]]

        # Go through each row.
        for row, col in itertools.product(range(nrow), range(ncol)):
            
            # Make the plot.
            legendHandle = axes[row, col].plot(t, data[row][col], color=color)

            # Y-axis labels.
            axes[row, col].set_ylabel(ylabels[row][col], rotation=False)
            axes[row, col].get_yaxis().set_label_coords(ylabelXcoordinate, 0.5)
        
            # X axis label and limits.
            if row == 2:
                axes[row, col].set_xlabel('Time (Hour)')
                axes[row, col].set_xlim([np.min(t), np.max(t)])

        # Add the legend handle to the list. 
        legendHandles += legendHandle

    # Figure legends. 
    if legendNames is not None:
        figure.legend(handles=legendHandles,
                      labels=legendNames, 
                      loc=titleLoc, ncol=len(legendNames)//2)

    # Return.
    return [figure]

def plot_costHistogram(*, cost_list, legendColors, legendNames,
                          xlabel, ylabel, nBins, xlims, ylims, figureSize):
    """ Make a histogram of the costs computer over different 
        closed-loop trajectories. """
    
    # Create figures.
    figure, axes = plt.subplots(figsize=figureSize)

    # Loop over the cost list.
    for cost, color in zip(cost_list, legendColors):

        # Make the histogram.
        axes.hist(cost, bins=nBins, range=xlims, color=color)

    # Legend.
    if legendNames is not None:
        axes.legend(legendNames)

    # X and Y labels.
    axes.set_ylabel(ylabel)
    axes.set_xlabel(xlabel)

    # X limits.
    axes.set_xlim(xlims)
    axes.set_ylim(ylims)

    # Return the figure.
    return [figure]

def print_metrics(controllerNames, lamTmean_optlqg, lamTmeans, estTimes):
    """ Print performance metrics (mean-of-the closed-loop cost, and controller 
        estimation time) for the four controllers (Optimal LQG, SYSID, 
        LSPI-KQW, LSPI-UQW). 
    """

    # Loop through the provided controller names, mean of the performance 
    # metric, and the estimation times. 
    for (controllerName, lamTmean, estTime) in zip(controllerNames, 
                                                   lamTmeans, estTimes):

        # Compute the performance loss compared to the optimal LQG controller.
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
 
    # Load the closed-loop calculations with the optimal LQR.
    with open("hvac_sfeed_optlqg.pickle", "rb") as stream:
        hvac_sfeed_optlqg = pickle.load(stream)

    # Load the calculatoins with the controller gain estimated using 
    # system identification (MLE).
    with open("hvac_sfeed_sysid.pickle", "rb") as stream:
        hvac_sfeed_sysid = pickle.load(stream)

    # Load the calculations with the controller gain estimated using the 
    # LSPI algorithm with known noise covariance of the process noise.
    with open("hvac_sfeed_lspi_kqw.pickle", "rb") as stream:
        hvac_sfeed_lspi_kqw = pickle.load(stream)

    # Load the calculations with the controller gain estimated using the 
    # LSPI algorithm with unknown noise covariance of the process noise.
    with open("hvac_sfeed_lspi_uqw.pickle", "rb") as stream:
        hvac_sfeed_lspi_uqw = pickle.load(stream)

    # Extract computations for the optimal LQR. 
    xseq_list_optlqg = hvac_sfeed_optlqg['xseq_list']
    useq_list_optlqg = hvac_sfeed_optlqg['useq_list']
    lamT_optlqg = hvac_sfeed_optlqg['lamT']

    # Specify the data calculation ID and plot.
    dataCalcID = 19

    # Get the closed-loop simulation dictionaries for the three types of 
    # calculations. 
    clSimData_sysid = hvac_sfeed_sysid['clSimsData_list'][dataCalcID]
    clSimData_lspi_kqw = hvac_sfeed_lspi_kqw['clSimsData_list'][dataCalcID]
    clSimData_lspi_uqw = hvac_sfeed_lspi_uqw['clSimsData_list'][dataCalcID]

    # Extract computations for the system identification. 
    xseq_list_sysid = clSimData_sysid['xseq_list']
    useq_list_sysid = clSimData_sysid['useq_list']
    lamT_sysid = clSimData_sysid['lamT']

    # Load computations for the LSPI with known noise covariance. 
    xseq_list_lspikqw = clSimData_lspi_kqw['xseq_list']
    useq_list_lspikqw = clSimData_lspi_kqw['useq_list']
    lamT_lspikqw = clSimData_lspi_kqw['lamT']

    # Load computations for the LSPI with unknown noise covariance. 
    xseq_list_lspiuqw = clSimData_lspi_uqw['xseq_list']
    useq_list_lspiuqw = clSimData_lspi_uqw['useq_list']
    lamT_lspiuqw = clSimData_lspi_uqw['lamT']

    # Create an empty list to store all the figures. 
    figures = []

    # Construct lists to plot the closed-loop trajectories. 
    simIdx = 20
    Nt = useq_list_sysid[simIdx].shape[0]
    xlist = [xseq_list_optlqg[simIdx], xseq_list_sysid[simIdx], 
             xseq_list_lspikqw[simIdx], xseq_list_lspiuqw[simIdx]]
    ulist = [useq_list_optlqg[simIdx], useq_list_sysid[simIdx], 
             useq_list_lspikqw[simIdx], useq_list_lspiuqw[simIdx]]
    t = np.arange(0, Nt, 1)/60 # (Convert the time array to hours).

    # Make a plot of the closed-loop trajectories. 
    figures += plot_xmudata(t=t, xlist=xlist, ulist=ulist, 
                            legendNames=['OPTLQG', 'SYSID', 
                                         'LSPI-KQW', 'LSPI-UQW'], 
                            legendColors=['r', 'b', 'm', 'g'], 
                            figureSize=(5, 5), 
                            ylabelXcoordinate=-0.2,
                            titleLoc=(0.28, 0.9))

    # Make a plot of the histogram of the performance metric 
    # obtained over different closed-loop trajectories. 
    cost_list = [lamT_optlqg, lamT_sysid, lamT_lspikqw, lamT_lspiuqw]
    figures += plot_costHistogram(cost_list=cost_list, 
                                   legendNames=['OPTLQG', 'SYSID', 
                                                'LSPI-KQW', 'LSPI-UQW'], 
                                   legendColors=['r', 'b', 'm', 'g'], 
                                   figureSize=(5, 5), 
                                   xlabel=r'$\Lambda_T$', 
                                   ylabel='Frequency', nBins=100, 
                                   xlims=[0., 50.], ylims=[0, 25])

    # Specify the controller names, list of the performanc metrics, and 
    # controller estimation times. 
    controllerNames = ['SYSID', 'LSPI-KQW', 'LSPI-UQW']
    lamTmean_optlqg = np.mean(lamT_optlqg)
    lamTmeans = [np.mean(lamT_sysid), np.mean(lamT_lspikqw), 
                 np.mean(lamT_lspiuqw)]
    estTimes = [hvac_sfeed_sysid['estTimes'][dataCalcID], 
                hvac_sfeed_lspi_kqw['estTimes'][dataCalcID], 
                hvac_sfeed_lspi_uqw['estTimes'][dataCalcID]]
    # Call the print function. 
    print_metrics(controllerNames, lamTmean_optlqg, lamTmeans, estTimes)

    # Save the figures.
    with PdfPages('hvac_sfeed_clanalysis_plots.pdf', 'w') as pdf_file:
        for figure in figures:
            pdf_file.savefig(figure)

# Execute main.
main()