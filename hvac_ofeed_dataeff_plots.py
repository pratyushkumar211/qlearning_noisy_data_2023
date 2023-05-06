# [depends] hvac_ofeed_optlqg.pickle hvac_ofeed_sysid.pickle 
# [depends] hvac_ofeed_lspi.pickle
import sys
sys.path.append('lib/')
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_dataeffmetrics(*, t, perfloss_list, legendNames, 
                              lineColors, figureSize, 
                              ylabelXcoordinate, titleLoc, 
                              left_frac=0.12, right_frac=0.95, top_frac=0.95):
    """ Plot measured states and control inputs. """
    
    # Create figure/axes.
    (figure, axes) = plt.subplots(nrows=1, ncols=1,
                                  sharex=True, figsize=figureSize, 
                                  gridspec_kw=dict(left=left_frac, 
                                                   right=right_frac, 
                                                   top=top_frac))

    # Go through all the simulation datasets.
    for (perfloss, lineColor, 
         legendName) in zip(perfloss_list, lineColors, legendNames):

        # Make the plot of the legend.
        axes.plot(t, perfloss, color=lineColor, label=legendName)

    # Axes legends.
    axes.legend()

    # Y axis limits and labels.
    axes.set_ylabel(r'$\%$ Loss')
    axes.get_yaxis().set_label_coords(ylabelXcoordinate, 0.5)

    # X axis label and limits.
    axes.set_xlabel('Training data size (hours)')
    axes.set_xlim([np.min(t), np.max(t)])
    # Xtick locations. Increase them every 2 hours. 
    xticks = list(np.arange(np.min(t), np.max(t), 2)) + [24]
    axes.set_xticks(xticks) 

    # Return.
    return figure

def main():
    """ Load the pickle files and plot. """
    
    # Optimal linear quadratic Gaussian.
    with open("hvac_ofeed_optlqg.pickle", "rb") as stream:
        hvac_optlqg = pickle.load(stream)

    # Maximum likelihood.
    with open("hvac_ofeed_sysid.pickle", "rb") as stream:
        hvac_sysid = pickle.load(stream)

    # Unknown noise covariance.
    with open("hvac_ofeed_lspi.pickle", "rb") as stream:
        hvac_lspi = pickle.load(stream)

    # Load results for the optimal lqg computations.
    lamT_optlqg = hvac_optlqg['lamT']
    lamTmean_optlqg = np.mean(lamT_optlqg)

    # Load computations for the system identification computations. 
    t = hvac_sysid['t']/3600 # (scale to hours).
    lamTmean_sysid = hvac_sysid['lamTmeans']
    perfloss_sysid = 100*(lamTmean_sysid - lamTmean_optlqg)/lamTmean_optlqg

    # Load computations for the LSPI algorithm with unknown noise covariance. 
    lamTmean_lspi = hvac_lspi['lamTmeans']
    perfloss_lspi = 100*(lamTmean_lspi - lamTmean_optlqg)/lamTmean_optlqg

    # Create a list of errors in S, K, and Lam. 
    perfloss_list = [perfloss_sysid, perfloss_lspi]
    
    # Make the figure. 
    figure = plot_dataeffmetrics(t=t, 
                                perfloss_list=perfloss_list,
                                legendNames=[ 'SYSID', 'LSPI'], 
                                lineColors=['b', 'g'], 
                                figureSize=(4.5, 4.5), ylabelXcoordinate=-0.1, 
                                titleLoc=(0.4, 0.9))

    # Save the figures.
    with PdfPages('hvac_ofeed_dataeff_plots.pdf', 'w') as pdf_file:
        pdf_file.savefig(figure)

# Execute main.
main()