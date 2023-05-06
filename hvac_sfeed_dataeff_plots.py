# [depends] hvac_sfeed_optlqg.pickle hvac_sfeed_sysid.pickle 
# [depends] hvac_sfeed_lspi_kqw.pickle hvac_sfeed_lspi_uqw.pickle
# [depends] hvac_sfeed_lspi_regl.pickle
import sys
sys.path.append('lib/')
import itertools
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_dataeffmetrics(*, t, errS_list, errK_list, errLam_list, perfloss_list, 
                    legendNames, lineColors, figureSize, 
                    ylabelXcoordinate, titleLoc, 
                    left_frac=0.18, right_frac=0.95, top_frac=0.86,
                    wspace=0.6, hspace=0.1, 
                    xticks = [3, 15, 30, 45, 60], ntitlecol=2):
    """ Plot measured states and control inputs. """

    # Y axes labels.
    ylabels = [[r'$\dfrac{|S - S^{*}|}{|S^{*}|}$', 
                r'$\%$Loss'],
               [r'$\dfrac{|K - K^{*}|}{|K^{*}|}$', 
                r'$\dfrac{|\beta - \beta^{*}|}{|\beta^{*}|}$']]

    # Number of rows and columns, 
    nrow, ncol = 2, 2
    
    # Create figure/axes.
    (figure, axes) = plt.subplots(nrows=nrow, ncols=ncol,
                                  sharex=True, figsize=figureSize, 
                                  gridspec_kw=dict(left=left_frac, 
                                                   right=right_frac, 
                                                   top=top_frac,
                                                   wspace=wspace, 
                                                   hspace=hspace))

    # Legend handles. 
    legendHandles = []

    # Go through all the simulation datasets.
    for (errS, errK, errLam, 
         perfloss, lineColor) in zip(errS_list, errK_list, errLam_list, 
                                     perfloss_list, lineColors):

        data = [[errS, perfloss], [errK, errLam]]

        # Go through each row.
        for row, col in itertools.product(range(nrow), range(ncol)):

            # Plot the data. 
            legendHandle = axes[row, col].plot(t, data[row][col], 
                                                   color=lineColor)
            
            # Y-axis scaling. 
            if (row == 0 and col == 1):
                axes[row, col].set_yscale('symlog')
            else: 
                axes[row, col].set_yscale('log')
            
            # Plot depending on the row.
            if row == 0 and col == 0:
                legendHandles += legendHandle

            # Y axis limits and labels.
            axes[row, col].set_ylabel(ylabels[row][col], rotation=False)
            axes[row, col].get_yaxis().set_label_coords(ylabelXcoordinate, 0.5)

            # X axis label, limits, and xticks.
            if row == nrow - 1:
                axes[row, col].set_xlabel('Training data size (hours)')
                axes[row, col].set_xlim([np.min(t), np.max(t)])
                axes[row, col].set_xticks(xticks)

    # Figure legends. 
    if legendNames is not None:
        figure.legend(handles=legendHandles,
                      labels=legendNames,
                      loc=titleLoc, ncol=ntitlecol)

    # Return.
    return [figure]

def main():
    """ Load the pickle files and plot. """
    
    # Load the closed-loop calculations with the optimal LQR.
    with open("hvac_sfeed_optlqg.pickle", "rb") as stream:
        hvac_sfeed_optlqg = pickle.load(stream)

    # Maximum likelihood.
    with open("hvac_sfeed_sysid.pickle", "rb") as stream:
        hvac_sysid = pickle.load(stream)

    # Kwown noise covariance.
    with open("hvac_sfeed_lspi_kqw.pickle", "rb") as stream:
        hvac_lspi_kqw = pickle.load(stream)

    # Unknown noise covariance.
    with open("hvac_sfeed_lspi_uqw.pickle", "rb") as stream:
        hvac_lspi_uqw = pickle.load(stream)
    
    # Unknown noise covariance regularized.
    with open("hvac_sfeed_lspi_regl.pickle", "rb") as stream:
        hvac_lspi_regl = pickle.load(stream)

    # Load results for the optimal lqg computations.
    lamT_optlqg = hvac_sfeed_optlqg['lamT']
    lamTmean_optlqg = np.mean(lamT_optlqg)

    # Load computations for the system identification computations. 
    t = hvac_sysid['t']/3600 # (scale to hours).
    errS_sysid = hvac_sysid['errS']
    errK_sysid = hvac_sysid['errK']
    errLam_sysid = hvac_sysid['errLam']
    lamTmean_sysid = hvac_sysid['lamTmeans']
    perfloss_sysid = 100*(lamTmean_sysid - lamTmean_optlqg)/lamTmean_optlqg

    # Load computations for the LSPI algorithm with unknown noise covariance. 
    errS_lspi_kqw = hvac_lspi_kqw['errS']
    errK_lspi_kqw = hvac_lspi_kqw['errK']
    errLam_lspi_kqw = hvac_lspi_kqw['errLam']
    lamTmean_lspi_kqw = hvac_lspi_kqw['lamTmeans']
    perfloss_lspi_kqw = 100*(lamTmean_lspi_kqw - lamTmean_optlqg)/lamTmean_optlqg

    # Load computations for the LSPI algorithm with known noise covariance. 
    errS_lspi_uqw = hvac_lspi_uqw['errS']
    errK_lspi_uqw = hvac_lspi_uqw['errK']
    errLam_lspi_uqw = hvac_lspi_uqw['errLam']
    lamTmean_lspi_uqw = hvac_lspi_uqw['lamTmeans']
    perfloss_lspi_uqw = 100*(lamTmean_lspi_uqw - lamTmean_optlqg)/lamTmean_optlqg

    # Load computations for the LSPI algorithm that does regularization. 
    errS_lspi_regl = hvac_lspi_regl['errS']
    errK_lspi_regl = hvac_lspi_regl['errK']
    errLam_lspi_regl = hvac_lspi_regl['errLam']
    lamTmean_lspi_regl = hvac_lspi_regl['lamTmeans']
    perfloss_lspi_regl = 100*(lamTmean_lspi_regl - lamTmean_optlqg)/lamTmean_optlqg

    # Create an empty list to store all the figures. 
    figures = []

    # Create a list of errors in S, K, and Lam.
    errS_list = [errS_sysid, errS_lspi_kqw, errS_lspi_uqw, errS_lspi_regl]
    errK_list = [errK_sysid, errK_lspi_kqw, errK_lspi_uqw, errK_lspi_regl]
    errLam_list = [errLam_sysid, errLam_lspi_kqw, errLam_lspi_uqw, errLam_lspi_regl]
    perfloss_list = [perfloss_sysid, perfloss_lspi_kqw, perfloss_lspi_uqw, perfloss_lspi_regl]
    perflossp25_list = None
    
    # Make two figures, without and with the regularization idea. The latter  
    # goes in the review reply. 
    # Without regularization.
    figures += plot_dataeffmetrics(t=t, errS_list=errS_list[:-1],
                            errK_list=errK_list[:-1],
                            errLam_list=errLam_list[:-1],
                            perfloss_list=perfloss_list[:-1],
                            legendNames=[ 'SYSID', 'LSPI-KQW', 'LSPI-UQW'], 
                            lineColors=['b', 'm', 'g'], 
                            figureSize=(5, 4.5), ylabelXcoordinate=-0.4, 
                            titleLoc=(0.2, 0.88), ntitlecol=3)

    # With Regularization.
    figures += plot_dataeffmetrics(t=t, errS_list=errS_list,
                        errK_list=errK_list,
                        errLam_list=errLam_list,
                        perfloss_list=perfloss_list,
                        legendNames=[ 'SYSID', 'LSPI-KQW', 'LSPI-UQW', 'LSPI-Reg'], 
                        lineColors=['b', 'm', 'g', 'tomato'], 
                        figureSize=(5, 4.5), ylabelXcoordinate=-0.4, 
                        titleLoc=(0.3, 0.88))


    # Save the figures.
    with PdfPages('hvac_sfeed_dataeff_plots.pdf', 'w') as pdf_file:
        for figure in figures:
            pdf_file.savefig(figure)

# Execute main.
main()