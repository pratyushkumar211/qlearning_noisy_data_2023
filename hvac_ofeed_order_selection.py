# [depends] hvac_ofeed_lspi_fxddata.pickle 
import sys
sys.path.append('lib/')
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def plot_datam_cond_numbers(*, Np_vals, cond_numbers, figureSize, 
                               ylabelXcoordinate, 
                               left_frac=0.18, right_frac=0.95, top_frac=0.95):
    """ Plot measured states and control inputs. """
    
    # Create figure/axes.
    (figure, axes) = plt.subplots(nrows=1, ncols=1,
                                  sharex=True, figsize=figureSize, 
                                  gridspec_kw=dict(left=left_frac, 
                                                   right=right_frac, 
                                                   top=top_frac))

    # Make the condition number plot.
    axes.semilogy(Np_vals, cond_numbers)

    # Y axis limits and labels.
    axes.set_ylabel('Condition number of data matrix')
    axes.get_yaxis().set_label_coords(ylabelXcoordinate, 0.5)

    # X axis label and limits.
    axes.set_xlabel(r'$N_p$')
    axes.set_xlim([np.min(Np_vals), np.max(Np_vals)])
    axes.set_xticks(Np_vals)

    # Return.
    return figure

def main():
    """ Load the pickle files and plot. """
    
    # Optimal linear quadratic Gaussian.
    with open("hvac_ofeed_lspi_fxddata.pickle", "rb") as stream:
        [_, order_selec_data] = pickle.load(stream)

    # Load results for the optimal lqg computations.
    Np_vals = order_selec_data['Np_vals']
    cond_numbers = order_selec_data['cond_numbers']

    # Make the figure. 
    figure = plot_datam_cond_numbers(Np_vals=Np_vals, 
                                     cond_numbers=cond_numbers,
                                     figureSize=(5, 5), ylabelXcoordinate=-0.18)

    # Save the figures.
    with PdfPages('hvac_ofeed_order_selection.pdf', 'w') as pdf_file:
        pdf_file.savefig(figure)

# Execute main.
main()