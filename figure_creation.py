#%%
###############################################################################
""" Import what is needed """
###############################################################################
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import os
from joblib import load
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator, LogLocator

# Import predefined functions from other files
import simulation_functions as sim          # functions for simulating the light curve
from plot_layout import *
###############################################################################
#%%
###############################################################################
""" Define the Classes """
###############################################################################
class CI:
    def __init__(self, model, error, truths, priors, mcmc_params, samples, flux, intensity, tdd):
        self.model = model
        self. error = error
        self.truths = truths
        self.priors = priors
        self. mcmc_params = mcmc_params
        self.samples = samples
        self.flux = flux
        self.intensity = intensity
        self.tdd = tdd
    def __repr__(self):
        if self.truths['u1']==0:
            limb = 'No Limb Darkening'
        else:
            limb = 'Minimal Limb Darkening'
        return f"Confidence: {self.model}, Error: {self.error} ppm, Truths:{self.truths}, {limb}"

class Posterior_Samples:
    def __init__(self, model, error, truths, priors, mcmc_params, samples, tdd):
        self.model = model
        self. error = error
        self.truths = truths
        self.priors = priors
        self. mcmc_params = mcmc_params
        self.samples = samples
        self.tdd = tdd
    def __repr__(self):
        if self.truths['u1']==0:
            limb = 'No Limb Darkening'
        else:
            limb = 'Minimal Limb Darkening'
        return f"Posterior: {self.model}, Error: {self.error} ppm, Truths:{self.truths}, {limb}"
###############################################################################
#%%
###############################################################################
""" Function to read the Datasets """
###############################################################################
def read_datasets(file_path, category):
    """
    Reads all .joblib files in the given directory that match the specified category.

    Args:
        file_path (str): Path to the directory containing .joblib files.
        category (str): Prefix of the files to load (e.g., 'confidence', 'posterior').

    Returns:
        list: A list of loaded datasets.
    """
    # Get all files in the directory with .joblib extension and the specified category prefix
    list_of_files = [f for f in os.listdir(file_path) if f.endswith('.joblib') and f.startswith(category)]
    
    # Load each file and store the objects in a list
    list_of_datasets = []
    for f in list_of_files:
        try:
            file_path_full = os.path.join(file_path, f)
            data = load(file_path_full)  # Load the .joblib file
            list_of_datasets.append(data)
        except Exception as e:
            print(f"Error loading file {f}: {e}")
    
    return list_of_datasets
###############################################################################
#%%
###############################################################################
""" Function to filter the list of datasets """
###############################################################################
def filter_datasets(list_of_datasets, model=None, error=None, limb=None):
    """
    Filters a list of datasets based on the specified criteria.

    Args:
        list_of_datasets (list): A list of datasets to filter.
        model (str, optional): Model name to filter by. Defaults to None (no filter).
        error (int, optional): Error value to filter by. Defaults to None (no filter).
        limb (str, optional): Limb darkening condition ('No' or 'Minimal') to filter by. Defaults to None (no filter).

    Returns:
        list or object: The filtered list of datasets, or a single dataset if only one matches.
    """
    filtered_list = []
    for d in list_of_datasets:
        # Determine the limb darkening condition
        if d.truths['u1'] == 0:
            dark = 'No'
        else:
            dark = 'Minimal'
        
        # Apply filters
        if ((model is None or d.model == model) and 
            (error is None or d.error == error) and 
            (limb is None or dark == limb)):
            filtered_list.append(d)
    
    # Return the dataset directly if there's only one match
    if len(filtered_list) == 1:
        return filtered_list[0]
    else:
        return filtered_list
###############################################################################
#%%
###############################################################################
""" Function to generate the true ligth curve """
###############################################################################
def true_lightcurve(limb='No'):
    if limb=='No':
        truths = {
            'ps':0.1,                        # planet-to-star radius ratio = planet radius (in units of stellar radii)
            'u1':0.,                          # limb-darkening coefficients: u1, 
            'u2':0.,                          # u2 (no limb-darkening: u1=0, u2=0)
        }
    else:
        truths = {
            'ps':0.1,                        # planet-to-star radius ratio = planet radius (in units of stellar radii)
            'u1':0.05,                          # limb-darkening coefficients: u1, 
            'u2':0.05,                          # u2 (no limb-darkening: u1=0, u2=0)
        }
    fixed_params = {
        # orbital parameters:
        't0':0,                          # time of inferior conjunction
        'period':1,                      # orbital period (in days)
        'a':4,                           # semi-major axis in stellar radii
        'inc':90,                        # orbital inclination in degrees
        'ecc':0,                         # eccentricity
        'omega':90,                      # longitude of periastron (in degrees)
        'limb_dark_model':"quadratic",   # limb-darkening model
        # simulated observation parameters:
        'n_points':1000,                 # number of points in the light curve
        't_min':-0.25,                   # minimum time in days
        't_max':0.25,                    # maximum time in days
    }
    params, t_array = sim.initialize_parameters(truths, fixed_params)
    model, time_truth, flux_truth = sim.initialize_model(params, t_array)
    return time_truth, flux_truth
###############################################################################
#%%
###############################################################################
""" Create Figure 1 """
###############################################################################
def create_figure_1(file_path, error, limb, save_path=None):
    """
    Creates and saves a multi-panel figure summarizing intensity profiles, 
    normalized flux, and probability density for various models.

    Args:
        file_path (str): Path to the directory containing input data files.
        error (int): Error level (e.g., 1000, 300) to filter datasets.
        limb (str): Specifies limb darkening ('No' or other values for limb treatment).
        save_path (str, optional): Directory to save the figure. If None, displays the plot.

    Returns:
        None
    """
    ###########################################################################
    # Check if input and output paths are valid
    if not os.path.exists(file_path):
        print('Invalid input file path')
        return
    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)
    ###########################################################################
    # Initialize models, errors, and plotting configurations
    models = ['quadratic','kipping']
    errors = [1000,300,100,30,10,1]
    colors = [plotcolors(model,len(errors)) for model in models]
    colors.append(plotcolors('truth',len(errors)))  # Add truth color
    figsize, fontsize = plotsizes()                 # Get figure and font sizes
    ###########################################################################
    # Labels for intensity and flux plots
    intensity_label = [f"[$u_1$,$u_2$]", f"[$q_1$,$q_2$]"]
    flux_label = ["Truth", "Truth"]
    ###########################################################################
    # Determine error index for color selection
    err_idx = errors.index(error)
    ###########################################################################
     # Load datasets and calculate truth light curve
    confidence_data = read_datasets(file_path,'confidence')
    posterior_data = read_datasets(file_path,'posterior')
    time_truth, flux_truth = true_lightcurve(limb=limb)
    mu = np.linspace(0,1,1000)                 # Radial coordinate
    ###########################################################################
    # Create figure layout
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(14,6), 
                            gridspec_kw={'width_ratios':[2,2,3]})
    # Configure the large subplot
    gs = axs[1,2].get_gridspec()
    for ax in axs[0:,-1]:
        ax.remove()
    axbig = fig.add_subplot(gs[0:,-1])
    ###########################################################################
    # Iterate through models and plot intensity, flux, and probability density
    for i, model in enumerate(models):
        # Extract relevant data
        flux = filter_datasets(confidence_data, model=model, error=error, limb=limb).flux
        intensity = filter_datasets(confidence_data, model=model, error=error, limb=limb).intensity
        tdd = filter_datasets(posterior_data, model=model, error=error, limb=limb).tdd
        #######################################################################
        # Plot intensity profile
        axs[0,i].plot(mu, intensity['median'], color=colors[i][err_idx], label=intensity_label[i])
        axs[0,i].fill_between(mu,intensity['1 sigma'][0],intensity['1 sigma'][1], color=colors[i][err_idx], alpha = 0.3)
        axs[0,i].fill_between(mu,intensity['2 sigma'][0],intensity['2 sigma'][1], color=colors[i][err_idx], alpha = 0.1)
        #######################################################################
        # Configure intensity plot layout
        axs[0,i].set_xlim((-0.05,1.05))        
        axs[0,i].xaxis.set_major_locator(MultipleLocator(0.25))
        axs[0,i].set_xlabel(r'Radial Coordinate $\mu$', fontsize=fontsize['axis'])
        axs[0,i].legend(fontsize=fontsize['legend'])
        #######################################################################
        # Plot flux data
        axs[1,i].plot(time_truth,flux['median'], color=colors[i][err_idx])
        axs[1,i].fill_between(time_truth,flux['1 sigma'][0],flux['1 sigma'][1], color=colors[i][err_idx], alpha = 0.3)
        axs[1,i].fill_between(time_truth,flux['2 sigma'][0],flux['2 sigma'][1], color=colors[i][err_idx], alpha = 0.1)
        axs[1,i].plot(time_truth,flux_truth, color=colors[-1][err_idx], label=flux_label[i])
        #######################################################################
        # Configure flux plot layou        
        axs[1,i].set_xlim((-0.05,0.05))
        axs[1,i].xaxis.set_major_locator(MultipleLocator(0.02))
        axs[1,i].set_xlabel(f"Time [days]", fontsize=fontsize['axis'])
        axs[1,i].legend(fontsize=fontsize['legend'])
        #######################################################################
        # Configure intensity and flux plot layout y-axis
        if limb=='No':
            # l = int(len(time_truth)/2)-30
            # u = int(len(time_truth)/2)+30
            # max_diff = np.max(np.concatenate((
            #     (flux['median'][l:u]-flux['2 sigma'][0][l:u]), 
            #     (flux['2 sigma'][1][l:u]-flux['median'][l:u])
            # )))
            # exp = int(f"{max_diff:.20e}".split('e')[1])
            # step_int = 10**(exp+3)
            # step_flux = 5*10**exp
            # mid_flux = np.min(flux_truth)
            
            # axs[0,i].set_ylim((1-2*step_int+0.1*step_int,1+2*step_int+0.1*step_int))
            # axs[0,i].yaxis.set_major_locator(MultipleLocator(step_int))
            # axs[1,i].set_ylim((mid_flux-step_flux,mid_flux+2*step_flux))
            # axs[1,i].yaxis.set_major_locator(MultipleLocator(step_flux))
            
            axs[0,i].set_ylim((0.82,1.18))
            axs[0,i].yaxis.set_major_locator(MultipleLocator(0.1))
            axs[1,i].set_ylim((0.9895,0.9910))
            axs[1,i].yaxis.set_major_locator(MultipleLocator(0.0005))
        else:
            axs[0,i].set_ylim((0.82,1.18))
            axs[0,i].yaxis.set_major_locator(MultipleLocator(0.1))
            axs[1,i].set_ylim((0.9895,0.9910))
            axs[1,i].yaxis.set_major_locator(MultipleLocator(0.0005))
        #######################################################################
        # Plot probability density
        axbig.hist(tdd, bins = 50, density=True, color=colors[i][err_idx], 
                   alpha=0.5)
        axbig.hist(tdd, bins = 50, density=True, edgecolor=colors[i][err_idx], 
                   linewidth=1.5, histtype='step')
    ###########################################################################
    # Configure probability density layout
    axbig.tick_params(axis='y',labelleft=False, which='both',length=0)
    axbig.axvline(x=0, color=colors[-1][err_idx],linestyle='--')
    # Set the Axis Labels
    axs[0,0].set_ylabel(f"Intensity Profile", fontsize=fontsize['axis'])
    axs[1,0].set_ylabel(f"Normalized Flux", fontsize=fontsize['axis'])
    axbig.set_xlabel(f"Deviation from True Transit Depth [ppm]", fontsize=fontsize['axis'])
    axbig.set_ylabel(f"Probability Density", fontsize=fontsize['axis'])
    fig.tight_layout()
    ###########################################################################
    # Save or display the figure
    if save_path != None:
        filename_add = 'no' if limb=='No' else 'min_limb'
        filename = os.path.join(save_path,f"Figure_1_{filename_add}_{error}_ppm")
        fig.savefig(filename, dpi=300)
    else:
        plt.show()

    return
###############################################################################
#%%
###############################################################################
""" Create Figure 2 """
###############################################################################
def create_figure_2(file_path, save_path=None, confidence='1 sigma', limb='both'):
    """
    Creates and saves a figure comparing deviations from the true transit depth 
    across different photometric scatter levels and limb darkening models.

    Args:
        file_path (str): Path to the directory containing input data files.
        save_path (str, optional): Directory to save the figure. If None, displays the plot.
        confidence (str): Confidence interval for error bars ('1 sigma' or '2 sigma').
        limb (str): Limb darkening treatment ('both', 'No', or 'Minimal').

    Returns:
        None
    """
    ###########################################################################
    # Check if input and output paths are valid
    if not os.path.exists(file_path):
        print('Invalid input file path')
        return
    if save_path is not None and not os.path.exists(save_path):
        os.makedirs(save_path)
    ###########################################################################
    # Configure limb darkening settings
    if limb=='both':
        n = 2
        limbs = ['No','Minimal']
        filename_add='both'
    elif limb=='No':
        n = 1
        limbs = [limb]
        filename_add='no'
    elif limb=='Minimal':
        n = 1
        limbs = [limb]
        filename_add='min_limb'
    else:
        print("Limb not recognized, must be 'both', 'No' or 'Minimal'")
        return
    ###########################################################################
    # Initialize models, errors, colors, and plotting configurations
    models = ['quadratic','kipping']
    errors = [1,10,30,100,300,1000]                         # Photometric scatter levels in ppm
    colors = [plotcolors(model,n) for model in models]
    figsize, fontsize = plotsizes()
    figsize = tuple(e*0.75 for e in figsize)                # Adjust figure size
    confidence_data = read_datasets(file_path,'confidence') # Load dataset
    intensity_label = [f"[$u_1$,$u_2$]", f"[$q_1$,$q_2$]"]  # Labels for models
    ###########################################################################
    # Create figure and add horizontal reference line
    fig, ax = plt.subplots(figsize=figsize)
    ax.axhline(y=0, color='black', linestyle=':', linewidth=1)
    ###########################################################################
    # Iterate over limb darkening options and models to plot data
    for j,limb in enumerate(limbs):
        for i,model in enumerate(models):
            median = []
            lower_err = []
            upper_err = []
            # Collect median and error bounds for each photometric scatter level
            for error in errors:
                tdd_CI = filter_datasets(confidence_data, model=model, error=error, limb=limb).tdd
                
                median.append(tdd_CI['median'])
                lower_err.append(tdd_CI['median']-tdd_CI[confidence][0])
                upper_err.append(tdd_CI[confidence][1]-tdd_CI['median'])
            # Plot data with error bars
            ax.errorbar(np.array(errors), np.array(median), 
                        yerr=[np.array(lower_err),np.array(upper_err)], 
                        fmt='o', color=colors[i][j],capsize=5, 
                        label=f"{intensity_label[i]}, {limb} Limb Darkening", 
                        linestyle='--', alpha=0.8)
    ###########################################################################
    # Configure axes and layout
    ax.set_xscale('log')                 # Logarithmic scale for x-axis (photometric scatter)
    ax.set_yscale('symlog',linthresh=1)  # Symmetrical log scale for y-axis
    ax.set_xlabel(f"Photometric Scatter [ppm]", fontsize=fontsize['axis'])
    ax.set_ylabel(f"Deviation from True Transit Depth [ppm]", fontsize=fontsize['axis'])
    ax.tick_params(axis='both',labelsize=fontsize['tick_label'])
    ax.legend(fontsize=fontsize['legend'],ncol=n)
    
    ###########################################################################
    # Save or display the figure
    if save_path != None:
        filename = os.path.join(save_path,f"Figure_2_{filename_add}")
        fig.savefig(filename, dpi=300)
    else:
        plt.show()

    return
###############################################################################
#%%
###############################################################################