#%%

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import batman
import os

# Import predefined functions from other files
from simulation_functions import *          # functions for simulating the light curve
from model_functions import *               # functions for evaluation of the model
from mcmc_functions import *                # functions used for the MCMC analysis
from check_convergence import *             # functions used for checking convergence
from TESS_functions import *                # functions used for TESS data preprocessing and plotting
#%%

def run_full_routine(truths, model_params, model, priors, mcmc, 
                     time_data, flux_data, errors_dict, 
                     transform=False, save=None):
    
    for key in errors_dict:
        # iterate through all available error envelopes for the default quadratic parameterization
        if save == None:
            posterior_samples, unflattened_samples = run_mcmc(time_data, flux_data, errors_dict[key], model,
                                                                model_params, priors, mcmc,
                                                                transform=transform)
        else:
            posterior_samples, unflattened_samples = run_mcmc(time_data, flux_data, errors_dict[key], model,
                                                                model_params, priors, mcmc,
                                                                transform=transform, save=save+'_'+key)

        # Plot the corner plot
        create_corner_plot(posterior_samples, truths, errors_dict[key][0]*1e6, transform=transform)

        if transform:
            model_name = f"%.0fppm_kipping_model" % (errors_dict[key][0]*1e6)
        else:
            model_name = f"%.0fppm_quadratic_model" % (errors_dict[key][0]*1e6)

        # Use unflattened samples to check convergence
        gr_stat = check_convergence(unflattened_samples, model_name, truths.keys())

        # Check if Gelman-Rubin statistic is below convergence threshold
        if gr_stat.max() < 1.1:
            print("Chains are well-mixed.")
        else:
            print("Chains may not have converged. Check diagnostics.")

    return
    

#%%
################################################################################
#################### TESS DATA PREPROCESSING & CLEANING ########################
################################################################################

# Planet Name: 					HD-209458b

# Load the data from the csv file
data = np.genfromtxt('outputs/data_TESS/data.csv', delimiter=',')
time_data, flux_data, flux_err_data = data[1:, 0], data[1:, 1], data[1:, 2] # Skip the header & extract the relevant columns

# Clean the data (remove NaN values)
time_data_noNaN, flux_data_noNaN, flux_err_noNaN = clean_data(time_data, flux_data, flux_err_data)

# Normalize the time data & flux data & err data for easier post-processing
time_data_norm_noNaN = time_data_noNaN - time_data_noNaN[0]

min_val = np.min(flux_data_noNaN)
max_val = np.max(flux_data_noNaN)

flux_data_norm_noNaN = (flux_data_noNaN - min_val) / (max_val - min_val)
flux_err_norm_noNaN = flux_err_noNaN / (max_val - min_val)

# Plot the cleaned data, zoom_range=(2826.65, 2826.92)
fig, ax = plot_single_light_curve_with_zoom(time_data_norm_noNaN, flux_data_norm_noNaN, flux_err_norm_noNaN, plt_size=(20, 10),zoom_range=(5.7, 7.3))

# Save the plot
output_plot_dir = pathlib.Path("outputs/plots_TESS")
tess_data_plot_name = output_plot_dir / "tess_data_plot.png"
if not os.path.exists(tess_data_plot_name):
    fig.savefig(tess_data_plot_name, dpi=300)


#%%
# This is the original data with NaN values but normalized: 
time_data_norm = time_data - time_data[0]
flux_data_norm = (flux_data - min_val) / (max_val - min_val)
flux_err_norm = flux_err_data / (max_val - min_val)

# Check for NaN values only for the first three columns
if np.isnan(flux_err_norm).any():
    # Count the number of NaN entries
    nan_count = np.isnan(flux_err_norm).sum()
    print(f"Number of NaN entries: {nan_count}")
else:
    print("No NaN entries found.")

# Check if everything still makes sense and no additional NaN values are present
assert len(flux_data_norm)-len(flux_data_norm_noNaN) == np.isnan(flux_err_norm).sum()

# %%
# Apparently the data contains some jumps (not the NaN values, but actual jumps also for the time). first need to deal with this: 
import sys
np.set_printoptions(threshold=sys.maxsize)

# Step 0: Check that time data is mononically increasing
assert np.all(np.diff(time_data_norm) > 0)

# Step 1: Compute the differences between consecutive time points
# Define the threshold
threshold = 0.00139  # Detect increments larger than this value

# Compute the differences between consecutive time points
time_diffs = np.diff(time_data_norm)

# Detect where increments are larger than the threshold
large_jump_indices = np.where(time_diffs > threshold)[0]  # Indices of large jumps

# print("Time differences:", time_diffs)
print("Indices where increments are larger than threshold:", large_jump_indices)
print("Positions in time array with large jumps:", time_data_norm[large_jump_indices])

#print the size of the jumps
print("Size of the jumps:", time_diffs[large_jump_indices])


# Step 2: Calculate the average size of time increments
# Calculate the average size of time increments
average_time_increment = np.mean(time_diffs[:2021])
print("Average time increment:", average_time_increment)

# Step 3: Correct the time data by replacing the large jumps with the average time increment    
# Manually replace each of the three gaps with the average time increment
desired_incremenet = average_time_increment

# Replace the first gap
print(time_data_norm[4656]-time_data_norm[4657])





#%%
################################################################################
################## TESS DATA - GAUSSIAN PROCESS FIT to GAPS ####################
################################################################################

""" 
Before starting here a short summary: 
- 'time_data_norm_noNaN', 'flux_data_norm_noNaN' and 'flux_err_norm_noNaN' are the normalized data without NaN values.
- 'time_data_norm', 'flux_data_norm' and 'flux_err_norm' are the normalized data with NaN values.
- min_val and max_val are the minimum and maximum values of the flux data which we can use to normalize back (we have no need to normalize back the time data)
"""

from tinygp import GaussianProcess, kernels
import jax.numpy as jnp


fig, ax = plot_simple_light_curve(time_data_norm, flux_data_norm, plt_size=(10, 5))

# Now the goal here is to first save all NaN entries (for which we want to do the GP fit). Then, we want to set those values to 0 and then discard all TRANSITS from the dataset (including those NaN values which are now 0). Then we do the Gaussian Process fit to the gaps and then we can predict the missing values and again put back in the transits.

# Save the NaN entries
# Step 1: Detect NaN values and save their indices
original_NaN_indices = np.where(np.isnan(flux_data_norm))[0]
print(len(original_NaN_indices))

# Step 2: Set NaN values to 0
flux_data_norm[original_NaN_indices] = 0
fig, ax = plot_simple_light_curve(time_data_norm, flux_data_norm, plt_size=(10, 5))




#%%

# define the length scale for the GP
l = 0.002
sigma = 500.0
period = 3.52474955

# Define the Kernel
kernel = (
    sigma**2 + kernels.ExpSquared(scale=l)  # RBF for smooth variations
    + sigma**2 * kernels.ExpSineSquared(scale=period, gamma=1.0)
)

# Normalize flux
flux_mean = np.mean(flux_data)
flux_std = np.std(flux_data)
flux_obs_normalized = (flux_data - flux_mean) / flux_std

# Initialize the GP
gp = GaussianProcess(kernel, time_data, diag=flux_err_data**2)  # Include observational noise in diag

# Condition the GP on data
gp_conditioned = gp.condition(y=flux_obs_normalized, X_test=time).gp

# Predict missing values
mean, std = gp_conditioned.loc, jnp.sqrt(gp_conditioned.variance)

# Denormalize predicted flux
mean = mean * flux_std + flux_mean
std = std * flux_std

# Plot Results
plt.figure(figsize=(10, 6))
plt.plot(time_data, flux_data, ".k", alpha=0.5, label="Observed Data")
plt.plot(time, mean, "-r", label="GP Prediction (mean)")
plt.fill_between(time, mean - std, mean + std, color="r", alpha=0.2, label="Uncertainty")
plt.legend()
plt.xlabel("Time")
plt.ylabel("Flux")
plt.title("Transit Lightcurve Gap Filling with GP")
plt.show()


#%%
################### Step 2 - Initialize Model Parameters #######################
# Define & initialize the parameters for the light curve simulation. Based on the parameters of the paper

# TRUE VALUES (those are the parameters we want to estimate with MCMC)
# most references are from https://exoplanets.nasa.gov/exoplanet_watch_results/hd_209458_b/?utm_source=chatgpt.com

truths = {
    'ps':0.1268,                     # Ref: ExoFOP-TESS TOI
    'u1':0,                          # NO REF (assuming 0 is wrong!)
    'u2':0,                          # NO REF (assuming 0 is wrong!)
    'ecc':0.01,                      # Ref: Rosenthal et al. 2021
    't0':0,                          # NO REF (assuming 0 is wrong!)
}

# FIXED VALUES (those are the parameters we assume to be known)
fixed_params = {
    # orbital parameters:
    'period':3.52474955,             # orbital period (in days):                REF: Kokori et al. 2023
    'a':8.76,                        # semi-major axis in stellar radii:        REF: Kokori et al. 2023
    'inc':86.71,                     # orbital inclination in degrees           REF: Kokori et al. 2023
    'omega':0,                       # longitude of periastron (in degrees)     REF: Rosenthal et al. 2021
    'limb_dark_model':"quadratic",   # limb-darkening model                     REF: https://arxiv.org/pdf/0802.3764
    # observation parameters from dataset:
    'n_points':1000,                 # number of points in the light curve
    't_min':-0.25,                   # minimum time in days
    't_max':0.25,                    # maximum time in days
}
# Initialize the parameters accordingly, to match the "syntax" of the batman package
params, t_array = initialize_parameters(truths, fixed_params)


#%%
################ Step 2 - Create Model & Simulate Light Curve ##################

# Initialize the batman model using the parameters from above
model, time_data, flux_data = initialize_model(params, t_array)    #initializes model for the simulation

# Define the error envelopes for the light curve ranging from 1-1000 ppm & store them in a dictionary. 
all_errors_dict = {
    "1 ppm":    (1    / 1e6) * flux_data,       # error envelope for 1 ppm
    "10 ppm":   (10   / 1e6) * flux_data,       # error envelope for 10 ppm
    "30 ppm":   (30   / 1e6) * flux_data,       # error envelope for 30 ppm
    "100 ppm":  (100  / 1e6) * flux_data,       # error envelope for 100 ppm,
    "300 ppm":  (300  / 1e6) * flux_data,       # error envelope for 300 ppm, 
    "1000 ppm": (1000 / 1e6) * flux_data,       # error envelope for 1000 ppm
    }  

save_simdata(time_data, flux_data, all_errors_dict, 'no_ld')
# the reload below is unnecessary, but here to illustrate how to load the data if necessary later on:
time_data, flux_data, all_errors_dict = load_simdata_all_errs('no_ld')



#%%
##################### Step 3 - Plot & Save Light Curve #######################

# Plot the simulated light curve. This is the light curve we assume to be the true data. The error envelopes represent our "observed" data
fig, ax = plot_single_light_curve(flux_data, time_data, all_errors_dict, plt_size=(15, 8))

# Dynamically generate the filename with all truths parameters
output_plot_dir = pathlib.Path("outputs/plots")
truths_str = get_name_str(truths=truths)
light_curve_plot_name = output_plot_dir / f"light_curve_plot_sim_{truths_str}.png"

# Save the light curve plot
if not os.path.exists(light_curve_plot_name):
    fig.savefig(light_curve_plot_name, dpi=300)


#%%
################################################################################
######################### Run Quadratic Limb-Drkening ##########################
################################################################################

# choose Priors
# convention:
# [uniform, lower bound, upper bound]
# [gauss, mean, sigma]
param_priors = {
    'ps':        ['uni', 0., 0.5],     # stellar radii
    'u1':        ['uni', -3., 3],      # limb darkening
    'u2':        ['uni', -3., 3.],     # limb darkening
}

# MCMC parameters
mcmc_params = {
    'ndim'        :len(param_priors),
    'nwalkers'    :4*len(param_priors),
    'nsteps'      :100000,
    'burn_in_frac':0.6,
}

# iterate over all errors, creating output plots, may take a while!
# if you want to save the file, additionally pass the argument save=get_name_str(truths)
# run_full_routine(truths, params, model, param_priors, mcmc_params, 
#                  time_data, flux_data, all_errors_dict, 
#                  transform=False)#, save=get_name_str(truths))
#%%
################################################################################
######################### Run Kipping Limb-Drkening ############################
################################################################################

# update priors and ground truth to Kipping
param_priors = {
    'ps':        ['uni', 0., 0.5],    # stellar radii
    'u1':        ['uni', 0., 1.],     # limb darkening
    'u2':        ['uni', 0., 1.],     # limb darkening
}

truths = {
    'ps':0.1,                        # planet-to-star radius ratio = planet radius (in units of stellar radii)
    'u1':0,                          # limb-darkening coefficients: q1 
    'u2':None,                       # limb-darkening coefficients: q2
    # TODO: q2 is not actually well defined... need to calculate the limes of 0.5 * u1 / (u1 + u2) for u1 & u2 -> 0
    # but it diverges (gives different results depending which param you let go to 0 first)
}

# iterate over all errors, creating output plots, may take a while!
# if you want to save the file, additionally pass the argument save=get_name_str(truths)
run_full_routine(truths, params, model, param_priors, mcmc_params, 
                 time_data, flux_data, all_errors_dict, 
                 transform=True)#, save=get_name_str(truths))

#%%
