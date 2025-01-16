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
#################### SIMULATING TRANSIT LIGHT CURVE(S) #########################
################################################################################


################### Step 1 - Initialize Model Parameters #######################
# Define & initialize the parameters for the light curve simulation. Based on the parameters of the paper

# TRUE VALUES (those are the parameters we want to estimate with MCMC)
truths = {
    'ps':0.1,                        # planet-to-star radius ratio = planet radius (in units of stellar radii)
    'u1':0,                          # limb-darkening coefficients: u1, 
    'u2':0,                          # u2 (no limb-darkening: u1=0, u2=0)
}

# FIXED VALUES (those are the parameters we assume to be known)
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

