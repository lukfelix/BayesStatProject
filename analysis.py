#%%

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import batman
import corner
import pathlib
import os

# Import predefined functions from other files
from simulation_functions import *          # functions for simulating the light curve
from model_functions import *               # functions for evaluation of the model
from mcmc_functions import *                # functions used for the MCMC analysis
from check_convergence import *             # functions used for checking convergence
#%%
################################################################################
#################### SIMULATING TRANSIT LIGHT CURVE(S) #########################
################################################################################


################### Step 1 - Initialize Model Parameters #######################

# Define & initialize the parameters for the light curve simulation. Based on the parameters of the paper
# TODO: Double check if these parameters are correct from the paper => DONE!

# TRUE VALUES (those are the parameters we want to estimate with MCMC)
truths = {
    'ps':0.1,                        # planet-to-star radius ratio = planet radius (in units of stellar radii)
    'u':[0, 0]                       # limb-darkening coefficients: u1, u2 (no limb-darkening = [0, 0])
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
ncpu = 1 # number of cores available for batman (set to 1 if you either want to be slow or don't have openMP)
model = batman.TransitModel(params, t_array, nthreads = int(ncpu))    #initializes model for the simulation

# Generate data
flux_data = simulate_light_curve(model, params)         # Simulate the light curve using the batman model and the parameters to generate the data
time_data = t_array                                     # Time array for the simulation

# Define the (half-) error envelopes for the light curve ranging from 1-1000 ppm & store them in a dictionary. Half because we want the error to be symmetric around the flux_data
all_errors_dict = {
    "1 ppm":    (1    / 1e6) * flux_data,       # error envelope for 1 ppm
    "10 ppm":   (10   / 1e6) * flux_data,       # error envelope for 10 ppm
    "100 ppm":  (100  / 1e6) * flux_data,       # error envelope for 100 ppm, 
    "1000 ppm": (1000 / 1e6) * flux_data,       # error envelope for 1000 ppm
    }  

# TODO: implement a way to save the data under "outputs/data". Best to probably use numpy's "np.save" and "np.load" to save, resp. load the data
# => DONE!
# Create the directory
output_dir = pathlib.Path("outputs/data")
output_dir.mkdir(parents=True, exist_ok=True)

# Save time data and flux data
np.save(output_dir / "time_data.npy", time_data)
np.save(output_dir / "flux_data.npy", flux_data)

# Save error envelopes
for key, error_data in all_errors_dict.items():
    filename = key.replace(" ", "_") + "_error.npy"  # e.g., "1_ppm_error.npy"
    np.save(output_dir / filename, error_data)


#Can later load data like this:
# Load time and flux data
time_data_loaded = np.load(output_dir / "time_data.npy")
flux_data_loaded = np.load(output_dir / "flux_data.npy")

# Load error envelopes
all_errors_loaded = {}
for key in all_errors_dict.keys():
    filename = key.replace(" ", "_") + "_error.npy"
    all_errors_loaded[key] = np.load(output_dir / filename)






#%%
##################### Step 3 - Plot & Save Light Curve #######################

# Plot the simulated light curve. This is the light curve we assume to be the true data. The error envelopes represent our "observed" data
fig, ax = plot_single_light_curve(flux_data, time_data, all_errors_dict, plt_size=(15, 8))

# Save the light curve plot
#TODO change naming above to match all the model parameters (not very important, only relevant if we actually change them) => DONE!
# Dynamically generate the filename with all truths parameters
# Bereinigen des Strings und Hinzufügen einer Dateiendung

output_plot_dir = pathlib.Path("outputs/plots")
truths_str = "_".join([f"{key}_{str(value).replace('[', '').replace(']', '').replace(', ', '_').replace(' ', '_')}" for key, value in truths.items()])
light_curve_plot_name = output_plot_dir / f"light_curve_plot_sim_{truths_str}.png"

if not os.path.exists(light_curve_plot_name):
    fig.savefig(light_curve_plot_name, dpi=300)

#%%

################################################################################
######################### FITTING DATA TO A MODEL ##############################
################################################################################

# choose Priors
# convention:
# [uniform, lower bound, upper bound]
# [gauss, mean, sigma]
param_priors = {
    # TODO: adapt these depending on simdata => DONE, see suggestion below!
    'ps':        ['uni', 0., 0.5],      # stellar radii
    'u1':        ['uni', -3., 3],     # limb darkening
    'u2':        ['uni', -3., 3.],     # limb darkening
    # 't0':        ['uni', t_0+0.9,  t_0+1.1], # days
    # 'a':         ['uni', 10.,  50.],        # stellar radii
    # 'a':         ['gauss', 41, 20.],         # stellar radii
    # 'inc':       ['uni', 80.,  90.],         # degrees maybe convert this to b? b=cos(i) * a/R*
    # 'c':         ['uni', 0.9,  1.1],   # factor allowing vertical offset (TODO: I don't think they allow for this, but they probably should, also slope below)
    # 'v':         ['gauss', 0, 1e3],    # allow for slope !=0 in time (dy/dt), needs a gaussian prior to converge consistently
    }


# MCMC parameters
# TODO: just a working selection right now, could probably be improved
mcmc_params = {
    'ndim'        :len(param_priors),
    'nwalkers'    :4*len(param_priors),
    'nsteps'      :100000,
    'burn_in_frac':0.6,
}

#%%
################################################################################
######################### Run Quadratic Limb-Drkening ##########################
################################################################################
param_names = ['ps', 'u1', 'u2']

for key in all_errors_dict:
    # iterate through all available error envelopes for the default quadratic parameterization
    #TODO: maybe nicer way to pass param_names
    posterior_samples, unflattened_samples = run_mcmc(time_data, flux_data, all_errors_dict[key], model,
                                params, param_priors, mcmc_params, param_names,
                                transform=False)

    # Plot the corner plot
    create_corner_plot(posterior_samples, truths, all_errors_dict[key][0]*1e6, transform=False)

    model_name = f"%.0fppm_quadratic_model" % (all_errors_dict[key][0]*1e6)

    # Use unflattened samples to check convergence
    gr_stat = check_convergence(unflattened_samples, model_name)

    # Check if Gelman-Rubin statistic is below convergence threshold
    if gr_stat.max() < 1.1:
        print("Chains are well-mixed.")
    else:
        print("Chains may not have converged. Check diagnostics.")
#%%
################################################################################
######################### Run Kipping Limb-Drkening ############################
################################################################################

# update priors and ground truth to Kipping
param_priors = {
    # TODO: adapt these depending on simdata
    'ps':        ['uni', 0., 0.5],      # stellar radii
    'u1':        ['uni', 0., 1.],     # limb darkening
    'u2':        ['uni', 0., 1.],     # limb darkening
}
truths = {
    'ps':0.1,                        # planet-to-star radius ratio = planet radius (in units of stellar radii)
    'u':[0, None]                    # limb-darkening coefficients: q1, q2 (no limb-darkening = [0, ?])
    # TODO: q2 is not actually well defined... need to calculate the limes of 0.5 * u1 / (u1 + u2) for u1 & u2 -> 0, i guess it's 0.25?
    # 
}

for key in all_errors_dict:
    param_names = ['ps', 'q1', 'q2']
    # iterate through all available error envelopes for the kipping parameterization
    #TODO: maybe nicer way to pass param_names
    posterior_samples, unflattened_samples = run_mcmc(time_data, flux_data, all_errors_dict[key], model,
                                params, param_priors, mcmc_params, param_names,
                                transform=True)

    # Plot the corner plot
    create_corner_plot(posterior_samples, truths, all_errors_dict[key][0]*1e6, transform=True)
    
    model_name = f"%.0fppm_kipping_model" % (all_errors_dict[key][0]*1e6)

    # Use unflattened samples to check convergence
    gr_stat = check_convergence(unflattened_samples, model_name)


    if gr_stat.max() < 1.1:
        print("Chains are well-mixed.")
    else:
        print("Chains may not have converged. Check diagnostics.")
#%%

# TODO: Implement further statistics & plots, e.g. gelman-rubin, autocorrelation, etc. to check for convergence
# => DONE!


#%%

