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

def run_full_routine_NO_err_dic(truths, model_params, model, priors, mcmc, 
                     time_data, flux_data, err_data, 
                     transform=False, save=None):
    
    # iterate through all available error envelopes for the default quadratic parameterization
    if save == None:
        posterior_samples, unflattened_samples = run_mcmc(time_data, flux_data, err_data, model,
                                                            model_params, priors, mcmc,
                                                            transform=transform)
    else:
        posterior_samples, unflattened_samples = run_mcmc(time_data, flux_data, err_data, model,
                                                            model_params, priors, mcmc,
                                                            transform=transform, save=save)

    # Plot the corner plot
    create_corner_plot_NO_err_dic(posterior_samples, truths, transform=transform)

    # if transform:
    #     model_name = f"%.0fppm_kipping_model" % (err_data[0]*1e6)
    # else:
    #     model_name = f"%.0fppm_quadratic_model" % (err_data[0]*1e6)

    # # Use unflattened samples to check convergence
    # gr_stat = check_convergence(unflattened_samples, model_name, truths.keys())

    # # Check if Gelman-Rubin statistic is below convergence threshold
    # if gr_stat.max() < 1.1:
    #     print("Chains are well-mixed.")
    # else:
    #     print("Chains may not have converged. Check diagnostics.")

    return posterior_samples
    
#%%
# Load the preprocessed data

FINAL_time_data = np.load("outputs/data_TESS/FINAL_time_data.npy")
FINAL_flux_data = np.load("outputs/data_TESS/FINAL_flux_data.npy")
FINAL_flux_err_data = np.load("outputs/data_TESS/FINAL_flux_err_data.npy")

#%%
# plot the data 
"""
remember: Data is 

    - normalized (time is set to 0 at the first data point, and flux is normalized by min+0.01 &  max of flux data)
    - normalized again by the Gaussian Process (solar flux on average approx. 1)
    - specifically cleaned up for the critical time around 13 days
    - gaps are still present which is ok, no need to worry. 
"""

fig, ax = plot_single_light_curve_with_zoom(FINAL_time_data, FINAL_flux_data, FINAL_flux_err_data, plt_size=(20, 10),zoom_range=(1.4, 1.65))



#%%


# choose only the first TRANSIT to test the MCMC: 

# time_data = FINAL_time_data[900:1300]
# flux_data = FINAL_flux_data[900:1300]
# flux_err_data = FINAL_flux_err_data[900:1300]

time_data = FINAL_time_data
flux_data = FINAL_flux_data
flux_err_data = FINAL_flux_err_data

# Plot the results
plt.figure(figsize=(15, 6))
plt.plot(time_data, flux_data, label="TESS HD-209458b flux data", color="blue")
plt.axvline(x = 1.518, color = 'black', linestyle='--', label = 'approx. middle of transit')
plt.ylabel("Flux")
plt.title("First Transit of TESS HD-209458b")
plt.legend()
plt.show()

#%%

# Check if there are any NaN values in the data
if np.isnan(time_data).any() or np.isnan(flux_data).any() or np.isnan(flux_err_data).any():
    print("Error: NaN entries still present in the data.")
    print("Removing NaN entries and discard the corresponding data points...")
    print("Length of data before: ", len(time_data))

    time_data, flux_data, flux_err_data = fix_nans(time_data, flux_data, flux_err_data)
    print("Length of data after: ", len(time_data))

else:
    print("No NaN entries present in the data.")
    print("Length of data: ", len(time_data))

# # #check if the data is equally spaced
# # diffs = np.diff(time_data)

# # # Check if the differences are constant
# # is_equally_spaced = np.allclose(diffs, diffs[0])  # Allow for small numerical errors
# # print("Equally spaced:" if is_equally_spaced else "Not equally spaced")

# Plot the results
plt.figure(figsize=(15, 6))
plt.plot(time_data, flux_data, label="TESS HD-209458b flux data", color="blue")
plt.axvline(x = 1.518, color = 'black', linestyle='--', label = 'approx. middle of transit')
plt.ylabel("Flux")
plt.title("First Transit of TESS HD-209458b")
plt.legend()
plt.show()

#%%

################### Step 1 - Initialize Model Parameters #######################
# Define & initialize the parameters for the light curve simulation. Based on the parameters of the paper

# TRUE VALUES (those are the parameters we want to estimate with MCMC)
# most references are from https://exoplanets.nasa.gov/exoplanet_watch_results/hd_209458_b/?utm_source=chatgpt.com

truths = {
    'ps':0.1268,                     # Ref: ExoFOP-TESS TOI
    'u1':0.27741,                          # NO REF (assuming 0 is wrong!)
    'u2':0.23677,                          # NO REF (assuming 0 is wrong!)
    'a':8.81,                        # semi-major axis in stellar radii:        REF: Stassun et al. 2017
    'period':3.52474955,             # orbital period (in days):                REF: Kokori et al. 2023
}

# FIXED VALUES (those are the parameters we assume to be known)
fixed_params = {
    # orbital parameters:
    't0':1.518,                      # checked on plot
    'ecc':0.01,                      # Ref: Rosenthal et al. 2021
    'inc':86.71,                     # orbital inclination in degrees           REF: Kokori et al. 2023
    'omega':0,                       # longitude of periastron (in degrees)     REF: Rosenthal et al. 2021
    'limb_dark_model':"quadratic",   # limb-darkening model                     REF: https://arxiv.org/pdf/0802.3764
    # observation parameters from dataset:
    'n_points':len(time_data),       # number of points in the light curve
    't_min':min(time_data),                   # minimum time in days
    't_max':max(time_data),                    # maximum time in days
}

# Initialize the parameters accordingly, to match the "syntax" of the batman package
params, t_array = initialize_parameters(truths, fixed_params)

#%%
assert t_array.all() == time_data.all(), "Time array and data must be identical."

#%%
################ Step 2 - Create Model & Simulate Light Curve ##################

# Initialize the batman model using the parameters from above
model, time_sim, flux_sim = initialize_model(params, t_array)    #initializes model for the simulation

print(flux_sim[50])
# # Define the error envelopes for the light curve ranging from 1-1000 ppm & store them in a dictionary. 
# all_errors_dict = {
#     "1 ppm":    (1    / 1e6) * flux_data,       # error envelope for 1 ppm
#     "10 ppm":   (10   / 1e6) * flux_data,       # error envelope for 10 ppm
#     "30 ppm":   (30   / 1e6) * flux_data,       # error envelope for 30 ppm
#     "100 ppm":  (100  / 1e6) * flux_data,       # error envelope for 100 ppm,
#     "300 ppm":  (300  / 1e6) * flux_data,       # error envelope for 300 ppm, 
#     "1000 ppm": (1000 / 1e6) * flux_data,       # error envelope for 1000 ppm
#     }  

# save_simdata(time_data, flux_data, all_errors_dict, 'no_ld')
# the reload below is unnecessary, but here to illustrate how to load the data if necessary later on:
# time_data, flux_data, all_errors_dict = load_simdata_all_errs('no_ld')

#%%
# Just for overview:
model = model               # initialized batman model
flux_sim = flux_sim         # simulated flux data array (NOT IMPORTANT)

time_data = time_data       # time data array
flux_data = flux_data       # flux data array
err_data = flux_err_data    # error data array

# #%%
#################### Step 3 - Plot & Save Light Curve #######################

# Plot the results
plt.figure(figsize=(15, 6))
plt.plot(time_data[1000:1200], flux_data[1000:1200],'.', label="TESS HD-209458b flux data", color="blue")
plt.plot(time_sim[1000:1200], flux_sim[1000:1200], label="Simulated Light Curve", color="black", linestyle="--")
plt.ylabel("Flux")
plt.title("First Transit of TESS HD-209458b")
plt.legend()
plt.show()


# # Plot the simulated light curve. This is the light curve we assume to be the true data. The error envelopes represent our "observed" data
# fig, ax = plot_single_light_curve(flux_data, time_data, all_errors_dict, plt_size=(15, 8))

# # Dynamically generate the filename with all truths parameters
# output_plot_dir = pathlib.Path("outputs/plots")
# truths_str = get_name_str(truths=truths)
# light_curve_plot_name = output_plot_dir / f"light_curve_plot_sim_{truths_str}.png"

# # Save the light curve plot
# if not os.path.exists(light_curve_plot_name):
#     fig.savefig(light_curve_plot_name, dpi=300)


#%%
################################################################################
######################### Run Quadratic Limb-Drkening ##########################
################################################################################

# choose Priors
# convention:
# [uniform, lower bound, upper bound]
# [gauss, mean, sigma]
param_priors = {
    'ps':        ['uni', 0.1, 0.15],       # stellar radii
    'u1':        ['uni', -3., 3],        # limb darkening
    'u2':        ['uni', -3., 3.],       # limb darkening
    'a':         ['uni', 8.0, 10],        # semi-major axis in stellar radii
    'period':    ['uni', 3.45, 3.6 ],     # orbital period
    # 'ecc':       ['uni', 0., 0.1],     # eccentricity
    # 't0':        ['uni', 0.5, 2.0],    # time of inferior conjunction
}

# MCMC parameters
mcmc_params = {
    'ndim'        :len(param_priors),
    'nwalkers'    :4*len(param_priors),
    # 'nwalkers'    :10,
    'nsteps'      :150000,
    'burn_in_frac':0.5,
}
#%%
# iterate over all errors, creating output plots, may take a while!
# if you want to save the file, additionally pass the argument save=get_name_str(truths)
flattened_samples = run_full_routine_NO_err_dic(truths, params, model, param_priors, mcmc_params, 
                 time_data, flux_data, err_data, 
                 transform=False)#, save=get_name_str(truths))
#%%
sample_name = "outputs/mcmc_TESS_samples/samples_quadratic_%s_%s_%s_%s.npy" % (mcmc_params['nsteps'], mcmc_params['burn_in_frac'], mcmc_params['nwalkers'], mcmc_params['ndim'])
np.save(sample_name, flattened_samples)

#%%
sample_name = "outputs/mcmc_TESS_samples/samples_quadratic_%s_%s_%s_%s.npy" % (mcmc_params['nsteps'], mcmc_params['burn_in_frac'], mcmc_params['nwalkers'], mcmc_params['ndim'])
flat_samples = np.load(sample_name)
# Plot the corner plot
create_corner_plot_NO_err_dic(flat_samples, truths, transform=False)


#%%

# #%%
# ################################################################################
# ######################### Run Kipping Limb-Drkening ############################
# ################################################################################

# # update priors and ground truth to Kipping
# param_priors = {
#     'ps':        ['uni', 0., 0.5],    # stellar radii
#     'u1':        ['uni', 0., 1.],     # limb darkening
#     'u2':        ['uni', 0., 1.],     # limb darkening
# }

# truths = {
#     'ps':0.1,                        # planet-to-star radius ratio = planet radius (in units of stellar radii)
#     'u1':0,                          # limb-darkening coefficients: q1 
#     'u2':None,                       # limb-darkening coefficients: q2
#     # TODO: q2 is not actually well defined... need to calculate the limes of 0.5 * u1 / (u1 + u2) for u1 & u2 -> 0
#     # but it diverges (gives different results depending which param you let go to 0 first)
# }

# # iterate over all errors, creating output plots, may take a while!
# # if you want to save the file, additionally pass the argument save=get_name_str(truths)
# run_full_routine(truths, params, model, param_priors, mcmc_params, 
#                  time_data, flux_data, all_errors_dict, 
#                  transform=True)#, save=get_name_str(truths))

#%%




















###### PLOT THE RESULTS NICELY ########

best_fit = {
    'ps':0.12763,                     # Ref: ExoFOP-TESS TOI
    'u1':0.3771,                          # NO REF (assuming 0 is wrong!)
    'u2':0.05549,                          # NO REF (assuming 0 is wrong!)
    'a':8.8493,                        # semi-major axis in stellar radii:        REF: Stassun et al. 2017
    'period':3.52477,             # orbital period (in days):                REF: Kokori et al. 2023
}

fixed_params = {
    # orbital parameters:
    't0':1.518,                      # checked on plot
    'ecc':0.01,                      # Ref: Rosenthal et al. 2021
    'inc':86.71,                     # orbital inclination in degrees           REF: Kokori et al. 2023
    'omega':0,                       # longitude of periastron (in degrees)     REF: Rosenthal et al. 2021
    'limb_dark_model':"quadratic",   # limb-darkening model                     REF: https://arxiv.org/pdf/0802.3764
    # observation parameters from dataset:
    'n_points':len(time_data),       # number of points in the light curve
    't_min':min(time_data),                   # minimum time in days
    't_max':max(time_data),                    # maximum time in days
}

# Initialize the parameters accordingly, to match the "syntax" of the batman package
params_best_fit, t_array = initialize_parameters(best_fit, fixed_params)


#%%
################ Step 2 - Create Model & Simulate Light Curve ##################

# Initialize the batman model using the parameters from above
model, time_sim, flux_best_fit = initialize_model(params_best_fit, t_array)    #initializes model for the simulation


# Plot the results
plt.figure(figsize=(15, 5))
plt.plot(time_sim[1000:1190], flux_best_fit[1000:1190], label="Best Fit (MCMC)", color="mediumseagreen", linestyle="-", linewidth=2.5)
plt.plot(time_data[1000:1190], flux_data[1000:1190],'.', label="TESS flux data", color="black", markersize=8)
plt.ylabel("Relative Flux", fontsize=17)
plt.xlabel("Time [days]", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.title("First Transit of TESS Dataset", fontsize=18)
plt.legend(loc='best', fontsize=17)
plt.show()
# %%

