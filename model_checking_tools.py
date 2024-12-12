"""
This file is still WIP.
Contains extra functions to make developement easier.
"""

# Import necessary libraries
import numpy as np
import scipy
import matplotlib.pyplot as plt

from model_functions import full_model
from mcmc_functions import log_posterior
from simulation_functions import load_single_simdata, initialize_parameters, initialize_model

###################################
#extra stuff
###################################

#load sim data
t, y, yerr = load_single_simdata('simdata_no_ld_10_ppm.npy')

#prepare model
truths = {
    'ps':0.1,                        # planet-to-star radius ratio = planet radius (in units of stellar radii)
    'u1':0,                          # limb-darkening coefficients: u1, 
    'u2':0,                          # u2 (no limb-darkening: u1=0, u2=0)
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
# Initialize the parameters accordingly, to match the "syntax" of the batman package
params, t_array = initialize_parameters(truths, fixed_params)

# Initialize the batman model using the parameters from above
model, time_data, flux_data = initialize_model(params, t_array)    #initializes model for the simulation

priors = {
    'ps':        ['uni', 0., 0.5],     # stellar radii
    'u1':        ['uni', -3., 3],      # limb darkening
    'u2':        ['uni', -3., 3.],     # limb darkening
}










#calculate MAP
def negative_log_posterior(theta, t, y, yerr, params, model, priors, transform):
        return -log_posterior(theta, t, y, yerr, params, model, priors, transform)

theta_init = np.array([0.1, 0.1, 0.1])
transform=False

MAP_result = scipy.optimize.minimize(
        fun=negative_log_posterior,
        x0=theta_init,
        args=(t, y, yerr, params, model, priors, transform)
        )

theta_MAP = MAP_result.x

print("MAP results")
print(theta_MAP)

y_MAP = full_model(theta_MAP, params, model)

#plt.plot(t, y)
#plt.plot(t, y_MAP)
#plt.show()



###################################


#Implementation of the chi squared statistic
def chi_squared(y, sigma_y, mu):
    return np.sum((y - mu)**2/sigma_y**2)

chi_squared_quadratic = chi_squared(y, yerr, y_MAP)

n_data = len(y)
n_param_quad = 3
PTE_quad = scipy.stats.chi2(df=n_data - n_param_quad).sf(chi_squared_quadratic)

print(f"Quadratic: χ²={chi_squared_quadratic:.1f}, "
      f"ndof={n_data}-{n_param_quad}, PTE={PTE_quad:.3f}")





