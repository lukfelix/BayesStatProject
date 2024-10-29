#%%

# Importing the necessary libraries
import numpy as np
import batman
import matplotlib.pyplot as plt

# Importing the necessary functions
from simulation import initialize_parameters, simulate_light_curve, plot_single_light_curve


################################################################################
##################### SIMULATING TRANSIT LIGHT CURVES ##########################
################################################################################

#%%

# Define & initialize the parameters for the light curve simulation
t0 = 0                              # time of inferior conjunction
period = 1                          # orbital period (in days)
P_S_ratio = 0.1                     # planet-to-star radius ratio
a = 4                               # semi-major axis in stellar radii
inc = 90                            # orbital inclination in degrees
e = 0                               # eccentricity
omega = 90                          # longitude of periastron (in degrees)
u = [0, 0]                          # limb-darkening coefficients: u1, u2 (no limb-darkening)
limb_dark_model = "quadratic"       # limb-darkening model

n_points = 1000                     # number of points in the light curve
t_min = -0.25                       # minimum time in days
t_max = 0.25                        # maximum time in days

params, t_array = initialize_parameters(t0, period, P_S_ratio, a, inc, e, omega, u, limb_dark_model, n_points, t_min, t_max)

#%%

# Initialize the batman model & simulate the light curve
flux = simulate_light_curve(params, t_array)

#%%

# Plot a single light curve
plot_single_light_curve(t_array, flux)


################################################################################
######################### FITTING DATA TO A MODEL ##############################
################################################################################

#%%




