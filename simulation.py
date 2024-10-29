import numpy as np
import batman
import matplotlib.pyplot as plt

# Initializes the parameters for the transit simulation with the batman package
def initialize_parameters(t0, period, P_S_ratio, a, inc, e, omega, u, limb_dark_model, n_points, t_min, t_max):
    
    # Define parameters for the transit simulation
    params = batman.TransitParams()       
    params.t0 = t0                          # time of inferior conjunction
    params.per = period                     # orbital period (in days)
    params.rp = P_S_ratio                   # planet-to-star radius ratio
    params.a = a                            # semi-major axis in stellar radii
    params.inc = inc                        # orbital inclination in degrees
    params.ecc = e                          # eccentricity
    params.w = omega                        # longitude of periastron (in degrees)
    params.u = u                            # limb-darkening coefficients (no limb-darkening)
    params.limb_dark = limb_dark_model      # limb-darkening model

    # Time array for the simulation
    t = np.linspace(t_min, t_max, n_points)    # time from t_min to t_max days, n points

    return params, t

# Create a batman model and simulate the light curve
def simulate_light_curve(params, t):

    m = batman.TransitModel(params, t)    # initialize the model with parameters and time array
    flux = m.light_curve(params)          # calculate the light curve

    return flux

# Plot the simulated light curve
def plot_single_light_curve(t, flux):

    plt.plot(t, flux, label="Simulated Light Curve")
    plt.xlabel("Time from Mid-Transit (days)")
    plt.ylabel("Relative Flux")
    plt.title("Simulated Transit Light Curve")
    plt.legend()
    plt.show() 



