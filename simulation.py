import numpy as np
import batman
import matplotlib.pyplot as plt


# Define parameters for the transit simulation
params = batman.TransitParams()       
params.t0 = 0                         # time of inferior conjunction
params.per = 1                        # orbital period (in days)
params.rp = 0.1                       # planet-to-star radius ratio
params.a = 4                          # semi-major axis in stellar radii
params.inc = 90                       # orbital inclination in degrees
params.ecc = 0                        # eccentricity
params.w = 90                         # longitude of periastron (in degrees)
params.u = []                         # limb-darkening coefficients (no limb-darkening)
params.limb_dark = "uniform"          # limb-darkening model


# Generate time array for the simulation
t = np.linspace(-0.25, 0.25, 1000)    # time from -0.25 to 0.25 days, 1000 points


# Create a batman model and simulate the light curve
m = batman.TransitModel(params, t)    # initialize the model with parameters and time array
flux = m.light_curve(params)          # calculate the light curve


# Plot the simulated light curve
plt.plot(t, flux, label="Simulated Light Curve")
plt.xlabel("Time from Mid-Transit (days)")
plt.ylabel("Relative Flux")
plt.title("Simulated Transit Light Curve")
plt.legend()
plt.show() 


