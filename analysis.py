#%%

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import batman
import emcee
import corner
import pathlib
import os

# Import predefined functions from other files
from simulation_functions import *          # functions for simulating the light curve
from model_functions import *               # functions for evaluation of the model
from mcmc_functions import *                # functions used for the MCMC analysis

#%%
################################################################################
#################### SIMULATING TRANSIT LIGHT CURVE(S) #########################
################################################################################


################### Step 1 - Initialize Model Parameters #######################

# Define & initialize the parameters for the light curve simulation. Based on the parameters of the paper
# TODO: Double check if these parameters are correct from the paper

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

# Define the different errors that can be used in the light curve simulation
err_1_ppm = 1                        # 1 ppm value for the error
err_10_ppm = 10                      # 10 ppm value for the error
err_100_ppm = 100                    # 100 ppm value for the error
err_1000_ppm = 1000                  # 1000 ppm value for the error

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
    "1 ppm":    (err_1_ppm    / 1000000) * flux_data,       # error envelope for 1 ppm
    "10 ppm":   (err_10_ppm   / 1000000) * flux_data,       # error envelope for 10 ppm
    "100 ppm":  (err_100_ppm  / 1000000) * flux_data,       # error envelope for 100 ppm, 
    "1000 ppm": (err_1000_ppm / 1000000) * flux_data,       # error envelope for 1000 ppm
    }  

# TODO: implement a way to save the data under "outputs/data". Best to probably use numpy's "np.save" and "np.load" to save, resp. load the data

#%%
##################### Step 3 - Plot & Save Light Curve #######################

# Plot the simulated light curve. This is the light curve we assume to be the true data. The error envelopes represent our "observed" data
fig, ax = plot_single_light_curve(flux_data, time_data, all_errors_dict, plt_size=(15, 8))

# Save the light curve plot
light_curve_plot_name = "outputs/plots/light_curve_plot_sim"
#TODO change naming above to match all the model parameters (not very important)
if not os.path.exists(light_curve_plot_name):
    fig.savefig(light_curve_plot_name, dpi=300)

#%%

# For the remainder, we choose the 1000 ppm error envelope to be the error data
# TODO change code below to allow to do MCMC for all 4 error cases

error_data = all_errors_dict['1000 ppm']
# error_data = all_errors_dict['100 ppm']
# error_data = all_errors_dict['10 ppm']
# error_data = all_errors_dict['1 ppm']

#%%

################################################################################
######################### FITTING DATA TO A MODEL ##############################
################################################################################

# Our goal is to estimate the parameters of the model that best fit the data.
# The parameters we want to esimate are (for the moment): 'ps', 'u1' and 'u2'. 

# 1. ** Define the Model **:                Set up the physical model that you will fit to the data. This involves specifying the transit light curve model with the parameters you wish to estimate. We already did this through the batman model.
# 2. ** Choose Priors **:                   Define prior distributions for each parameter. Priors represent your beliefs about the parameters before seeing the data.
# 3. ** Set Up the Likelihood Function **:  This function evaluates how likely the observed data is, given a set of parameters. 
# 4. ** Initialize the MCMC Sampler **:     Choose an MCMC algorithm (e.g., Metropolis-Hastings, Gibbs sampling, or Hamiltonian Monte Carlo) and set the initial positions of the chains, typically starting from different points to cover the parameter space well.
# 5. ** Run the MCMC Sampler **:            Execute the chains to explore the parameter space. The sampler will generate a sequence of samples for each parameter.
# 6. ** Analyze the Results **:             After the MCMC run, analyze the chains for convergence, discard the burn-in phase, and use the remaining samples to compute the posterior distributions and statistics (mean, median, confidence intervals) for each parameter.

# TODO: Check that everything is correct --> play with the priors a bit, see what happens

######################## Step 1 - Define the model #############################

# DONE (see above), here just as recap:
model = model
params = params

######################### Step 2 - Choose Priors ###############################

# convention:
# [uniform, lower bound, upper bound]
# [gauss, mean, sigma]
param_priors = {
    # TODO: adapt these depending on simdata
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

#################### Step 3 - Set Up the Likelihood Function #####################

# p(D|theta)
def log_likelihood(theta, t, y, yerr):
    model_prediction = full_model(theta, params, model)
    return -0.5 * np.sum(((y - model_prediction) / yerr) ** 2)

# p(theta)
def log_prior(theta):
    """
    evaluate priors for given paramter theta, looks unusual due to dictionary structure of priors above
    """
    lp = 0.
    for val, prior in zip(theta, param_priors):
        # print(val, prior)
        if param_priors[prior][0] == 'uni':
            ret = log_uni(val, param_priors[prior][1], param_priors[prior][2])
        elif param_priors[prior][0]=='gauss':
            ret = log_gauss(val, param_priors[prior][1], param_priors[prior][2])
        else:
            print('Error: prior-distribution "{}" not accepted by current code.'.format(param_priors[prior][0]))
            return None
        if ret == -np.inf:# or ret < 1e-20:
            # print('inf')
            return -np.inf
        else:
            lp += ret
    return lp

# Define the final probability function as likelihood * prior ('+' due to log).
# p(theta|D) = p(D|theta) * p(theta)
def log_posterior(theta, t, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    # print('not inf')
    return lp + log_likelihood(theta, t, y, yerr)

#%%
##################### Step 4 - Initialize the MCMC Sampler #######################

# MCMC parameters
# TODO adapt these to the specific problem
ndim            =   len(param_priors)
nwalkers        =   32
nsteps          =   10000
burn_in_frac    =   0.4

# Initialize the walkers
pos = np.zeros((nwalkers, ndim))
for i, param in enumerate(param_priors):
    if param_priors[param][0]=='uni':
        pos[:, i] = np.random.uniform(param_priors[param][1], param_priors[param][2], nwalkers)
    elif param_priors[param][0]=='gauss':
        pos[:, i] = np.random.normal(param_priors[param][1], param_priors[param][2], nwalkers)
    else:
        print('prior not recognized')
        break

# Initialize the sampler (at the moment time_data is not being used, see "model_functions.py/full_model()" )
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(time_data, flux_data, error_data))

#%%
####################### Step 5 - Run the MCMC Sampler ############################

# Run the sampler
sampler.run_mcmc(pos, nsteps, progress=True)

# Get the samples
samples = sampler.get_chain(discard=int(burn_in_frac * nsteps))     # discard the burn-in phase

# TODO: save samples in some format for more flexible plotting and post-processing

#%%
####################### Step 6 - Analyze the Results ############################

# Flatten the samples (remove the walkers)
flattened_samples = samples.reshape(-1, len(param_priors))  # flatten the samples for plotting

# print(flattened_samples.shape)

#%%
# Plot the corner plot

fig = corner.corner(

    flattened_samples, 
    title_fmt='.5f',
    bins=50,
    show_titles=True,
    labels=[r"$P_S$", r"$u_1$", r"$u_2$"], 
    truths=[truths['ps'], truths['u'][0], truths['u'][1]],
    plot_density=True,
    plot_datapoints=True,
    fill_contours=False,
    smooth=True,
    levels=(0.6827, 0.90, 0.9545),              # shows the 1, 1.5 and 2 sigma contours in the 2D plots
    quantiles=[0.16, 0.5, 0.84],                # shows the 1 sigma interval in the 1D plots
    title_kwargs={"fontsize": 10},
    truth_color='cornflowerblue',

);
#%%
# Save the corner plot
corner_plot_name = "outputs/plots/corner_plot_%.0fparameters_%.0fppm_no_linear_model" % (len(param_priors), err_1000_ppm )

if not os.path.exists(corner_plot_name):
    fig.savefig(corner_plot_name, dpi=300)

#%%

# TODO: Implement further statistics & plots, e.g. galman-rubin, autocorrelation, etc. to check for convergence


#%%




