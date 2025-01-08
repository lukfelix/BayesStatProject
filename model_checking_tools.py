"""
This file is still WIP.
Contains extra functions to make developement easier.
"""

# Import necessary libraries
import numpy as np
import scipy
import matplotlib.pyplot as plt

from model_functions import full_model
from mcmc_functions import log_posterior, log_likelihood
from simulation_functions import load_single_simdata, initialize_parameters, initialize_model

###################################
#extra stuff
###################################

#load sim data
#t, y, yerr = load_single_simdata('simdata_no_ld_10_ppm.npy')

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

# =============================================================================
# theta_init = np.array([0.1, 0.1, 0.1])
# transform=False
# 
# MAP_result = scipy.optimize.minimize(
#         fun=negative_log_posterior,
#         x0=theta_init,
#         args=(t, y, yerr, params, model, priors, transform)
#         )
# 
# theta_MAP = MAP_result.x
# 
# print("MAP results")
# print(theta_MAP)
# 
# y_MAP = full_model(theta_MAP, params, model)
# =============================================================================

#plt.plot(t, y)
#plt.plot(t, y_MAP)
#plt.show()



###################################


#Implementation of the chi squared statistic
def chi_squared(y, sigma_y, mu):
    return np.sum((y - mu)**2/sigma_y**2)

# =============================================================================
# chi_squared_quadratic = chi_squared(y, yerr, y_MAP)
# 
# n_data = len(y)
# n_param_quad = 3
# PTE_quad = scipy.stats.chi2(df=n_data - n_param_quad).sf(chi_squared_quadratic)
# 
# print(f"Quadratic: χ²={chi_squared_quadratic:.1f}, "
#       f"ndof={n_data}-{n_param_quad}, PTE={PTE_quad:.3f}")
# =============================================================================



#implementic DIC
def DIC(theta_star, theta_samples, log_likelihood):
    # Compute log likelihood at theta_star and the samples theta_i
    log_likelihood_star = log_likelihood(theta_star)
    log_likelihood_samples = np.array(
        [log_likelihood(theta) for theta in theta_samples]
    )
    p_D = 2*(log_likelihood_star - np.mean(log_likelihood_samples))
    p_V = 2*np.var(log_likelihood_samples)
    return -2*(log_likelihood_star - p_D), p_D, p_V






print('Loading Sim Data ...')
t1, y1, yerr1 = load_single_simdata('simdata_no_ld_1_ppm.npy')
t10, y10, yerr10 = load_single_simdata('simdata_no_ld_10_ppm.npy')
t30, y30, yerr30 = load_single_simdata('simdata_no_ld_30_ppm.npy')
t100, y100, yerr100 = load_single_simdata('simdata_no_ld_100_ppm.npy')
t300, y300, yerr300 = load_single_simdata('simdata_no_ld_300_ppm.npy')
t1000, y1000, yerr1000 = load_single_simdata('simdata_no_ld_1000_ppm.npy')


print('Loading Samples from Quadratic Runs ...')
quad_1ppm = np.load('outputs/samples/samples_ps_0.1_u1_0_u2_0_1 ppm_100000steps.npy')
quad_10ppm = np.load('outputs/samples/samples_ps_0.1_u1_0_u2_0_10 ppm_100000steps.npy')
quad_30ppm = np.load('outputs/samples/samples_ps_0.1_u1_0_u2_0_30 ppm_100000steps.npy')
quad_100ppm = np.load('outputs/samples/samples_ps_0.1_u1_0_u2_0_100 ppm_100000steps.npy')
quad_300ppm = np.load('outputs/samples/samples_ps_0.1_u1_0_u2_0_300 ppm_100000steps.npy')
quad_1000ppm = np.load('outputs/samples/samples_ps_0.1_u1_0_u2_0_1000 ppm_100000steps.npy')

print('Loading Samples from Kipping Runs ...')
kipping_1ppm = np.load('outputs/samples/samples_ps_0.1_u1_0_u2_None_1 ppm_100000steps.npy')
kipping_10ppm = np.load('outputs/samples/samples_ps_0.1_u1_0_u2_None_10 ppm_100000steps.npy')
kipping_30ppm = np.load('outputs/samples/samples_ps_0.1_u1_0_u2_None_30 ppm_100000steps.npy')
kipping_100ppm = np.load('outputs/samples/samples_ps_0.1_u1_0_u2_None_100 ppm_100000steps.npy')
kipping_300ppm = np.load('outputs/samples/samples_ps_0.1_u1_0_u2_None_300 ppm_100000steps.npy')
kipping_1000ppm = np.load('outputs/samples/samples_ps_0.1_u1_0_u2_None_1000 ppm_100000steps.npy')



# =============================================================================
# import warnings
# warnings.filterwarnings("ignore")
# 
# print('Calculating quadratic MAPs ...')
# theta_init_quad = np.array([0.1, 0.1, 0.1])
# priors_quad = {
#     'ps':        ['uni', 0., 0.5],     # stellar radii
#     'u1':        ['uni', -3., 3],      # limb darkening
#     'u2':        ['uni', -3., 3.],     # limb darkening
# }
# 
# 
# MAP_result_quad1 = scipy.optimize.minimize(
#         fun=negative_log_posterior,
#         x0=theta_init_quad,
#         args=(t1, y1, yerr1, params, model, priors_quad, False)
#         )
# theta_MAP_quad1 = MAP_result_quad1.x
# print('MAP quadratic 1: ', theta_MAP_quad1)
# 
# MAP_result_quad10 = scipy.optimize.minimize(
#         fun=negative_log_posterior,
#         x0=theta_init_quad,
#         args=(t10, y10, yerr10, params, model, priors_quad, False)
#         )
# theta_MAP_quad10 = MAP_result_quad10.x
# print('MAP quadratic 10: ', theta_MAP_quad10)
# 
# MAP_result_quad30 = scipy.optimize.minimize(
#         fun=negative_log_posterior,
#         x0=theta_init_quad,
#         args=(t30, y30, yerr30, params, model, priors_quad, False)
#         )
# theta_MAP_quad30 = MAP_result_quad30.x
# print('MAP quadratic 30: ', theta_MAP_quad30)
# 
# MAP_result_quad100 = scipy.optimize.minimize(
#         fun=negative_log_posterior,
#         x0=theta_init_quad,
#         args=(t100, y100, yerr100, params, model, priors_quad, False)
#         )
# theta_MAP_quad100 = MAP_result_quad100.x
# print('MAP quadratic 100: ', theta_MAP_quad100)
# 
# MAP_result_quad300 = scipy.optimize.minimize(
#         fun=negative_log_posterior,
#         x0=theta_init_quad,
#         args=(t300, y300, yerr300, params, model, priors_quad, False)
#         )
# theta_MAP_quad300 = MAP_result_quad300.x
# print('MAP quadratic 300: ', theta_MAP_quad300)
# 
# MAP_result_quad1000 = scipy.optimize.minimize(
#         fun=negative_log_posterior,
#         x0=theta_init_quad,
#         args=(t1000, y1000, yerr1000, params, model, priors_quad, False)
#         )
# theta_MAP_quad1000 = MAP_result_quad1000.x
# print('MAP quadratic 1000: ', theta_MAP_quad1000)
# 
# 
# 
# print('Calculating Kipping MAPs ...')
# theta_init_kipping = np.array([0.1, 0.0, 0.3])
# priors_kipping = {
#     'ps':        ['uni', 0., 0.5],    # stellar radii
#     'u1':        ['uni', 0., 1.],     # limb darkening
#     'u2':        ['uni', 0., 1.],     # limb darkening
# }
# 
# MAP_result_kipping1 = scipy.optimize.minimize(
#         fun=negative_log_posterior,
#         x0=theta_init_kipping,
#         args=(t1, y1, yerr1, params, model, priors_kipping, True)
#         )
# theta_MAP_kipping1 = MAP_result_kipping1.x
# print('MAP Kipping 1: ', theta_MAP_kipping1)
# 
# MAP_result_kipping10 = scipy.optimize.minimize(
#         fun=negative_log_posterior,
#         x0=theta_init_kipping,
#         args=(t10, y10, yerr10, params, model, priors_kipping, True)
#         )
# theta_MAP_kipping10 = MAP_result_kipping10.x
# print('MAP Kipping 10: ', theta_MAP_kipping10)
# 
# MAP_result_kipping30 = scipy.optimize.minimize(
#         fun=negative_log_posterior,
#         x0=theta_init_kipping,
#         args=(t30, y30, yerr30, params, model, priors_kipping, True)
#         )
# theta_MAP_kipping30 = MAP_result_kipping30.x
# print('MAP Kipping 30: ', theta_MAP_kipping30)
# 
# MAP_result_kipping100 = scipy.optimize.minimize(
#         fun=negative_log_posterior,
#         x0=theta_init_kipping,
#         args=(t100, y100, yerr100, params, model, priors_kipping, True)
#         )
# theta_MAP_kipping100 = MAP_result_kipping100.x
# print('MAP Kipping 100: ', theta_MAP_kipping100)
# 
# MAP_result_kipping300 = scipy.optimize.minimize(
#         fun=negative_log_posterior,
#         x0=theta_init_kipping,
#         args=(t300, y300, yerr300, params, model, priors_kipping, True)
#         )
# theta_MAP_kipping300 = MAP_result_kipping300.x
# print('MAP Kipping 300: ', theta_MAP_kipping300)
# 
# MAP_result_kipping1000 = scipy.optimize.minimize(
#         fun=negative_log_posterior,
#         x0=theta_init_kipping,
#         args=(t1000, y1000, yerr1000, params, model, priors_kipping, True)
#         )
# theta_MAP_kipping1000 = MAP_result_kipping1000.x
# print('MAP Kipping 1000: ', theta_MAP_kipping1000)
# 
# 
# 
# 
# #TODO check log_likelihood inputs
# print('Calculating quadratic DICs ...')
# DIC_quad1, p_D_quad1, p_V_quad1 = DIC(
#     theta_star=theta_MAP_quad1,
#     theta_samples=quad_1ppm,
#     log_likelihood=lambda theta: log_likelihood(theta, t1, y1, yerr1, params, model, False)
# )
# 
# DIC_quad10, p_D_quad10, p_V_quad10 = DIC(
#     theta_star=theta_MAP_quad10,
#     theta_samples=quad_10ppm,
#     log_likelihood=lambda theta: log_likelihood(theta, t10, y10, yerr10, params, model, False)
# )
# 
# DIC_quad30, p_D_quad30, p_V_quad30 = DIC(
#     theta_star=theta_MAP_quad30,
#     theta_samples=quad_30ppm,
#     log_likelihood=lambda theta: log_likelihood(theta, t30, y30, yerr30, params, model, False)
# )
# 
# DIC_quad100, p_D_quad100, p_V_quad100 = DIC(
#     theta_star=theta_MAP_quad100,
#     theta_samples=quad_100ppm,
#     log_likelihood=lambda theta: log_likelihood(theta, t100, y100, yerr100, params, model, False)
# )
# 
# DIC_quad300, p_D_quad300, p_V_quad300 = DIC(
#     theta_star=theta_MAP_quad300,
#     theta_samples=quad_300ppm,
#     log_likelihood=lambda theta: log_likelihood(theta, t300, y300, yerr300, params, model, False)
# )
# 
# DIC_quad1000, p_D_quad1000, p_V_quad1000 = DIC(
#     theta_star=theta_MAP_quad1000,
#     theta_samples=quad_1000ppm,
#     log_likelihood=lambda theta: log_likelihood(theta, t1000, y1000, yerr1000, params, model, False)
# )
# 
# 
# 
# print('Calculating Kipping DICs ...')
# DIC_kipping1, p_D_kipping1, p_V_kipping1 = DIC(
#     theta_star=theta_MAP_kipping1,
#     theta_samples=kipping_1ppm,
#     log_likelihood=lambda theta: log_likelihood(theta, t1, y1, yerr1, params, model, True)
# )
# 
# DIC_kipping10, p_D_kipping10, p_V_kipping10 = DIC(
#     theta_star=theta_MAP_kipping10,
#     theta_samples=kipping_10ppm,
#     log_likelihood=lambda theta: log_likelihood(theta, t10, y10, yerr10, params, model, True)
# )
# 
# DIC_kipping30, p_D_kipping30, p_V_kipping30 = DIC(
#     theta_star=theta_MAP_kipping30,
#     theta_samples=kipping_30ppm,
#     log_likelihood=lambda theta: log_likelihood(theta, t30, y30, yerr30, params, model, True)
# )
# 
# DIC_kipping100, p_D_kipping100, p_V_kipping100 = DIC(
#     theta_star=theta_MAP_kipping100,
#     theta_samples=kipping_100ppm,
#     log_likelihood=lambda theta: log_likelihood(theta, t100, y100, yerr100, params, model, True)
# )
# 
# DIC_kipping300, p_D_kipping300, p_V_kipping300 = DIC(
#     theta_star=theta_MAP_kipping300,
#     theta_samples=kipping_300ppm,
#     log_likelihood=lambda theta: log_likelihood(theta, t300, y300, yerr300, params, model, True)
# )
# 
# DIC_kipping1000, p_D_kipping1000, p_V_kipping1000 = DIC(
#     theta_star=theta_MAP_kipping1000,
#     theta_samples=kipping_1000ppm,
#     log_likelihood=lambda theta: log_likelihood(theta, t1000, y1000, yerr1000, params, model, True)
# )
# 
# 
# 
# print('Printing Results ...')
# 
# print(f"Quadratic 1 ppm: DIC = {DIC_quad1:.3f}, p_D = {p_D_quad1:.1f}, p_V = {p_V_quad1:.1f}")
# print(f"Kipping 1 ppm: DIC = {DIC_kipping1:.3f}, p_D = {p_D_kipping1:.1f}, p_V = {p_V_kipping1:.1f}")
# print('\n')
# print(f"Quadratic 10 ppm: DIC = {DIC_quad10:.3f}, p_D = {p_D_quad10:.1f}, p_V = {p_V_quad10:.1f}")
# print(f"Kipping 10 ppm: DIC = {DIC_kipping10:.3f}, p_D = {p_D_kipping10:.1f}, p_V = {p_V_kipping10:.1f}")
# print('\n')
# print(f"Quadratic 30 ppm: DIC = {DIC_quad30:.3f}, p_D = {p_D_quad30:.1f}, p_V = {p_V_quad30:.1f}")
# print(f"Kipping 30 ppm: DIC = {DIC_kipping30:.3f}, p_D = {p_D_kipping30:.1f}, p_V = {p_V_kipping30:.1f}")
# print('\n')
# print(f"Quadratic 100 ppm: DIC = {DIC_quad100:.3f}, p_D = {p_D_quad100:.1f}, p_V = {p_V_quad100:.1f}")
# print(f"Kipping 100 ppm: DIC = {DIC_kipping100:.3f}, p_D = {p_D_kipping100:.1f}, p_V = {p_V_kipping100:.1f}")
# print('\n')
# print(f"Quadratic 300 ppm: DIC = {DIC_quad300:.3f}, p_D = {p_D_quad300:.1f}, p_V = {p_V_quad300:.1f}")
# print(f"Kipping 300 ppm: DIC = {DIC_kipping300:.3f}, p_D = {p_D_kipping300:.1f}, p_V = {p_V_kipping300:.1f}")
# print('\n')
# print(f"Quadratic 1000 ppm: DIC = {DIC_quad1000:.3f}, p_D = {p_D_quad1000:.1f}, p_V = {p_V_quad1000:.1f}")
# print(f"Kipping 1000 ppm: DIC = {DIC_kipping1000:.3f}, p_D = {p_D_kipping1000:.1f}, p_V = {p_V_kipping1000:.1f}")
# 
# =============================================================================




#implementing WAIC
def pointwise_log_likelihood(theta, t, y, yerr, params, model, transform, index):
    model_prediction = full_model(theta, params, model, transform)[index]
    return -0.5 * np.sum(((y - model_prediction) / yerr) ** 2)

import tqdm
def WAIC(theta_samples, pointwise_log_likelihood, y_partitions, x_partitions, y_err_partitons, params, model, transform):
    # Compute the log likelihood for each partition separately
    pointwise_log_likelihood_samples = np.array(
        [[pointwise_log_likelihood(theta, x_partitions[i], y_partitions[i], y_err_partitons[i], params, model, transform, i)
            for i in range(len(y_partitions))] 
         for theta in tqdm.tqdm(theta_samples)]
    )

    # Compute the lppd and p_waic for each partition
    #print('\n')
    #print(pointwise_log_likelihood_samples)
    lppd = np.log(np.mean(np.exp(pointwise_log_likelihood_samples), axis=0))
    p_waic = np.var(pointwise_log_likelihood_samples, axis=0)
    # Check if the any of the terms in p_waic are too large, which indicates
    # a problem
    if np.any(p_waic > 0.4):
        print(f"Warning: Var[log p(y_i|theta)] > 0.4 for data points "
              f"{np.argwhere(p_waic > 0.4)}. p_WAIC unreliable!")
    # Sum up the partitions
    lppd = np.sum(lppd)
    p_waic = np.sum(p_waic)

    return -2*(lppd - p_waic), p_waic, pointwise_log_likelihood_samples


# =============================================================================
# WAIC_quad, p_WAIC_quad, pointwise_log_likelihood_samples_quad = WAIC(
#     theta_samples=quad_100ppm,
#     log_likelihood=log_likelihood,
#     y_partitions=y100,
#     x_partitions=t100,
#     y_err_partitons=yerr100,
#     params=params,
#     model=model,
#     transform=False
# )
# print(f"Quadratic: WAIC = {WAIC_quad:.1f}, p_WAIC = {p_WAIC_quad:.1f}")
# =============================================================================
#this runs extremely slowly

#try WAIC & LOOC with arviz

import arviz

#shape = (480000, 3)

#make the number of samples smaller to make calculating faster
number_of_samples = 1000
index = np.random.choice(quad_100ppm.shape[0], number_of_samples, replace=False)


quad_100ppm_short = quad_100ppm[index]
kipping_100ppm_short = kipping_100ppm[index]

WAIC_quad, p_WAIC_quad, pointwise_log_likelihood_samples_quad = WAIC(
    theta_samples=quad_100ppm_short,
    pointwise_log_likelihood=pointwise_log_likelihood,
    y_partitions=y100,
    x_partitions=t100,
    y_err_partitons=yerr100,
    params=params,
    model=model,
    transform=False
)
WAIC_kipping, p_WAIC_kipping, pointwise_log_likelihood_samples_kipping = WAIC(
    theta_samples=kipping_100ppm_short,
    pointwise_log_likelihood=pointwise_log_likelihood,
    y_partitions=y100,
    x_partitions=t100,
    y_err_partitons=yerr100,
    params=params,
    model=model,
    transform=True
)
print('\n')
print(f"Quadratic: WAIC = {WAIC_quad:.1f}, p_WAIC = {p_WAIC_quad:.1f}")
print(f"Kipping: WAIC = {WAIC_kipping:.1f}, p_WAIC = {p_WAIC_kipping:.1f}")

inference_data_kipping = arviz.from_dict(
    posterior={"kipping": kipping_100ppm_short[None, ...]},
    log_likelihood={"kipping": pointwise_log_likelihood_samples_kipping[None, ...]},
)

inference_data_quad = arviz.from_dict(
    posterior={"quad": quad_100ppm_short[None, ...]},
    log_likelihood={"quad": pointwise_log_likelihood_samples_quad[None, ...]},
)
printer = arviz.compare({"kipping": inference_data_kipping, "quad": inference_data_quad}, ic="loo", scale="deviance")










