#%%
# Import necessary libraries

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import norm



# Import predefined functions from other files
import simulation_functions as sim
import model_functions as mod_fun

###############################################################################
def light_curve(posterior_samples, initial_params, model, transform):
    posterior_flux = []
    for theta in posterior_samples:
        params = initial_params
        params.rp = theta[0]
        if transform:
            params.u = mod_fun.kipping_to_quad(theta[1],theta[2])
        else:
            params.u = [theta[1],theta[2]]
        
        post_flux = model.light_curve(params)
        posterior_flux.append(post_flux)
    return posterior_flux
###############################################################################
#%%
###############################################################################
def intensity_profile(posterior_samples, n, transform):
    posterior_intensity = []
    mu = np.linspace(0,1,n)
    for post_truth in posterior_samples:
        if transform:
            q1,q2 = post_truth[1],post_truth[2]
            I = 1-np.sqrt(q1)*(2*q2*(1-mu)+(1-q2)*(1-mu)**2)
        else:
            u1,u2 = post_truth[1],post_truth[2]
            I = 1 - u1*(1-mu) - u2*(1-mu)**2
        posterior_intensity.append(I)
    return posterior_intensity
###############################################################################
#%%
###############################################################################
def transit_depth_deviation(posterior_samples, truths):
    tdd = (posterior_samples.T[0]**2-truths['ps']**2)*1e6
    return tdd
###############################################################################
#%%
###############################################################################
def sigma_to_interval(n):
    sigma_perc = norm.cdf(n)-norm.cdf(-n)
    lower = np.round(100*(1-sigma_perc)/2,int(np.ceil(n)-1))
    upper = np.round(100*(1+sigma_perc)/2,int(np.ceil(n)-1))
    interval = [lower,upper]
    return interval
###############################################################################
#%%
###############################################################################
def confidence_interval(data, x_value=None, sigmas=[1,2]):
    CI = {}
    if x_value != None:
        CI['x'] = np.median(x_value, axis=0)
    
    CI['median'] = np.median(data, axis=0)
    for sig in sigmas:
        interval = sigma_to_interval(sig)
        CI[f"{sig} sigma"] = np.percentile(data, interval, axis=0)
    return CI
###############################################################################