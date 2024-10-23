import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import batman
import emcee
import corner
import pathlib
import os

def get_data(filepath):
    # TODO: for now this just returns a very simple toy dataset with a box transit
    # need to see how simulation output files look like first
    time = np.linspace(-1, 1, num=400)
    depth = np.ones(400)
    depth[120:-120] *= 1 - 1e-4
    errs = np.ones(400)*2e-5

    return time, depth, errs

# define some necessary math for priors later
def log_uni(x, a, b):
    if a<x<b:
        return 0.0
    else: return -np.inf

def log_gauss(x, mean, sigma):
    return -np.log(np.sqrt(2*np.pi)*sigma) - (x-mean)**2 / (2*sigma**2)

###############################################################
# performance settings and basic MCMC parameters
###############################################################
ncpu = 1 # number of cores available for batman, set to 1 if you either want to be slow or don't have openMP
nsteps = 10000
nwalkers = 64
burn_in_frac = 0.4

##################
#     data       #
##################
this_directory = pathlib.Path().resolve()
# print(this_directory)

# example that assumes that there is data in the data folder named lightcurve.dat, maybe do something similar so it works on everyones system
filename = os.path.join(pathlib.Path().resolve(), 'outputs', 'data', 'lightcurve.dat') 

time, data, errs  = get_data(filename)  # TODO: get actual data here

##############################
#    fixed Parameters        #
##############################

limb_darkening = ['quadratic', [0.001, 0.001]] # just the initialization, caclulated with https://exoctk.stsci.edu/limb_darkening for a specific target, TODO: needs to be adapted for the Kipping case!
t_0 = time[0]       # start time of dataset
per = 60.           # need to fix period due to degeneracy with semi-major axis   TODO: might not be necessary for simulated dataset
ecc =  0.           # go for circular orbit for now
w =    0.           # irrelevant if circular, so set to zero for now
inc = 90.

###############################
#  priors for fitted params   #
###############################

# convention:
# [uni, lower bound, upper bound]
# [gauss, mean, sigma]
param_priors = {
    # TODO: adapt these depending on simdata
    't0':   ['uni', t_0+0.9,  t_0+1.1], # days
    'rp':   ['uni', 0.005, 0.015],      # stellar radii
    'a':    ['uni', 10.,  50.],        # stellar radii
    # 'a':    ['gauss', 41, 20.],         # stellar radii
    # 'inc':  ['uni', 80.,  90.],         # degrees maybe convert this to b? b=cos(i) * a/R*
    # others
    'u1':     ['uni', -3., 3],     # limb darkening
    'u2':     ['uni', -3., 3.],     # limb darkening
    'c':      ['uni', 0.9,  1.1],   # factor allowing vertical offset (TODO: I don't think they allow for this, but they probably should, also slope below)
    'v':      ['gauss', 0, 1e3],    # allow for slope !=0 in time (dy/dt), needs a gaussian prior to converge consistently
    }



"""
Create batman model with some initial parameters

Must be done before fitting, otherwise we would initialize every time
but values that are fittet will be overwirtten anyway

This is also where we impose the times where the fitted model is evaluated, that's why "time" is an input.
"""
params = batman.TransitParams()         # object to store transit parameters
params.t0 = t_0 + 1.                    # time of inferior conjunction
params.per = per                        # orbital period in days
params.rp = 0.05                        # planet radius (in units of stellar radii)
params.a = 35                           # semi-major axis (in units of stellar radii)
params.inc = inc                        # orbital inclination (in degrees)
params.ecc = ecc                        # eccentricity
params.w = w                            # longitude of periastron (in degrees)
params.limb_dark = limb_darkening[0]    # set at the beginning
params.u = limb_darkening[1]               

model = batman.TransitModel(params, time, nthreads = int(ncpu))    #initializes model

def eval_transit(theta):
    """
    evaluate the batman model by overwriting paramaters with ccurrent guess theta,
    get normalized flux values as lightcurve,
    reset batman model and return flux
    """
    temp = params.t0, params.per, params.rp, params.a, params.inc, params.ecc, params.w, params.u

    # update model parameters, commented parameters are not fitted
    params.t0 = theta[0]
    # params_d.per= 
    params.rp = theta[1]
    params.a  = theta[2]
    # params.inc= theta[3]
    # params_d.ecc= 
    # params_d.w  = 
    params.u = [theta[-4], theta[-3]]           # indices are weird to simplify adding more orbital parameters    


    flux = model.light_curve(params)
    params.t0, params.per, params.rp, params.a, params.inc, params.ecc, params.w, params.u = temp   # reset model

    return flux

def full_model(t, theta):
    """
    evaluate batman model and allow for fitting of linear trend in the data
    """
    flux = (theta[-2] + theta[-1]*(t-t_0)) * eval_transit(theta)
    return flux

# necessary emcee functions
def log_likelihood(theta, t, y, yerr):
    model_prediction = full_model(t, theta)

    return -0.5 * np.sum(((y - model_prediction) / yerr) ** 2)

def log_prior(theta):
    """
    evaluate priors for given paramter theta, looks unusual due to dict structure of priors above
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

def log_probability(theta, t, y, yerr):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    # print('not inf')
    return lp + log_likelihood(theta, t, y, yerr)

def do_mcmc(time, data, err, theta0, var_names, n_steps=1000, n_walkers=64, frac_burnin=0.3, walker_noise=1e-3):
    """
    Runs mcmc and returns parameters for all non-burnin steps averaged over walkers
    """
    # prepare sampler
    ndim = len(theta0)
    sampler = emcee.EnsembleSampler(n_walkers, ndim, log_probability, args=(time, data, err))
    
    # Run the sampler
    pos = theta0 + walker_noise * np.random.randn(n_walkers, ndim)
    
    sampler.run_mcmc(pos, n_steps, progress=True)
    
    # Get the samples
    samples = sampler.get_chain(discard=int(frac_burnin*n_steps))
    
    # TODO: save samples in some format for more flexible plotting and post-processing

    flattened = samples.reshape(-1, len(theta0))

    # average over walkers
    averaged = np.mean(samples, axis=1)
    # print(averaged)
    for i, name in enumerate(var_names):
        # save some more diagnostic plots
        plt.plot(averaged[:, i])
        plt.title(name)
        plt.savefig(os.path.join(this_directory, 'outputs', 'plots', 'param_{}.jpg').format(name))
        plt.close()
    
    return averaged, flattened



##################################
# actual code execution below
##################################        

theta0 = np.array([
    # d planet
    t_0+1.,    # 't0_d' days
    0.01,  # 'rp_d' stellar radii
    35.8,   # 'a_d' stellar radii
    # others
    0.001, 	    # 'u1':     ['uni', 0., 1.],
    0.001,       # 'u2':     ['uni', 0., 1.],
    1.0001,     # 'c':      ['uni', 0.9,  1.1],
    0.,    # 'v':      ['uni', -0.05, 0.05],   # dy/dt
])

param_names = list(param_priors.keys())
averaged_samples, flattened_samples = do_mcmc(time, data, errs, theta0, param_names, n_walkers=nwalkers, n_steps=nsteps, frac_burnin=burn_in_frac)

corner.corner(flattened_samples, labels=param_names, show_titles=True)#, truths=theta0)
plt.savefig(os.path.join(this_directory, 'outputs', 'plots', 'corner_lcfit.pdf'))
plt.close()


print('Median Values: ', np.median(averaged_samples, axis=0))
print('Percentile-Errors (16%, 84%): ', 
      np.median(averaged_samples, axis=0) - np.percentile(averaged_samples, [16], axis=0),
      np.percentile(averaged_samples, [84], axis=0) - np.median(averaged_samples, axis=0))
print('Creating figure ...')
fig, (ax1, ax2) = plt.subplots(2, sharex=True, height_ratios=[5, 2], figsize=(19,7))
fig.subplots_adjust(hspace=0)

for sample in averaged_samples:
    model_res = full_model(time, sample)
    ax1.plot(time-t_0, model_res, alpha=0.01, zorder=1, color='bisque')
ax1.errorbar(time-t_0, data, yerr=errs, linestyle='', marker='o', alpha=0.4, zorder=2)

median_model = np.median(averaged_samples, axis=0)
print('MAD =',      np.mean(abs(full_model(time, median_model) - data))*1e6, 'ppm')
print('$\sigma$ =', np.std(     full_model(time, median_model) - data) *1e6, 'ppm')

ax1.plot((time-t_0), full_model(time, median_model), alpha=1., color='tab:orange', zorder=4, label='Model', lw=3)


ax2.plot(time-t_0, np.zeros_like(time), color='tab:orange')
ax2.plot(time-t_0, (data - full_model(time, median_model))*1e6, linestyle='', marker='.', alpha=0.4, color='tab:blue')

ax2.text(0.01, 0.18, 'MAD = {:.2f} ppm'.format(np.mean(abs(full_model(time, median_model) - data))*1e6), fontsize=10,
         horizontalalignment='left', verticalalignment='center', transform=ax2.transAxes)
ax2.text(0.01, 0.06, '$\sigma$ = {:.2f} ppm'.format(np.std(full_model(time, median_model) - data) *1e6), fontsize=10,
         horizontalalignment='left', verticalalignment='center', transform=ax2.transAxes)


ax2.set_xlabel('Time since start of observation [mystery units]')   # TODO: adjust units
ax1.set_ylabel('Normalized Flux')
ax2.set_ylabel('Residuals [ppm]')
ax1.legend(loc='lower left')
ax1.set_xlim((time[0]-t_0, time[-1]-t_0))
plt.tight_layout()
plt.savefig(os.path.join(this_directory, 'outputs', 'plots', 'lcfit.pdf'))
plt.show()