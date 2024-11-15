import numpy as np
import emcee
import corner
import os

from model_functions import full_model, kipping_to_quad

# define some necessary math for priors functions
def log_uni(x, a, b):
    if a<x<b:
        return 0.0
    else: return -np.inf

def log_gauss(x, mean, sigma):
    return -np.log(np.sqrt(2*np.pi)*sigma) - (x-mean)**2 / (2*sigma**2)

# p(D|theta)
def log_likelihood(theta, t, y, yerr, params, model, transform):
    model_prediction = full_model(theta, params, model, transform)
    return -0.5 * np.sum(((y - model_prediction) / yerr) ** 2)

# p(theta)
def log_prior(theta, priors):
    """
    evaluate priors for given paramter theta, looks unusual due to dictionary structure of priors
    """
    lp = 0.
    for val, prior in zip(theta, priors):
        # print(val, prior)
        if priors[prior][0] == 'uni':
            ret = log_uni(val, priors[prior][1], priors[prior][2])
        elif priors[prior][0]=='gauss':
            ret = log_gauss(val, priors[prior][1], priors[prior][2])
        else:
            print('Error: prior-distribution "{}" not accepted by current code.'.format(priors[prior][0]))
            return None
        if ret == -np.inf:# or ret < 1e-20:
            # print('inf')
            return -np.inf
        else:
            lp += ret
    return lp

# Define the final probability function as likelihood * prior ('+' due to log).
# p(theta|D) = p(D|theta) * p(theta)
def log_posterior(theta, t, y, yerr, params, model, priors, transform):
    lp = log_prior(theta, priors)
    if not np.isfinite(lp):
        return -np.inf
    # print('not inf')
    return lp + log_likelihood(theta, t, y, yerr, params, model, transform)

def run_mcmc(time_data, flux_data, error_data, model,
             model_params, priors, mcmc,
             transform=False):
    """
    In this function we take all the necessary inputs for running the mcmc provided in the analysis.py file.
    From there we initialize the mcmc-run and execute it, resulting in posterior samples that are returned.
    time_data       :   array containing the time values associated with each flux datapoint
    flux_data       :   array of flux values as simulated by the lightcurve modeling
    error_data      :   array of errors associated with each simulated flux value
    model           :   batman model instance (has to be initiated with the same time_data array)
    model_params    :   batman param instance (needs to already contain the fixed parameter values)
    priors          :   dict of planet radii and limb-drakening priors, can either be gaussian or uniform atm
    mcmc            :   dict of mcmc parameters ( 'ndim', 'nwalkers', 'nsteps', 'burn_in_frac')
    transform=False :   whether or not the limb-darkening is analysed in 
                            the transformed Kipping parameterization (True) 
                            or default quadratic (False)

    Returns         :   flattened posterior samples
    """

    # Initialize the walkers
    pos = np.zeros((mcmc['nwalkers'], mcmc['ndim']))
    for i, param in enumerate(priors):
        if priors[param][0]=='uni':
            pos[:, i] = np.random.uniform(priors[param][1], priors[param][2], mcmc['nwalkers'])
        elif priors[param][0]=='gauss':
            pos[:, i] = np.random.normal(priors[param][1], priors[param][2], mcmc['nwalkers'])
        else:
            print('prior not recognized')
            break

    # Initialize the sampler (at the moment time_data is not being used, see "model_functions.py/full_model()", but would be necessary if one also wants to fit a linear slope)
    # args are additional inputs besides theta in the log_posterior calculation
    sampler = emcee.EnsembleSampler(mcmc['nwalkers'], mcmc['ndim'], log_posterior,
                                    args=(time_data, flux_data, error_data, model_params, model, priors, transform))

    # Run the sampler
    sampler.run_mcmc(pos, mcmc['nsteps'], progress=True)

    # Get the samples
    samples = sampler.get_chain(discard=int(mcmc['burn_in_frac'] * mcmc['nsteps']))     # discard the burn-in phase

    # Flatten the samples (remove the walkers)
    flattened_samples = samples.reshape(-1, len(priors))  # flatten the samples for plotting
    
    # TODO: save samples in some format for more flexible plotting and post-processing

    return flattened_samples

def create_corner_plot(posterior_samples, truths, errval, transform=False):
    """
    Plotand save the corner plot of the posterior samples, showing the underlying truth values
    """
    fig = corner.corner(
        posterior_samples, 
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
    )
    if transform:
        corner_plot_name = "outputs/plots/corner_plot_kipping_%.0fppm_ld%.2f.png" % (errval, kipping_to_quad(truths['u'][0], truths['u'][1])[0])
    else:
        corner_plot_name = "outputs/plots/corner_plot_quadratic_%.0fppm_ld%.2f.png" % (errval, truths['u'][0])

    if not os.path.exists(corner_plot_name):
        fig.savefig(corner_plot_name, dpi=300)
        return