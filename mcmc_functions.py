import numpy as np
import emcee
import corner
import os
import pathlib

from model_functions import full_model
from model_functions import compute_fisher_information
from model_functions import kipping_to_quad

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

    # Check for NaNs in the model prediction
    if np.any(np.isnan(model_prediction)):
        print(f"NaN detected in model prediction for theta={theta}")
        return -np.inf
    
    return -0.5 * np.sum(((y - model_prediction) / yerr) ** 2)

# p(theta)
def log_prior(theta, priors, params, model, time_data, flux_data, error_data, use_jeffrey=False, transform=False):
    """
    evaluate priors for given paramter theta, looks unusual due to dictionary structure of priors
    
    Args:
        theta: Current parameter values.
        priors: Dictionary of prior distributions.
        params, model, time_data, flux_data, error_data: Parameters for Jeffreys prior.
        use_jeffreys: Boolean to toggle Jeffreys prior.
        transform: Boolean, true for Kipping, false for quadratic

    Returns:
        Log of the prior probability.
    """


    # If we want to use Jeffrey priors
    if use_jeffrey:
        if transform:
            return log_jeffreys_prior_kipping(theta, params, model, time_data, flux_data, error_data)
        else: 
            return log_jeffreys_prior(theta, params, model, time_data, flux_data, error_data)
    
    lp = 0.0
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

def log_jeffreys_prior_kipping(theta, params, model, time_data, flux_data, error_data):
    """
    Compute the Jeffreys prior for Kipping parameterization.
    """
    params.u = kipping_to_quad(theta[1], theta[2])  # Convert Kipping parameters back to quadratic
    fisher_matrix = compute_fisher_information(params, model, time_data, flux_data, error_data)
    
    determinant = np.linalg.det(fisher_matrix)
    if determinant <= 1e-10:  # Small threshold to avoid log of zero or negative
        return -np.inf
    
    return 0.5 * np.log(determinant)

def log_jeffreys_prior(theta, params, model, time_data, flux_data, error_data):
    """
    Compute the Jeffreys prior for the limb-darkening parameters.

    Args:
        theta: Current parameter estimates.
        params: batman.TransitParams object.
        model: batman.TransitModel object.
        time_data: Time points of the light curve.
        flux_data: Observed flux values.
        error_data: Flux error values.

    Returns:
        Log of Jeffreys prior.
    """
    #print("Theta =", theta)
    params.u = theta  # Set parameters to current guess
    fisher_matrix = compute_fisher_information(params, model, time_data, flux_data, error_data)
    
    # Regularize Fisher matrix for stability
    #fisher_matrix += np.eye(fisher_matrix.shape[0]) * 1e-8

    # Compute determinant of the Fisher information matrix
    determinant = np.linalg.det(fisher_matrix)

    if determinant <= 1e-10:
        #print(f"Small determinant: {determinant}, Theta = {theta}")
        return -np.inf  # Log prior undefined for non-positive determinant

    return 0.5 * np.log(determinant) # p(theta) ~ sqrt(determinante(fisher_matrix))


# Define the final probability function as likelihood * prior ('+' due to log).
# p(theta|D) = p(D|theta) * p(theta)
# New argument (boolean) use_jeffrey
def log_posterior(theta, t, y, yerr, params, model, priors, transform, use_jeffrey=False):
    lp = log_prior(theta, priors, params, model, t, y, yerr, use_jeffrey, transform)
    if not np.isfinite(lp):
        return -np.inf
    
    # = log_likelihood(theta, t, y, yerr, params, model, transform)
    #if not np.isfinite(ll):
    #    print(f"Invalid likelihood: Theta = {theta}, ll = {ll}")
    #    return -np.inf
    # print('not inf')
    return lp + log_likelihood(theta, t, y, yerr, params, model, transform)

def run_mcmc(time_data, flux_data, error_data, model,
             model_params, priors, mcmc, param_names,
             transform=False, use_jeffrey=False):
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
    param_names     :   names of parameters to be fitted
    transform=False :   whether or not the limb-darkening is analysed in 
                            the transformed Kipping parameterization (True) 
                            or default quadratic (False)
    use_jeffrey=False:  whether or not jeffrey priors should be used, default: False

    Returns:   
        flattened_samples: flattened posterior samples
        samples: unflattened posterior samples for convergence checks
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
    # new argument use_jeffrey in args
    sampler = emcee.EnsembleSampler(mcmc['nwalkers'], mcmc['ndim'], log_posterior,
                                    args=(time_data, flux_data, error_data, model_params, model, priors, transform, use_jeffrey))

    # Run the sampler
    sampler.run_mcmc(pos, mcmc['nsteps'], progress=True)

    # Get the samples
    samples = sampler.get_chain(discard=int(mcmc['burn_in_frac'] * mcmc['nsteps']))     # discard the burn-in phase

    # Get autocorrelation-time
    tau = sampler.get_autocorr_time()
    print("Integrated auto-correlation time")
    for name, iat in zip(param_names, tau):
        print(f"{name}: {iat:.1f}")

    # Flatten the samples (remove the walkers)
    flattened_samples = samples.reshape(-1, len(priors))  # flatten the samples for plotting
    
    # TODO: save samples in some format for more flexible plotting and post-processing =>DONE!
    samples_output_dir = pathlib.Path("outputs/samples")
    samples_output_dir.mkdir(parents=True, exist_ok=True)

    sample_file_name = samples_output_dir / f"samples_{mcmc['ndim']}params_{mcmc['nsteps']}steps.npy"
    np.save(sample_file_name, flattened_samples)
    print(f"Saved samples to {sample_file_name}")

    return flattened_samples, samples

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
        corner_plot_name = "outputs/plots/corner_plot_kipping_%.0fppm_no_linear_model" % (errval)
    else:
        corner_plot_name = "outputs/plots/corner_plot_quadratic_%.0fppm_no_linear_model" % (errval)

    if not os.path.exists(corner_plot_name):
        fig.savefig(corner_plot_name, dpi=300)
    return