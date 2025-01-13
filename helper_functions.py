# Import predefined functions from other files
from simulation_functions import *          # functions for simulating the light curve
from model_functions import *               # functions for evaluation of the model
from mcmc_functions import *                # functions used for the MCMC analysis
from check_convergence import *             # functions used for checking convergence
from helper_functions import *

def run_full_routine(truths, model_params, model, priors, mcmc, 
                     time_data, flux_data, errors_dict, 
                     transform=False, save=None):
    
    for key in errors_dict:
        # iterate through all available error envelopes for the default quadratic parameterization
        if save == None:
            posterior_samples, unflattened_samples = run_mcmc(time_data, flux_data, errors_dict[key], model,
                                                                model_params, priors, mcmc,
                                                                transform=transform)
        else:
            posterior_samples, unflattened_samples = run_mcmc(time_data, flux_data, errors_dict[key], model,
                                                                model_params, priors, mcmc,
                                                                transform=transform, save=save+'_'+key)

        # Plot the corner plot
        create_corner_plot(posterior_samples, truths, errors_dict[key][0]*1e6, transform=transform)

        if transform:
            model_name = f"kipping_model_%.0fppm" % (errors_dict[key][0]*1e6)
        else:
            model_name = f"quadratic_model_%.0fppm" % (errors_dict[key][0]*1e6)

        # Use unflattened samples to check convergence
        gr_stat = check_convergence(unflattened_samples, model_name, truths.keys())

        # Check if Gelman-Rubin statistic is below convergence threshold
        if gr_stat.max() < 1.1:
            print("Chains are well-mixed.")
        else:
            print("Chains may not have converged. Check diagnostics.")

    return
    
def change_to_kipping_run(truths, priors):
    """
    Returns a new truth and prior dict with changed limb-darkening behaviour.
    This is simply to ensure that everything is changed consistently and saves some space in the analysis file.
    """
    # for priors we only need to change parameterspace
    priors_kip = priors.copy()
    priors_kip['u1'] = ['uni', 0., 1.]
    priors_kip['u2'] = ['uni', 0., 1.]
    
    # limb-darkening coefficients q1, q2 are then transformed from initial quadratic values
    truths_kip = truths.copy()
    truths_kip['u1'], truths_kip['u2'] = quad_to_kipping(truths['u1'], truths['u2'])

    # TODO: q2 is not actually well defined... need to calculate the limes of 0.5 * u1 / (u1 + u2) for u1 & u2 -> 0 
    # but the results diverge depending which parameter you let go to 0 first...
    # For now this is handled in quad_to_kipping by setting q2=None

    return truths_kip, priors_kip  

def run_quad_kipping_routine(truths, model_params, model, priors, mcmc, 
                             time_data, flux_data, errors_dict):
    """
    Takes the set of dicts and simulated data for the typical quadratic case and runs that, as well as the correctly transformed kipping case.
    This does currently only save plots, if you want to save the flattened chains you need to call run_full_routine seperately.
    """
    run_full_routine(truths, model_params, model, priors, mcmc, 
                 time_data, flux_data, errors_dict, 
                 transform=False)
    
    truths_kip, priors_kip = change_to_kipping_run(truths, priors)

    run_full_routine(truths_kip, model_params, model, priors_kip, mcmc, 
                 time_data, flux_data, errors_dict, 
                 transform=True)

    return
