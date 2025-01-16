#%%
###############################################################################
""" Import what is needed """
###############################################################################
# Import necessary libraries
import os
from joblib import dump

# Import predefined functions from other files
import simulation_functions as sim          # functions for simulating the light curve
import model_functions as mod_fun           # functions for evaluation of the model
import mcmc_functions as mcmc_fun           # functions used for the MCMC analysis
import check_convergence as conv            # functions used for checking convergence
import posterior_functions as post_fun      # functions for analyzing the posterior samples
###############################################################################
#%%
###############################################################################
""" Define the Classes """
###############################################################################
class CI:
    def __init__(self, model, error, truths, priors, mcmc_params, samples, flux, intensity, tdd):
        self.model = model
        self. error = error
        self.truths = truths
        self.priors = priors
        self. mcmc_params = mcmc_params
        self.samples = samples
        self.flux = flux
        self.intensity = intensity
        self.tdd = tdd
    def __repr__(self):
        if self.truths['u1']==0:
            limb = 'No Limb Darkening'
        else:
            limb = 'Minimal Limb Darkening'
        return f"Confidence: {self.model}, Error: {self.error} ppm, Truths:{self.truths}, {limb}"

class Posterior_Samples:
    def __init__(self, model, error, truths, priors, mcmc_params, samples, tdd):
        self.model = model
        self. error = error
        self.truths = truths
        self.priors = priors
        self. mcmc_params = mcmc_params
        self.samples = samples
        self.tdd = tdd
    def __repr__(self):
        if self.truths['u1']==0:
            limb = 'No Limb Darkening'
        else:
            limb = 'Minimal Limb Darkening'
        return f"Posterior: {self.model}, Error: {self.error} ppm, Truths:{self.truths}, {limb}"
###############################################################################
#%%
###############################################################################
""" Combine the different functions to run the full routine """
###############################################################################
def full_routine(model, truths, priors=None, fixed=None, errors=None, 
                 mcmc_steps=100000, folder_name='max_outputs', save_plots=True, 
                 save_data=True,save_intermediate_results=False):
    """
    In this function we combine the different functions to initialize the model,
    run the MCMC and save the results in various ways.
    INPUTS
    model      :              either 'quadratic' or 'kipping'
    truths     :              a dict containing the true values 
                                using the quadratic parametrization
    priors     :              priors of the truth, 
                                if None then the default_priors are used
    fixed      :              the fixed parameters, 
                                if None then the default_parameters are used
    errors     :              define the ppm of the error envelopes, 
                                if None the default_errors are used
    mcmc_steps :              the number of steps the MCMC uses,
                                default = 100'000
    folder_name:              The Name of the folder where everything gets saved
                                default = 'max_outputs'
    save_plots :              boolean if the plots are saved or not
    save_data  :              boolean if the data is saved
    save_intermediate_reults: boolean if intermediate results are saved 
                                (WARNING: This requires a lot of storage)
    RETURNS
    Depending on what is enabled for the save_... it saves
    for _plots: The corner, trace and gelman-rubin plots
    for _data:  Two Data Classes containing info about
                The confidence intervals of different values (CI dataclass)
                Posterior Samples and Posterior TDD (Posterior dataclass)
    for _intermediate_results: The posterior flux and posterior intensity
    FUNCTIONS
    From model_functions.py:
        -quad_to_kipping()
    From simulation_functions.py:
        - initialize_parameters()
        - initialize_model()
    From mcmc_functions.py:
        - run_mcmc()
        - create_corner_plot()
    From posterior_functions.py:
        - light_curve()
        - intensity()
        - transit_depth_deviation()
        - confidence_interval()
    From check_convergence.py:
        -check_convergence()
    """
    ###########################################################################
    if model == 'quadratic':
        transform = False
    elif model == 'kipping':
        transform = True
    else:
        print("Invalid model")
        return
    ###########################################################################
    # Set the Default values
    default_parameters = {
        # orbital parameters:
        'ps':0.1,                        # planet-to-star radius ratio
        't0':0,                          # time of inferior conjunction
        'period':1,                      # orbital period (in days)
        'a':4,                           # semi-major axis in stellar radii
        'inc':90,                        # orbital inclination in degrees
        'ecc':0,                         # eccentricity
        'omega':90,                      # longitude of periastron (in degrees)
        'limb_dark_model':"quadratic",   # limb-darkening model
        'u1':0,                          # limb-darkening coefficient u1
        'u2':0,                          # limb-darkening coefficient u2
        # simulated observation parameters:
        'n_points':1000,                 # number of points in the light curve
        't_min':-0.25,                   # minimum time in days
        't_max':0.25,                    # maximum time in days
    }
    
    default_priors = {
        'ps':['uni',0.,0.5],             # planet-to-star radius ratio
        'period':['uni',0.1,10.],        # orbital period (in days)
        'a':['uni',1.,10.],              # semi-major axis in stellar radii
        'inc':['uni',60.,120.],          # orbital inclination in degrees
        'ecc':['uni',0.,1.],             # eccentricity
        'omega':['uni',60.,120.],        # longitude of periastron (in degrees)
        'u1':['uni',-3.,3.],             # limb-darkening coefficient u1
        'u2':['uni',-3.,3.],             # limb-darkening coefficient u2
        'q1':['uni',0.,1.],              # limb-darkening coefficient q1
        'q2':['uni',0.,1.],              # limb-darkening coefficient q2
    }
    
    default_errors = [1000,300,100,30,10,1]
    ###########################################################################
    # Check if truth values are valid
    if not set(truths.keys()).issubset(set(default_parameters.keys())):
        print("Truth values not recognized")
        print("Allowed Values are:")
        print(list(default_parameters.keys()))
        return
    if not (('u1' in truths) and ('u2' in truths)):
        print("Limb Darkening Truths are missing")
        return
    # Check if priors are valid
    if priors != None:
        if not set(priors.keys()).issubset(set(default_priors.keys())):
            print("Priors not recognized")
            print("Allowed Priors are:")
            print(list(default_priors.keys()))
            return
        updated_priors = default_priors | priors
    else:
        updated_priors = default_priors
    # Check if Fixed values are valid
    if fixed != None:
        if not set(fixed.keys()).issubset(set(default_parameters.keys())):
            print("Fixed Values not recognized")
            print("Allowed fixed values are:")
            print(list(default_parameters.keys()))
            return
        if not set(fixed.keys()).isdisjoint(set(truths.keys())):
            print("Truth Values and Fixed Values Overlap")
            print(f"Overlap: {set(fixed.keys()) & set(truths.keys())}")
            return
        updated_fixed = default_parameters | fixed
    else:
        updated_fixed = default_parameters
    ###########################################################################
    # Crete the output folders if they do not exist
    if save_plots:
        folder_plots = os.path.join(folder_name,'plots')
        if not os.path.exists(folder_plots):
            os.makedirs(folder_plots)
    if save_data:
        folder_data = os.path.join(folder_name,'data')
        if not os.path.exists(folder_data):
            os.makedirs(folder_data)
    if save_intermediate_results:
        folder_intermediate = os.path.join(folder_name,'data','no_sync')
        if not os.path.exists(folder_intermediate):
            os.makedirs(folder_intermediate)
    ###########################################################################
    # Get the Priors and Truths
    initialize_truths = truths.copy()
    mcmc_truths = truths.copy()
    mcmc_priors = {}
    for key in list(truths.keys()):
        mcmc_priors[key] = updated_priors[key]
    # For the Kipping Model
    if model=='kipping':
        mcmc_truths['u1'],mcmc_truths['u2'] = mod_fun.quad_to_kipping(mcmc_truths['u1'],mcmc_truths['u2'])
        mcmc_priors['u1'] = updated_priors['q1']
        mcmc_priors['u2'] = updated_priors['q2']
    # Get the Errors
    if errors == None:
        errors_ppm = default_errors
    else:
        errors_ppm = errors
    # Get the mcmc values
    mcmc_parameters = {
            'ndim': len(mcmc_priors),
            'nwalkers': 4*len(mcmc_priors),
            'nsteps': mcmc_steps,
            'burn_in_frac': 0.6,
        }
    ###########################################################################
    # Initialize the model and get the true time and flux data
    batman_model_parameters, true_time = sim.initialize_parameters(initialize_truths, updated_fixed)
    batman_model, true_time, true_flux = sim.initialize_model(batman_model_parameters, true_time)
    ###########################################################################
    # Iterate over all error envelopes
    for err in errors_ppm:
        error_data = (err/1e6)*true_flux
        err_str = f"{err} ppm"
        #######################################################################
        # Run the MCMC
        posterior_samples, unflattened_samples = mcmc_fun.run_mcmc(
            true_time, true_flux, error_data, batman_model, batman_model_parameters, 
            mcmc_priors, mcmc_parameters, transform = transform, save = None
        )
        
        #######################################################################
        posterior_flux = post_fun.light_curve(posterior_samples, 
                                              batman_model_parameters, 
                                              batman_model, transform)
        posterior_intensity = post_fun.intensity_profile(posterior_samples, 
                                                         updated_fixed['n_points'], 
                                                         transform)
        posterior_tdd = post_fun.transit_depth_deviation(posterior_samples, mcmc_truths)
        #######################################################################
        # Compute the confidence intervals
        CI_samples = post_fun.confidence_interval(posterior_samples)
        CI_flux = post_fun.confidence_interval(posterior_flux)
        CI_intensity = post_fun.confidence_interval(posterior_intensity)
        CI_tdd = post_fun.confidence_interval(posterior_tdd)
        #######################################################################
        # Save the data
        filename = f"{model}_u1{initialize_truths['u1']}_{err}_ppm"
        Posterior_Samples_DC = Posterior_Samples(model, err, initialize_truths, 
                                                 mcmc_priors, mcmc_parameters, 
                                                 posterior_samples, posterior_tdd)
        CI_DC = CI(model, err, initialize_truths, mcmc_priors, mcmc_parameters, 
                   CI_samples, CI_flux, CI_intensity, CI_tdd)
        if save_data:
            dump(Posterior_Samples_DC,os.path.join(folder_data,f"posterior_{filename}.joblib"))
            dump(CI_DC,os.path.join(folder_data,f"confidence_{filename}.joblib"))
        if save_intermediate_results:
            dump(posterior_flux,os.path.join(folder_intermediate,f"posterior_flux_{filename}.joblib"))
            dump(posterior_intensity,os.path.join(folder_intermediate,f"posterior_intensity_{filename}.joblib"))
        #######################################################################
        # Plot the Corner Plot and the Gelman-Rubin Plot
        if save_plots:
            mcmc_fun.create_corner_plot(posterior_samples, mcmc_truths, errors_ppm, 
                                        err, folder_plots, filename, transform)
            ###################################################################
            gr_stat = conv.check_convergence(unflattened_samples, mcmc_truths.keys(), 
                                             folder_plots, filename)
            # Check if Gelman-Rubin statistic is below convergence threshold
            if gr_stat.max() < 1.1:
                print("Chains are well-mixed.")
            else:
                print("Chains may not have converged. Check diagnostics.")
        #######################################################################
    return