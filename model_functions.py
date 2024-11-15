import numpy as np

def kipping_to_quad(q1, q2):
    """
    Transforms the two parameters of the Kipping parameterization to the quadratic parameterization.
    (q1, q2) -> (u1, u2)
    Source: https://ui.adsabs.harvard.edu/abs/2013MNRAS.435.2152K/abstract
    Parameter spaces for q are [0, 1] but for u it's [-3, 3]
    """
    u1 = 2*np.sqrt(q1)*q2
    u2 = np.sqrt(q1)*(1 - 2*q2)
    return u1, u2

def quad_to_kipping(u1, u2):
    """
    Probably not used (maybe for plotting at some point?)

    Transforms the two parameters of the quadratic parameterization to the Kipping parameterization.
    (u1, u2) -> (q1, q2)
    Source: https://ui.adsabs.harvard.edu/abs/2013MNRAS.435.2152K/abstract
    Parameter spaces for q are [0, 1] but for u it's [-3, 3]
    """
    q1 = (u1 + u2)**2

    # the case where u1 and u2 are 0 is handled here 
    # (only comes up when converting between the truth dict as it's never exactly 0 otherwise)
    try:
        q2 = 0.5 * u1 / (u1 + u2)
    except:
        q2 = None
    return q1, q2

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
    truths_kip['u'] = quad_to_kipping(truths['u'][0], 
                                      truths['u'][1])
    # TODO: q2 is not actually well defined... need to calculate the limes of 0.5 * u1 / (u1 + u2) for u1 & u2 -> 0 
    # but the results diverge depending which parameter you let go to 0 first...
    # For now this is handled in quad_to_kipping by setting q2=None

    return truths_kip, priors_kip    

def eval_transit(theta, initial_params, model, transform):
    """
    evaluate the batman model by overwriting paramaters with current guess theta,
    get normalized flux values as lightcurve,
    reset batman model and return flux.

    "initial_params" is a batman.TransitParams object which we need to modify to evaluate the model. We want to always use the initial model parameters again in every step
    "model" is a batman.TransitModel object which we use to evaluate the model.
    "transform" determines whether or not the priors use the Kipping parameterization
    """
    params = initial_params             # copy the initial model, not really necessary, just nicer

    # update model parameters, commented parameters are not fitted
    params.rp = theta[0]
    if transform:
        params.u = kipping_to_quad(theta[1], theta[2])
    else:
        params.u = [theta[1], theta[2]]
    # TODO: add further parameters to be fitted

    # evaluate model with current guesses theta
    flux = model.light_curve(params)

    return flux

def full_model(theta, initial_params, model, transform=False):
    """
    evaluate batman model and allow for fitting of linear trend in the data
    """
    # TODO implement & uncomment linear trend --> need to adapt the function inputs
    # flux = (theta[-2] + theta[-1]*(t-t_0)) * eval_transit(params, theta, model, transform)  
    
    flux = eval_transit(theta, initial_params, model, transform)
    return flux