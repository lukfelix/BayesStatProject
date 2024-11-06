def eval_transit(theta, initial_params, model):
    """
    evaluate the batman model by overwriting paramaters with current guess theta,
    get normalized flux values as lightcurve,
    reset batman model and return flux.

    "initial_params" is a batman.TransitParams object which we need to modify to evaluate the model. We want to always use the initial model parameters again in every step
    "model" is a batman.TransitModel object which we use to evaluate the model.

    """
    params = initial_params             # copy the initial model, not really necessary, just nicer

    # update model parameters, commented parameters are not fitted
    params.rp = theta[0]
    params.u = [theta[1], theta[2]]         
    # TODO: add further parameters to be fitted

    # evaluate model with current guesses theta
    flux = model.light_curve(params)

    return flux

def full_model(theta, initial_params, model):
    """
    evaluate batman model and allow for fitting of linear trend in the data
    """
    # TODO implement & uncomment linear trend --> need to adapt the function inputs
    # flux = (theta[-2] + theta[-1]*(t-t_0)) * eval_transit(params, theta, model)  
    
    flux = eval_transit(theta, initial_params, model)
    return flux