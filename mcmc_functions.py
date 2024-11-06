import numpy as np

# define some necessary math for priors later
def log_uni(x, a, b):
    if a<x<b:
        return 0.0
    else: return -np.inf

def log_gauss(x, mean, sigma):
    return -np.log(np.sqrt(2*np.pi)*sigma) - (x-mean)**2 / (2*sigma**2)




def eval_transit(params, theta, model):
    """
    evaluate the batman model by overwriting paramaters with ccurrent guess theta,
    get normalized flux values as lightcurve,
    reset batman model and return flux
    """
    temporary_params = params.rp, params.u

    # update model parameters, commented parameters are not fitted
    params.rp = theta[0]
    params.u = [theta[1], theta[2]]         
    # TODO: add further parameters to be fitted

    # evaluate model with current guesses theta
    flux = model.light_curve(params)

    params.rp, params.u = temporary_params   # reset model the way it was

    return flux

def full_model(params, theta, model):
    """
    evaluate batman model and allow for fitting of linear trend in the data
    """
    # flux = (theta[-2] + theta[-1]*(t-t_0)) * eval_transit(params, theta, model)   # TODO implement linear trend?
    flux = eval_transit(params, theta, model)
    return flux


# necessary emcee functions
def log_likelihood(params, theta, model, y, yerr, full_model):
    model_prediction = full_model(params, theta, model)

    return -0.5 * np.sum(((y - model_prediction) / yerr) ** 2)

def log_prior(theta, param_priors):
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

def log_probability(theta, params, param_priors, model, y, yerr, full_model):
    lp = log_prior(theta, param_priors)
    if not np.isfinite(lp):
        return -np.inf
    # print('not inf')
    return lp + log_likelihood(theta, params, model, y, yerr, full_model)