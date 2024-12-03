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
    q2 = 0.5 * u1 / (u1 + u2)
    return q1, q2


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

def compute_fisher_information(params, model, time_data, flux_data, error_data):
    """
    Compute the Fisher Information Matrix for the given parameters.

    Args:
        params: batman.TransitParams object.
        model: batman.TransitModel object.
        time_data: Time points of the light curve.
        flux_data: Observed flux values.
        error_data: Flux error values.

    Returns:
        Fisher information matrix for the parameters.
    """
    n_params = 2  # Only considering two parameters (e.g., u1, u2 or q1, q2)
    fisher_matrix = np.zeros((n_params, n_params))

    # Generate the model flux
    initial_flux = model.light_curve(params)

    # Calculate numerical derivatives
    for i in range(n_params):
        for j in range(n_params):
            # Slight perturbations for numerical derivatives
            theta = np.array([params.u[0], params.u[1]])
            delta = 1e-5

            theta[i] += delta
            params.u = theta
            flux_plus = model.light_curve(params)

            theta[i] -= 2 * delta
            params.u = theta
            flux_minus = model.light_curve(params)

            params.u = [params.u[0], params.u[1]]  # Reset

            # Compute second derivative
            d2_logL = (
                (flux_plus - initial_flux) - (flux_minus - initial_flux)
            ) / (delta ** 2)

            # I_{i,j} = sum_k ( 1/error_k^2 * (del^2 f_k) / (del theta_i del theta_j) )
            fisher_matrix[i, j] = np.sum(d2_logL / error_data ** 2)

    return fisher_matrix
