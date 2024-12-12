import numpy as np
import emcee
import corner
import os
import pathlib

from model_functions import full_model

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
             transform=False, save=None):
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
    save=None       :   whether ot not to save the flattened samples in an npy file
                            It'a enabled by entering any string which will end up in the name of the file.
                            Enabling this slows the script considerably, files are of order 10s of MB

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

    # Get autocorrelation-time
    tau = sampler.get_autocorr_time()
    print("Integrated auto-correlation time")
    for name, iat in zip(priors.keys(), tau):
        print(f"{name}: {iat:.1f}")

    # Flatten the samples (remove the walkers)
    flattened_samples = samples.reshape(-1, len(priors))  # flatten the samples for plotting

    if save != None:
        # save flattened samples for more flexible plotting and post-processing
        samples_output_dir = pathlib.Path("outputs/samples")
        samples_output_dir.mkdir(parents=True, exist_ok=True)

        sample_file_name = samples_output_dir / f"samples_{save}_{mcmc['nsteps']}steps.npy"
        np.save(sample_file_name, flattened_samples)
        print(f"Saved samples to {sample_file_name}")

    return flattened_samples, samples
from matplotlib.ticker import FuncFormatter

def create_corner_plot_NO_err_dic(posterior_samples, truths, transform=False):
    """
    Plotand save the corner plot of the posterior samples, showing the underlying truth values
    """
    fig = corner.corner(
        posterior_samples, 
        title_fmt='.4f',
        bins=50,
        show_titles=True,
        labels=[r"$R_P / R_*$", r"$u_1$", r"$u_2$", r"$a / R_*$", r"Period"], # leave this in since we want fancier plots, but at some point go with keys for flexibility
        # labels = truths.keys(),
        # truths=[truths['ps'], truths['u1'], truths['u2'], truths['a'], truths['period']],
        plot_density=True,
        plot_datapoints=True,
        fill_contours=False,
        smooth=True,
        levels=(0.6827, 0.90, 0.9545),              # shows the 1, 1.5 and 2 sigma contours in the 2D plots
        quantiles=[0.16, 0.5, 0.84],                # shows the 1 sigma interval in the 1D plots
        title_kwargs={"fontsize": 14},
        truth_color='cornflowerblue',
    )
    # Adjusting the size of x tick labels and ticks
    for ax in fig.axes:
        ax.tick_params(axis='x', labelsize=12, length=4, width=2)  # Adjust label size, tick length, and width
        ax.tick_params(axis='y', labelsize=12, length=4, width=2)  # Similarly for y-axis ticks if needed
        # ax.xaxis.get_offset_text().set_fontsize(14)
    for i, ax in enumerate(fig.axes):
        if ax.xaxis.get_label().get_text():  # Check if x-axis has a label
            ax.set_xlabel(ax.xaxis.get_label().get_text(), fontsize=15)  # Adjust x-axis label
        if ax.yaxis.get_label().get_text():  # Check if y-axis has a label
            ax.set_ylabel(ax.yaxis.get_label().get_text(), fontsize=15)

    # Formatter for x ticks to set decimal places
    def format_ticks(x, pos):
        return f"{x:.4f}"  # Adjust to show 2 decimal places

    formatter = FuncFormatter(format_ticks)

    # Apply the formatter to all x-axes
    for i, ax in enumerate(fig.axes):
        if i == len(fig.axes) - 5:
            ax.xaxis.set_major_formatter(formatter)

    fig.subplots_adjust(left=-0.03, bottom=-0.03)  # Adjust the spacing between the edges of the figure and the subplots


    # if transform:
    #     corner_plot_name = "outputs/plots/corner_plot_kipping_%.0fppm" % (errval)
    # else:
    #     corner_plot_name = "outputs/plots/corner_plot_quadratic_%.0fppm" % (errval)

    # if not os.path.exists(corner_plot_name):
    #     fig.savefig(corner_plot_name, dpi=300)

    fig.show()
    return
