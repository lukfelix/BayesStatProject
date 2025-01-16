import matplotlib.pyplot as plt
import numpy as np
import emcee

#TODO: this function is implemented from scratch, I found a package PyMC3 which has a function pymc3.diagnostics.gelman_rubin => might want to try this out
def gelman_rubin(samples):
    """
    Calculate the Gelman-Rubin statistic for each parameter.

    Parameters:
    samples : array-like
        MCMC chains of shape (nwalkers, nsteps, ndim).

    Returns:
    array
        Gelman-Rubin statistics for each parameter.
    """
    n_steps, n_walkers, ndim = samples.shape
    print()
    chain_means = np.mean(samples, axis=1)
    chain_variances = np.var(samples, axis=1, ddof=1)
    overall_mean = np.mean(chain_means, axis=0)

    # Between-chain variance
    B = n_walkers * np.var(chain_means, axis=0, ddof=1)
    # Within-chain variance
    W = np.mean(chain_variances, axis=0)

    # Gelman-Rubin statistic
    R_hat = np.sqrt((W * (n_walkers - 1) / n_walkers + B / n_walkers) / W)
    return R_hat



def trace_plot(param_names, n_walker, chain_per_walker, title, save_path, show=False):
    """
    Plot the trace of the MCMC chain for all walkers.

    Parameters:
    param_names : list of str
        Names of the parameters being plotted.
    n_walker : int
        Number of walkers.
    chain_per_walker : ndarray
        The MCMC chain of shape (n_steps, n_walkers, n_params).
    title : str, optional
        Title of the plot. Defaults to None.
    save_path : str or Path, optional
        Path to save the plot. Defaults to None.
    """
    # Create subplots for each parameter
    fig, ax = plt.subplots(len(param_names), 1, sharex=True, figsize=(10, 2 * len(param_names)))
    plt.subplots_adjust(hspace=0.3)

    # Plot traces
    for i, name in enumerate(param_names):
        for j in range(n_walker):
            ax[i].plot(chain_per_walker[:, j, i], lw=0.5, alpha=0.7, color=f"C{j}")
        ax[i].set_ylabel(name)
    ax[-1].set_xlabel("step")

    # Set the title
    fig.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.9)  # Adjust layout to fit the title

    # Save the plot
    fig.savefig(save_path, dpi=300)
    print(f"Plot saved to {save_path}")

    if show:
        plt.show()  # Display the plot


def check_convergence(samples, model_name, param_names):
    """
    Check convergence of MCMC samples by plotting traceplots, autocorrelation-time and 
    calculating the Gelman-Rubin diagnostic.
    
    Parameters:
    samples : ndarray
        Unflattened MCMC samples with shape (nwalkers, nsteps, ndim).
    covergence_plot_name : f-string
        Directory/Name to save the convergence plots.
    
    Returns:
    ndarray
        Gelman-Rubin statistics for each parameter.
    """
    nsteps, n_walker, ndim = samples.shape
    print(samples.shape)

    trace_plot(param_names, n_walker, samples, "Trace Plot "+model_name, "outputs/plots/traces_"+model_name+".png")

    # Gelman-Rubin Diagnostic (use the unflattened samples)
    gr_stat = gelman_rubin(samples)
    print(f"Gelman-Rubin Statistic: {gr_stat}")

    # Gelman-Rubin bar plot
    fig, ax = plt.subplots()
    ax.bar(range(ndim), gr_stat)
    ax.set_title("Gelman-Rubin Diagnostic")
    ax.set_xlabel("Parameter Index")
    ax.set_ylabel("GR Statistic")

    # Save Gelman-Rubin plot
    gr_plot_path = "outputs/plots/gelman_rubin_"+str(model_name) +".png"
    fig.savefig(gr_plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved Gelman-Rubin plot to {gr_plot_path}")

    return gr_stat
