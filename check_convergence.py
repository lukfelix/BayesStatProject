import matplotlib.pyplot as plt
import numpy as np
import emcee

def autocorr(x, lag=50):
    """
    Compute the autocorrelation of a 1D array up to a specified lag.
    
    Parameters:
    x : array-like
        The input array (e.g., single MCMC chain for a parameter).
    lag : int
        Maximum lag to compute autocorrelation for.

    Returns:
    array
        Autocorrelation values for each lag.
    """
    n = len(x)
    mean = np.mean(x)
    var = np.var(x)
    autocorr_values = []

    for l in range(lag):
        if l == 0:
            autocorr_values.append(1.0)  # Autocorrelation at lag 0 is always 1
        else:
            if l < n:  # Only calculate for valid lag values
                autocorr_values.append(
                    np.correlate(x[:-l] - mean, x[l:] - mean)[0] / (var * (n - l))
                )
            else:
                break  # Stop if lag exceeds the length of the array
    
    return np.array(autocorr_values)

def gelman_rubin(chains):
    """
    Calculate the Gelman-Rubin statistic for each parameter.

    Parameters:
    chains : array-like
        MCMC chains of shape (nwalkers, nsteps, ndim).

    Returns:
    array
        Gelman-Rubin statistics for each parameter.
    """
    num_chains, num_samples, num_params = chains.shape
    chain_means = np.mean(chains, axis=1)
    chain_variances = np.var(chains, axis=1, ddof=1)
    overall_mean = np.mean(chain_means, axis=0)

    # Between-chain variance
    B = num_samples * np.var(chain_means, axis=0, ddof=1)
    # Within-chain variance
    W = np.mean(chain_variances, axis=0)

    # Gelman-Rubin statistic
    R_hat = np.sqrt((W * (num_samples - 1) / num_samples + B / num_samples) / W)
    return R_hat



def check_convergence(samples, covergence_plot_name):
    """
    Check convergence of MCMC samples by plotting autocorrelation and 
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
    nwalkers, nsteps, ndim = samples.shape
    param_names = ['ps', 'u1', 'u2']
    tau = emcee.autocorr.integrated_time(samples)
    print("Integrated auto-correlation time")
    for name, iat in zip(param_names, tau):
        print(f"{name}: {iat:.1f}")
    # Autocorrelation plots for each parameter
    for dim in range(ndim):
        fig, ax = plt.subplots()
        for walker in range(nwalkers):
            autocorr_vals = autocorr(samples[walker, :, dim])
            ax.plot(autocorr_vals, label=f'Walker {walker}')
        ax.set_title(f"Autocorrelation of Chains (Param {dim})")
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.legend()

        # Save autocorrelation plot
        autocorr_plot_path = covergence_plot_name +"_param{dim}.png"
        fig.savefig(autocorr_plot_path, dpi=300)
        plt.close(fig)
        print(f"Saved autocorrelation plot for Param {dim} to {autocorr_plot_path}")

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
    gr_plot_path = output_dir / "gelman_rubin_plot.png"
    fig.savefig(gr_plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved Gelman-Rubin plot to {gr_plot_path}")

    return gr_stat
