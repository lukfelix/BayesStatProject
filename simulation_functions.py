import numpy as np
import batman
import matplotlib.pyplot as plt

# Initializes the parameters for the transit simulation with the batman package
def initialize_parameters(truths, fixed):
    
    # Define parameters for the transit simulation
    params = batman.TransitParams()       
    params.t0 = fixed['t0']                          # time of inferior conjunction
    params.per = fixed['period']                     # orbital period (in days)
    params.rp = truths['ps']                   # planet-to-star radius ratio = planet radius (in units of stellar radii)
    params.a = fixed['a']                            # semi-major axis in stellar radii
    params.inc = fixed['inc']                        # orbital inclination in degrees
    params.ecc = fixed['ecc']                          # eccentricity
    params.w = fixed['omega']                        # longitude of periastron (in degrees)
    params.u = truths['u']                            # limb-darkening coefficients (no limb-darkening)
    params.limb_dark = fixed['limb_dark_model']      # limb-darkening model

    # Time array for the simulation
    t = np.linspace(fixed['t_min'], 
                    fixed['t_max'], 
                    fixed['n_points'])    # time from t_min to t_max days, n points

    return params, t

# Create a batman model and simulate the light curve
def simulate_light_curve(model, params):

    flux = model.light_curve(params)          # calculate the light curve

    return flux

# # Plot the simulated light curve
# def plot_single_light_curve(flux_data, time_data, all_errors_dic, plt_size=(10, 5)):

#     # Determine the number of keys to scale alpha values
#     num_keys = len(all_errors_dic)
#     alpha_values = np.linspace(0.8, 0.1, num_keys)  # Scale from 0.1 to 0.9


#     plt.figure(figsize=plt_size)
#     plt.plot(time_data, flux_data, color='black', label="Simulated Light Curve",lw = 0.5)   # plot the simulated light curve
#     for i, (key, error) in enumerate(all_errors_dic.items()):
#         plt.fill_between(time_data, flux_data - error/2, flux_data + error/2, alpha=alpha_values[i], label=key, color='forestgreen')
#     plt.xlabel("Time from Mid-Transit (days)")
#     plt.ylabel("Relative Flux")
#     plt.title("Simulated Transit Light Curve with Error Enevelopes")
#     plt.legend()
#     plt.show()

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Function to plot the simulated light curve with a zoomed-in inset
def plot_single_light_curve(flux_data, time_data, all_errors_dic, plt_size=(10, 5)):
    # Determine the number of keys to scale alpha values
    num_keys = len(all_errors_dic)
    alpha_values = np.linspace(0.8, 0.1, num_keys)  # Scale from 0.9 to 0.1

    # Create main plot
    fig, ax = plt.subplots(figsize=plt_size)
    ax.plot(time_data, flux_data, color='black', label="Simulated Light Curve", lw=0.5)  # plot the simulated light curve
    for i, (key, error) in enumerate(all_errors_dic.items()):
        ax.fill_between(time_data, flux_data - error/2, flux_data + error/2, alpha=alpha_values[i], label=key, color='forestgreen')

    ax.set_xlabel("Time from Mid-Transit (days)")
    ax.set_ylabel("Relative Flux")
    ax.set_title("Simulated Transit Light Curve with Error Envelopes")
    ax.legend()

    # Create inset (zoomed view)
    zoom_range = 0.05
    axins = inset_axes(ax, width="30%", height="30%", loc='lower right', borderpad=2)
    axins.plot(time_data, flux_data, color='black', lw=0.5)
    for i, (key, error) in enumerate(all_errors_dic.items()):
        axins.fill_between(time_data, flux_data - error/2, flux_data + error/2, alpha=alpha_values[i], color='forestgreen')

    axins.set_xlim(-zoom_range, zoom_range)
    axins.set_ylim(min(flux_data)-0.0007, min(flux_data)+0.0007)  # Set y-limits; adjust the factor as needed
    axins.set_title("Zoom at Transit")

    plt.show()

    return fig, ax



