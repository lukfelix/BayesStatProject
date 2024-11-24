import numpy as np
import batman
import matplotlib.pyplot as plt
import pathlib
import os

# Initializes the parameters for the transit simulation with the batman package
def initialize_parameters(truths, fixed):
    
    # Define parameters for the transit simulation
    params = batman.TransitParams()       
    params.t0 = fixed['t0']                          # time of inferior conjunction
    params.per = fixed['period']                     # orbital period (in days)
    params.rp = truths['ps']                         # planet-to-star radius ratio = planet radius (in units of stellar radii)
    params.a = fixed['a']                            # semi-major axis in stellar radii
    params.inc = fixed['inc']                        # orbital inclination in degrees
    params.ecc = fixed['ecc']                        # eccentricity
    params.w = fixed['omega']                        # longitude of periastron (in degrees)
    params.u = [truths['u1'], truths['u2']]          # limb-darkening coefficients (no limb-darkening)
    params.limb_dark = fixed['limb_dark_model']      # limb-darkening model

    # Time array for the simulation
    t = np.linspace(fixed['t_min'], 
                    fixed['t_max'], 
                    fixed['n_points'])    # time from t_min to t_max days, n points

    return params, t

def initialize_model(params, time, ncpu=1):
    """
    Initialize a batman model using the given parameters and time array.
    ncpu=1  : number of cores available for batman (set to 1 if you either want to be slow or don't have openMP)
    """
    model = batman.TransitModel(params, time, nthreads = int(ncpu))    #initializes model for the simulation
    # Generate data
    flux_data = simulate_light_curve(model, params)    # Simulate the light curve using the batman model and the parameters to generate the data
    time_data = time                                   # Time array for the simulation

    return model, time_data, flux_data

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
    
    # dynamically adapt alpha value to get around the random sorting of dicts that ,essed up coloring...
    # (TODO: there (hopefully) is a better solution to this, but this was simpler for now)
    minalpha, maxalpha = 0.1, 0.8
    err_list = [int(key.replace(' ppm', '')) for key in list(all_errors_dic.keys())]
    minerr, maxerr = min(err_list), max(err_list)
    
    # Create main plot
    fig, ax = plt.subplots(figsize=plt_size)
    ax.plot(time_data, flux_data, color='black', label="Simulated Light Curve", lw=0.5)  # plot the simulated light curve
    
    # Create inset (zoomed view)
    zoom_range = 0.05
    axins = inset_axes(ax, width="30%", height="30%", loc='lower right', borderpad=2)
    axins.plot(time_data, flux_data, color='black', lw=0.5)

    for i, (key, error) in enumerate(all_errors_dic.items()):
        alpha = minalpha + (maxalpha-minalpha)*(np.log(maxerr) - np.log(error[0]*1e6))/np.log(maxerr)

        ax.fill_between(time_data, flux_data - error, flux_data + error, alpha=alpha, label=key, color='forestgreen')
        axins.fill_between(time_data, flux_data - error, flux_data + error, alpha=alpha, color='forestgreen')

    ax.set_xlabel("Time from Mid-Transit (days)")
    ax.set_ylabel("Relative Flux")
    ax.set_title("Simulated Transit Light Curve with Error Envelopes")
    ax.legend()

    
    

    axins.set_xlim(-zoom_range, zoom_range)
    axins.set_ylim(min(flux_data)-0.0012, min(flux_data)+0.0012)  # Set y-limits; adjust the factor as needed
    axins.set_title("Zoom at Transit")

    plt.show()

    return fig, ax

def save_simdata(time, flux, error_dict, name):
    # Create the directory
    output_dir = pathlib.Path("outputs/data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save time data, flux data and error envelopes
    for key, error_data in error_dict.items():
        filename = 'simdata_' + name + '_' + key.replace(" ", "_") + ".npy"  # e.g., "simdata_quadratic_1_ppm.npy"
        np.save(output_dir / filename, np.array([time, flux, error_data]))
    return

def load_single_simdata(filename):
    """Loads one single data file into the three equal length arrays"""
    [time, flux, err] = np.load('outputs/data/' + filename)
    return time, flux, err

def load_simdata_all_errs(name):
    """
    loads time, flux and errordict in same format as used in analysis.py 
    """
    time, flux, err = {}, {}, {}
    
    available_files = os.listdir('outputs/data' )
    for file in available_files:
        if ('simdata_' + name + '_' in file) and (file.endswith('_ppm.npy')):
            loaded_time, loaded_flux, loaded_err = load_single_simdata(file)
            ppms = file.replace('simdata_' + name + '_', '').replace('_ppm.npy', ' ppm')

            time[ppms] = loaded_time
            flux[ppms] = loaded_flux
            err[ppms] = loaded_err

    # TODO: potential safety measure to make sure that only same data arrays have been loaded
    # at the moment just returns the last time / flux arrays, which should be fine unless we 

    # only return errors as dict
    return time[ppms], flux[ppms], err

def get_name_str(truths):
    if truths==None:
        return None
    return "_".join([f"{key}_{str(value).replace('[', '').replace(']', '').replace(', ', '_').replace(' ', '_')}" for key, value in truths.items()])