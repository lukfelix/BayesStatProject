
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# Function to preprocess and clean the TESS data
def clean_data(time_data, flux_data, flux_err_data):

    data = np.column_stack((time_data, flux_data, flux_err_data))

    # Check for NaN values only for the first three columns
    if np.isnan(data).any():
        # Count the number of NaN entries
        nan_count = np.isnan(data).sum()
        print(f"Number of NaN entries: {nan_count}")
    else:
        print("No NaN entries found.")

    # clean the data from NaN values 
    data = data[~np.isnan(data).any(axis=1)] # Finds rows without NaN and keeps only those rows
    print("New shape of dataset: ", data.shape)

    # Check for NaN values only for the first three columns
    if np.isnan(data).any():
        return "Error: NaN entries still present in the data."
    else:
        print("Succesfully removed all NaN entries.")

    return data[:,0], data[:,1], data[:,2]



#Plotting the TESS data

# Function to plot the simulated light curve with a zoomed-in inset
def plot_simple_light_curve(time_data, flux_data, plt_size=(10, 5)):
    
    # Create main plot
    fig, ax = plt.subplots(figsize=plt_size)

    # Overlay the main line
    ax.plot(time_data, flux_data, linestyle='-', linewidth=0.5, color='black', label='Flux')

    # Customize appearance
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Flux", fontsize=12)
    ax.set_title("Flux vs Time with Error Bars", fontsize=14)
    ax.legend(fontsize=10)

    # Tweak axes for better presentation
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(visible=True, linestyle='--', alpha=0.5)  # Add a subtle grid for reference

    # Save the figure for publication
    plt.tight_layout()  # Adjust layout to avoid overlap
    ax.legend()

    plt.show()

    return fig, ax


# Function to plot the simulated light curve with a zoomed-in inset
def plot_single_light_curve_with_zoom(time_data, flux_data, flux_err_data, plt_size=(10, 5), zoom_range=(2825, 2830)):

    # Create main plot
    fig, ax = plt.subplots(figsize=plt_size)

    # Plot the flux with error bars
    ax.errorbar(
        time_data, flux_data, yerr=flux_err_data, fmt='o', capsize=1, capthick=0.5, elinewidth=0.5, 
        color='orange', label='Flux with Errors', markersize=1, alpha=0.6
    )

    # Overlay the main line
    ax.plot(time_data, flux_data, linestyle='-', linewidth=1.5, color='black', label='Flux')

    # Customize appearance
    ax.set_xlabel("Time [days]", fontsize=12)
    ax.set_ylabel("Flux", fontsize=12)
    ax.set_title("Flux vs Time with Error Bars", fontsize=14)
    ax.legend(fontsize=10)

    # Tweak axes for better presentation
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax.grid(visible=True, linestyle='--', alpha=0.9)  # Add a subtle grid for reference

    # Create zoomed-in inset
    axins = inset_axes(ax, width="100%", height="100%", bbox_to_anchor=(0.57, 0.05, 0.40, 0.40), 
                       bbox_transform=ax.transAxes, loc="lower right")  # Adjust position and size of inset
    axins.errorbar(
        time_data, flux_data, yerr=flux_err_data, fmt='o', capsize=2, capthick=1, elinewidth=1, 
        color='orange', markersize=2, alpha=1.0
    )
    axins.plot(time_data, flux_data, linestyle='-', linewidth=1.5, color='black')

    # Set limits for zoom-in on the first transit
    axins.set_xlim(zoom_range[0], zoom_range[1])
    axins.set_ylim(
        np.min(flux_data[(time_data >= zoom_range[0]) & (time_data <= zoom_range[1])]) - 0.002, 
        np.max(flux_data[(time_data >= zoom_range[0]) & (time_data <= zoom_range[1])]) + 0.002
    )

    # Remove ticks for inset for simplicity
    axins.tick_params(axis='both', which='major', labelsize=8)
    axins.grid(visible=True, linestyle='--', alpha=0.5)  # Add a subtle grid for reference
    axins.set_title("Zoomed-In View of First Transit", fontsize=10)

    # Highlight zoom range on main plot
    ax.indicate_inset_zoom(axins, edgecolor="red", lw=3)

    # Save the figure for publication
    plt.tight_layout()  # Adjust layout to avoid overlap
    plt.show()

    return fig, ax



########### FUNCTIONS FOR GAUSSIAN PROCESS REGRESSION ############








