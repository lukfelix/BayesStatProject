#%%

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import batman
import os

# Import predefined functions from other files
from simulation_functions import *          # functions for simulating the light curve
from model_functions import *               # functions for evaluation of the model
from mcmc_functions import *                # functions used for the MCMC analysis
from check_convergence import *             # functions used for checking convergence
from TESS_functions import *                # functions used for TESS data preprocessing and plotting

    

#%%
################################################################################
#################### TESS DATA PREPROCESSING & CLEANING ########################
################################################################################

# Planet Name: 					HD-209458b

# Load the data from the csv file
data = np.genfromtxt('outputs/data_TESS/data.csv', delimiter=',')
time_data, flux_data, flux_err_data = data[1:, 0], data[1:, 1], data[1:, 2] # Skip the header & extract the relevant columns

# Clean the data (remove NaN values)
time_data_noNaN, flux_data_noNaN, flux_err_noNaN = clean_data(time_data, flux_data, flux_err_data)

# Plot the cleaned data with errors
plt.figure(figsize=(15, 10))
plt.errorbar(time_data_noNaN, flux_data_noNaN, yerr=flux_err_noNaN, fmt='.', color='orange', alpha=0.3, label="Flux Errors", markersize=1)
plt.plot(time_data_noNaN, flux_data_noNaN,'.', label="Observed Flux", color="black", markersize=2)
plt.ylabel("Flux [e-/s]")
plt.xlabel("Time [days]")
plt.title("TESS DATA: HD-209458b")
plt.legend()
plt.show()

fig, ax = plot_single_light_curve_with_zoom(time_data_noNaN, flux_data_noNaN, flux_err_noNaN, plt_size=(15, 10),zoom_range=(2826.45, 2827.1))


#%%

# if already preprocessed, plot the data before and after preprocessing
# Plot the cleaned data with errors
plt.figure(figsize=(15, 5))
# plt.errorbar(time_data_noNaN, flux_data_noNaN, yerr=flux_err_noNaN, fmt='.', color='orange', alpha=0.3, label="Flux Errors", markersize=1)
plt.plot(time_data_noNaN, flux_data_noNaN,'.', label="Observed Flux", color="black", markersize=0.8)
plt.ylabel("Flux [e-/s]", fontsize=17)
plt.xlabel("Time [days]", fontsize=17)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title("TESS DATA - HD-209458b - Before Preprocessing", fontsize=20)
# plt.legend(loc='best', fontsize=20)
plt.show()

if os.path.exists("outputs/data_TESS/FINAL_time_data.npy"):

    x_final = np.load("outputs/data_TESS/FINAL_time_data.npy")
    y_final = np.load("outputs/data_TESS/FINAL_flux_data.npy")

    plt.figure(figsize=(15, 5))
    # plt.errorbar(time_data_noNaN, flux_data_noNaN, yerr=flux_err_noNaN, fmt='.', color='orange', alpha=0.3, label="Flux Errors", markersize=1)
    plt.plot(x_final, y_final,'.', label="Observed Flux", color="black", markersize=0.8)
    plt.ylabel("Relative Flux", fontsize=17)
    plt.xlabel("Time [days]", fontsize=17)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.title("TESS DATA - HD-209458b - After Preprocessing", fontsize=20)
    # plt.legend(loc='best', fontsize=20)
    plt.show()


#%%


# Normalize the time data & flux data & err data for easier post-processing
time_data_norm_noNaN = time_data_noNaN - time_data_noNaN[0]

min_val = np.min(flux_data_noNaN) + 0.01 # Add a small offset to avoid division by zero
max_val = np.max(flux_data_noNaN)

print("Min value:", min_val)
print("Max value:", max_val)

flux_data_norm_noNaN = (flux_data_noNaN - min_val) / (max_val - min_val)
flux_err_norm_noNaN = flux_err_noNaN / (max_val - min_val)

# # normalize by the average flux
# normalization_factor = np.mean(flux_data_noNaN)
# flux_data_norm_noNaN = flux_data_noNaN / normalization_factor
# flux_err_norm_noNaN = flux_err_noNaN / normalization_factor


# Plot the cleaned data, zoom_range=(2826.65, 2826.92)
fig, ax = plot_single_light_curve_with_zoom(time_data_norm_noNaN, flux_data_norm_noNaN, flux_err_norm_noNaN, plt_size=(15, 8),zoom_range=(5.7, 7.3))

# Save the plot
output_plot_dir = pathlib.Path("outputs/plots_TESS")
tess_data_plot_name = output_plot_dir / "tess_data_plot.png"
if not os.path.exists(tess_data_plot_name):
    fig.savefig(tess_data_plot_name, dpi=300)



#%%
# This is the original data with NaN values but normalized: 
time_data_norm = time_data - time_data[0]
flux_data_norm = (flux_data - min_val) / (max_val - min_val)
flux_err_norm = flux_err_data / (max_val - min_val)

# Check for NaN values only for the first three columns
if np.isnan(flux_err_norm).any():
    # Count the number of NaN entries
    nan_count = np.isnan(flux_err_norm).sum()
    print(f"Number of NaN entries: {nan_count}")
else:
    print("No NaN entries found.")

# Check if everything still makes sense and no additional NaN values are present
assert len(flux_data_norm)-len(flux_data_norm_noNaN) == np.isnan(flux_err_norm).sum()

# %%
# Apparently the data contains some jumps (not the NaN values, but actual jumps also for the time). first need to deal with this: 
import sys
np.set_printoptions(threshold=sys.maxsize)

# Step 0: Check that time data is mononically increasing
assert np.all(np.diff(time_data_norm) > 0)

# Step 1: Compute the differences between consecutive time points
# Define the threshold
threshold = 0.003  # Detect increments larger than this value

# Compute the differences between consecutive time points
time_diffs = np.diff(time_data_norm)

# Detect where increments are larger than the threshold
large_jump_indices = np.where(time_diffs > threshold)[0]  # Indices of large jumps

# print("Time differences:", time_diffs)
print("Indices where increments are larger than threshold:", large_jump_indices)
print("Positions in time array with large jumps:", time_data_norm[large_jump_indices])

#print the size of the jumps
print("Size of the jumps:", time_diffs[large_jump_indices])


# Step 2: Calculate the average size of time increments
# Calculate the average size of time increments
average_time_increment = np.mean(time_diffs[:2021])
print("Average time increment:", average_time_increment)

# Step 3: Correct the time data by replacing the large jumps with the average time increment    
# Manually replace each of the three gaps with the average time increment

#################################################### 1st gap
# Replace the first gap
gap_1 = time_data_norm[4657] - time_data_norm[4656]  # Size of the gap
num_intervals_1 = int(gap_1 // average_time_increment)  # Number of intervals that fit
adjusted_interval_1 = gap_1 / (num_intervals_1 + 1)

# Generate new points with the adjusted interval
new_points_1 = np.linspace(time_data_norm[4656] + adjusted_interval_1, time_data_norm[4657] - adjusted_interval_1, num_intervals_1)

# Insert new points into the array
time_data_norm_longer_1 = np.insert(time_data_norm, 4657, new_points_1)

print("Resulting Array:", time_data_norm_longer_1[4657] - time_data_norm_longer_1[4656])
print("Length of original Array:", len(time_data_norm))
print("Length of new Array:", len(time_data_norm_longer_1))

# Create an array of NaNs for the new points
nan_array = np.full(len(new_points_1), np.nan)

# Insert NaNs into the data array
flux_data_norm_longer_1 = np.insert(flux_data_norm, 4657, nan_array)
flux_err_norm_longer_1 = np.insert(flux_err_norm, 4657, nan_array)

print("Length of original flux Array:", len(flux_data_norm))
print("Length of new flux Array:", len(flux_data_norm_longer_1))

# check if it worked
time_diffs = np.diff(time_data_norm_longer_1)
large_jump_indices = np.where(time_diffs > threshold)[0]  # Indices of large jumps
print("Indices where increments are larger than threshold:", large_jump_indices)


#################################################### 2nd gap
# Replace the second gap
gap_2 = time_data_norm_longer_1[9544] - time_data_norm_longer_1[9543]  # Size of the gap
num_intervals_2 = int(gap_2 // average_time_increment)  # Number of intervals that fit
adjusted_interval_2 = gap_2 / (num_intervals_2 + 1)

# Generate new points with the adjusted interval
new_points_2 = np.linspace(time_data_norm_longer_1[9543] + adjusted_interval_2, time_data_norm_longer_1[9544] - adjusted_interval_2, num_intervals_2)

# Insert new points into the array
time_data_norm_longer_2 = np.insert(time_data_norm_longer_1, 9544, new_points_2)

print("Resulting Array:", time_data_norm_longer_2[9544] - time_data_norm_longer_2[9543])
print("Length of original Array:", len(time_data_norm_longer_1))
print("Length of new Array:", len(time_data_norm_longer_2))

# Create an array of NaNs for the new points
nan_array = np.full(len(new_points_2), np.nan)

# Insert NaNs into the data array
flux_data_norm_longer_2 = np.insert(flux_data_norm_longer_1, 9544, nan_array)
flux_err_norm_longer_2 = np.insert(flux_err_norm_longer_1, 9544, nan_array)

print("Length of original flux Array:", len(flux_data_norm_longer_1))
print("Length of new flux Array:", len(flux_data_norm_longer_2))

# check if it worked
time_diffs = np.diff(time_data_norm_longer_2)
large_jump_indices = np.where(time_diffs > threshold)[0]  # Indices of large jumps
print("Indices where increments are larger than threshold:", large_jump_indices)


#################################################### 3rd gap
# Replace the third gap
gap_3 = time_data_norm_longer_2[14847] - time_data_norm_longer_2[14846]  # Size of the gap

num_intervals_3 = int(gap_3 // average_time_increment)  # Number of intervals that fit
adjusted_interval_3 = gap_3 / (num_intervals_3 + 1)

# Generate new points with the adjusted interval
new_points_3 = np.linspace(time_data_norm_longer_2[14846] + adjusted_interval_3, time_data_norm_longer_2[14847] - adjusted_interval_3, num_intervals_3)

# Insert new points into the array
time_data_norm_longer_3 = np.insert(time_data_norm_longer_2, 14847, new_points_3)

print("Resulting Array:", time_data_norm_longer_3[14847] - time_data_norm_longer_3[14846])
print("Length of original Array:", len(time_data_norm_longer_2))
print("Length of new Array:", len(time_data_norm_longer_3))

# Create an array of NaNs for the new points
nan_array = np.full(len(new_points_3), np.nan)

# Insert NaNs into the data array
flux_data_norm_longer_3 = np.insert(flux_data_norm_longer_2, 14847, nan_array)
flux_err_norm_longer_3 = np.insert(flux_err_norm_longer_2, 14847, nan_array)


print("Length of original flux Array:", len(flux_data_norm_longer_2))
print("Length of new flux Array:", len(flux_data_norm_longer_3))

# check if it worked
time_diffs = np.diff(time_data_norm_longer_3)
large_jump_indices = np.where(time_diffs > threshold)[0]  # Indices of large jumps
print("Indices where increments are larger than threshold:", large_jump_indices)


NEW_time_data_norm = time_data_norm_longer_3
NEW_flux_data_norm = flux_data_norm_longer_3
NEW_flux_err_norm = flux_err_norm_longer_3


x_time = time_data_norm_longer_3
y_flux = flux_data_norm_longer_3
y_err = flux_err_norm_longer_3

np.save("outputs/data_TESS/NEW_time_data_norm.npy", NEW_time_data_norm)
np.save("outputs/data_TESS/NEW_flux_data_norm.npy", NEW_flux_data_norm)
np.save("outputs/data_TESS/NEW_flux_err_norm.npy", NEW_flux_err_norm)

# Plot the results
plt.figure(figsize=(15, 6))
plt.plot(x_time, y_flux,'.', label="Observed Flux / GP Fitted FLux", color="blue", markersize=2)
plt.ylabel("Flux")
plt.title("Observed Flux divided by Gaussian Process Flux")
plt.legend()
plt.show()

      


#%%
################################################################################
################## TESS DATA - GAUSSIAN PROCESS FIT to GAPS ####################
################################################################################

""" 
Before starting here a short summary: 
- 'time_data_norm_noNaN', 'flux_data_norm_noNaN' and 'flux_err_norm_noNaN' are the normalized data without NaN values.
- 'time_data_norm', 'flux_data_norm' and 'flux_err_norm' are the normalized data with NaN values.
- min_val and max_val are the minimum and maximum values of the flux data which we can use to normalize back (we have no need to normalize back the time data)
"""

from tinygp import GaussianProcess, kernels
import jax.numpy as jnp


fig, ax = plot_simple_light_curve(NEW_time_data_norm, NEW_flux_data_norm, plt_size=(10, 5))

# Now the goal here is to first save all NaN entries (for which we want to do the GP fit). Then, we want to set those values to 0 and then discard all TRANSITS from the dataset (including those NaN values which are now 0). Then we do the Gaussian Process fit to the gaps and then we can predict the missing values and again put back in the transits.

# Save the NaN entries
# Step 1: Detect NaN values and save their indices
original_NaN_indices = np.where(np.isnan(NEW_flux_data_norm))[0]
print(len(original_NaN_indices))

# Step 2: Set NaN values to 0
NEW_flux_data_norm[original_NaN_indices] = -0.5
fig, ax = plot_simple_light_curve(NEW_time_data_norm, NEW_flux_data_norm, plt_size=(10, 5))


# Step 3: Discard all transits & 0 values from the dataset (including the NaN values which are now 0)
# First:  Discard the transits by first determining the indice of the "middle transit time", then determine all time corresponding to the transit and then discard them and also discard a bit more to be sure

# Define the transit time

# fig, ax = plt_simple_light_curve(NEW_time_data_norm[(1000+2530*7):(1200+2530*7)], NEW_flux_data_norm[(1000+2530*7):(1200+2530*7)], plt_size=(10, 5))

# transit one: 1.45 - 1.59
# transit two: 4.97 - 5.12
# transit three: 8.49 - 8.64
# transit four: 12.02 - 12.17
# transit five: 15.54 - 15.69
# transit six: 19.07 - 19.22
# transit seven: 22.59 - 22.74
# transit eight: 26.12 - 26.27

# Transit intervals
transit_intervals = [
    (1.45, 1.59),
    (4.97, 5.12),
    (8.49, 8.64),
    (12.02, 12.17),
    (15.54, 15.69),
    (19.07, 19.22),
    (22.59, 22.74),
    (26.12, 26.27),
]
cleaned_time_data = NEW_time_data_norm
cleaned_flux_data = NEW_flux_data_norm

transit_indices = []
# Set flux to -0.5 for transit intervals
for start, end in transit_intervals:
    # Find indices where time is within the interval
    indices_transit = np.where((cleaned_time_data >= start) & (cleaned_time_data <= end))[0]
    transit_indices.append(indices_transit)
    # Update flux values at these indices
    cleaned_flux_data[indices_transit] = -0.9

# Find the indices of the transits
fig, ax = plot_simple_light_curve(cleaned_time_data, cleaned_flux_data, plt_size=(10, 5))

#%%

# original_NaN_indices is the list of all indices where we did not have a data entry originally
# indices_transit is the list of all indices where we have a transit occuring

# Filter the array
filtered_flux = cleaned_flux_data[cleaned_flux_data <= 0.5]

# Get indices of values above the threshold
indices_stellar_flux = np.where(cleaned_flux_data > 0.5)[0]


# Plot only the stellar flux: 
fig, ax = plot_simple_light_curve(cleaned_time_data[indices_stellar_flux], cleaned_flux_data[indices_stellar_flux], plt_size=(12, 3))



# cleaned_time_data[indices_stellar_flux]  &  cleaned_flux_data[indices_stellar_flux is now the dataset only with the stellar flux. 
# cleaned_time_data is the time range for which we want to have all stellar flux values. 

print(len(cleaned_time_data))
print(len(cleaned_time_data[indices_stellar_flux]))

#%%

# Assume cleaned_time_data, cleaned_flux_data, cleaned_flux_err, and indices_stellar_flux are already loaded

# Extract the data for stellar flux
time_stellar_flux = cleaned_time_data[indices_stellar_flux]
flux_stellar_flux = cleaned_flux_data[indices_stellar_flux]
flux_err_stellar_flux = NEW_flux_err_norm[indices_stellar_flux]

# Extend data
extend = 1300

# extend extended time data
interval = average_time_increment
start_extension = np.linspace(cleaned_time_data[0] - extend * interval,cleaned_time_data[0] - interval,num=extend)
end_extension = np.linspace(cleaned_time_data[-1] + interval,cleaned_time_data[-1] + extend * interval,num=extend)
print("Length of start_extension:", len(start_extension))
print("Length of end_extension:", len(end_extension))
extended_time_data = np.concatenate([start_extension, cleaned_time_data, end_extension])

start_extension_obs = np.linspace(time_stellar_flux[0] - extend * interval,time_stellar_flux[0] - interval,num=extend)
end_extension_obs = np.linspace(time_stellar_flux[-1] + interval, time_stellar_flux[-1] + extend * interval,num=extend)
extended_time_data_obs = np.concatenate([start_extension_obs, time_stellar_flux, end_extension_obs])

# extend flux data
mean_flux = np.mean(flux_stellar_flux)
std_flux = np.sqrt(np.var(flux_stellar_flux))  # Standard deviation
synthetic_left_flux = np.random.normal(loc=mean_flux, scale=std_flux, size=extend)
synthetic_right_flux = np.random.normal(loc=mean_flux, scale=std_flux, size=extend)
extended_flux_data = np.concatenate([synthetic_left_flux, flux_stellar_flux, synthetic_right_flux])

# extend error data
mean_flux_err = np.mean(flux_err_stellar_flux)
synthetic_flux_err = np.full(extend, mean_flux_err)
extended_flux_err = np.concatenate([synthetic_flux_err, flux_err_stellar_flux, synthetic_flux_err])


# Plot extended solar flux data
plt.figure(figsize=(15, 4))
plt.plot(extended_time_data_obs, extended_flux_data, ".", label="Extended Data", color="green", markersize=2)
plt.plot(time_stellar_flux, flux_stellar_flux, ".", label="Observed Data", color="black", markersize=2)
plt.ylabel("Relative Flux", fontsize=14)
plt.xlabel("Time [days]", fontsize=14)
plt.title("Extended Stellar Flux Data", fontsize=16)
plt.legend(fontsize=16, loc="lower left", markerscale=3)
plt.show()

#%%

# Define the kernel: a combination of periodic and squared exponential kernels
kernel = (
    1.0 * kernels.ExpSquared(scale=4.0)  # Smooth variation         # 1.0, 5.0
    + 0.5 * kernels.ExpSineSquared(scale=10.0, gamma=0.5)  # Periodic variation
    + 0.3 * kernels.RationalQuadratic(scale=2.0, alpha=0.6)  # Adaptive smoothness, best so far: 0.3, 2.0, 0.6
)
slope = 0.01
mean_func = lambda x: jnp.mean(flux_stellar_flux) + slope * x

# Define the Gaussian Process model
gp = GaussianProcess(kernel, extended_time_data_obs, diag=extended_flux_err**2, mean=mean_func)

# Compute the GP conditioned on the data
gp_conditioned = gp.condition(y=extended_flux_data, X_test=extended_time_data).gp

mu, std = gp_conditioned.loc, jnp.sqrt(gp_conditioned.variance)

#%%
# use rolling average to smooth out the Gaussian Fit
# Define a rolling average function
def rolling_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='same')

# Smooth the GP mean (mu) using a rolling average
window_size = 500  # Adjust this to control the smoothness
smoothed_mu = rolling_average(mu, window_size)


# %%
# Plot the results
plt.figure(figsize=(15, 4))
plt.plot(extended_time_data_obs, extended_flux_data, ".", label="Extended Data", color="green", markersize=2)
plt.plot(time_stellar_flux, flux_stellar_flux,'.', label="Observed Data", color="black", markersize=2)
# plt.fill_between(extended_time_data, mu-std, mu+std, color = "red", alpha=0.5, label="GP Uncertainty")
plt.plot(extended_time_data, mu,'.', alpha=0.5, label="GP fit", color="red", markersize=3)
plt.plot(extended_time_data[500:-500], smoothed_mu[500:-500], alpha=1.0, label="GP fit averaged", color="yellow")
plt.ylabel("Relative Flux", fontsize=14)
plt.xlabel("Time [days]", fontsize=14)
plt.title("Extended Stellar Flux Data with Gaussian Process Fit", fontsize=16)
plt.legend(fontsize=16, loc="lower right", markerscale=3, ncol=2)
plt.show()

#%%

print(len(mu))

flux = smoothed_mu[1300:-1300]
print(len(flux))
print(len(cleaned_time_data))

flux_with_gaps = flux[indices_stellar_flux]
print(len(flux_with_gaps))

print(len(flux_stellar_flux))

# print(flux_stellar_flux/flux_with_gaps)

#transit points

print(np.average(flux_stellar_flux/flux_with_gaps))

# Plot the results
plt.figure(figsize=(15, 4))
# plt.plot(time_stellar_flux[7400:-7500], (flux_with_gaps/flux_stellar_flux)[7400:-7500],'.', label="Observed Flux / GP Fitted FLux", color="blue", markersize=2)
plt.plot(time_stellar_flux, (flux_stellar_flux/flux_with_gaps),'.', label="Observed Flux / GP Fit Avg ", color="black", markersize=2)
plt.axhline(y=np.average(flux_stellar_flux/flux_with_gaps), color='r', linestyle='--', label="Mean", linewidth=2)
plt.ylabel("Relative Flux", fontsize=14)
plt.xlabel("Time [days]", fontsize=14)
plt.title("Observed Flux nromalized by Gaussian Process Fit", fontsize=16)
plt.legend(fontsize=16, loc="best", markerscale=3, ncol=2)
plt.show()

#%%


x_time = np.load("outputs/data_TESS/NEW_time_data_norm.npy")
y_flux = np.load("outputs/data_TESS/NEW_flux_data_norm.npy")
y_err = np.load("outputs/data_TESS/NEW_flux_err_norm.npy")

# Plot the results
plt.figure(figsize=(15, 6))
plt.plot(x_time, y_flux,'.', label="Observed Flux / GP Fitted FLux", color="blue", markersize=2)
plt.ylabel("Flux")
plt.title("Observed Flux divided by Gaussian Process Flux")
plt.legend()
plt.show()


# time
final_time_data = x_time
print(len(final_time_data))

# flux
final_flux_data = y_flux
final_flux_data[indices_stellar_flux] = flux_with_gaps/flux_stellar_flux

for i in range (len(transit_indices)):
    final_flux_data[transit_indices[i]] = y_flux[transit_indices[i]] / flux[transit_indices[i]]

# flux_err
final_flux_err_data = y_err

# Plot the results
plt.figure(figsize=(15, 6))
plt.plot(final_time_data, final_flux_data, label="Observed Flux / GP Fitted FLux", color="blue")
plt.ylabel("Flux")
plt.title("Observed Flux divided by Gaussian Process Flux")
plt.legend()
plt.show()

#%%

# Plot the cleaned data, zoom_range=(2826.65, 2826.92)
fig, ax = plot_single_light_curve_with_zoom(final_time_data, final_flux_data, final_flux_err_data, plt_size=(20, 10),zoom_range=(1.4, 1.65))


print(len(np.where(np.isnan(final_flux_data))[0]))

print(len(final_time_data))
# %%
# time --> 12.75 - 13.4 


# FILTER OUT THE FINAL TIME: we do not want.

indices_good_flux = np.where((final_time_data < 12.75) | (final_time_data > 13.4))[0]

print(len(indices_good_flux))

fig, ax = plot_single_light_curve_with_zoom(final_time_data[indices_good_flux], final_flux_data[indices_good_flux], final_flux_err_data[indices_good_flux], plt_size=(20, 10),zoom_range=(1.4, 1.65))

print(np.average(final_flux_data[indices_good_flux]))



#%%
# Need to "unnormalize" the data to put it back into the original scale
FINAL_time_data = final_time_data
FINAL_flux_data = final_flux_data * (max_val - min_val) + min_val
FINAL_flux_err_data = final_flux_err_data * (max_val - min_val)

# Plot the results
plt.figure(figsize=(15, 6))
plt.plot(FINAL_time_data, FINAL_flux_data, label="Unnormlized Flux", color="blue")
plt.ylabel("Flux")
plt.title("Observed Flux divided by Gaussian Process Flux")
plt.legend()
plt.show()

#%%
print(len(FINAL_time_data))
print(len(NEW_flux_data_norm))
# now normalize the data such that the average stellar flux is 1
mean_stellar_flux = np.mean(FINAL_flux_data[indices_stellar_flux])

FINAL_flux_data = FINAL_flux_data / mean_stellar_flux
FINAL_flux_err_data = FINAL_flux_err_data / mean_stellar_flux

#%%


# Create a mask for all indices
mask = np.ones_like(final_time_data, dtype=bool)
mask[indices_good_flux] = False  # Mark good stellar indices as False

# Set all non-good indices to NaN
FINAL_flux_data[mask] = np.nan
FINAL_flux_err_data[mask] = np.nan


# # FINAL DATA is wheen we discard this small patch that is not so pretty.
# FINAL_time_data = final_time_data[indices_good_flux]
# FINAL_flux_data = FINAL_flux_data[indices_good_flux]
# FINAL_flux_err_data = FINAL_flux_err_data[indices_good_flux]

FINAL_time_data = final_time_data
FINAL_flux_data = FINAL_flux_data
FINAL_flux_err_data = FINAL_flux_err_data


# Plot the results
plt.figure(figsize=(15, 6))
plt.plot(FINAL_time_data, FINAL_flux_data, label="Unnormlized Flux", color="blue")
plt.ylabel("Flux")
plt.title("Observed Flux divided by Gaussian Process Flux")
plt.legend()
plt.show()


#%%

#check if there are NaN values
print(len(np.where(np.isnan(FINAL_flux_data))[0]))
print(len(final_time_data)-len(FINAL_flux_data))

#%%
########################################################

########################################################

########################################################


## FINISHED PREPROCESSING - SAVING THE PREPROCESSED TESS DATA - ##

########################################################

########################################################

########################################################


# Save the preprocessed data
np.save("outputs/data_TESS/FINAL_time_data.npy", FINAL_time_data)
np.save("outputs/data_TESS/FINAL_flux_data.npy", FINAL_flux_data)
np.save("outputs/data_TESS/FINAL_flux_err_data.npy", FINAL_flux_err_data)

#%%
