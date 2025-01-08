

import numpy as np
import matplotlib.pyplot as plt

from simulation_functions import load_simdata_all_errs, initialize_parameters, initialize_model
from model_functions import full_model

import dynesty
from dynesty import plotting as dyplot
import matplotlib.pyplot as plt
import corner

import matplotlib.lines as mlines



#prepare model
truths = {
    'ps':0.1,                        # planet-to-star radius ratio = planet radius (in units of stellar radii)
    'u1':0,                          # limb-darkening coefficients: u1, 
    'u2':0,                          # u2 (no limb-darkening: u1=0, u2=0)
}

fixed_params = {
    # orbital parameters:
    't0':0,                          # time of inferior conjunction
    'period':1,                      # orbital period (in days)
    'a':4,                           # semi-major axis in stellar radii
    'inc':90,                        # orbital inclination in degrees
    'ecc':0,                         # eccentricity
    'omega':90,                      # longitude of periastron (in degrees)
    'limb_dark_model':"quadratic",   # limb-darkening model
    # simulated observation parameters:
    'n_points':1000,                 # number of points in the light curve
    't_min':-0.25,                   # minimum time in days
    't_max':0.25,                    # maximum time in days
}
# Initialize the parameters accordingly, to match the "syntax" of the batman package
params, t_array = initialize_parameters(truths, fixed_params)

# Initialize the batman model using the parameters from above
model, time_data, flux_data = initialize_model(params, t_array)    #initializes model for the simulation


# Define the error envelopes for the light curve ranging from 1-1000 ppm & store them in a dictionary. 
all_errors_dict = {
    "1 ppm":    (1    / 1e6) * flux_data,       # error envelope for 1 ppm
    "10 ppm":   (10   / 1e6) * flux_data,       # error envelope for 10 ppm
    "30 ppm":   (30   / 1e6) * flux_data,       # error envelope for 30 ppm
    "100 ppm":  (100  / 1e6) * flux_data,       # error envelope for 100 ppm,
    "300 ppm":  (300  / 1e6) * flux_data,       # error envelope for 300 ppm, 
    "1000 ppm": (1000 / 1e6) * flux_data,       # error envelope for 1000 ppm
    }














#prepares the priors to be used with dynesty assuming uniform priors
def prior_transformation(u, *priors):
    
    x = np.array(u)
    
    prior_dict = priors[0]
    
    for key in prior_dict:
        
        a = prior_dict[key][1]
        b = prior_dict[key][2]
        index = prior_dict[key][3]
        
        x[index] = (b-a)*u[index] + a
        
    
    return x

def ptransform_simple(u):
    
    x = np.array(u)
    
    x[0] = (0.5 - 0.0)*u[0] + 0.0
    x[1] = (3.0 - (-3.0))*u[1] - 3.0
    x[2] = (3.0 - (-3.0))*u[2] - 3.0
    return x

def ptransform_kipping(u):
    
    x = np.array(u)
    
    x[0] = (0.5 - 0.0)*u[0] + 0.0
    x[1] = (1.0 - (0.0))*u[1] + 0.0
    x[2] = (1.0 - (-0.0))*u[2] + 0.0
    
    return x


#the log likelihood in a for dynesty usable form
def log_likelihood(theta, t, y, yerr, params, model, transform):
    model_prediction = full_model(theta, params, model, transform)
    return -0.5 * np.sum(((y - model_prediction) / yerr) ** 2)



#logl_args=(t, y, yerr, params, model, priors, transform)
#ptform_args


#dynesty run for quadratic parametrization

#prepare priors for the quadratic parametrization

priors_quadratic = {
    'ps':        ['uni', 0., 0.5, 0],     # stellar radii
    'u1':        ['uni', -3., 3, 1],      # limb darkening
    'u2':        ['uni', -3., 3., 2],     # limb darkening
}

priors_kipping = {
    'ps':        ['uni', 0., 0.5],    # stellar radii
    'u1':        ['uni', 0., 1.],     # limb darkening
    'u2':        ['uni', 0., 1.],     # limb darkening
}


ndim = len(priors_quadratic)

# =============================================================================
# dsampler_quad = dynesty.DynamicNestedSampler(log_likelihood, ptransform_simple, ndim,
#                                         logl_args=(time_data, flux_data, all_errors_dict["1 ppm"], params, model, False),
#                                         )#ptform_args=(priors))
# dsampler_quad.run_nested()
# dresults_quad = dsampler_quad.results
# =============================================================================
#print(prior_transformation([0,1,2],priors))

# =============================================================================
# fg, ax = dyplot.cornerplot(dresults_quad, color='dodgerblue',# truths=np.array([0.1, 0, 0]),
#                            truth_color='black', show_titles=True,
#                            quantiles=None)#, max_n_ticks=3)
# =============================================================================

# =============================================================================
# dsampler_kipping = dynesty.DynamicNestedSampler(log_likelihood, ptransform_kipping, ndim,
#                                         logl_args=(time_data, flux_data, all_errors_dict["10 ppm"], params, model, True),
#                                         )
# dsampler_kipping.run_nested()
# dresults_kipping = dsampler_kipping.results
# =============================================================================


#dsampler_quad.save('outputs/dynesty samples/dsampler_quad_1ppm')
#dsampler_kipping.save('outputs/dynesty samples/dsampler_kipping_10ppm')



# =============================================================================
# # initialize figure
# fig, axes = plt.subplots(3, 7, figsize=(35, 15))
# axes = axes.reshape((3, 7))  # reshape axes
# 
# # add white space
# [a.set_frame_on(False) for a in axes[:, 3]]
# [a.set_xticks([]) for a in axes[:, 3]]
# [a.set_yticks([]) for a in axes[:, 3]]
# 
# # plot initial run (quadratic; left)
# fg, ax = dyplot.cornerplot(dresults_quad, color='blue', truths=np.array([0.1, 0, 0]),
#                            truth_color='black', show_titles=True,
#                            fig=(fig, axes[:, :3]))
# 
# # plot extended run (kipping; right)
# fg, ax = dyplot.cornerplot(dresults_kipping, color='red', truths=np.array([0.1, 0, 0]),
#                            truth_color='black', show_titles=True,
#                            fig=(fig, axes[:, 4:]))
# =============================================================================




# =============================================================================
# log_R = []
# for i in range(100):
#     log_z_quad = dynesty.utils.jitter_run(dresults_quad).logz[-1]
#     log_z_kipping = dynesty.utils.jitter_run(dresults_kipping).logz[-1]
#     log_R.append(log_z_quad - log_z_kipping)
# 
# log_R = np.array(log_R)
# 
# R_mean = np.mean(np.exp(log_R))
# R_std = np.std(np.exp(log_R))
# print(f"Bayes' ratio: {R_mean:.2f} ± {R_std:.2f}")
# =============================================================================



#load samples from file
quad_1 = dynesty.DynamicNestedSampler.restore('outputs/dynesty samples/dsampler_quad_1ppm').results
quad_10 = dynesty.DynamicNestedSampler.restore('outputs/dynesty samples/dsampler_quad_10ppm').results
quad_30 = dynesty.DynamicNestedSampler.restore('outputs/dynesty samples/dsampler_quad_30ppm').results
quad_100 = dynesty.DynamicNestedSampler.restore('outputs/dynesty samples/dsampler_quad_100ppm').results
quad_300 = dynesty.DynamicNestedSampler.restore('outputs/dynesty samples/dsampler_quad_300ppm').results
quad_1000 = dynesty.DynamicNestedSampler.restore('outputs/dynesty samples/dsampler_quad_1000ppm').results

kipping_1 = dynesty.DynamicNestedSampler.restore('outputs/dynesty samples/dsampler_kipping_1ppm').results
kipping_10 = dynesty.DynamicNestedSampler.restore('outputs/dynesty samples/dsampler_kipping_10ppm').results
kipping_30 = dynesty.DynamicNestedSampler.restore('outputs/dynesty samples/dsampler_kipping_30ppm').results
kipping_100 = dynesty.DynamicNestedSampler.restore('outputs/dynesty samples/dsampler_kipping_100ppm').results
kipping_300 = dynesty.DynamicNestedSampler.restore('outputs/dynesty samples/dsampler_kipping_300ppm').results
kipping_1000 = dynesty.DynamicNestedSampler.restore('outputs/dynesty samples/dsampler_kipping_1000ppm').results



# initialize figure
# =============================================================================
# fig, axes = plt.subplots(3, 7, figsize=(35, 15))
# axes = axes.reshape((3, 7))  # reshape axes
# 
# # add white space
# [a.set_frame_on(False) for a in axes[:, 3]]
# [a.set_xticks([]) for a in axes[:, 3]]
# [a.set_yticks([]) for a in axes[:, 3]]
# 
# # plot initial run (quadratic; left)
# fg, ax = dyplot.cornerplot(quad_1000, color='blue', truths=np.array([0.1, 0, 0]),
#                            truth_color='black', show_titles=True, title_fmt='.9f',
#                            fig=(fig, axes[:, :3]))
# 
# # plot extended run (kipping; right)
# fg, ax = dyplot.cornerplot(kipping_1000, color='red', truths=np.array([0.1, 0, 0]),
#                            truth_color='black', show_titles=True, title_fmt='.9f',
#                            fig=(fig, axes[:, 4:]))
# =============================================================================



# =============================================================================
# fig = corner.corner(
#     kipping_1000['samples'],
#     weights=kipping_1000.importance_weights(),
#     title_fmt='.5f',
#     bins=50,
#     show_titles=True,
#     labels=[r"$P_S$", r"$u_1$", r"$u_2$"], # leave this in since we want fancier plots, but at some point go with keys for flexibility
#     # labels = truths.keys(),
#     #truths=[truths['ps'], truths['u1'], truths['u2']],
#     truths=[truths['ps'], None, None],
#     plot_density=True,
#     plot_datapoints=True,
#     fill_contours=False,
#     smooth=True,
#     levels=(0.6827, 0.90, 0.9545),              # shows the 1, 1.5 and 2 sigma contours in the 2D plots
#     quantiles=[0.16, 0.5, 0.84],                # shows the 1 sigma interval in the 1D plots
#     title_kwargs={"fontsize": 10},
#     truth_color='cornflowerblue',
#     range=[(0.095,0.105), (0,0.04), (0,1)]
# )
# #fig.savefig('corner_plot_kipping_1000ppm.png', dpi=300)
# =============================================================================

emcee_quad_1000ppm = np.load('outputs/samples/samples_ps_0.1_u1_0_u2_0_1000 ppm_100000steps.npy')
emcee_kipping_1000ppm = np.load('outputs/samples/samples_ps_0.1_u1_0_u2_None_1000 ppm_100000steps.npy')

emcee_quad_1ppm = np.load('outputs/samples/samples_ps_0.1_u1_0_u2_0_1 ppm_100000steps.npy')
emcee_kipping_1ppm = np.load('outputs/samples/samples_ps_0.1_u1_0_u2_None_1 ppm_100000steps.npy')


# =============================================================================
# fig = corner.corner(
#     emcee_quad_1000ppm,
#     #weights=quad_1000.importance_weights(),
#     weights=0.04*quad_1000.importance_weights().max()*np.ones(len(emcee_quad_1000ppm)),
#     title_fmt='.5f',
#     bins=50,
#     show_titles=True,
#     labels=[r"$P_S$", r"$u_1$", r"$u_2$"], # leave this in since we want fancier plots, but at some point go with keys for flexibility
#     # labels = truths.keys(),
#     truths=[truths['ps'], truths['u1'], truths['u2']],
#     #truths=[truths['ps'], None, None],
#     plot_density=True,
#     plot_datapoints=True,
#     fill_contours=False,
#     #smooth=True,
#     levels=(0.6827, 0.90, 0.9545),              # shows the 1, 1.5 and 2 sigma contours in the 2D plots
#     quantiles=[0.16, 0.5, 0.84],                # shows the 1 sigma interval in the 1D plots
#     title_kwargs={"fontsize": 10},
#     truth_color='cornflowerblue',
#     #range=[(0.095,0.105), (0,0.04), (0,1)]
# )
# 
# 
# fig = corner.corner(
#     quad_1000['samples'],
#     weights=quad_1000.importance_weights(),
#     #plot_density=True,
#     #plot_datapoints=True,
#     #fill_contours=False,
#     #smooth=True,
#     levels=(0.6827, 0.90, 0.9545),              # shows the 1, 1.5 and 2 sigma contours in the 2D plots
#     quantiles=[0.16, 0.5, 0.84],                # shows the 1 sigma interval in the 1D plots
#     #title_kwargs={"fontsize": 10},
#     range=[(0.098,0.102), (-0.5,0.5), (-0.7,0.7)],
#     color='teal',
#     #hist_kwargs={"linestyle": '--'},#, "alpha": 0.1},
#     #alpha=0.1,
#     fig=fig
#     )
# plt.legend(handles=[mlines.Line2D([], [], color='k', label='EMCEE'),
#                     mlines.Line2D([], [], color='teal', label='Dynesty')],
#            loc="upper right",
#            bbox_to_anchor=(1, 2))
# #fig.savefig('double_corner_plot_quad_1000ppm.png', dpi=300)
# =============================================================================




#monkey patch for title setting
import matplotlib.transforms as mtransforms

from types import MethodType


# Matplotlib text doesn't support multiple colors, so we need to perform
# some hacks to place text neatly

# Ideally should probably be implemented as a new Artist class holding
# text objects
def get_neighbor_transform(text, direction="top", npad=0):
    """
    Determines the transform for the _next_ Text plotted with
    set_title_artist. For alignment purposes, requires rendering
    the text and getting its extent.
    """
    text.draw(text.figure.canvas.get_renderer())
    ex = text.get_window_extent()

    if direction == "up":
        x = 0
        y = ex.height + npad
    elif direction == "down":
        x = 0
        y = -ex.height - npad
    elif direction == "right":
        halign = text.get_horizontalalignment()
        if halign == "left":
            x = ex.width + npad
        elif halign == "center":
            x = ex.width / 2 + npad
        elif halign == "right":
            x = 0
        y = 0
    elif direction == "left":
        halign = text.get_horizontalalignment()
        if halign == "left":
            x = 0
        elif halign == "center":
            x = -ex.width / 2 - npad
        elif halign == "right":
            x = -ex.width - npad
        y = 0
    else:
        raise ValueError("loc must be top, bottom, or right")

    return mtransforms.offset_copy(
        text.get_transform(), x=x, y=y,
        fig=text.figure, units='points'
    )

# Determines the attribute of an Axes arist corresponding to the correct
# Text artist for the 'loc' parameter
LOC_ATTR = {
    "center": "title",
    "right": "_right_title",
    "left": "_left_title"
}

def set_title_neighbor(ax, label, loc="center", **kwargs):
    """
    corner.py is inflexible and only outputs the medians and quartiles
    through calling Axes.set_title. We monkey patch in the collation
    behavior using this method, deferring to the original implementation
    whern the user hasn't called init_neighbor_title on that axis and
    title loc.
    """
    neighbor_entry = ax._neighbor_titles.get(loc, None)
    if neighbor_entry is None:
        return ax._orig_set_title(label, loc=loc, **kwargs)

    neighbor_artists = neighbor_entry["artists"]
    direction = neighbor_entry["direction"]
    if not neighbor_artists:
        text = ax._orig_set_title(label, loc=loc, **kwargs)
        neighbor_artists.append(text)
    else:
        base_title = ax.__getattribute__(LOC_ATTR[loc])
        kwargs.pop("horizontalignment", None)
        kwargs.pop("verticalalignment", None)
        if direction == "up" or direction == "down":
            halign = base_title.get_horizontalalignment()
        elif direction == "right":
            halign = "left"
        elif direction == "left":
            halign = "right"
        else:
            raise ValueError("direction must be up, down, right or left")

        text = ax.text(
            *base_title.get_position(), label,
            transform=neighbor_entry["next_transform"],
            horizontalalignment=halign,
            verticalalignment=base_title.get_verticalalignment(),
            fontproperties=base_title.get_fontproperties(),
            **kwargs
        )
        neighbor_artists.append(text)

    neighbor_entry["next_transform"] = get_neighbor_transform(
        text,
        direction=direction,
        npad=neighbor_entry["npad"]
    )

def get_all_title_artists(ax):
    """
    Returns all title artists.
    Useful for the "bbox_extra_artists" parameter to Figure.savefig
    or modifying attributes of all titles.
    """
    _loc_attr = dict(**LOC_ATTR)
    if hasattr(ax, "_neighbor_titles"):
        for loc, neighbor_entry in ax._neighbor_titles.items():
            neighbor_artists = neighbor_entry["artists"]
            if not neighbor_artists:
                continue
            _loc_attr.pop(loc)
            yield from neighbor_artists

    for attr in _loc_attr.values():
        yield ax.__getattribute__(attr)

def init_neighbor_title(ax, loc="center", direction="right", npad=2):
    """
    Initializes the neighbor title behavior, allowing set_title to create
    multiple artists with, say, different colors, by monkey-patching the
    set_title method.
    
    Only affects the title with the loc specified, so if you have both
    a left title and a center title you need to call the method on each.
    direction can be up, down, left, or right and determines where each
    subsequent neighbor will be placed.
    
    npad determines the spacing between each Text artist.
    
    Can be called multiple times to change the direction and npad settings,
    although the changes will only affect future Text artists.
    """
    if getattr(ax.set_title, "func", None) is not set_title_neighbor:
        ax._orig_set_title = ax.set_title
        ax.set_title = MethodType(set_title_neighbor, ax)
        ax.get_all_title_artists = MethodType(get_all_title_artists, ax)
        ax._neighbor_titles = dict()

    neighbor_entry = ax._neighbor_titles.setdefault(loc, dict())
    neighbor_entry["direction"] = direction
    neighbor_entry["npad"] = npad

    neighbor_artists = neighbor_entry.setdefault("artists", list())
    if not neighbor_artists and ax.get_title(loc=loc):
        title = ax.__getattribute__(LOC_ATTR[loc])
        neighbor_artists.append(title)
        title.draw(ax.get_figure().canvas.get_renderer())

    if neighbor_artists:
        neighbor_entry["next_transform"] = get_neighbor_transform(
            neighbor_artists[-1],
            direction=direction,
            npad=npad
        )

#fig = plt.figure(figsize=(8, 8))

# =============================================================================
# fig, axes = plt.subplots(3, 7, figsize=(35, 15))
# axes = axes.reshape((3, 7))  # reshape axes
# 
# # add white space
# [a.set_frame_on(False) for a in axes[:, 3]]
# [a.set_xticks([]) for a in axes[:, 3]]
# [a.set_yticks([]) for a in axes[:, 3]]
# =============================================================================

fig_quad = plt.figure(1, figsize=(8, 8))
fig_kipping = plt.figure(2, figsize=(8, 8))



kwargs_quad = dict(
    title_fmt='.5f',
    #title_fmt='.2E',
    bins=50,
    show_titles=True,
    labels=[r"$P_S$", r"$u_1$", r"$u_2$"],
    truths=[truths['ps'], truths['u1'], truths['u2']],
    plot_density=True,
    plot_datapoints=True,
    fill_contours=False,
    smooth=True,
    levels=(0.6827, 0.90, 0.9545),              # shows the 1, 1.5 and 2 sigma contours in the 2D plots
    quantiles=[0.16, 0.5, 0.84],                # shows the 1 sigma interval in the 1D plots
    #title_kwargs={"fontsize": 10},
    truth_color='g',
    range=[(0.098,0.102), (-0.5,0.5), (-0.7,0.7)]
)

kwargs_kipping = dict(
    title_fmt='.5f',
    #title_fmt='.2E',
    bins=50,
    show_titles=True,
    labels=[r"$P_S$", r"$q_1$", r"$q_2$"],
    truths=[truths['ps'], None, None],
    plot_density=True,
    plot_datapoints=True,
    fill_contours=False,
    smooth=True,
    levels=(0.6827, 0.90, 0.9545),              # shows the 1, 1.5 and 2 sigma contours in the 2D plots
    quantiles=[0.16, 0.5, 0.84],                # shows the 1 sigma interval in the 1D plots
    #title_kwargs={"fontsize": 10},
    truth_color='g',
    range=[(0.096,0.104), (0,0.05), (0,1)]
)

# Ensures the first dataset we plot is the one on top.
primary = dict(
    contour_kwargs=dict(
        zorder=10,
    ),
    contourf_kwargs=dict(
        zorder=10,
    ),
    pcolor_kwargs=dict(
        zorder=10,
    )
)

# Sets both the color of the plotted elements and the
# associated title.
def _color_and_title(color, **title_kwargs):
    return dict(
        color=color,
        title_kwargs=dict(
            color=color,
            **title_kwargs
        )
    )



corner.corner(
    emcee_quad_1000ppm,
    weights=0.018*quad_1000.importance_weights().max()*np.ones(len(emcee_quad_1000ppm)),
    fig=fig_quad,
    **_color_and_title("k", loc="center", fontsize=10),#, pad=10),
    **kwargs_quad,
    #labels=[r"$P_S$", r"$u_1$", r"$u_2$"]
)

for ax in fig_quad.axes:
    init_neighbor_title(ax, loc="center", direction='up')#, npad=1)

corner.corner(
    quad_1000['samples'],
    weights=quad_1000.importance_weights(),
    fig=fig_quad,
    **_color_and_title("b", loc="center", fontsize=10),
    #**primary,
    **kwargs_quad
)
plt.figure(1)
plt.legend(handles=[mlines.Line2D([], [], color='k', label='EMCEE'),
                    mlines.Line2D([], [], color='b', label='Dynesty')],
           loc="upper right",
           bbox_to_anchor=(1, 3))


corner.corner(
    emcee_kipping_1000ppm,
    weights=0.018*kipping_1000.importance_weights().max()*np.ones(len(emcee_kipping_1000ppm)),
    fig=fig_kipping,
    **_color_and_title("k", loc="center", fontsize=10),#, pad=10),
    **kwargs_kipping,
    #labels=[r"$P_S$", r"$u_1$", r"$u_2$"]
)

for ax in fig_kipping.axes:
    init_neighbor_title(ax, loc="center", direction='up')#, npad=1)

corner.corner(
    kipping_1000['samples'],
    weights=kipping_1000.importance_weights(),
    fig=fig_kipping,
    **_color_and_title("r", loc="center", fontsize=10),
    #**primary,
    **kwargs_kipping
)
plt.figure(2)
plt.legend(handles=[mlines.Line2D([], [], color='k', label='EMCEE'),
                    mlines.Line2D([], [], color='r', label='Dynesty')],
           loc="upper right",
           bbox_to_anchor=(1, 3))


#fig_quad.savefig('double_cornerplot_quadratic_1000ppm.png', dpi=320, bbox_inches='tight')
#fig_kipping.savefig('double_cornerplot_kipping_1000ppm.png', dpi=320, bbox_inches='tight')




# =============================================================================
# log_R = []
# for i in range(100):
#     log_z_quad = dynesty.utils.jitter_run(quad_1).logz[-1]
#     log_z_kipping = dynesty.utils.jitter_run(kipping_1).logz[-1]
#     log_R.append(log_z_quad - log_z_kipping)
# log_R = np.array(log_R)
# R_mean_1 = np.mean(np.exp(log_R))
# R_std_1 = np.std(np.exp(log_R))
# 
# 
# log_R = []
# for i in range(100):
#     log_z_quad = dynesty.utils.jitter_run(quad_10).logz[-1]
#     log_z_kipping = dynesty.utils.jitter_run(kipping_10).logz[-1]
#     log_R.append(log_z_quad - log_z_kipping)
# log_R = np.array(log_R)
# R_mean_10 = np.mean(np.exp(log_R))
# R_std_10 = np.std(np.exp(log_R))
# 
# 
# log_R = []
# for i in range(100):
#     log_z_quad = dynesty.utils.jitter_run(quad_30).logz[-1]
#     log_z_kipping = dynesty.utils.jitter_run(kipping_30).logz[-1]
#     log_R.append(log_z_quad - log_z_kipping)
# log_R = np.array(log_R)
# R_mean_30 = np.mean(np.exp(log_R))
# R_std_30 = np.std(np.exp(log_R))
# 
# 
# log_R = []
# for i in range(100):
#     log_z_quad = dynesty.utils.jitter_run(quad_100).logz[-1]
#     log_z_kipping = dynesty.utils.jitter_run(kipping_100).logz[-1]
#     log_R.append(log_z_quad - log_z_kipping)
# log_R = np.array(log_R)
# R_mean_100 = np.mean(np.exp(log_R))
# R_std_100 = np.std(np.exp(log_R))
# 
# 
# log_R = []
# for i in range(100):
#     log_z_quad = dynesty.utils.jitter_run(quad_300).logz[-1]
#     log_z_kipping = dynesty.utils.jitter_run(kipping_300).logz[-1]
#     log_R.append(log_z_quad - log_z_kipping)
# log_R = np.array(log_R)
# R_mean_300 = np.mean(np.exp(log_R))
# R_std_300 = np.std(np.exp(log_R))
# 
# 
# log_R = []
# for i in range(100):
#     log_z_quad = dynesty.utils.jitter_run(quad_1000).logz[-1]
#     log_z_kipping = dynesty.utils.jitter_run(kipping_1000).logz[-1]
#     log_R.append(log_z_quad - log_z_kipping)
# log_R = np.array(log_R)
# R_mean_1000 = np.mean(np.exp(log_R))
# R_std_1000 = np.std(np.exp(log_R))
# 
# 
# 
# R_mean = np.array([R_mean_1, R_mean_10, R_mean_30, R_mean_100, R_mean_300, R_mean_1000])
# R_std = np.array([R_std_1, R_std_10, R_std_30, R_std_100, R_std_300, R_std_1000])
# 
# 
# plt.errorbar(np.array([1, 2, 3, 4, 5, 6]), R_mean, yerr=R_std, fmt='o')
# plt.xticks(np.array([1, 2, 3, 4, 5, 6]), np.array(['1 ppm', '10 ppm', '30 ppm', '100 ppm', '300 ppm', '1000 ppm']), rotation=45)
# plt.title('Parametrization Comparison')
# plt.ylabel('Bayes\' Ratio')
# #plt.savefig('BayesRatio.png', dpi=300, bbox_inches="tight")
# plt.show()
# =============================================================================






#calculate DIC, WAIC, LOOC

import scipy
from model_functions import full_model
from mcmc_functions import log_posterior


#time, flux, err = load_simdata_all_errs('no_Id')





def negative_log_posterior(theta, t, y, yerr, params, model, priors, transform):
        return -log_posterior(theta, t, y, yerr, params, model, priors, transform)

# =============================================================================
# theta_init = np.array([0.1, 0.1, 0.1])
# transform=False
# 
# MAP_result = scipy.optimize.minimize(
#         fun=negative_log_posterior,
#         x0=theta_init,
#         args=(time_data, flux_data, all_errors_dict['1 ppm'], params, model, priors_quadratic, transform)
#         )
# 
# theta_MAP = MAP_result.x
# =============================================================================

# =============================================================================
# print("MAP results")
# print(theta_MAP)
# 
# y_MAP = full_model(theta_MAP, params, model)
# 
# plt.plot(time_data, flux_data)
# plt.plot(time_data, y_MAP)
# plt.show()
# =============================================================================


def DIC(theta_star, theta_samples, log_likelihood):
    # Compute log likelihood at theta_star and the samples theta_i
    log_likelihood_star = log_likelihood(theta_star)
    log_likelihood_samples = np.array(
        [log_likelihood(theta) for theta in theta_samples]
    )
    p_D = 2*(log_likelihood_star - np.mean(log_likelihood_samples))
    p_V = 2*np.var(log_likelihood_samples)
    return -2*(log_likelihood_star - p_D), p_D, p_V


# =============================================================================
# chain_quad = quad_1.samples
# 
# 
# DIC_quad, p_D_quad, p_V_quad = DIC(
#     theta_star=theta_MAP,
#     theta_samples=chain_quad[np.random.choice(chain_quad.shape[0], size=500)],
#     log_likelihood=lambda theta: log_likelihood(theta, time_data, flux_data, all_errors_dict['1 ppm'], params, model, transform)
# )
# 
# print(f"Quadratic: DIC = {DIC_quad:.1f}, p_D = {p_D_quad:.1f}, p_V = {p_V_quad:.1f}")
# 
# =============================================================================














