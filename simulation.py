import numpy as np
import matplotlib.pyplot as plt
import batman


# JUST A TEST FUNCTION
def simulate_planet_transit(period, t0, duration, rp, a, inc, ecc, w, u, t):
    """
    Simulate a planet transit using the batman package.
    """
    # Create a transit model
    params = batman.TransitParams()
    params.t0 = t0  # time of inferior conjunction
    params.per = period  # orbital period
    params.rp = rp  # planet radius (in units of stellar radii)
    params.a = a  # semi-major axis (in units of stellar radii)
    params.inc = inc  # orbital inclination (in degrees)
    params.ecc = ecc  # eccentricity
    params.w = w  # longitude of periastron (in degrees)
    params.u = u  # limb darkening coefficients
    params.limb_dark = "quadratic"  # limb darkening model

    # Create a transit model
    m = batman.TransitModel(params, t)

    # Generate the light curve
    flux = m.light_curve(params)
    return flux


