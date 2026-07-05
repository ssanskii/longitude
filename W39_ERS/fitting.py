import batman
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
def setu(LD):
    global u
    u =LD

def bat(times, rp):
    global u
    params = batman.TransitParams()       #object to store transit parameters
    params.t0 = 0.8356471737                      #time of inferior conjunction
    params.per = 4.0552842518                       #orbital period
    params.rp = rp                      #planet radius (in units of stellar radii)
    params.a = 11.3904973685                    #semi-major axis (in units of stellar radii)
    params.inc = 87.7369030491272                      #orbital inclination (in degrees)
    params.ecc = 0                       #eccentricity
    params.w = 90                        #longitude of periastron (in degrees)
    params.limb_dark = "quadratic"        #limb darkening model
    params.u = u
    m = batman.TransitModel(params, times)
    flux = m.light_curve(params)

    return flux

def bat_lin(times, rp, m, c):
    flux = bat(times, rp)
    lin = (m * times) + c
    flux_lin = [f / l for f, l in zip(flux, lin)]
    return flux_lin
        

def fit(func, times, fluxes, guesses):
    popt, pcov = curve_fit(func, times, fluxes, p0 = guesses)
    return popt, pcov

