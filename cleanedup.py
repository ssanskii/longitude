import batman
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from dataclasses import dataclass
from dataclasses import replace
from functools import partial

@dataclass
class TransitParams:
    '''
    Dataclass for transit parameters, default values set assumed as W39 parameters. 
    Main purpose to allow for easy expansion of code to test for other parameters (only really used to change u in this analysis)
    '''
    rp: float = 0.1
    t0: float = 0.8356471737
    per: float = 4.0552842518
    a: float = 11.3904973685
    inc: float = 87.7369030491272
    ecc: float = 0
    w: float = 90
    phi: float = 90
    limb_dark: str = "quadratic"
    u: list[float] = (0.1, 0.3)

    def to_batman(self, rp: float) -> batman.TransitParams:
        '''
        Builds a batman.TransitParams object with the fixed values and variable rp.
        '''
        params = batman.TransitParams()
        params.t0 = self.t0
        params.per = self.per
        params.rp = rp
        params.a = self.a
        params.inc = self.inc
        params.ecc = self.ecc
        params.w = self.w
        params.limb_dark = "quadratic"
        params.u = self.u
        return params


def constr_bat(times, rp, tp):
    '''
    Returns fluxes for a constrained batman model for transit radius fitting.
    '''
    # Set variables
    params = tp.to_batman(rp)

    # Create batman model and return flux based on params and given time vars
    m = batman.TransitModel(params, times)
    flux = m.light_curve(params)

    return flux

def bat_lin(times, rp, m, c, tp):
    '''
    Returns the flux for a linear trended batman model curve. 
    Calls upon constr_bat() and involved in kinda ad hoc method to detrend lightcurves for analysis.
    '''
    # Calculate flux from batman and superposes a linear trend 
    flux = constr_bat(times, rp, tp)
    lin = (m * times) + c
    flux_lin = [f / l for f, l in zip(flux, lin)]

    return flux_lin

def create_plot(x, y, ax = None, fig = None, show = False, **kwargs):
    '''
    Takes general x and y data (and kwargs for plotting) and returns matplotlib ax object for editing, saving, etc and figure object.
    Optionally shows the plot if requested.
    Can also take an existing ax and figure object to plot on (for example plotting fitted curves on top of data).
    '''        
    # Set defaults
    defaults = {
        'markerSize': 0.5,
        'color': 'black',
        'label': 'JWST flux data',
        'fontsize': 13,
        'fontfamily': 'Times New Roman',
        'figsize': (10, 6),
        'label': '',
        'fmt': 'x'
    }

    # Address **kwargs
    defaults.update(kwargs)

    # Set font properties :D
    plt.rcParams["font.family"] = defaults['fontfamily']
    plt.rcParams["font.size"] = defaults['fontsize']

    # Create a new figure and axis if none are provided and plot data 
    if ax is None:      
        fig, ax = plt.subplots(figsize = defaults['figsize'])
    ax.plot(x, y, defaults['fmt'], markersize = defaults['markerSize'], color = defaults['color'], label = defaults['label']) 

    #Optionally display plot and then close it
    if show:
        plt.show()
    plt.close(fig)    

    return fig, ax    

def cut_from_midtransit(time: np.ndarray, flux: np.ndarray, phase, t0, per, right = False):
    '''
    Cuts light curve data from given mid transit point.
    Option cut towards the left (default) or right cut curve for ingress and egress respectively.
    '''
    # Splices ndarrays using boolean mask of time array
    offset = (phase / (2* np.pi)) * per
    cutPoint = offset + t0 if right else t0 - offset
    mask = time >= cutPoint if right else time <= cutPoint

    return time[mask], flux[mask]

def main(times:np.ndarray, fluxes:np.ndarray, tp: TransitParams, u, ephase, iphase):
    # Remove nans
    mask = ~np.isnan(fluxes)
    times = times[mask]
    fluxes = fluxes[mask]

    # Set u value of tp object and fit a linear batman model to find parameters for detrending
    tp = replace(tp, u=u)
    linBatModel = partial(bat_lin, tp = tp) # Partial used so curvefit ignores the tp object when fitting
    poptLin, _ = curve_fit(linBatModel, times, fluxes, p0 = [0.1, 0, 1]) 
    rpGuess, m, c = poptLin
   
    # Define linearly dtrended batman model
    def bat_detrended(times, rp ,tp):
        lin = m * times + c
        return constr_bat(times, rp, tp) / lin

    # cut lightcurve into egress and ingress
    egressTime, egressFlux = cut_from_midtransit(times, fluxes, phase=ephase, t0=tp.t0, per=tp.per, right=True)
    ingressTime, ingressFlux = cut_from_midtransit(times, fluxes, phase=iphase, t0=tp.t0, per=tp.per, right=False)

    # fit batman curves on egress and ingress
    poptEgress, pcovEgress = curve_fit(partial(bat_detrended, tp = tp), egressTime, egressFlux, p0 = [rpGuess]) # used linbat fitted rp as a guess
    rpEgress = poptEgress[0]
    poptIngress, pcovIngress = curve_fit(partial(bat_detrended, tp = tp), ingressTime, ingressFlux, p0 = [rpGuess])
    rpIngress = poptIngress[0]

    #Find out about SNR and errors

    return rpEgress, rpIngress
