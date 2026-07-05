import matplotlib.pyplot as plt
import numpy as np
import batman 
from scipy.optimize import curve_fit
# from lightcurve import LightCurve
import bisect
import catwoman

tranWL = [0.775792618340347, 0.8954958116810303]
enclosedTranWL = [0.7944304761622334, 0.8768579547904665]
t0 = 0.8356471737
per = 4.0552842518
a = 11.3904973685
inc = 87.7369030491272
ecc = 0
w = 90
phi = 90

def phase_range(times, timeLimits):
    timeStart = times[bisect.bisect_right(times, timeLimits[0])]
    timeEnd = times[bisect. bisect_left(times, timeLimits[1])]  
    phaseStart = (timeStart - t0) * ((2 * np.pi) / per)
    phaseEnd = (timeEnd - t0) * ((2 * np.pi) / per)
    return phaseStart, phaseEnd

def set_u(LD_coeff):
    global u
    u = LD_coeff

def bat(times, rp):
    params = batman.TransitParams()      #object to store transit parameters
    params.t0 = t0                      #time of inferior conjunction
    params.per = per                       #orbital period
    params.rp = rp                      #planet radius (in units of stellar radii)
    params.a = a                    #semi-major axis (in units of stellar radii)
    params.inc = inc                      #orbital inclination (in degrees)
    params.ecc = ecc                       #eccentricity
    params.w = w                        #longitude of periastron (in degrees)
    params.limb_dark = "quadratic"        #limb darkening model
    params.u = u

    m = batman.TransitModel(params, times)
    flux = m.light_curve(params)
    return flux
    
def cat(timeFrame, rp1, rp2, noise = True, noiseVal = 0, plot = False):
    params = catwoman.TransitParams()       #object to store transit parameters
    params.t0 = t0                         #time of inferior conjuction (in days)
    params.per = per                         #orbital period (in days)
    params.rp = rp1                         #top semi-circle radius (in units of stellar radii)
    params.rp2 = rp2                     #bottom semi-circle radius (in units of stellar radii)
    params.a = a                          #semi-major axis (in units of stellar radii)
    params.inc = inc                        #orbital inclination (in degrees)
    params.ecc = ecc                         #eccentricity
    params.w = w                          #longitude of periastron (in degrees)
    params.u = u                  #limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic"          #limb darkening model
    params.phi = phi                #angle of rotation of top semi-circle

    m = catwoman.TransitModel(params, timeFrame, max_err = 0.1)      #initialises the model
    flux = m.light_curve(params)
    if noise:
        n = np.random.normal(0, noiseVal, len(timeFrame))
        flux = [f + n for f, n in zip(flux, n)]
    
    if plot:
        plt.plot(timeFrame, flux, 'x', markersize = 0.5, color = 'black')
    return flux

def bat_lin(times, rp, m, c):
    flux = bat(times, rp)
    lin = (m * times) + c
    flux_lin = [f / l for f, l in zip(flux, lin)]
    return flux_lin
        
def fit(func, times, fluxes, guesses):
    popt, pcov = curve_fit(func, times, fluxes, p0 = guesses)
    return popt, pcov

def cut(flux, timeFrame, iPhase, ePhase):
    iCutPoint = ((iPhase / (2 * np.pi)) * per ) + t0 #period and t0 taken from data
    eCutPoint = ((ePhase / (2 * np.pi)) * per ) + t0
    i, e = bisect.bisect_left(timeFrame, iCutPoint), bisect.bisect_right(timeFrame, eCutPoint)
    start, end = bisect.bisect_left(timeFrame, tranWL[0]), bisect.bisect_right(timeFrame, tranWL[1])
    iCurve = [np.append(timeFrame[:i], timeFrame[end:]), np.append(flux[:i], flux[end:])]
    eCurve = [np.append(timeFrame[:start], timeFrame[e:]), np.append(flux[:start], flux[e:])]
    return iCurve, eCurve

def plot_fit(func, times, fluxes, guesses, plot = True):
    popt, pcov = fit(func, times, fluxes, guesses)
    y_fit = func(times, *popt)
    if plot:
        plt.figure()
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 13
        plt.plot(times, fluxes,'x', markersize = 0.5, color = 'black', label = 'JWST flux data' )
        plt.plot(times, y_fit, color = 'red', label = 'batman fit')
        plt.xlabel('Time (days)')
        plt.ylabel('Normalised Flux')
        plt.legend()
        plt.savefig('FullTransit.pdf')
        plt.show()
    return popt, pcov

def plot_cut_fit(func, flux, timeFrame, mPhase, ePhase, guesses, split = False, plot = True):
    iCurve, eCurve = cut(flux, timeFrame, mPhase, ePhase)

    ipopt, ipcov = fit(func, iCurve[0],iCurve[1], guesses)
    epopt, epcov = fit(func, eCurve[0], eCurve[1], guesses)
    if plot:
        plt.figure()
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 13
        plt.plot(iCurve[0], iCurve[1], 'x', markersize = 0.5, color = 'green', label = 'cut JWST flux data')
        plt.plot(timeFrame, func(timeFrame, *ipopt), color = 'lime', label = 'batman fit')
        plt.xlabel('Time (days)')
        plt.ylabel('Normalised Flux')
        if split:
            plt.legend()
            plt.savefig('IngressTransit.pdf')
            plt.show()
        plt.plot(eCurve[0],eCurve[1], 'x', markersize = 0.5, color = 'orange', label = 'cut JWST flux data')
        plt.plot(timeFrame, func(timeFrame, *epopt), color = 'tomato', label = 'batman fit')
        plt.xlabel('Time (days)')
        plt.ylabel('Normalised Flux')
        plt.legend()
        plt.savefig('EgressTransit.pdf')
        plt.show()
    return [ipopt, ipcov], [epopt, epcov]

def snr_plot(func, flux, timeFrame, sample, guesses, plot = True):
    startEnc, endEnc = phase_range(timeFrame, enclosedTranWL)
    start, end = phase_range(timeFrame, tranWL)
    if abs(start) >= end:
        bigPhase = end
    elif end > abs(start):
        bigPhase = abs(start)
    else:
        raise Exception('how')
    
    if abs(startEnc) >= endEnc:
        smallPhase = abs(startEnc)
    elif endEnc > abs(startEnc):
        smallPhase = endEnc
    else:
        raise Exception('how2')
    
    phases = np.linspace(smallPhase, bigPhase, sample)
    SNRs = np.zeros(sample)
    iUnc = np.zeros(sample)
    eUnc = np.zeros(sample)
    
    for n, phase in enumerate(phases):
        i, e = plot_cut_fit(func, flux, timeFrame, -phase, phase, guesses, split = False, plot = False)
        iRelErr = np.sqrt(i[1]) / i[0] 
        eRelErr = np.sqrt(e[1]) / e[0]
        iTD = i[0] * i[0]
        eTD = e[0] * e[0]
        diff = abs(iTD - eTD)
        diffAbsErr = (2 * iRelErr * iTD) + (2 * eRelErr * eTD)
        SNRs[n] = diff / (diffAbsErr / 2)
        iUnc[n] = 2 * iRelErr 
        eUnc[n] = 2 * eRelErr 
    if plot:
        plt.plot(phases, SNRs, 'x')
        plt.xlabel('Phases probed')
        plt.ylabel('SNR')
        plt.savefig('SNR.pdf')
        plt.show()
    return SNRs, phases, iUnc, eUnc
    
def bat_test(times, rp, iPhase, ePhase):
    fluxes = bat(times, rp)
    i, e = cut(fluxes, times, iPhase, ePhase)
    ipopt, ipcov = fit(bat, i[0], i[1], 0.1437859369)
    epopt, epcov = fit(bat, e[0], e[1], 0.1437859369)
    plt.plot(i[0], i[1], 'x')
    plt.plot(times, bat(times, ipopt))
    plt.show()
    plt.plot(e[0], e[1], 'x')
    plt.plot(times, bat(times, epopt))
    plt.show()
    snr_plot(bat, fluxes, times, 100,0.1437859369)
    return ipcov, epcov

def cat_cut_fit(timeFrame, rp1, rp2, noiseVal, mPhase, ePhase,  phaseRange, sample = 20, repeat = False):
    catFlux = cat(timeFrame, rp1, rp2, noise = True, noiseVal = noiseVal, plot = False)  
    phases = np.linspace(phaseRange[0], phaseRange[1], sample)  
    plt.plot(timeFrame, catFlux, 'x', markersize = 1)
    if repeat:
        rp1s = np.zeros(sample)
        rp2s = np.zeros(sample)
        rp1Errs = np.zeros(sample)
        rp2Errs = np.zeros(sample)
        for n, i in enumerate(phases):
            i, e = plot_cut_fit(bat, catFlux, timeFrame, -i, i, 0.1437, split=True, plot=False) #chosen phi=90 means rp2 represents the ingress portion and rp1 the egress
            rp1s[n] = e[0]
            rp1Errs[n] = e[1][0]
            rp2s[n] = i[0]
            rp2Errs[n] = i[1][0]
        
        relrp1 = [(rp - rp1)/rp1 for rp in rp1s]
        relrp2 = [(rp - rp2)/rp2 for rp in rp2s]
        plt.figure()
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams["font.size"] = 13
        plt.plot(phases, relrp1, 'x', label = 'egress')
        plt.ylim(0,0.2)
        # plt.show()
        plt.plot(phases, relrp2, 'x', label = 'ingress')
        plt.title(f' rp1 ={rp1} ' + f' rp2 = {rp2} ' + f' Limb Darkening coefficients = {u}')
        plt.xlabel('Phases')
        plt.ylabel('Relative Rp deviation')
        plt.legend()
        plt.savefig('Catwoman.pdf')
        plt.show()
        return phases, relrp1, relrp2

    else:
        i, e = plot_cut_fit(bat, catFlux, timeFrame, mPhase, ePhase, 0.1437, split=True, plot=True)
        return i, e

'''    
def I(x):
    return 1

def cremini(times, r, rp1, rp2):
    b = 0.443
    r = r
    rp1 = rp1
    rp2 = rp2
    def I(x):
        return 1
    r_sep = 11.37
    per = 4.055259
    phase_diff = times[-1]/per

    model = LightCurve(b,r,rp1,rp2,I,r_sep)
    model.simulate(phase_diff / 2, len(times))
    small_fluxes = model.small_curve
    big_fluxes = model.big_curve
    planet_fluxes = model.planet_curve
    total_fluxes = 1 - (small_fluxes + big_fluxes + planet_fluxes) / np.pi
    return total_fluxes
'''