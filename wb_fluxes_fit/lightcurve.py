'''
comment
'''

import numpy as np
import matplotlib.pyplot as plt 
from cross_section import two_shapes_upright, get_z_range
from band_area import Area
from numba import jit

def intense(phi:np.float64, theta_n:np.float64, d:np.float64, n_bands:np.int64, n_slices:np.int64, rp1:np.float64, rp2:np.float64, r:np.float64, x:np.float64,y:np.float64, I:np.float64):
    big_flux = np.zeros(n_slices)
    small_flux = np.zeros(n_slices)
    for n, z in enumerate(get_z_range(rp1, rp2, r, theta_n, number = n_slices)):
        r1, h1, r2, h2 = two_shapes_upright(rp1, rp2, r, theta_n, z)
        if x == 0:
            if y > 0:
                theta_o = np.pi/2
            else:
                theta_o = - np.pi/2
        else:
            theta_o = np.arctan(-y / abs(x)) #original angle
        
        if x < 0:
            theta_p = theta_o - phi + np.pi/2
        else:
            theta_p = theta_o + phi - np.pi/2
        # now focusing on big atmosphere, exclude nucleus from it for calculation of the flux
        areas = np.zeros(n_bands + 1)
        for m, i in enumerate(np.linspace(0, 1, n_bands + 1)):
            a = Area(i, r1, d, theta_p, h1) - Area(i, r, d, theta_p, h1)
            areas[m] = round(a, 10)
        areas = areas[1:] - areas[:-1]
        mid_I = np.zeros(n_bands)
        for m, i in enumerate(np.linspace(0, 1 - 1/n_bands, n_bands)):
            mid_I[m] = I((2*i+1/n_bands)/2)
        big_flux[n] = np.sum(mid_I * areas)
        
        # now look at small atmosphere
        areas = np.zeros(n_bands + 1)
        for m, i in enumerate(np.linspace(0, 1, n_bands + 1)):
            a = Area(i, r2, d, -theta_p, h2) - Area(i, r, d, -theta_p, h2)
            areas[m] = round(a, 10)
        areas = areas[1:] - areas[:-1]
        small_flux[n] = np.sum(mid_I * areas)
    # look at the nucleus
    areas = np.zeros(n_bands + 1)
    for m, i in enumerate(np.linspace(0, 1, n_bands + 1)):
        a = Area(i, r, d, theta_p, -r)
        areas[m] = round(a, 10)
    areas = areas[1:] - areas[:-1]

    return mid_I, areas, small_flux, big_flux



class LightCurve():
    '''
    This class combines all the functions and generate the lightcurve.

    Args:
        b: impact parameter of planet
        r: radius of the nucleus relative to the star
        rp1: radius of the planet's hotter (greater) atmosphere relative to the star
        rp2: radius of the planet's cooler (smaller) atmosphere relative to the star
        I: intensity function describing limb darkening
        r_sep: orbital radius
        rotate: function describes the rotation of the planet
        theta: incline of the orbital plane
    '''

    def __init__(self, b, r, rp1, rp2, I, r_sep, rotate = 'tidal_lock', wind = 0):
        self.b = b
        self.r = r
        self.rp1 = rp1
        self.rp2 = rp2
        self.I = I
        self.r_sep = r_sep
        self.rotate = rotate
        self.theta = np.arcsin(b / r_sep)
        self.planet_pos = np.zeros(3)
        self.wind = wind

    def orbit(self, phase):
        '''
        Args:
            phase: phase of the planet relative to the star
        
        Assign the attributes.
        '''
        x = self.r_sep * np.sin(np.pi*2*phase)
        y = np.sin(self.theta) * self.r_sep * np.cos(2*np.pi*phase)
        z = np.cos(self.theta) * self.r_sep * np.cos(2*np.pi*phase)
        self.planet_pos = np.array([x, y, z])

    def long(self, d):
        '''
        Finds the longitudes cut across by light at current phase.
        '''
        x = self.planet_pos[0]
        y = self.planet_pos[1]
        xy_sq = (x * x) + (y * y)
        dr_sq = (1 - self.r) * (1 - self.r) 
        c = -dr_sq + xy_sq
        det = (c * c * y * y) - (xy_sq * ((c * c) - (4 * self.r * self.r * x * x)))

        #for planet inside and outside case
        while det <= 0:
            if d > 1:
                
                return 0, 0
            elif d < 1:
               
                return 0, 360
        
        y_pos = ((c * y) + np.sqrt(det)) / (2 * xy_sq)
        y_neg = ((c * y) - np.sqrt(det)) / (2 * xy_sq)

        return y_pos, y_neg



    def simulate(self, phase, sample, n_bands = 10, n_slices = 10):
        '''
        Args:
        '''
        
        n_b = n_bands
        n_s = n_slices
        phases = np.linspace(-phase, phase, sample)
        big_curve = np.zeros(sample)
        small_curve = np.zeros(sample)
        planet_curve = np.zeros(sample)
        longitudes1 = np.zeros(sample)
        longitudes2 = np.zeros(sample)
        for p, i in enumerate(phases):
            self.orbit(i)
            d = np.sqrt(self.planet_pos[0]**2 + self.planet_pos[1]**2)
            if self.rotate == 'tidal_lock':
                normal = np.array([-self.planet_pos[0], -self.planet_pos[1], -self.planet_pos[2]]) # pointing toward the observer
                if self.wind != 0:
                    rot = np.array([[np.cos(self.wind), 0, np.sin(self.wind)], [0, 1, 0],[-np.sin(self.wind), 0, np.cos(self.wind)]])
                    normal = np.matmul(rot, normal)
                    normal = np.array([normal[0], normal[1], -normal[2]])

            else:
                raise Exception('We cannot deal with not tidally-locked case.')
            if normal[1] >= 0:
                phi = np.arccos(normal[0] / np.sqrt(normal[0]**2 + normal[1]**2))
            else:
                phi = - np.arccos(normal[0] / np.sqrt(normal[0]**2 + normal[1]**2)) + 2*np.pi
            theta_n = np.arccos(normal[2] / np.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2))

            mid_I, areas, small_flux, big_flux = intense(
                n_bands = n_b, n_slices = n_s, theta_n= theta_n, phi=phi, d=d, rp1=self.rp1, rp2=self.rp2, r=self.r, x=self.planet_pos[0], y=self.planet_pos[1], I=self.I
                )
            
            #this next bit is for the longitude
            longitude = self.long(d = d)
            longitudes1[p] = longitude[0]
            longitudes2[p] = longitude[1]
            self.longitudes1 =longitudes1
            self.longitudes2 = longitudes2

            planet_curve[p] = np.sum(mid_I * areas)
            small_curve[p] = np.mean(small_flux)
            big_curve[p] = np.mean(big_flux)
            self.phases = phases
            self.planet_curve = planet_curve
            self.small_curve = small_curve
            self.big_curve = big_curve
            


    def plots(self):
        '''plot'''
        fig, ax = plt.subplots(2, 1, sharex = 1, figsize = (5,5))
    
        ax[0].plot(self.phases, np.pi - self.big_curve, '.', label='hotter atmosphere', markersize=0.7)
        ax[0].plot(self.phases, np.pi - self.small_curve, '.',label='cooler atmosphere', markersize=0.7)
        ax[0].plot(self.phases, np.pi - self.planet_curve, '.', label='planet', markersize=0.7)
        ax[0].legend()
        ax[1].plot(self.phases, np.pi - (self.planet_curve+ self.small_curve + self.big_curve),'.', label='combined', markersize=0.5)
        ax[1].legend()
        ax[1].set_ylabel('flux')
        plt.savefig('lightcurve.pdf')
        plt.show()
        
        plt.plot(self.phases, self.longitudes1, 'x', markersize = 0.5)
        plt.plot(self.phases, self.longitudes2, 'x', markersize = 0.5)
        plt.savefig('test.pdf')
        plt.show()