import numpy as np
import matplotlib.pyplot as plt
from numba import jit
"""
place the centre of the segment (radius R_s) to (0,0) and let the cut-off line be y=h
make sure the centre of the complete circle (radius R) has negative x-coordinate
let distance between centres be d, measure theta clockwise from the left leg of the segment to the
line connecting the centre
"""
 
@jit
def SegmentArea(radius, a1, b1, a2, b2):
    alpha = np.arccos(1 - ((a1 - a2)**2 + (b1 - b2)**2) / (2 * radius**2))
    return float(radius**2 / 2 * (alpha - np.sin(alpha)))
 
@jit
def Area(R:float, R_s:float, d:float, theta:float, h:float) -> float:
    if h >= R_s:
        return 0
    if h < -R_s:
        h = -R_s
    while theta < -np.pi:
        theta += 2*np.pi
    while theta > np.pi:
        theta -= 2*np.pi
    if np.isclose(theta, np.pi/2):
        theta = np.pi/2
    if np.isclose(theta, -np.pi/2):
        theta = -np.pi/2
    if theta > np.pi/2:
        theta = np.pi - theta
    if theta < -np.pi/2:
        theta = -np.pi + theta

    if d == 0: # concentric circles
        if R_s > R:
            r = R_s
            R_s = R
            R = r
            if h > 0:
                return SegmentArea(R_s, -np.sqrt(R_s**2 - h**2), h, np.sqrt(R_s**2 - h**2), h)
            return R_s**2*np.pi - SegmentArea(R_s, -np.sqrt(R_s**2 - h**2), h, np.sqrt(R_s**2 - h**2), h)
    # calculating the intersection of two circles
    if (R_s**2*(4 * d**2) - (R**2 - R_s**2 - d**2) **2)  <= 0 or np.isclose((R_s**2*(4 * d**2) - (R**2 - R_s**2 - d**2) **2),0):# if R_s**2 - A**2 <= 0
        intersect = False
    else:
        intersect = True
        A = (R**2 - R_s**2 - d**2) / (2 * d)
        if np.isclose(theta, np.pi/2):
            b1 = - A
            a1 = - np.sqrt(R_s**2 - b1**2)
            b2 = - A
            a2 = np.sqrt(R_s**2 - b2**2)
        elif np.isclose(theta, -np.pi/2):
            b1 = A
            a1 = - np.sqrt(R_s**2 - b1**2)
            b2 = A
            a2 = np.sqrt(R_s**2 - b2**2)
        else:
            b1 = - A * np.sin(theta) - np.cos(theta) * np.sqrt(R_s**2 - A**2) #lower y
            a1 = b1 * np.tan(theta) + A / np.cos(theta)
            b2 = - A * np.sin(theta) + np.cos(theta) * np.sqrt(R_s**2 - A**2) #higher y
            a2 = b2 * np.tan(theta) + A / np.cos(theta)
    # the extent of the base (on y=h)
    delta = np.sqrt(R_s**2 - h**2)
    # for h < 0, first find the area between two arcs
    if h < 0:
        if not intersect: # pretent to complete the segment, and have no intersection between two circles
            if d>= R_s + R: # either they don't contain each other
                return 0
            if R > R_s: # or one is containing another, take the pure intersection area
                return np.pi*R_s**2 - SegmentArea(R_s, -delta, h , delta, h)
            else:
                return np.pi*R**2  
        else:
            # check if two centre are on the same side of the connecting line of two circles
            # if yes, the bigger segment cannot be found using SegmentArea, but cirle - SegmentArea
            n = np.array([a1 + d*np.cos(theta),b1 -  d*np.sin(theta) ]) # circle centre to (a1, b1)
            n = n/np.sqrt((n[0]**2 + n[1]**2))
            m = np.array([a1, b1]) # segment centre to (a1, b1)
            m = m/np.sqrt((m[0]**2 + m[1]**2))
            l = np.array([a1-a2, b1-b2]) # cutting line from (a1, b1) to (a2, b2)
            l = l/np.sqrt((l[0]**2 + l[1]**2))
   
            twopoints = np.arccos(np.dot(n,m)) #angle between two centres
            circle = np.arccos(np.dot(n,l))
            seg = np.arccos(np.dot(m,l))
            if circle > twopoints or seg > twopoints:
                switch = False
                if R_s > R:
                    switch = True
                    r = R
                    R = R_s
                    R_s = r
                big_segment = SegmentArea(R, a1, b1, a2, b2) + np.pi*R_s**2 - SegmentArea(R_s, a1, b1, a2, b2)
                if switch is True: # change two radius back
                    r = R
                    R = R_s
                    R_s = r
            else:
                big_segment = SegmentArea(R, a1, b1, a2, b2) + SegmentArea(R_s, a1, b1, a2, b2)
       
        # then flip the segment to find the take-away area
        theta *= -1
        h *= -1
        if np.isclose(big_segment - Area(R, R_s, d, theta, h),0):
            return 0
        return big_segment - Area(R, R_s, d, theta, h)
 
    # now only h>=0 can remain
    if np.abs(h - d * np.sin(theta)) >= R: # the complete circle touches y=h at most once
        if h >= d * np.sin(theta): # circle is below y=h
            return 0
        if not intersect: # no intersection and above could only be two cases
            if d-R >= R_s: # either not containing each other
                return 0
            return np.pi*R**2 # not possible for the segment to be contained since circle not touching y=h
        # if it is above and intersects, is just the area of two circles
        # careful to check if both centres lie on the same side
        n = np.array([a1 + d*np.cos(theta),b1 -  d*np.sin(theta) ]) # circle centre to (a1, b1)
        n = n/np.sqrt((n[0]**2 + n[1]**2))
        m = np.array([a1, b1]) # segment centre to (a1, b1)
        m = m/np.sqrt((m[0]**2 + m[1]**2))
        l = np.array([a1-a2, b1-b2]) # cutting line from (a1, b1) to (a2, b2)
        l = l/np.sqrt((l[0]**2 + l[1]**2))
        twopoints = np.arccos(np.dot(n,m)) #angle between two centre
        circle = np.arccos(np.dot(n,l))
        seg = np.arccos(np.dot(m,l))
        if circle > twopoints or seg > twopoints:
            switch = False
            if R_s > R:
                switch = True
                r = R
                R = R_s
                R_s = r
            big_segment = SegmentArea(R, a1, b1, a2, b2) + np.pi*R_s**2 - SegmentArea(R_s, a1, b1, a2, b2)
            if switch is True: # change two radius back
                r = R
                R = R_s
                R_s = r
        else:
            big_segment = SegmentArea(R, a1, b1, a2, b2) + SegmentArea(R_s, a1, b1, a2, b2)
        return big_segment
    # intercepts of the (complete) circular arc with the y=h
    x1 = - d * np.cos(theta) - np.sqrt(R**2 - (h - d * np.sin(theta))**2) #x1: intercept with h, x1 < 0
    x2 = - d * np.cos(theta) + np.sqrt(R**2 - (h - d * np.sin(theta))**2) #x2: intercept with h, x2 > 0
    if x1 <= -delta and abs(x2)>=delta: # complete circle always touches y=h twice, and now look at x2 outside
        if x2 < - delta: # this is the case with both intercepts to the left
            return 0
        # then this is automatically x2 to the right of the shape
        if intersect and b1 > h:
            return SegmentArea(R_s, -delta, h, delta, h) - SegmentArea(R_s, a1, b1, a2, b2) + SegmentArea(R, a1, b1, a2, b2)
        return  SegmentArea(R_s, -delta, h, delta, h) #cover the whole thing
    if x1 >= -delta: # the x2 can only be inside now, and first check both inside by saying x1>=-delta
        a3 = -d*np.cos(theta) + R
        b3 = d*np.sin(theta)
        if not intersect or b1 < h: # but not touching the top
            if b3 <= h: # (a3, b3) is below y=h
                return SegmentArea(R, x1, h, x2, h)
            return R**2*np.pi - SegmentArea(R, x1, h, x2, h) # (a3, b3) is above y=h
        if a1 > a2:# but touching the arc meaning four intersections
            a3 = a1
            a1 = a2
            a2 = a3
            b3 = b1
            b1 = b2
            b2 = b3
        a = np.sqrt((a1-x1)**2 + (h-b1)**2)
        b = np.sqrt((a2-x1)**2 + (b2-h)**2)
        c = np.sqrt((a1-a2)**2 + (b1-b2)**2)
        s = (a+b+c)/2
        return np.sqrt(s*(s-a)*(s-b)*(s-c)) + (x2-x1)*(b2-h)/2 + SegmentArea(R, a1, b1, x1, h) + SegmentArea(R, a2, b2, x2, h) + SegmentArea(R_s, a1, b1, a2, b2)
   
    # only thing left is one outside and one inside
    return SegmentArea(R, x2, h, a2, b2) + SegmentArea(R_s, -delta, h, a2, b2) + (x2+delta)*(b2-h)/2
 
"""
Numerical Verification
"""
@jit
def put_in_points(multiplier, R_s, h):
    points = np.zeros((int((2*R_s*multiplier)**2), 2))
    num = int(R_s*multiplier)
    x = np.linspace(-num, num, 2*num)
    count = 0
    for i in x:
        for j in x:
            if j/multiplier >h and (i/multiplier)**2 + (j/multiplier)**2 < R_s**2:
                points[count] = np.array([i/multiplier,j/multiplier])
                count += 1
    return points[:count]
@jit
def doit(R, d, theta, points):
    count = 0
    for i in points:
        if (i[0]+d*np.cos(theta))**2 + (i[1]-d*np.sin(theta))**2 < R**2:
            count += 1
    return count
 
class NumericalSegment():
    def __init__(self, R_s, h, multiplier = 1000):
        self.R_s = R_s
        self.h = h
        self.multiplier = multiplier
        self.points= put_in_points(self.multiplier, self.R_s, self.h)
 
    def area_segment(self):
        return len(self.points)/self.multiplier**2
    def area_intersect(self, R,d, theta, plot = False):
        if plot == True:
            self.plot(R=R, d=d, theta=theta)
        return doit(R, d, theta, self.points)/self.multiplier**2
    def plot(self, R=0, d=0, theta=0):
        fig, ax = plt.subplots(figsize=(5,5))
        ax.plot(*np.array(self.points).transpose(),'.',markersize=0.1,color="darkseagreen")
        ax.set_xlim(-1.5*self.R_s, 1.5*self.R_s)
        ax.set_ylim(-1.5*self.R_s, 1.5*self.R_s)
        if R != 0:
            angle = np.linspace(0, 2* np.pi, 1000)
            ax.plot(-d*np.cos(theta) + R*np.cos(angle),d*np.sin(theta) + R*np.sin(angle),color='lightsalmon')
        ax.axis("off")
        plt.show()
 
 
"""
Comparison
"""
def plot_compare(R,R_s, d, theta, h, plot=False, multiplier= 1000):
    s = NumericalSegment(R_s = R_s, h=h, multiplier=multiplier)
    numerical_value= s.area_intersect(R=R,d=d,theta=theta, plot=plot)
    analytical_value = Area(R,R_s, d, theta, h)
    print(f"Numerical value is {numerical_value}")
    print(f"Analytical value is {analytical_value}")
    print(f"With multiplier = {multiplier}, difference is one part per {int(analytical_value/abs(numerical_value-analytical_value))}")
def vary_theta(start, end, step, R, R_s, d, h, plot=False, multiplier = 1000):
    for i in np.linspace(start, end, step):
        plot_compare(R,R_s, d, i,h, plot = plot, multiplier= multiplier)