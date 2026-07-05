"""
Deals with resulting shapes in the cross section after slicing the atmosphere orthogonal to the observation direction.
 
"""
 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation
from numba import jit
 
@jit
def two_shapes_upright(r1:np.float64, r2:np.float64, r:np.float64, theta:np.float64, z:np.float64) -> tuple[np.float64, np.float64, np.float64, np.float64]:
    """
    Geometrical information of the resulting shapes of arbitrary inclication and slice.

    Args:
        r1: true radius of first atmosphere
        r2: true radius of second atmosphere
        r: true radius of the planet (nucleus)
        theta: polar (inclination) angle of the planet measured between the normal of the hemisphere and the observation direction
        z: z-position of the slice

    Returns:
        lists of apparent radius and the horizontal cuf-off height for each of the two atmospheres
    """
    if theta < 0 or theta > np.pi:
        raise Exception("Polar angle out of range.")
    if theta > np.pi/2:
        theta = np.pi - theta
    if theta < np.arcsin(np.sqrt(r2*r2 - r*r)/r1): # region 1
        if z > np.sqrt(r1*r1 - r*r) or z < -np.sqrt(r2*r2 - r*r):
            print('z',z)
            raise Exception("z out of range. (region 1)")
        if z >=-np.sqrt(r2*r2 - r*r) and z < -r1*np.sin(theta): # region 1a
            return r, r, np.sqrt(r2*r2 - z*z), -np.sqrt(r2*r2 - z*z)
        if z >= -r1*np.sin(theta) and z < -r2*np.sin(theta): # region 1b
            return  np.sqrt(r1*r1 - z*z), -z/np.tan(theta), np.sqrt(r2*r2 - z*z), -np.sqrt(r2*r2 - z*z)
        if abs(z) <= r2*np.sin(theta): # region 1c
            return np.sqrt(r1*r1 - z*z), -z/np.tan(theta), np.sqrt(r2*r2 - z*z), z/np.tan(theta)
        if z > r2*np.sin(theta) and z < r1*np.sin(theta): # region 1d
            return np.sqrt(r1*r1 - z*z), -z/np.tan(theta),  r, r
        if z >= r1*np.sin(theta): # region 1e
            return np.sqrt(r1*r1 - z*z), -np.sqrt(r1*r1 - z*z),  r, r
    elif theta >= np.arcsin(np.sqrt(r2*r2 - r*r)/r1) and theta < np.arccos(r/r2): #region 2
        if z< -r1*np.sin(theta) or z > np.sqrt(r1*r1 - r*r):
            raise Exception("z out of range. (region 2)")
        if z >= -r1*np.sin(theta) and z < -np.sqrt(r2*r2 - r*r): # region 2a
            return np.sqrt(r1*r1 - z*z),  -z/np.tan(theta), r, r
        if z >= -np.sqrt(r2*r2 - r*r) and z < -r2*np.sin(theta): # region 2b
            return np.sqrt(r1*r1 - z*z), -z/np.tan(theta), np.sqrt(r2*r2 - z*z), -np.sqrt(r2*r2 - z*z)
        if abs(z) <= r2*np.sin(theta): # region 2c
            return np.sqrt(r1*r1 - z*z), -z/np.tan(theta), np.sqrt(r2*r2 - z*z), z/np.tan(theta)
        if z > r2*np.sin(theta) and z < r1*np.sin(theta): # region 2d
            return np.sqrt(r1*r1 - z*z), -z/np.tan(theta), r, r
        if z >= r1*np.sin(theta): # region 2e
            return np.sqrt(r1*r1 - z*z), -np.sqrt(r1*r1 - z*z), r, r
    elif theta >= np.arccos(r/r2) and theta < np.arccos(r/r1): # region 3
        if z < -r1*np.sin(theta) or z > np.sqrt(r1*r1 - r*r):
            raise Exception("z out of range. (region 3)")
        if z >= -r1*np.sin(theta) and z < -np.sqrt(r2*r2 - r*r): # region 3a
            return np.sqrt(r1*r1 - z*z), -z/np.tan(theta), r, r
        if abs(z) <= np.sqrt(r2*r2 - r*r): #region 3b
            return np.sqrt(r1*r1 - z*z), -z/np.tan(theta), np.sqrt(r2*r2 - z*z), z/np.tan(theta)
        if z > np.sqrt(r2*r2 - r*r) and z < r1*np.sin(theta): # region 3c
            return np.sqrt(r1*r1 - z*z), -z/np.tan(theta), r, r
        if z >= r1*np.sin(theta): # region 3d
            return np.sqrt(r1*r1 - z*z), -np.sqrt(r1*r1 - z*z), r, r
    elif theta >= np.arccos(r/r1) and theta <= np.pi/2: # region 4
        if abs(z) > np.sqrt(r1*r1 - r*r):
            raise Exception("z out of range. (region 4)")
        if z >= -np.sqrt(r1*r1 - r*r) and z < -np.sqrt(r2*r2 - r*r): # region 4a
            return np.sqrt(r1*r1 - z*z), -z/np.tan(theta), r, r
        if abs(z) <= np.sqrt(r2*r2 - r*r): # region 4b
            return np.sqrt(r1*r1 - z*z), -z/np.tan(theta), np.sqrt(r2*r2 - z*z), z/np.tan(theta)
        if z > np.sqrt(r2*r2 - r*r): # region 4c
            return np.sqrt(r1*r1 - z*z), -z/np.tan(theta), r, r
    else:
        raise Exception("You found a bug!")
 
@jit
def get_z_range(r1:float,r2:float,r:float,theta:float, number = 200):
    if theta < 0 or theta > np.pi:
        raise Exception("Theta out of range.")
    if theta > np.pi/2:
        theta = np.pi - theta
    if theta < np.arcsin(np.sqrt(r2*r2 - r*r)/r1): # region 1
        return np.linspace(-np.sqrt(r2*r2 - r*r), np.sqrt(r1*r1 - r*r), number)
    elif theta >= np.arcsin(np.sqrt(r2*r2 - r*r)/r1) and theta < np.arccos(r/r2): # region 2
        return np.linspace(-r1*np.sin(theta), np.sqrt(r1*r1 - r*r), number)
    elif theta >= np.arccos(r/r2) and theta < np.arccos(r/r1): # region 3
        return np.linspace(-r1*np.sin(theta), np.sqrt(r1*r1 - r*r), number)
    elif theta >= np.arccos(r/r1) and theta <= np.pi/2: # region 4
        return np.linspace(-np.sqrt(r1*r1 - r*r), np.sqrt(r1*r1 - r*r), number)
 
def draw_slice_upright_moving(r1, r2, r, theta):
    """
    Quick visualisation of the slicing process.
    """
    if theta < 0 or theta > 180:
        raise Exception("Theta out of range.")
    if theta > 90:
        theta = 180 - theta
    if theta < np.arcsin(np.sqrt(r2*r2 - r*r)/r1): # region 1
        zi = np.linspace(-np.sqrt(r2*r2 - r*r), np.sqrt(r1*r1 - r*r), 200)
    elif theta >= np.arcsin(np.sqrt(r2*r2 - r*r)/r1) and theta < np.arccos(r/r2): # region 2
        zi = np.linspace(-r1*np.sin(theta), np.sqrt(r1*r1 - r*r), 200)
    elif theta >= np.arccos(r/r2) and theta < np.arccos(r/r1): # region 3
        zi = np.linspace(-r1*np.sin(theta), np.sqrt(r1*r1 - r*r), 200)
    elif theta >= np.arccos(r/r1) and theta <= np.pi/2: # region 4
        zi = np.linspace(-np.sqrt(r1*r1 - r*r), np.sqrt(r1*r1 - r*r), 200)
    for i in zi:
        a1 = [ two_shapes_upright(r1,r2,r,theta, i)[0], two_shapes_upright(r1,r2,r,theta, i)[1] ]
        a2 =  [ two_shapes_upright(r1,r2,r,theta, i)[2], two_shapes_upright(r1,r2,r,theta, i)[3] ]
        fig= plt.figure(figsize = (10,10))
        ax = fig.add_subplot(projection='3d',aspect="equal")
        planex = i*np.ones(1000)
        planey,planez  = np.meshgrid(np.linspace(-5,5,1000),np.linspace(-5,5,1000))
        ax.plot_surface(planex, planey, planez,alpha=0.5, color='black',zorder=1)
        angle= np.linspace(0,2*np.pi,1000)
        if a1 [0] != 0:
            if a1[1] != None:
                line1 = Line2D(np.linspace(-np.sqrt(a1[0]*a1[0]-a1[1]*a1 [1]),np.sqrt(a1 [0]*a1 [0]-a1 [1]*a1 [1]),1000),ydata = np.ones(1000)*a1[1],linewidth=1,color='r')
                ax.add_line(line1)
                art3d.line_2d_to_3d(line1,zs=-5, zdir='x')
            x1 = a1 [0]*np.cos(angle)
            y1 = a1 [0]*np.sin(angle)
            if a1 [1] != None:
                x1 = x1[y1>a1 [1]]
                y1 = y1[y1>a1 [1]]
            line2 = Line2D(x1, y1,linewidth=1, color = "r")
            ax.add_line(line2)
            art3d.line_2d_to_3d(line2,zs=-5, zdir='x')
        if a2 [0] != 0:
            if a2[1] != None:
                line3 = Line2D(np.linspace(-np.sqrt(a2[0]*a2[0]-a2[1]*a2[1]),np.sqrt(a2[0]*a2[0]-a2[1]*a2[1]),1000),ydata = np.ones(1000)*a2[1],linewidth=1,color='b')
                ax.add_line(line3)
                art3d.line_2d_to_3d(line3, zs=-5, zdir='x')
            x2 = a2 [0]*np.cos(angle)
            y2 = a2 [0]*np.sin(angle)
            if a2 [1] != None:
                x2 = x2[y2<a2[1]]
                y2 = y2[y2<a2[1]]
            line4 = Line2D(x2, y2,linewidth=1,color='b')
            ax.add_line(line4)
            art3d.line_2d_to_3d(line4, zs=-5, zdir='x')
        circle = Circle((0,0),r,color='k')
        ax.set_xlim3d(-5, 5)
        ax.set_ylim3d(-5, 5)
        ax.set_zlim3d(-5, 5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.add_patch(circle)
        art3d.pathpatch_2d_to_3d(circle, z=-5, zdir="x")
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(np.size(u)), np.cos(v))
        u1 = np.linspace(np.pi/2 , 3 * np.pi/2, 100)
        v1 = np.linspace(0, np.pi, 100)
        u2 = np.linspace(-np.pi/2,np.pi/2, 100)
        v2 = np.linspace(0, np.pi, 100)
        x1 = r2 * np.outer(np.cos(u1), np.sin(v1))
        y1 = r2 * np.outer(np.sin(u1), np.sin(v1))
        z1 = r2 * np.outer(np.ones(np.size(u1)), np.cos(v1))
        x2 = r1* np.outer(np.cos(u2), np.sin(v2))
        y2 = r1 * np.outer(np.sin(u2), np.sin(v2))
        z2 = r1 * np.outer(np.ones(np.size(u2)), np.cos(v2))
        ax.plot_surface(x, y, z,color='k')
        for i in range(len(x1)):
            x1[i], z1[i] = np.cos(theta)*x1[i] - np.sin(theta)*z1[i] , +np.sin(theta)*x1[i] + np.cos(theta)*z1[i]
        for i in range(len(x2)):
            x2[i], z2[i] = np.cos(theta)*x2[i] - np.sin(theta)*z2[i] , +np.sin(theta)*x2[i] + np.cos(theta)*z2[i]
        ax.plot_surface(x1, y1, z1, edgecolor='royalblue', lw=0.8, rstride=8, cstride=8,
                    alpha=0.3)
        ax.plot_surface(x2, y2, z2, edgecolor='r', lw=0.8, rstride=8, cstride=8,
                    alpha=0.3)
        plt.pause(0.01)
        plt.close()
 
 
def animate_slice_upright(r1,r2,r,theta,sample):
    """
    Generates a movie out of the slicing process
    """
    if theta < 0 or theta > 180:
        raise Exception("Theta out of range.")
    if theta > 90:
        theta = 180 - theta
    if theta < np.arcsin(np.sqrt(r2*r2 - r*r)/r1): # region 1
        zi = np.linspace(-np.sqrt(r2*r2 - r*r), np.sqrt(r1*r1 - r*r), sample)
    elif theta >= np.arcsin(np.sqrt(r2*r2 - r*r)/r1) and theta < np.arccos(r/r2): # region 2
        zi = np.linspace(-r1*np.sin(theta), np.sqrt(r1*r1 - r*r), sample)
    elif theta >= np.arccos(r/r2) and theta < np.arccos(r/r1): # region 3
        zi = np.linspace(-r1*np.sin(theta), np.sqrt(r1*r1 - r*r), sample)
    elif theta >= np.arccos(r/r1) and theta <= np.pi/2: # region 4
        zi = np.linspace(-np.sqrt(r1*r1 - r*r), np.sqrt(r1*r1 - r*r), sample)
    fig= plt.figure(figsize = (10,10))
    ax = fig.add_subplot(projection='3d',aspect="equal")
    ax.set_xlim3d(-5, 5)
    ax.set_ylim3d(-5, 5)
    ax.set_zlim3d(-5, 5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    planex = zi[0]*np.ones(1000)
    planey,planez  = np.meshgrid(np.linspace(-5,5,1000),np.linspace(-5,5,1000))
    plane = ax.plot_surface(planex, planey, planez,alpha=0.5, color='black',zorder=1)
    line1 = Line2D([],[])
    ax.add_line(line1)
    line2 = Line2D([],[])
    ax.add_line(line2)
    line3 =Line2D([],[])
    ax.add_line(line3)
    line4 = Line2D([],[])
    ax.add_line(line4)
    
    def function(frame):
        ax.clear()
        a1 = [two_shapes_upright(r1,r2,r,theta, zi[frame])[0], two_shapes_upright(r1,r2,r,theta,  zi[frame])[1] ]
        a2 =  [two_shapes_upright(r1,r2,r,theta,  zi[frame])[2], two_shapes_upright(r1,r2,r,theta,  zi[frame])[3] ]
        planex = zi[frame]*np.ones(1000)
        planey,planez  = np.meshgrid(np.linspace(-5,5,1000),np.linspace(-5,5,1000))
        ax.plot_surface(planex, planey, planez,alpha=0.5, color='black',zorder=1)
        angle= np.linspace(0,2*np.pi,1000)
        if a1 [0] != 0:
            if a1[1] != None:
                line1 = Line2D(np.linspace(-np.sqrt(a1[0]*a1[0]-a1[1]*a1 [1]),np.sqrt(a1 [0]*a1 [0]-a1 [1]*a1 [1]),1000),ydata = np.ones(1000)*a1[1],linewidth=1,color='r')
                ax.add_line(line1)
                art3d.line_2d_to_3d(line1,zs=-5, zdir='x')
            x1 = a1 [0]*np.cos(angle)
            y1 = a1 [0]*np.sin(angle)
            if a1 [1] != None:
                x1 = x1[y1>a1 [1]]
                y1 = y1[y1>a1 [1]]
            line2 = Line2D(x1, y1,linewidth=1, color = "r")
            ax.add_line(line2)
            art3d.line_2d_to_3d(line2,zs=-5, zdir='x')
        if a2 [0] != 0:
            if a2[1] != None:
                line3 = Line2D(np.linspace(-np.sqrt(a2[0]*a2[0]-a2[1]*a2[1]),np.sqrt(a2[0]*a2[0]-a2[1]*a2[1]),1000),ydata = np.ones(1000)*a2[1],linewidth=1,color='b')
                ax.add_line(line3)
                art3d.line_2d_to_3d(line3, zs=-5, zdir='x')
            x2 = a2 [0]*np.cos(angle)
            y2 = a2 [0]*np.sin(angle)
            if a2 [1] != None:
                x2 = x2[y2<a2[1]]
                y2 = y2[y2<a2[1]]
            line4 = Line2D(x2, y2,linewidth=1,color='b')
            ax.add_line(line4)
            art3d.line_2d_to_3d(line4, zs=-5, zdir='x')
        circle = Circle((0,0),r,color='k')
        ax.set_xlim3d(-5, 5)
        ax.set_ylim3d(-5, 5)
        ax.set_zlim3d(-5, 5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.add_patch(circle)
        art3d.pathpatch_2d_to_3d(circle, z=-5, zdir="x")
 
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(np.size(u)), np.cos(v))
        u1 = np.linspace(np.pi/2 , 3 * np.pi/2, 100)
        v1 = np.linspace(0, np.pi, 100)
        u2 = np.linspace(-np.pi/2,np.pi/2, 100)
        v2 = np.linspace(0, np.pi, 100)
        x1 = r2 * np.outer(np.cos(u1), np.sin(v1))
        y1 = r2 * np.outer(np.sin(u1), np.sin(v1))
        z1 = r2 * np.outer(np.ones(np.size(u1)), np.cos(v1))
        x2 = r1* np.outer(np.cos(u2), np.sin(v2))
        y2 = r1 * np.outer(np.sin(u2), np.sin(v2))
        z2 = r1 * np.outer(np.ones(np.size(u2)), np.cos(v2))
        ax.plot_surface(x, y, z,color='k')
        for i in range(len(x1)):
            x1[i], z1[i] = np.cos(theta)*x1[i] - np.sin(theta)*z1[i] , +np.sin(theta)*x1[i] + np.cos(theta)*z1[i]
        for i in range(len(x2)):
            x2[i], z2[i] = np.cos(theta)*x2[i] - np.sin(theta)*z2[i] , +np.sin(theta)*x2[i] + np.cos(theta)*z2[i]
        ax.plot_surface(x1, y1, z1, edgecolor='royalblue', lw=0.8, rstride=8, cstride=8,
                    alpha=0.3)
        ax.plot_surface(x2, y2, z2, edgecolor='r', lw=0.8, rstride=8, cstride=8,
                    alpha=0.3)
        ax.set_title(f"Polar angle $\\theta = ${theta}, $r_1$ = {r1}, $r_2$ = {r2}, $r$ = {r}")
    ani = FuncAnimation(fig, function, interval = 200, frames=sample)
    ani.save('animations/slice.mp4', fps = 60)