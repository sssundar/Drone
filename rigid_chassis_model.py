#!python

# References
# 1. https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/

import sys
from numpy import sin, cos, pi, array, asarray, matmul, linspace
import numpy as np
import scipy.integrate as integrate
from time import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# This is a rigid body model of a symmetric bicopter. 
class RigidBicopter:

    # Compute the center of mass and moment of inertia relative to the center of mass
    def model(self):    
        # This is the total mass of the drone
        self.kg["d"] = self.kg["l"] + self.kg["r"] + self.kg["c"]
        # This is the line density of the drone
        self.p_kg_per_m = self.kg["c"] / self.l_m

        # This is the center of mass of the drone
        self.r_m["com"] = self.r_m["l"]*self.kg["l"] 
        self.r_m["com"] += self.r_m["r"]*self.kg["r"]
        self.r_m["com"] += 0.5*self.p_kg_per_m*(self.l_m**2)
        self.r_m["com"] /= self.kg["d"]        

        # This is the position of the left and right motors relative to the center of mass
        self.r_m["~l"] = self.r_m["com"]- self.r_m["l"]
        self.r_m["~r"] = self.r_m["r"] - self.r_m["com"]

        # This is the moment of inertia of the drone about its center of mass
        self.J_kg_m2 = (self.r_m["~l"]**2)*self.kg["l"]
        self.J_kg_m2 += (self.r_m["~r"]**2)*self.kg["r"]
        self.J_kg_m2 += ((1.0/3)*self.l_m**2 + self.r_m["com"]**2 - self.r_m["com"]*self.l_m)*self.p_kg_per_m*self.l_m
    
    # The state is 
    # [d/dt theta, d/dt x, d/dt y, theta, x, y]
    # where theta is the angle of the bicopter (CCW > 0) 
    # relative to the inertial x-axis
    def __init__(self, init_state = [0,0,0,0,0,0]):
        self.g_m_per_s2 = -9.8 
        self.l_m = 0.228 # This is about 9 inches
        
        # You can tweak these to make the bicopter asymmetric
        # Roughly speaking, expect 10% error in the positioning or mass of motors 1/10 the mass of the chassis
        # to cause a 1% change in the moment of inertia and center of mass.
        self.r_m = {"l": 0.000, "r" : self.l_m} # absolute distane of left motor, right motor from left edge of chassis
        self.kg = {"l" : 0.005, "r" : 0.005, "c" : 0.050} # left motor, right motor, chassis
        self.model()                
        
        # Simulation (time, state)
        self.t_s= 0
        self.state = asarray(init_state)

    # This draws the line that is the bicopter
    def draw(self):        
        (thetadot, xdot, ydot, theta, x, y) = self.state
        bcx = asarray([x - self.r_m["~l"]*cos(theta), x + self.r_m["~r"]*cos(theta)])
        bcy = asarray([y - self.r_m["~l"]*sin(theta), y + self.r_m["~r"]*sin(theta)])
        return (bcx,bcy)

    # This is a simple rigid body model of a bicopter ignoring drag
    # All we consider are the orientation of the bicopter, 
    # its moment of inertia and mass, and the forces from the propellers & gravity.
    def dstate_dt(self, state, t, F):        
        (theta_dot, x_dot, y_dot, theta, x, y) = state
        T = array([ [(self.r_m["~r"]/self.J_kg_m2),   -(self.r_m["~l"]/self.J_kg_m2),   0],
                    [-(1/self.kg["d"])*sin(theta),    -(1/self.kg["d"])*sin(theta),     0],
                    [(1/self.kg["d"])*cos(theta),     (1/self.kg["d"])*cos(theta),      (1/self.kg["d"])] ])        
        d2Ydt2 = matmul(T,F)
        return [d2Ydt2[0], d2Ydt2[1], d2Ydt2[2], theta_dot, x_dot, y_dot]        

    # For the moment, we fix the thrust forces and gravity.
    # We could introduce a controller by making F_n a function of the observed (aka noisy, estimated) state
    def step(self, dt):    
        F_g = self.kg["d"] * self.g_m_per_s2    
        F_n = array([-1.02*F_g/2, -1.00*F_g/2, F_g])     # Right motor thrust, left motor thrust, gravitational force
        self.state = integrate.odeint(self.dstate_dt, self.state, [0, dt], args=(F_n,))[1]
        self.t_s += dt

###################################################
# Animation code entirely copied from Reference 1 #
###################################################

bicopter = RigidBicopter()
dt = 1./60 # fps

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-30,10), ylim=(-20,10))
ax.grid()

line, = ax.plot([], [], 'o-', lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

def init():
    line.set_data([],[])
    time_text.set_text('')
    return line, time_text

def animate(i):
    global bicopter, dt
    # This lets us repeat the simulation instead of having the bicopter fly out of frame
    if i == 0:
        bicopter = RigidBicopter()
    bicopter.step(dt)    
    line.set_data(*bicopter.draw())        
    time_text.set_text('time = %0.1f' % bicopter.t_s)
    return line, time_text

# This delays between frames so time in real life is time in the animation
t0 = time()
animate(0)
t1 = time()
interval = 1000 * dt - (t1 - t0)

ani = animation.FuncAnimation(fig, animate, frames=200, repeat=True, interval=interval, blit=True, init_func=init)
#ani.save('bicopter.mp4', fps=60, extra_args=['-vcodec', 'libx264'])

plt.show()