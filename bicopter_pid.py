# @file bicopter_pid.py
# @details 
#  This is a rigid body model of an bicopter.
#   It must eventually be expanded to model control
#   by PWM drive of two asymmetric brushed DC motors.
#  
#  The fact that the motors are assumed radially symmetric 
#   about the COM is irrelevant; eventually, when we deal
#   in PWM inputs, instead of thrust inputs, the symmetry 
#   can be broken in the PWM->thrust transfer functions. 
# 
# @limitations
#  1. Assumes instantaneous perfect thrust control
#  2. Assumes perfect knowledge of state
# 
# @author Sushant Sundaresh
# @date 6 June 2017

##############
# References #
##############

# 1. https://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/

################
# Dependencies #
################

import sys
from numpy import sin, cos, pi, array, asarray, matmul, linspace
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time

############
# Dynamics #
############

# Let us consider the orientaton of a bicopter as (q,x,y) 
# which are, respectively, the CCW > 0 rotation in radians
# of the bicopter with respect to the Cartesian x-axis,
# the x-position, and the y-position, both in meters.
#
# Take the state to be k = (pq, q, 1/p e_q, px, x, 1/p e_x, py, y, 1/p e_y)
# where 
#   p = d/dt
#   1/p is the inverse of p (the integral from [0,t] with respect to t)
#   e_(q,x,y) = (q,x,y) - (q_r, x_r, y_r) 
#   *_r = set point for * 
#
# Consider pk = (p^2 q, pq, e_q, p^2 x, px, e_x, p^2 y, py, e_y)
# and note that if we can compute p^2 (q,x,y) from the kinematics & controllers,
# all of pk is known, so we can numerically integrate the 1st order system.
#
# So.. let's derive the kinematics & specify our controllers! 
#
# Consider a radially symmetric bicopter (motors equidistant at r from COM)
# with moment of inertia about the COM given by J and total mass m.
# The equations of motion become:
#
# p^2 q = (r/J) * Fdiff 
# p^2 x = -(1/m) * Fsum * sin(q)
# p^2 y = (1/m) * Fsum * cos(q) - g
#
# where r * Fdiff is the net torque around the COM and Fsum is the total thrust, perpendicular to the body.
# We must control both of these quantities along with q_r such that the thrust from each motor is
# strictly in 0 <= F <= mg and we approach (q,x,y) = (0, x_r, y_r) asymptotically.
# 
# Notice Fsum = Fr + Fl, Fdiff = Fr - Fl can be resolved into Fr, Fl as 
# Fr = 0.5 * (Fsum + Fdiff) and Fl = 0.5 * (Fsum - Fdiff)
# where we enforce |Fdiff| < Fsum & 0 <= F(r,l) <= mg always.
# 
# Consider independent PID control of (Fdiff, Fsum) as follows, assuming the set points change 
# as step functions, so p(q_r, x_r, y_r) ~ 0 for our purposes. Compute the following in order:
#
#  ie_x = 1/p e_x                                       bounded to [-awx, awx] meters as an anti-windup measure
#  ie_y = 1/p e_y                                       bounded to [-awy, awy] meters as an anti-windup measure
#  Fsum = Kpx * e_x + Kdx * px + Kix * ie_x +       
#         Kpy * e_y + Kdy * py + Kiy * ie_y             bounded to [0, max_thrust]
#  q_r = Kpx_ * e_x + Kdx_ * px                         bounded to [-max_q, max_q]
#  ie_q = 1/p * e_q                                     bounded to [-awq, awq] as an anti-windup measure
#  Fdiff = Kpq * e_q + Kdq * pq + Kiq * ie_q            bounded to |Fdiff| < Fsum & 0 <= F(r,l) <= max_thrust/2         
# 
#  We can impose the constraints as:
#   p ie_x = e_x if (ie_x >= -awx and ie_x <= awx) or (ie_x < -awx and e_x > 0) or (ie_x > awx and e_x < 0) else 0
#   p ie_y = e_y if (ie_y >= -awy and ie_y <= awy) or (ie_y < -awy and e_y > 0) or (ie_y > awy and e_y < 0) else 0
#   q_r = max(-max_q, min(Kpx_ * e_x + Kdx_ * px, max_q))
#   p ie_q = e_q if (ie_q >= -awq and ie_q <= awq) or (ie_q < -awq and e_q > 0) or (ie_q > awq and e_q < 0) else 0
# 
#   Fsum' = max(0, min(max_thrust, Fsum))
#   Fdiff' = sign(Fdiff) * min(max_thrust/2, Fsum', abs(Fdiff))
#
#   Fr = max(0, min(mg, 0.5 * (Fsum' + Fdiff'))
#   Fl = max(0, min(mg, 0.5 * (Fsum' - Fdiff'))
# 
#   Fsum'' = Fr + Fl
#   Fdiff'' = Fr - Fl

########
# Goal #
########

# Our goal is to find a stable controller within the space of our 11 parameters 
# which has an overshoot of <20%, a 2% settling time of <5 seconds for a unit meter step in any direction,
# and robustness to sensor noise & error in our estimation of r/J and m.

class RigidBicopter:

    def model(self):    
        # Compute the center of mass and moment of inertia relative to the center of mass
        self.g_m_per_s2 = 9.8 
        self.l_m = 0.228 # This is about 9 inches

        self.r_m = {"le": 0.000, "ri" : self.l_m} # absolute distance of left motor, right motor from left edge of chassis
        self.kg = {"l" : 0.005, "r" : 0.005, "c" : 0.050} # left motor, right motor, chassis

        # This is the total mass of the drone
        self.kg["d"] = self.kg["l"] + self.kg["r"] + self.kg["c"]
        # This is the line density of the drone
        self.p_kg_per_m = self.kg["c"] / self.l_m

        # This is the center of mass of the drone
        self.r_m["com"] = self.r_m["le"]*self.kg["l"] 
        self.r_m["com"] += self.r_m["ri"]*self.kg["r"]
        self.r_m["com"] += 0.5*self.p_kg_per_m*(self.l_m**2)
        self.r_m["com"] /= self.kg["d"]        

        # This is the position of the left and right motors relative to the center of mass
        self.r_m["~le"] = self.r_m["com"]- self.r_m["le"]
        self.r_m["~ri"] = self.r_m["ri"] - self.r_m["com"]
        self.r_m["r"] = self.r_m["~le"]               # Lever arms are equivalent, since this is a symmetric bicopter

        # This is the moment of inertia of the drone about its center of mass
        self.J_kg_m2 = (self.r_m["r"]**2)*self.kg["l"]
        self.J_kg_m2 += (self.r_m["r"]**2)*self.kg["r"]
        self.J_kg_m2 += ((1.0/3)*self.l_m**2 + self.r_m["com"]**2 - self.r_m["com"]*self.l_m)*self.p_kg_per_m*self.l_m

    def __init__(self):        
        self.model()    

        self.t_s = 0        # simulation time
        self.fl_g = 0       # left thrust in gravities
        self.fr_g = 0       # right thrust in gravities

        self.max_thrust  = 2*self.kg["d"]*self.g_m_per_s2
        self.max_q       = pi/12
        self.awq         = pi/2
        self.awx         = 3 
        self.awy         = 3
        self.Kpx         = 2
        self.Kdx         = 0.5
        self.Kix         = 0.05
        self.Kpy         = -2
        self.Kdy         = -0.5
        self.Kiy         = -0.05
        self.Kpx_        = 2
        self.Kdx_        = 0.5
        self.Kpq         = -2
        self.Kdq         = -0.5
        self.Kiq         = -0.05
        
        self.x_r = 0 # m
        self.y_r = 0 # m 
        self.x_0 = -1 # m 
        self.y_0 = -1 # m
        self.k = asarray([0, 0, 0, 0, self.x_0, 0, 0, self.y_0, 0])
    
    # This draws the line that is the bicopter
    def draw(self):        
        (pq, q, ie_q, px, x, ie_x, py, y, ie_y) = self.k
        bicopterx = asarray([x - self.r_m["~le"]*cos(q), x + self.r_m["~ri"]*cos(q)])
        bicoptery = asarray([y - self.r_m["~le"]*sin(q), y + self.r_m["~ri"]*sin(q)])
        return (bicopterx, bicoptery)

    # Here we have reduced the system to first order in k
    def pk(self, k_t, t):        
        (pq, q, ie_q, px, x, ie_x, py, y, ie_y) = k_t

        e_x = x - self.x_r
        e_y = y - self.y_r

        q_r = max(-self.max_q, min(self.Kpx_ * e_x + self.Kdx_ * px, self.max_q))
        e_q = q - q_r

        pie_x = e_x if ((ie_x >= -self.awx and ie_x <= self.awx) or (ie_x < -self.awx and e_x > 0) or (ie_x > self.awx and e_x < 0)) else 0
        pie_y = e_y if ((ie_y >= -self.awy and ie_y <= self.awy) or (ie_y < -self.awy and e_y > 0) or (ie_y > self.awy and e_y < 0)) else 0
        pie_q = e_q if ((ie_q >= -self.awq and ie_q <= self.awq) or (ie_q < -self.awq and e_q > 0) or (ie_q > self.awq and e_q < 0)) else 0

        f_sum =  self.Kpx * e_x + self.Kdx * px + self.Kix * ie_x
        f_sum += self.Kpy * e_y + self.Kdy * py + self.Kiy * ie_y 
        f_diff = self.Kpq * e_q + self.Kdq * pq + self.Kiq * ie_q

        f_sum = max(0, min(self.max_thrust, f_sum))
        sign_f_diff = 1 if f_diff >= 0 else -1
        f_diff = sign_f_diff * min(self.max_thrust/2, f_sum, abs(f_diff))

        self.fr = max(0, min(self.max_thrust/2, 0.5 * (f_sum + f_diff)))
        self.fl = max(0, min(self.max_thrust/2, 0.5 * (f_sum - f_diff)))

        f_sum = self.fr + self.fl 
        f_diff = self.fr - self.fl
        
        p2q = (self.r_m["r"] / self.J_kg_m2) * f_diff
        p2x = -(1/self.kg["d"])*f_sum*sin(q)
        p2y = (1/self.kg["d"])*f_sum*cos(q) - self.g_m_per_s2

        return [p2q, pq, pie_q, p2x, px, pie_x, p2y, py, pie_y]

    def step (self, dt):
        self.k = integrate.odeint(self.pk, self.k, [0, dt])[1]
        self.t_s += dt
        self.fl_g = self.fl / (self.kg["d"]*self.g_m_per_s2)
        self.fr_g = self.fr / (self.kg["d"]*self.g_m_per_s2)

###################################################
# Animation code entirely copied from Reference 1 #
###################################################

bicopter = RigidBicopter()
dt = 1./60      # 60 fps animation
                # Control loop is likely updated far faster (numerically 'continuous')
                #  as odeint is likely an adaptive step length integrator

fig = plt.figure()
ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-10,10), ylim=(-10,10))
ax.grid()

bicopter_body, = ax.plot([], [], 'o-', lw=2)
time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
left_thrust_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
right_thrust_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)

def init():
    bicopter_body.set_data([],[])
    time_text.set_text('')
    left_thrust_text.set_text('')
    right_thrust_text.set_text('')
    return bicopter_body, time_text, left_thrust_text, right_thrust_text

def animate(i):
    global bicopter, dt
    # This lets us repeat the simulation from the start, if we wish
    # if i == 0:
    #     bicopter = RigidBicopter()
    bicopter.step(dt)    
    bicopter_body.set_data(*bicopter.draw())        
    time_text.set_text('time = %0.1f s' % bicopter.t_s)
    left_thrust_text.set_text('Fl = %0.5f g' % bicopter.fl_g) 
    right_thrust_text.set_text('Fr = %0.5f g' % bicopter.fr_g)
    return bicopter_body, time_text, left_thrust_text, right_thrust_text

# This delays between frames so we hit our fps target
t0 = time()
animate(0)
t1 = time()
interval = 1000 * dt - (t1 - t0)

ani = animation.FuncAnimation(fig, animate, frames=200, repeat=True, interval=interval, blit=True, init_func=init)
#ani.save('bicopter.mp4', fps=60, extra_args=['-vcodec', 'libx264'])

plt.show()
