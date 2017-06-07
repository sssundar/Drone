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

################
# Dependencies #
################

import sys
from numpy import sin, cos, pi, array, asarray, matmul, linspace
import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

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
#  ie_x = 1/p e_x 										bounded to [-awx, awx] meters as an anti-windup measure
#  ie_y = 1/p e_y 										bounded to [-awy, awy] meters as an anti-windup measure
#  Fsum = Kpx * e_x + Kdx * px + Kix * ie_x + 		
#  		  Kpy * e_y + Kdy * py + Kiy * ie_y 			bounded to [0, max_thrust]
#  q_r = Kpx_ * e_x + Kdx_ * px 						bounded to [-max_q, max_q]
#  ie_q = 1/p * e_q 									bounded to [-awq, awq] as an anti-windup measure
#  Fdiff = Kpq * e_q + Kdq * pq + Kiq * ie_q 			bounded to |Fdiff| < Fsum & 0 <= F(r,l) <= max_thrust/2 		
# 
#  We can impose the constraints as:
#   p ie_x = e_x if (ie_x >= -awx and ie_x <= awx) or (ie_x < -awx and e_x > 0) or (ie_x > awx and e_x < 0) else 0
#   p ie_y = e_y if (ie_y >= -awy and ie_y <= awy) or (ie_y < -awy and e_y > 0) or (ie_y > awy and e_y < 0) else 0
#   q_r = max(-max_q, min(Kpx_ * e_x + Kdx_ * px, max_q))
#   p ie_q = e_q if (ie_q >= -awq and ie_q <= awq) or (ie_q < -awq and e_q > 0) or (ie_q > awq and e_q < 0) else 0
# 
#   Fsum' = max(0, min(max_thrust, Fsum))
#  	Fdiff' = sign(Fdiff) * min(max_thrust/2, Fsum', abs(Fdiff))
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

############################
# Rigid Body Approximation #
############################ 

# Compute the center of mass and moment of inertia relative to the center of mass
g_m_per_s2 = -9.8 
l_m = 0.228 # This is about 9 inches

r_m = {"le": 0.000, "ri" : l_m} # absolute distance of left motor, right motor from left edge of chassis
kg = {"l" : 0.005, "r" : 0.005, "c" : 0.050} # left motor, right motor, chassis

if (r_m["le"] != r_m["ri"]):
    print "Error: asymmetric bicopter not yet supported."
    sys.exit(1)

# This is the total mass of the drone
kg["d"] = kg["l"] + kg["r"] + kg["c"]
# This is the line density of the drone
p_kg_per_m = kg["c"] / l_m

# This is the center of mass of the drone
r_m["com"] = r_m["le"]*kg["l"] 
r_m["com"] += r_m["ri"]*kg["r"]
r_m["com"] += 0.5*p_kg_per_m*(l_m**2)
r_m["com"] /= kg["d"]        

# This is the position of the left and right motors relative to the center of mass
r_m["~le"] = r_m["com"]- r_m["le"]
r_m["~ri"] = r_m["ri"] - r_m["com"]
r_m["r"] = r_m["~le"]               # Lever arms are equivalnet, since this is a symmetric bicopter

# This is the moment of inertia of the drone about its center of mass
J_kg_m2 = (r_m["r"]**2)*kg["l"]
J_kg_m2 += (r_m["r"]**2)*kg["r"]
J_kg_m2 += ((1.0/3)*l_m**2 + r_m["com"]**2 - r_m["com"]*l_m)*p_kg_per_m*l_m

####################
# Parameterization #
####################

max_thrust 	= 2*mg
max_q 		= pi/12
awq 		= pi/2
awx 		= 3 
awy 		= 3
Kpx 		= 2
Kdx 		= 0.5
Kix 		= 0.05
Kpy 		= 2
Kdy 		= 0.5
Kiy 		= 0.05
Kpx_ 		= 2
Kdx_ 		= 0.5
Kpq 		= 2
Kdq 		= 0.5
Kiq 		= 0.05

###############
# Integration #
###############

# Simulate the system above for the parameterization given, for 100 seconds. 
# Sample the state (q, x, y) over time at 50 Hz. 

# Here we have reduced the system to first order in k
def pk(self, k, t):        
    (pq, q, ie_q, px, x, ie_x, py, y, ie_y) = k

    q_r =
    e_q = 

    e_x = 
    e_y = 

    f_sum = 
   	# ...
    
    fr = 
    fl = 

    fsum = 
    fdiff = 
    
    p2q = 
    p2x = 
    p2y = 

    return [p2q, pq, e_q, p2x, px, e_x, p2y, py, e_y]

#################
# Visualization #
#################

# Plot q,x,y over time. We may want to save integral state & internal references as well, from the controllers. 

#############
# Animation #
#############

# When we've found a suitable controller, let's animate it for fun, 
# as in the script rigid_chassis_model.py.