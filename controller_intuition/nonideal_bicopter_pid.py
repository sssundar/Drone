# @file nonideal_bicopter_pid.py
# @details 
#  This is a rigid body model of an bicopter.
#   It is actuated by PWM drive of two asymmetric brushed DC motors.
# 
# @limitations
#  1. Assumes perfect knowledge of state
# 
# @author Sushant Sundaresh
# @date 21 June 2017

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

################
# New Dynamics #
################

# The dynamics and controller equations for the ideal bicopter are described in bicopter_pid.py.
# Here, let's consider the impact of non-ideal motors which respond to a duty cycle input [0,1] asymmetrically.

# Let's neglect the electrical response of each motor; that will die down orders of magnitude faster than
# the mechanical response. In that case each motor can be modeled as a resistor in series with an EMF proportional
# to rotor frequency. 

# Let 
#  w be the motor frequency
#  J be the moment of inertia of the motor about it's rotational axis
#  d be the duty cycle input in [0,1]
#  T be the maximum torque (d = 1, w = 0)
#  ku be the dynamic friction (quadratic in w probably, but linearizable in practice at our speeds)
#  ke be the proportionality constant between w and that motor's EMF
#  R be the winding resistance of the motor
# 
# We will find, assuming motor current proportional to motor torque, and
# propellor drag proportional to motor frequency...
# 
#  J pw = d T - (ku + (ke/R) d) w     
#       
# which looks simple until you realize the time constant depends on the duty cycle.
# This means we may take significantly longer to slow down, than to speed up.
# 
# Let's pick J, T, ku, ke/R arbitrarily so that for d in [0,1] we have 
#  a. w_max = 20rps = 1200rpm
#  b. 0->1 taking 1 s to reach steady state in thrust
#  c. 1->0 taking 2 s to reach steady state in thrust
# 
# The model is self-contained (only its dynamics matter to the rest of the system) so 
# our numbers don't matter. 

# Let the force output of the motor-propeller system be proportional to w^2
# with proportionality constant km. Pick this so that w_max translates to f_max. Allow 
# the maximum force to be tuned to each motor, so we can introduce asymmetry.

# How do we integrate this into our motor for the bicopter & its controller?
# Well, in the ideal case, our controller output forces directly. 
# Let's scale those forces down to [0,1] by dividing by mg, our controller's idealization.
# Then we'll feed it through motor dynamics (new system state) to yield our actual force
# output. 

# Let's see how our simple PID controller holds up! 

####################
# Motor Simulation #
####################

# First, let's parameterize our motor.


class Motor:

    # J pw = d T - (ku + (ke/R) d) w     
    # Realize R doesn't matter. Just call ke/R 'ke'
    # At d=1, pw = 0, w = w_max.
    # This means T = (ku + ke) w_max, or 
    # let's just pick ku and ke and derive T from that,
    # then tune ku, ke to get our timing requirements.

    # For J, 0.00034656 kg*m^2 is the moment of inertia of a 50 g bicopter chassis with 5 gram motors.
    # Our motor moments are likely to be FAR smaller than that. Let's say.. 30x smaller.
    # So take J ~ 0.00001 kg * m^2
    def model(self):    
        self.w_max = 40*pi  # Maximum propellor angular velocity (at our operating voltage)
        self.f_max = 0.06*9.8 # Maximum propellor thrust (at w_max) in kg*m/s^2
        
        self.km = self.f_max / ((self.w_max)**2) # w^2-Thrust coefficient
        
        self.ku = 0.00001     # w-Drag Torque coefficient. kg*m^2/s
        self.ke = 0.00003     # w-EMF Torque coefficient. kg*m^2/s
        self.J =  0.00001     # Moment of inertia about rotational axis. kg*m^2
        
        self.T = (self.ku+self.ke)*self.w_max # Maximum electromagnetic torque. kg m^2 / s^2
        
    def thrust(self):
        return self.km*((self.w)**2)

    def __init__(self, ramp):        
        self.model()    

        self.t_s = 0        # simulation time

        if ramp == "up":
            self.d = 1          # fixed input duty cycle
            self.w = 0          # propellor rotational velocity        
        else:
            self.d = 0
            self.w = self.w_max

        self.f = self.thrust()

    def pw(self, w_t, t):        
        dwdt = (1.0/self.J) * (self.d*self.T - (self.ku + self.ke*self.d) * w_t)
        return dwdt

    def step (self, dt):
        self.w = integrate.odeint(self.pw, self.w, [0, dt])[1]
        self.t_s += dt
        self.f = self.thrust()
        return (self.t_s, self.w, self.f)

def SimulateMotor(ramp):
    m = Motor(ramp)
    dt = 0.020  # seconds
    t_max = 5.0 # seconds

    time_s = [0]
    omega = [m.w/m.w_max]
    thrust = [m.f/m.f_max]

    for k in xrange(int(t_max/dt)):
        (t_s, w, f) = m.step(dt)
        time_s.append(t_s)
        omega.append(w/m.w_max)
        thrust.append(f/m.f_max)

    return (time_s, omega, thrust)

def VisualizeMotor():
    (tu, wu, fu) = SimulateMotor("up")
    (td, wd, fd) = SimulateMotor("down")
    plt.plot(tu,wu,'r-')
    plt.plot(tu,fu,'r--')
    plt.plot(td,wd,'b-')
    plt.plot(td,fd,'b--')
    plt.xlabel("Seconds")
    plt.legend(["Ramp Up (w/w_max)", "Ramp up (f/f_max)", "Ramp Down (w/w_max)", "Ramp Down (f/f_max)"])
    plt.show()

if __name__ == "__main__":
    VisualizeMotor()
    sys.exit(0)