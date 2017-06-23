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
# output. Bear in mind... unless we give our controller a way to observe the state of the 
# motor (RPM) we don't really have any information other than what we were already
# using in bicopter_pid.py.

# Let's see how our simple PID controller holds up! 

##########################
# Motor Parametric Model #
##########################

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
    def model(self, w_max, f_max, ku, ke, J):    
        self.w_max = w_max  # Maximum propellor angular velocity (at our operating voltage)
        self.f_max = f_max  # Maximum propellor thrust (at w_max) in kg*m/s^2
        
        self.km = self.f_max / ((self.w_max)**2) # w^2-Thrust coefficient
        
        self.ku = ku     # w-Drag Torque coefficient. kg*m^2/s
        self.ke = ke     # w-EMF Torque coefficient. kg*m^2/s
        self.J =  J      # Moment of inertia about rotational axis. kg*m^2
        
        self.T = (self.ku+self.ke)*self.w_max # Maximum electromagnetic torque. kg m^2 / s^2
        
    def thrust(self, w):
        return self.km*(w**2)

    def set_dutycycle(self, dc):
        self.d = dc
    
    def set_w(self, w):
        self.w = w

    def __init__(self, w_max, f_max, ku, ke, J): 
        self.model(w_max, f_max, ku, ke, J)       
        self.t_s = 0        # simulation time

    def pw(self, w_t, t):        
        dwdt = (1.0/self.J) * (self.d*self.T - (self.ku + self.ke*self.d) * w_t)
        return dwdt

    def step (self, dt):
        self.w = integrate.odeint(self.pw, self.w, [0, dt])[1]
        self.t_s += dt
        self.f = self.thrust(self.w)
        return (self.t_s, self.w, self.f)

def SimulateMotor(ramp):
    m = Motor(w_max=40*pi, f_max=0.06*9.8, ku=0.00001, ke=0.00003, J=0.00001)

    if ramp == "up":
        m.set_dutycycle(1)            # fixed input duty cycle
        m.set_w(0)                    # propellor rotational velocity        
    else:
        m.set_dutycycle(0)            # fixed input duty cycle
        m.set_w(m.w_max)              # propellor rotational velocity        

    dt = 0.020  # seconds
    t_max = 5.0 # seconds

    time_s = [0]
    omega = [m.w/m.w_max]
    thrust = [m.thrust(m.w)/m.f_max]

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

#############################
# Updated Bicopter Dynamics #
#############################

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

        # Simulation State
        self.t_s = 0        # simulation time
        self.fl_g = 0       # left thrust in gravities
        self.fr_g = 0       # right thrust in gravities

        # Controller Parameters
        self.max_thrust  = 2*self.kg["d"]*self.g_m_per_s2       # motor limits
        self.max_q       = pi/12                                # sanity bound
        self.Kpy         = -5                                   # thrust controller
        self.Kdy         = -2
        self.Kiy         = -5
        self.Kpx_        = 2                                    # reference pitch controller 
        self.Kdx_        = 2
        self.Kpq         = -2                                   # torque controller
        self.Kdq         = -0.5
        self.Kiq         = -0.05
        self.awq         = pi/2                                 # antiwindup bounds       
        self.awy         = 1.2*self.kg["d"]*self.g_m_per_s2/abs(self.Kiy) if abs(self.Kiy) > 0 else 0

        # Left Motor 
        self.left_motor = Motor(w_max=40*pi, f_max=self.max_thrust/2, ku=0.00001, ke=0.00003, J=0.00001)

        # Right Motor
        self.right_motor = Motor(w_max=40*pi, f_max=self.max_thrust/2, ku=0.00001, ke=0.00003, J=0.00001)

        # Reference Input
        self.x_r = 0 # m
        self.y_r = 0 # m 

        # Initial Conditions
        self.x_0 = -2 # m 
        self.y_0 = 4 # m

        # The initial state.
        self.k = asarray([0, 0, 0, 0, self.x_0, 0, self.y_0, 0, 0, 0])
    
    # This draws the line that is the bicopter
    def draw(self):        
        (pq, q, ie_q, px, x, py, y, ie_y, w_r, w_l) = self.k
        bicopterx = asarray([x - self.r_m["~le"]*cos(q), x + self.r_m["~ri"]*cos(q)])
        bicoptery = asarray([y - self.r_m["~le"]*sin(q), y + self.r_m["~ri"]*sin(q)])
        return (bicopterx, bicoptery)

    # Here we have reduced the system to first order in k
    def pk(self, k_t, t):        
        (pq, q, ie_q, px, x, py, y, ie_y, w_r, w_l) = k_t

        # It's important to see whether we're close to wind up
        # so let's save the following state for display
        self.ie_y = ie_y
        self.ie_q = ie_q

        e_x = x - self.x_r
        e_y = y - self.y_r

        q_r = max(-self.max_q, min(self.Kpx_ * e_x + self.Kdx_ * px, self.max_q))
        e_q = q - q_r

        pie_y = e_y if ((ie_y >= -self.awy and ie_y <= self.awy) or (ie_y < -self.awy and e_y > 0) or (ie_y > self.awy and e_y < 0)) else 0
        pie_q = e_q if ((ie_q >= -self.awq and ie_q <= self.awq) or (ie_q < -self.awq and e_q > 0) or (ie_q > self.awq and e_q < 0)) else 0

        f_sum  = self.Kpy * e_y + self.Kdy * py + self.Kiy * ie_y 
        f_diff = self.Kpq * e_q + self.Kdq * pq + self.Kiq * ie_q

        f_sum = max(0, min(self.max_thrust, f_sum))        
        sign_f_diff = 1 if f_diff >= 0 else -1
        f_diff = sign_f_diff * min(self.max_thrust/2, f_sum, abs(f_diff))

        self.fr = max(0, min(self.max_thrust/2, 0.5 * (f_sum + f_diff)))
        self.fl = max(0, min(self.max_thrust/2, 0.5 * (f_sum - f_diff)))

        # Normalize the intended instantaneous force into a duty cycle output for each motor in [0,1]
        self.right_motor.set_dutycycle(self.fr / (self.max_thrust/2))
        self.left_motor.set_dutycycle(self.fl / (self.max_thrust/2))

        # Feed this input into each motor (electronics stablize more or less instantly from mechanical POV)
        pwr = self.right_motor.pw(w_r, t)
        pwl = self.left_motor.pw(w_l, t)
        self.fr = self.right_motor.thrust(w_r)
        self.fl = self.left_motor.thrust(w_l)

        f_sum = self.fr + self.fl 
        f_diff = self.fr - self.fl
        
        p2q = (self.r_m["r"] / self.J_kg_m2) * f_diff
        p2x = -(1/self.kg["d"])*f_sum*sin(q)
        p2y = (1/self.kg["d"])*f_sum*cos(q) - self.g_m_per_s2

        return [p2q, pq, pie_q, p2x, px, p2y, py, pie_y, pwr, pwl]

    def step (self, dt):
        self.k = integrate.odeint(self.pk, self.k, [0, dt])[1]
        self.t_s += dt
        self.fl_g = self.fl / (self.kg["d"]*self.g_m_per_s2)
        self.fr_g = self.fr / (self.kg["d"]*self.g_m_per_s2)

###################################################
# Animation code entirely copied from Reference 1 #
###################################################

def init():
    bicopter_body.set_data([],[])
    time_text.set_text('')
    target_text.set_text('')
    yi_text.set_text('')
    yd_text.set_text('')
    yp_text.set_text('')
    return bicopter_body, time_text, target_text, yi_text, yd_text, yp_text

def animate(i):
    global bicopter, dt, aim

    if i % 480 == 0:
        # When the animation is due to 'repeat,' shift our reference
        if aim == [-2,-2]:
            aim = [2,2]
        elif aim == [2,2]:
            aim = [2,-2]
        elif aim == [2,-2]:
            aim = [-2,2]
        elif aim == [-2,2]:
            aim = [-2,-2]
        bicopter.x_r = aim[0]
        bicopter.y_r = aim[1]

    bicopter.step(dt)    
    bicopter_body.set_data(*bicopter.draw())        
    time_text.set_text('time = %0.1f s' % bicopter.t_s)             # time in seconds
    target_text.set_text('(x, y) => (%d, %d) m' % (aim[0], aim[1]))  # reference (target position)
    
    (pq, q, ie_q, px, x, py, y, ie_y, w_r, w_l) = bicopter.k
    yi_text.set_text('i => %0.6f N' % (bicopter.Kiy * ie_y))
    yd_text.set_text('d => %0.6f N' % (bicopter.Kdy * py))
    yp_text.set_text('p => %0.6f N' % (bicopter.Kpy * (y - bicopter.y_r)))
    return bicopter_body, time_text, target_text, yi_text, yd_text, yp_text

if __name__ == "__main__":
    if sys.argv[1] == "m":
        # First, let's parameterize our motor.
        VisualizeMotor()
        sys.exit(0)
    elif sys.argv[1] == "a":
        # Now, let's animate the effect of the non-ideal motor dynamics on the bicopter PID controller
        bicopter = RigidBicopter()
        dt = 1./60      # 60 fps animation
                        # Control loop is likely updated far faster (numerically 'continuous')
                        #  as odeint is likely an adaptive step length integrator
        aim = [-2,-2]

        fig = plt.figure()
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False, xlim=(-5,5), ylim=(-5,5))
        ax.set_xlabel("x, meters")
        ax.set_ylabel("y, meters")
        ax.set_title("PID Controlled Bicopter With Realistic Motors")
        ax.grid()

        bicopter_body, = ax.plot([], [], 'o-', lw=2)
        time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
        target_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)
        yi_text = ax.text(0.02, 0.85, '', transform=ax.transAxes)
        yd_text = ax.text(0.02, 0.80, '', transform=ax.transAxes)
        yp_text = ax.text(0.02, 0.75, '', transform=ax.transAxes)

        # This delays between frames so we hit our fps target
        t0 = time()
        animate(0)
        t1 = time()
        interval = 1000 * dt - (t1 - t0)

        ani = animation.FuncAnimation(fig, animate, frames=1250, repeat=True, interval=interval, blit=True, init_func=init)
        plt.show()

        # ani = animation.FuncAnimation(fig, animate, frames=1250, repeat=False, interval=interval, blit=True, init_func=init)
        # ani.save('bicopter.mp4', fps=60, extra_args=['-vcodec', 'libx264'])

