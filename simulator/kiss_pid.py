# This is a K.I.S.S. prototyping script for PID control of 
# a 1D rotation in the y-jig with perfect observation.

import sys
import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt

dt_s = 0.05
n = 100

theta_ref = 0.

kp = 10.
kd = 2.
ki = 0.1
integrator_clip = 0.5
torque_clip = 1.2

beta = 3.87732955e+02
Jy = 0.834
T_friction = 4
T_g = 387*0.1
eq_phase = 0.08
T_pole = 3

asymmetry = 0.05
T_max = [beta*(1-asymmetry), beta*(1+asymmetry), beta*(1+asymmetry), beta*(1-asymmetry)],

theta_0 = +np.pi/6

cfg = [  [1, T_max[0], -1., 'r-'],
      [2, T_max[1], 1., 'g-'],
      [3, T_max[2], 1., 'b-'],
      [4, T_max[3], -1., 'k-'], ]

get_motorid = lambda annot: annot[0]
color = lambda mid: cfg[mid-1][-1]
motor_sign = lambda mid: cfg[mid-1][-2]
motor_max_torque = lambda mid: cfg[mid-1][-3]

def simulate():
  def ddt_state(state, t):
    theta, w, m1, m2, m3, m4, d1, d2, d3, d4 = state
    ddt = np.zeros(10)
    ddt[0] = w
    net_m = motor_sign(1)*m1 + motor_sign(2)*m2 + motor_sign(3)*m3 + motor_sign(4)*m4
    ddt[1] = (1./Jy)*(net_m + T_g*np.sin(theta+eq_phase) - np.sign(w)*np.abs(w)*T_friction)

    ddt[2] = -T_pole*m1 + d1*motor_max_torque(1)*T_pole
    ddt[3] = -T_pole*m2 + d2*motor_max_torque(2)*T_pole
    ddt[4] = -T_pole*m3 + d3*motor_max_torque(3)*T_pole
    ddt[5] = -T_pole*m4 + d4*motor_max_torque(4)*T_pole

    return ddt

  thetas = [theta_0]
  ws = [0.]
  ms = [[0.]*4]
  us = [0.]
  ds = [np.zeros(4)]

  integrator = 0.

  for idx in range(n):
    percent = (100.0*idx)/n
    complete = "%0.3f%%" % percent
    sys.stdout.write('\rSimulation ' + complete + " complete.")
    sys.stdout.flush()

    theta_0 = thetas[-1]
    w_0 = ws[-1]
    m_0 = ms[-1]
    d_0 = ds[-1]

    # Plant Dynamics
    state_0 = np.array([
                  theta_0, w_0,
                  m_0[0], m_0[1], m_0[2], m_0[3],
                  d_0[0], d_0[1], d_0[2], d_0[3],
                ])

    state_dt = odeint(ddt_state, state_0, t=[0, dt_s])[1]

    # Perfect Observation
    theta_dt = state_dt[0]
    w_dt = state_dt[1]
    m_dt = [state_dt[2], state_dt[3], state_dt[4], state_dt[5]]

    thetas.append(theta_dt)
    ws.append(w_dt)
    ms.append(m_dt)

    # Controller Update with theta_ref = 0
    theta_err = theta_ref - theta_dt
    proportional = kp*theta_err
    derivative = -kd*w_dt
    
    integrator += dt_s * theta_err
    integrator = np.maximum(np.minimum(integrator, integrator_clip), -integrator_clip)
    integral = ki*integrator

    u_dt = proportional + derivative + integral
    u_dt = np.maximum(np.minimum(u_dt, torque_clip), -torque_clip)
    us.append(u_dt)

    TODO
    d_dt = np.dot(Dp_inv, np.array(list(u_dt)+[0.8]))

    for k in range(4):
      if d_dt[k] > 1.0:
        d_dt[k] = 1.0
      elif d_dt[k] < 0.0:
        d_dt[k] = 0.0

    ds.append(d_dt)

  fig, ax = plt.subplots(2,1,sharex=True)
  ax[0].plot(np.asarray(range(n+1))*dt_s, ds)
  ax[0].legend(["d1", "d2", "d3", "d4"])
  ax[1].plot(np.asarray(range(n+1))*dt_s, us)
  ax[1].legend(["y"])
  plt.show()

if __name__ == "__main__":
  simulate()
