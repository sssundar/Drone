# This is a K.I.S.S. prototyping script for PID control of 
# a 1D rotation in the y-jig with perfect observation.

import sys
import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt

beta = 3.87732955e+02
Jy = 0.834
T_friction = 4
T_g = 387*0.1
eq_phase = 0.08
T_pole = 3

asymmetry = 0.05
T_max = [-beta*(1-asymmetry), beta*(1+asymmetry), beta*(1+asymmetry), -beta*(1-asymmetry)],

theta_0 = +np.pi/6
w_0 = 0
T_0 = 0

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
    ddt[1] = (1./Jy)*(net_m + T_g*s(theta+eq_phase) - np.sign(w)*np.abs(w)*T_friction)
    TODO
    ddt[2] = -T_pole*motor_torque + dc*T_max*T_pole
    return ddt

  t_sim = np.linspace(0, t_final_s, 1000)
  state_0 = np.array([theta_0, w_0, T_0])
  state = odeint(ddt_state, state_0, t=t_sim)

  for idx in range(n):
    percent = (100.0*idx)/n
    complete = "%0.3f%%" % percent
    sys.stdout.write('\rSimulation ' + complete + " complete.")
    sys.stdout.flush()

    q_0 = qs[-1]
    w_0 = ws[-1]
    d_0 = ds[-1]

    # Plant Dynamics
    state_0 = np.array([
                  w_0[0], w_0[1], w_0[2],
                  q_0[0], q_0[1][0], q_0[1][1], q_0[1][2],
                  d_0[0], d_0[1], d_0[2], d_0[3]
                ])

    state_dt = odeint(ddt_state, state_0, t=[0, dt_s])[1]

    # Perfect Observation
    w_dt = np.array([state_dt[0], state_dt[1], state_dt[2]])
    q_dt = [state_dt[3], np.array([state_dt[4], state_dt[5], state_dt[6]])]
    q_dt = quaternion_product(p=[1, np.asarray([0,0,0])], q=q_dt, normalize=True)

    qs.append(q_dt)
    ws.append(w_dt)

    # Controller Update
    q_error = quaternion_product(p=quaternion_inverse(q_dt), q=q_ref, normalize=True)
    theta = np.arccos(q_error[0])
    sin_theta = np.sin(theta)
    if abs(sin_theta) < 1e-4:
      w_error = np.array([0.0, 0.0, 0.0])
      theta = 0.0
    else:
      w_error = q_error[1] / np.sin(theta)

    proportional = kp*theta*np.dot(J, w_error)
    derivative = -kd*np.dot(J, w_dt)

    u_dt = proportional + derivative
    CLIP = 1.2
    u_dt = np.maximum(np.minimum(u_dt, CLIP), -CLIP)
    us.append(u_dt)

    # delta = np.dot(reduced_torque_inv, u_dt[0:2])
    # yaw = 0.2 if (proportional[2] > 0) else -0.2

    # d_dt = np.dot(D_inv, np.array([delta[0], delta[1], yaw, 0.6]))

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
  ax[1].legend(["x", "y", "z"])
  plt.show()

if __name__ == "__main__":
  simulate()
