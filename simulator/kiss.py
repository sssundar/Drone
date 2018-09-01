# This is a K.I.S.S. prototyping script for control of a 3D rotation with perfect
# observation in the absence of external fields.

import sys
import numpy as np
from scipy.integrate import odeint
from quaternions import *
from matplotlib import pyplot as plt
from animate import generate_body_frames, animate
import pdb

def simulate(visualize=True):
  # Simulation Configuraiton
  f_hz = 100.0
  dt_s = 1.0 / f_hz
  t_s = 0.0
  n = 500
  q_0 = np.array([1,1,-1])
  q_0 = q_0 / vector_norm(q_0)
  q_0 = axis_angle_to_quaternion(q_0, np.pi/2)

  # Dynamics Configuration
  J1 = 2.383e-05
  J2 = 2.383e-05
  J3 = 4.583e-05
  J = np.eye(3)
  J[0,0] = J1
  J[1,1] = J2
  J[2,2] = J3
  J_inv = np.linalg.inv(J)

  # Controller Configuration
  q_ref = [1, np.array([0.0, 0.0, 0.0])]
  kp = 20.0
  kd = 2.0

  # Observer
  qs = [q_0]
  ws = [np.array([0.0, 0.0, 0.0])]
  us = [np.array([0.0, 0.0, 0.0])]

  # Plant Dynamics
  def ddt_state(state, t):
    w = np.array([state[0], state[1], state[2]])
    q = [state[3], np.array([state[4], state[5], state[6]])]
    torque = np.array([state[7], state[8], state[9]])

    ddt_w = np.dot(J_inv, torque - np.cross(w, np.dot(J, w)))
    ddt_q = quaternion_times_scalar(scalar=0.5, quaternion=quaternion_product(p=q, q=[0, w], normalize=False))

    return np.array([
                ddt_w[0], ddt_w[1], ddt_w[2],
                ddt_q[0], ddt_q[1][0], ddt_q[1][1], ddt_q[1][2],
                0.0, 0.0, 0.0
              ])

  for idx in xrange(n):
    percent = (100.0*idx)/n
    complete = "%0.3f%%" % percent
    sys.stdout.write('\rSimulation ' + complete + " complete.")
    sys.stdout.flush()

    q_0 = qs[-1]
    w_0 = ws[-1]
    u_0 = us[-1]

    # Plant Dynamics
    state_0 = np.array([
                  w_0[0], w_0[1], w_0[2],
                  q_0[0], q_0[1][0], q_0[1][1], q_0[1][2],
                  u_0[0], u_0[1], u_0[2]
                ])

    state_dt = odeint(ddt_state, state_0, t=[0, dt_s])[1]

    # Perfect Observation
    w_dt = np.array([state_dt[0], state_dt[1], state_dt[2]])
    q_dt = [state_dt[3], np.array([state_dt[4], state_dt[5], state_dt[6]])]
    q_dt = quaternion_product(p=[1, np.asarray([0,0,0])], q=q_dt, normalize=True)

    qs.append(q_dt)
    ws.append(w_dt)

    # Controller Update
    q_error = quaternion_product(p=q_ref, q=quaternion_inverse(q_dt), normalize=True)

    theta = np.arccos(q_error[0])
    sin_theta = np.sin(theta)
    if abs(sin_theta) < 1e-4:
      w_error = np.array([0.0, 0.0, 0.0])
      theta = 0.0
    else:
      w_error = q_error[1] / np.sin(theta)

    proportional = kp*theta*np.dot(J,w_error)
    derivative = -kd*np.dot(J, w_dt)

    u_dt = proportional + derivative
    us.append(u_dt)

  if visualize:
    e0, e1, e2 = generate_body_frames(qs)
    animate(len(qs), e0, e1, e2, 5)
  else:
    plt.plot(np.asarray(range(n+1))*dt_s, us)
    plt.legend(["e1", "e2", "e3"])
    plt.show()

if __name__ == "__main__":
  simulate(visualize=True)
