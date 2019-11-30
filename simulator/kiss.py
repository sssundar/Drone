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
  f_hz = 20.
  dt_s = 1.0 / f_hz
  t_s = 0.0
  n = 1000
  q_0 = np.array([0,0,1])
  q_0 = q_0 / vector_norm(q_0)
  q_0 = axis_angle_to_quaternion(q_0, np.pi)

  # Dynamics Configuration
  alpha = 2.45550709e+02
  beta = 3.87732955e+02
  gamma = 0.335*2/0.1 # 5.10001156e-06
  torque = np.array([
    [-alpha, -alpha, alpha, alpha], 
    [-beta, beta, beta, -beta],
    [-gamma, gamma, -gamma, gamma] ])
  reduced_torque = np.array([
    [alpha, alpha],
    [beta, -beta] ])
  reduced_torque_inv = np.linalg.inv(reduced_torque)
  D = np.array([
    [-1., 0., 1., 0.],
    [0., -1., 0., 1.],
    [1., -1., 1., -1.],
    [1., 1., 1., 1.] ])
  D_inv = np.linalg.inv(D)
  Dp = np.array([
    [-alpha, -alpha, alpha, alpha], 
    [-beta, beta, beta, -beta],
    [-gamma, gamma, -gamma, gamma],
    [1., 1., 1., 1.] ])
  Dp_inv = np.linalg.inv(Dp)
  J = np.array([
    [ 1.50856451, -0.02084717,  0.02037978],
    [-0.02084717,  0.83391386,  0.09393599],
    [ 0.02037978,  0.09393599,  2.12029846] ])
  J_inv = np.linalg.inv(J)

  # Controller Configuration
  # q_ref = [1, np.array([0.0, 0.0, 0.0])]
  q_ref = [0.1927032619714737, np.array([0.        , 0.        , 0.98125708])]
  # q_ref = np.array([0,0,1])
  # q_ref = q_ref / vector_norm(q_ref)
  # q_ref = axis_angle_to_quaternion(q_ref, np.pi/2)
  kp = 10.0
  kd = 2.0

  # Observer
  qs = [q_0]
  ws = [np.array([0., 0., 0.])]
  us = [np.array([0., 0., 0.])]
  ds = [np.array([0., 0., 0., 0.])]

  # Plant Dynamics
  def ddt_state(state, t):
    w = np.array([state[0], state[1], state[2]])
    q = [state[3], np.array([state[4], state[5], state[6]])]
    d = np.array([state[7], state[8], state[9], state[10]])
    tqe = np.dot(torque, d)
    ddt_w = np.dot(J_inv, tqe - np.cross(w, np.dot(J, w)))
    ddt_q = quaternion_times_scalar(scalar=0.5, quaternion=quaternion_product(p=q, q=[0, w], normalize=False))

    return np.array([
                ddt_w[0], ddt_w[1], ddt_w[2],
                ddt_q[0], ddt_q[1][0], ddt_q[1][1], ddt_q[1][2],
                0., 0., 0., 0.,
              ])

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

  if visualize:
    e0, e1, e2 = generate_body_frames([q_ref] + qs)
    animate(len(qs), e0, e1, e2, 5)
  else:
    fig, ax = plt.subplots(2,1,sharex=True)
    ax[0].plot(np.asarray(range(n+1))*dt_s, ds)
    ax[0].legend(["d1", "d2", "d3", "d4"])
    ax[1].plot(np.asarray(range(n+1))*dt_s, us)
    ax[1].legend(["x", "y", "z"])
    plt.show()

if __name__ == "__main__":
  # Install ImageMagick on Ubunut then, after running this script, go to 'images' and run
  # convert -delay 0.05 -loop 0 *png stabilization.gif
  visualize = sys.argv[1] == "viz"
  simulate(visualize=visualize)
