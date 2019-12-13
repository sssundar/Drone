# This is a K.I.S.S. prototyping script for PID control of 
# a 1D rotation in the y-jig with perfect observation,
# gravity torque, and imperfectly known motor asymmetry.

import sys
import numpy as np
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from kiss_config import *

def simulate(kp, kd, ki):
  def ddt_state(state, t):
    theta, w, m1, m2, d1, d2 = state
    ddt = np.zeros(6)
    ddt[0] = w
    ddt[1] = (1./J)*(-m1 + m2 + gamma_g*np.sin(theta+phi) - gamma_f*w)
    ddt[2] = (1./tau) * (d1*beta1 - m1)
    ddt[3] = (1./tau) * (d2*beta2 - m2)
    return ddt

  thetas = [theta_initial]
  ws = [0.]
  ms = [[0.]*2]
  us = [0.]
  ds = [np.zeros(2)]
  zetas =[0.]

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
                  m_0[0], m_0[1],
                  d_0[0], d_0[1],
                ])
    state_dt = odeint(ddt_state, state_0, t=[0, dt_s])[1]

    # Perfect Observation
    theta_dt = state_dt[0]
    w_dt = state_dt[1]
    m_dt = [state_dt[2], state_dt[3]]
    thetas.append(theta_dt)
    ws.append(w_dt)
    ms.append(m_dt)

    # Controller Update
    theta_err = theta_ref - theta_dt
    proportional = kp*theta_err
    derivative = -kd*w_dt
    zeta = zetas[-1] + (theta_err * dt_s)
    zeta = np.maximum(np.minimum(zeta, CLIP["integrator"]), -CLIP["integrator"])
    zetas.append(zeta)
    integral = ki*zeta
    u_dt = proportional + derivative + integral
    u_dt = np.maximum(np.minimum(u_dt, CLIP["total"]), -CLIP["total"])
    us.append(u_dt)
    d_dt = 0.5*np.array([alpha - (u_dt/beta), alpha + (u_dt/beta)])
    for k in range(2):
      if d_dt[k] > 1.0:
        d_dt[k] = 1.0
      elif d_dt[k] < 0.0:
        d_dt[k] = 0.0
    ds.append(d_dt)

  t_s = np.asarray(range(n+1))*dt_s
  fig, ax = plt.subplots(6,1,sharex=True)
  ax[0].plot(t_s, us)
  ax[0].legend(["u"])
  ax[1].plot(t_s, zetas)
  ax[1].legend("zeta")
  ax[2].plot(t_s, ds)
  ax[2].legend(["d1", "d2"])
  ax[3].plot(t_s, [x[0] for x in ms], 'r-')
  ax[3].plot(t_s, [x[1] for x in ms], 'b-')
  ax[3].legend(["m1", "m2"])
  ax[4].plot(t_s, ws)
  ax[4].legend("w")
  ax[5].plot(t_s, thetas, "k-")
  ax[5].plot(t_s, np.ones(len(t_s))*theta_ref, 'r--')
  ax[5].legend(["theta", "ref"])
  plt.show()

# Eigenvalues of Near-Equilibrium Linearized Dynamics
def dynamics(kp, kd, ki):
  return np.linalg.eigvals(np.array([
    [0, 1, 0, 0, 0],
    [(1./J)*gamma_g*np.cos(theta_ref - phi), -gamma_f/J, -1./J, 1./J, 0],
    [(beta1/(2*tau*beta))*kp, (beta1/(2*tau*beta))*kd, -1./tau, 0, -(beta1/(2*tau*beta))*ki],
    [-(beta2/(2*tau*beta))*kp, -(beta2/(2*tau*beta))*kd, 0, -1./tau, (beta2/(2*tau*beta))*ki],
    [-1, 0, 0, 0, 0],
    ]))

def scan():
  kps = np.logspace(-2, 2, 20)
  kds = np.logspace(-2, 2, 20)
  kis = np.logspace(-2, 2, 20)
  x = []
  y = []
  z = []
  color = []
  for kp in kps:
    for kd in kds:
      for ki in kis:
        x.append(kp)
        y.append(kd)
        z.append(ki)
        color.append('k' if np.all(np.real(dynamics(kp, kd, ki)) <= 0) else 'r')

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.scatter(x, y, z, c=color, marker='o')
  ax.set_xlabel('kp')
  ax.set_ylabel('kd')
  ax.set_zlabel('ki')
  plt.show()

def equilibrium(ki):
  # Equilibrium
  M_inv = np.linalg.inv(np.array([
    [1, -1, 0],
    [1, 0, beta1*ki/(2*beta)],
    [0, 1, -beta2*ki/(2*beta)],
    ]))
  b = np.array([gamma_g*np.sin(theta_ref+phi), beta1*alpha/2, beta2*alpha/2])
  x_eq = np.array([0, 0] + list(np.dot(M_inv, b)))
  print("theta_eq: %0.2f\nw_eq: %0.2f\nm1_eq: %0.2f\nm2_eq: %0.2f\nzeta_eq: %0.2f\n" % tuple(x_eq))

if __name__ == "__main__":
  if (len(sys.argv) != 2):
    print("Usage: (laptop) python kiss_pid.py [scan|eq|sim]")
    sys.exit(1)

  if sys.argv[1] == "scan":
    scan()
  elif sys.argv[1] == "eq":
    equilibrium(ki=10)
  elif sys.argv[1] == "sim":
    kp = 120.
    kd = 40.
    ki = 20.
    equilibrium(ki)
    print(dynamics(kp, kd, ki))
    simulate(kp, kd, ki)
