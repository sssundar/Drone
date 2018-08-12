# Solves for the dynamics if a linear torque, quadratic drag model of our DC brushed motor.
# See notes from 8/4/2018-8/12/2018. We're trying to find the relations between Bd, Bm, Gamma, nd Jprop
# that give us linear relations between thrust and the duty cycle, which was measured,
# and which match a mechanical timescale of 100ms for the propellors, which was measured.

# Note, this is a non-linear system whose timescale depends linearly on the starting duty cycle and angular velocity.
# That means if we can find a range of d from w=0 with response timescales of ~100ms, any steps within that domain will also
# respond within 100ms.

import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint

def w_ss(d, cw, gamma, beta_m, beta_d):
  if d < 1E-9:
    return 0

  if cw:
    return ((d*beta_m)/(2*beta_d)) * (-1 + np.sqrt(1 + ((4*beta_d*gamma)/(beta_m*beta_m*d))))
  else:
    return ((d*beta_m)/(2*beta_d)) * (1 - np.sqrt(1 + ((4*beta_d*gamma)/(beta_m*beta_m*d))))

def sim(d, cw=True):
  Mps = 2.0/1000  # kg
  Rps = 2.54/100  # meters
  Mpp = 0.25/1000 # kg
  wpp = 0.5/100   # meters
  lpp = 2.54/100  # meters
  Jprop = 0.5*Mps*(Rps**2)
  Jprop += 0.1*Mpp*(9*(lpp**2) + 4*(wpp**2))

  RPM_max = 12000
  w_max = (2*np.pi*RPM_max)/60
  w_max *= 1 if cw else -1

  # Matching w_max and t_mech ~ 0.1s
  Bd = Jprop/120
  Bm = 10*Bd
  Gamma_int_max = (w_max**2)*Bd

  Gamma_int_max /= Jprop
  Bm /= Jprop
  Bd /= Jprop

  def dwdt(w, t):
    if cw:
      return d*Gamma_int_max - d*Bm*w - Bd*(w**2)
    else:
      return -d*Gamma_int_max - d*Bm*w + Bd*(w**2)

  tf = 0.3 #s which we would like to be three exponential timescales
  N = 100
  time_s = np.linspace(0, tf, N)

  w0 = w_ss(0, cw, Gamma_int_max, Bm, Bd)
  w = odeint(dwdt, w0, t = time_s)
  ws = w_ss(d, cw, Gamma_int_max, Bm, Bd) * np.ones(len(time_s))
  plt.plot(time_s, w, "k-")
  plt.plot(time_s, ws, 'r--')

if __name__ == "__main__":
  ds = np.linspace(0,1,10)
  for d in ds:
    sim(d)
  plt.xlabel("Time (s)")
  plt.ylabel("w (rad/s)")
  plt.title("Propellor Frequency for a DC Brushed Motor\nVarying PWM Duty Cycle with Linear Torque, Quadratic Drag\nGm ~ w_max^2 Bd; Bm ~ 10 Bd; Bd ~ Jp/120\nyield a t~100ms timescale for d varying in [0.1,1]")
  plt.show()


