# Solves for the steady state of a linear torque, quadratic drag model of our DC brushed motor.
# We know thrust is proportional to w^2 and we want to see if duty cycle is proportional to thrust.
# as we've measured, at steady state.

# See notes from 8/6/2018-8/8/2018.
import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint

def w_ss(d, cw=True):
  if d < 1E-9:
    return 0

  tm = 1
  km = 0.001
  kd = 0.0001
  if cw:
    return ((d*km)/(2*kd)) * (-1 + np.sqrt(1 + ((4*kd*tm)/(km*km*d))))
  else:
    return ((d*km)/(2*kd)) * (1 - np.sqrt(1 + ((4*kd*tm)/(km*km*d))))

def fit():
  d = np.linspace(0, 1, 100)
  w = [w_ss(duty) for duty in d]
  kfit = 95
  fit = kfit * np.sqrt(d)
  plt.plot(d, w, 'r-')
  plt.plot(d, fit, 'b-')
  plt.show()

# See notes from 8/8/2018-8/10/2018. We're trying to fit Bd to match a mechanical timescale of 100ms for the propellors.
# Simulating d=0 (s.s., w=0) step up to d.
def sim(d, cw=True):
  Mps = 2.0/1000  # kg
  Rps = 2.54/100  # meters
  Mpp = 0.25/1000 # kg
  wpp = 0.5/100   # meters
  lpp = 2.54/100  # meters
  Jprop = 0.5*Mps*(Rps**2)
  Jprop += 0.1*Mpp*(9*(lpp**2) + 4*(wpp**2))

  Bd = 0.0001 # Free Parameter
  RPM_max = 12000
  Bm = 10*Bd
  Gamma_int_max = (((2*np.pi*RPM_max)/60)**2)*Bd

  Gamma_int_max /= Jprop
  Bm /= Jprop
  Bd /= Jprop

  def dwdt(w, t):
    # TODO Add effect of d != 1, cw
    return Gamma_int_max - Bm*w - Bd*(w**2)

  time_s = np.linspace(0, 1, 100)
  w = odeint(dwdt, 0, t = time_s)

  plt.plot(time_s, w)
  plt.show()

if __name__ == "__main__":
  sim(1)

