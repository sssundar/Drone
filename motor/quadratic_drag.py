# Solves for the steady state of a linear torque, quadratic drag model of our DC brushed motor.
# We know thrust is proportional to w^2 and we want to see if duty cycle is proportional to thrust.
# as we've measured, at steady state.
#
# See notes from 8/6/2018-8/8/2018.

import sys
import numpy as np
from matplotlib import pyplot as plt

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

if __name__ == "__main__":
  fit()

