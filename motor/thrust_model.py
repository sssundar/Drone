# This script uses measurements and fits of my 8.5mmx20mm coreless brushed DC
# motors to predict how duty-cycle will affect its thrust. The idea is to use
# this script to guide battery voltage and discharge rating specification.

# See Google Drive/Drone/Motor Characterization/Test ESR/motor_thrust_model.xlsx
# for the fits. See Motor Characterization/Drive Circuit Rev B (near #16) for
# the math.

import sys
import numpy as np
from scipy import optimize as optimize
from matplotlib import pyplot as plt

# Parameters
N_Vp = 5
Vp_volts = {"1s":np.linspace(3.2,4.2,N_Vp), "2s":np.linspace(6.4,8.4,N_Vp)}
N_motors = 4.0
R_motor_ohms = 0.58
R_ds_ohms = 0.05
EMF2_TO_THRUST = 5.0
AVGI_TO_EMF = 1.5
R_battery_ohms = 0.08
duty_cycle = np.linspace(0.05,0.95,20)
alpha = (EMF2_TO_THRUST*(AVGI_TO_EMF**2))/(N_motors*R_battery_ohms + R_ds_ohms + R_motor_ohms)
beta = alpha/np.sqrt(EMF2_TO_THRUST)

# Solves alpha*supply_volts/g + beta/sqrt(g) - 1/d = 0 for g for each d in
# duty_cycle
def gsolve(supply_volts):
  thrusts = []
  g_start = 0.00001
  g_end = 100
  for d in duty_cycle:
    eqn = lambda g: (((alpha*supply_volts)/g) - (beta/np.sqrt(g)) - (1.0/d))
    g, r = optimize.bisect(eqn, 0.00001, 100, full_output=True)
    if r.converged:
      thrusts.append(N_motors*g)
    else:
      raise ValueError("Failed to converge at %d duty cycle." % d)
  return thrusts

def visualize():
  plt.subplot(211)
  # For each Vp in set "1s"
  for supply_volts in Vp_volts["1s"]:
    plt.plot(duty_cycle, gsolve(supply_volts), "r-")

  plt.subplot(212)
  # For each Vp in set "2s"
  for supply_volts in Vp_volts["2s"]:
    plt.plot(duty_cycle, gsolve(supply_volts), "b-")

  plt.show()

if __name__ == "__main__":
  visualize()
