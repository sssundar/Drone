#! /usr/bin/python

# Usage
# source activate hack
# python dc_brushed_ideal_model.py

import sys
from math import pi, exp
from matplotlib import pyplot as plt
from numpy import linspace

# Assumptions
# 1. Drag torque is linear in w and zero when w = 0
# 2. The motor is perfectly efficient. Torque and back-emf are linear in motor current and motor w, respectively.
# 3. Electrical inductance reaches steady state on us timescales and can be neglected in the mechanical dynamics.

# Free Parameters (chosen to hit f ~ 6000 RPM in ~0.1s and draw 2-4 A)
w_0 = 0.0 # rad/s
t_0 = 0.0 # s
t_sim_s = 1.0
t_pwm_s = 0.002
j_mp = 0.00003 # kg * m^2 / s
k_i = 0.4 # N * m / A
v_s = 7.4 # V
r_w = 0.5 # Ohms
k_d = 0.01 # N * m / (rev / s)

# Derived Parameters
k_v = k_i / (2*pi) # volts / (rev / s)
beta = lambda k_d: (1.0/(2*pi*j_mp)) * (k_d + ((k_i * k_v)/r_w))
alpha = lambda k_d: (1.0/(2*pi*j_mp)) * k_d
tau_max = k_i * v_s / r_w

def drive(t, w, d):
  global k_d
  return (t + t_pwm_s * d, w * exp(-beta(k_d)*t_pwm_s*d) + (tau_max/(j_mp*beta(k_d)))*(1.0-exp(-beta(k_d)*t_pwm_s*d)))

def freewheel(t, w, d):
  global k_d
  return (t + t_pwm_s * (1.0-d), w * exp(-alpha(k_d)*t_pwm_s*(1.0-d)))

# Plots a time series from t = [0, t_sim_s] of w(t,d)
def evolution(ax1, ax2, d, should_show):
  if d > 1.0 or d < 0.0:
    raise ValueError("Invalid duty cycle. d must be in [0,1].")
  time_s = [t_0]
  w = [w_0]
  just_drove = False
  while (time_s[-1] < t_sim_s):
    if not just_drove:
      t, v = drive(time_s[-1], w[-1], d)
      just_drove = True
    else:
      t, v = freewheel(time_s[-1], w[-1], d)
      just_drove = False
    time_s.append(t)
    w.append(v)
  f = [x / (2*pi) for x in w]
  rpm = [x * 60 for x in f]
  i = [(v_s - (k_v * x)) / r_w for x in f]

  ax1.plot(time_s, rpm, 'k.-')
  ax2.plot(time_s, i, 'k.-')

  if should_show:
    plt.show()

def evolutions():
  d = linspace(0, 1, 5)
  fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

  for duty in d:
    evolution(ax1, ax2, duty, False)

  ax1.set_ylabel("Motor RPM")
  ax2.set_ylabel("Motor Amps")
  ax2.set_xlabel("PWM Duty Cycle")
  plt.show()

def w_equil(d):
  global k_d
  w_equil_low = (tau_max/(j_mp * beta(k_d))) * (exp(-alpha(k_d)*t_pwm_s*(1.0-d)) * (1.0 - exp(-beta(k_d)*t_pwm_s*d))) / (1.0 - exp(-alpha(k_d)*t_pwm_s*(1.0-d)) * exp(-beta(k_d)*t_pwm_s*d))
  _, w_equil_high = drive(0, w_equil_low, d)
  return (w_equil_low, w_equil_high, (w_equil_low + w_equil_high)/2)

# Plots duty cycle (d) vs. <thrust> (w^2 as a proxy) of the motor
def transfer(should_show):
  d = linspace(0, 1, 100)
  w_eq_l = []
  w_eq_h = []
  w_eq = []
  for duty in d:
    w_l, w_h, w = w_equil(duty)
    w_eq_l.append(w_l)
    w_eq_h.append(w_h)
    w_eq.append(w)
  thrustify = lambda w: [x**2 for x in w]
  thrust = thrustify(w_eq)
  plt.plot(d, thrust, 'k.-')
  if should_show:
    plt.show()

def transfers():
  global k_d
  kds = linspace(k_i/100, k_i, 40)
  for k in kds:
    k_d = k
    transfer(False)
  plt.xlabel("PWM Duty Cycle")
  plt.ylabel("Equilibrium Thrust (w^2)")
  plt.show()

if __name__ == "__main__":
  if "evolution" in sys.argv[1]:
    evolutions()
  if "transfer" in sys.argv[1]:
    transfers()

