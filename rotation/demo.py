# Consider a body with dimensions L=5cm x W=5cm x H=1cm. The body is solid, with
# uniform density p = 6000 kg/m^3. Describe this body in aircraft orientation
# (z-down, x-forward, y-right) and 3-2-1 Euler axes. Simulate the rotation of
# the body axes in inertial space to get a feeling for quantization error in the
# mapping of angular velocity to instantaneous Euler angle derivatives.

# The body is subject to no external torques. Its center of mass is stationary.
# Its initial orientation is O = (roll = 0, pitch = 0, yaw = 0), and it is
# subject to the constraint d/dt pitch = 0 for all time, to avoid the
# singularity in the w-to-dO/dt inversion.

# The initial conditions O(0) = (0, 0, 0) and w_b(0) = (pi/4, 0, pi/4) rad/s
# guarantee that pitch will remain zero up to numerical error for all time.

import sys

import numpy as np
from numpy import pi as pi
from numpy import cos as c
from numpy import sin as s
from numpy import cross as cross
from numpy import dot as dot
from numpy import transpose as transpose
from numpy.linalg import inv as inverse
from numpy import linspace as linspace
from numpy import sqrt

from scipy.integrate import odeint

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Jb = None
Jb_inv = None
wb_0 = None
O_0 = None
AX_ROLL, AX_PITCH, AX_YAW = (0, 1, 2)

def setup():
  global Jb, Jb_inv, wb_0, O_0

  Jb = np.zeros((3,3))
  Jb[0,0] = 3.25
  Jb[1,1] = 3.25
  Jb[2,2] = 6.25

  Jb_inv = inverse(Jb)

  O_0 = np.zeros(3)

  wb_0 = np.zeros(3)
  wb_0[AX_ROLL] = pi/4
  wb_0[AX_PITCH] = 0
  wb_0[AX_YAW] = pi/4

def ddt_wb(wb, t):
  global Jb, Jb_inv
  return dot(Jb_inv, cross(dot(Jb, wb), wb))

def HIB(O):
  roll, pitch, yaw = O
  HI1 = np.zeros((3,3))
  HI1[0,0] = c(yaw)
  HI1[0,1] = s(yaw)
  HI1[0,2] = 0

  HI1[1,0] = -s(yaw)
  HI1[1,1] = c(yaw)
  HI1[1,2] = 0

  HI1[2,0] = 0
  HI1[2,1] = 0
  HI1[2,2] = 1

  H12 = np.zeros((3,3))
  H12[0,0] = c(pitch)
  H12[0,1] = 0
  H12[0,2] = -s(pitch)

  H12[1,0] = 0
  H12[1,1] = 1
  H12[1,2] = 0

  H12[2,0] = s(pitch)
  H12[2,1] = 0
  H12[2,2] = c(pitch)

  H2B = np.zeros((3,3))
  H2B[0,0] = 1
  H2B[0,1] = 0
  H2B[0,2] = 0

  H2B[1,0] = 0
  H2B[1,1] = c(roll)
  H2B[1,2] = s(roll)

  H2B[2,0] = 0
  H2B[2,1] = -s(roll)
  H2B[2,2] = c(roll)

  return dot(H2B, dot(H12, HI1))

def HBI(O):
  roll, pitch, yaw = O
  return transpose(HIB(roll,pitch,yaw))

# This was derived with the constraint that pitch = 0, all time.
# That turned out not to hold true when simulating.
# def Lbi(O):
#   roll, pitch, yaw = O
#   L = np.zeros((3,3))
#   L[0,0] = 1
#   L[1,1] = c(roll)
#   L[1,2] = -s(roll)
#   L[2,1] = s(roll)
#   L[2,2] = c(roll)
#   return L

def Lbi(O):
  roll, pitch, yaw = O
  L = np.zeros((3,3))
  L[0,0] = 1
  L[0,1] = s(roll) * s(pitch) / c(pitch)
  L[0,2] = c(roll) * s(pitch) / c(pitch)
  L[1,0] = 0
  L[1,1] = c(roll)
  L[1,2] = -s(roll)
  L[2,0] = 0
  L[2,1] = s(roll) / c(pitch)
  L[2,2] = c(roll) / c(pitch)
  return L

def Main():
  global wb_0, O_0

  setup()

  # Sampling period of body angular velocity
  T_wb_s = 0.02

  # Integrate d/dt wb = ddt_wb(wb, t). Sample it at T_wb.
  T_final_s = 100.0
  N_points = int(T_final_s / T_wb_s)
  time_s = linspace(0, T_final_s, N_points)
  wb = odeint(ddt_wb, wb_0, t = time_s)

  # Does the length of angular velocity without external torque increase with
  # time? Well, yes, but to the tune of 8E-7 per 100s. So... orders of magnitude
  # less than any noise. I'd ignore it.
  # wb_mag = [sqrt(dot(transpose(wb[k]),wb[k])) for k in xrange(len(wb))]
  # plt.plot(time_s, wb_mag)
  # plt.show()

  # Can you just plot wb over time in 3d space in aircraft orientation?
  # N_decimation = 80
  # X = [0 for k in wb[::N_decimation]]
  # Y = [0 for k in wb[::N_decimation]]
  # Z = [0 for k in wb[::N_decimation]]
  # U = [k[0] for k in wb[::N_decimation]]
  # V = [k[1] for k in wb[::N_decimation]]
  # W = [k[2] for k in wb[::N_decimation]]

  # fig = plt.figure()
  # ax = fig.add_subplot(111, projection='3d')
  # ax.quiver(Y,X,Z,V,U,W, length=0.1, arrow_length_ratio=0.05, pivot='tail')
  # ax.set_xlabel("y")
  # ax.set_ylabel("x")
  # ax.set_zlabel("z")
  # ax.invert_zaxis()
  # plt.show()

  # For each w_b(t), compute dO/dt and integrate it over time-steps of T_o <<
  # T_wb and To = T_wb (discrete step). Assume wb is constant over the T_wb
  # window. How do things change between the two?
  dT_discrete = T_wb_s
  O_discrete = [O_0]
  for k in xrange(len(wb[:-1])):
    dO_discrete = dot(Lbi(O_discrete[k]), wb[k])
    O_discrete.append(O_discrete[k] + (dO_discrete * dT_discrete))
    #O_discrete[k+1] = O_discrete[k+1] % (2*pi)

  N_infinitesimal = 10
  dT_infinitesimal = T_wb_s / N_infinitesimal
  O_infinitesimal = [O_0]
  for k in xrange(len(wb[:-1])):
    O_temp = O_infinitesimal[k]
    for j in xrange(N_infinitesimal):
      dO_temp = dot(Lbi(O_temp), wb[k])
      O_temp += dO_temp * dT_infinitesimal
    O_infinitesimal.append(O_temp)

    # It is super weird that adding this line makes us go from constant => periodic
    # O_infinitesimal[k+1] = O_infinitesimal[k+1] % (2*pi)

  print O_infinitesimal

  # plt.plot(time_s, [O_infinitesimal[k] - O_discrete[k] for x in xrange(len(O_infinitesimal))])
  # plt.plot(time_s, O_infinitesimal)
  # plt.legend(["AX_ROLL", "AX_PITCH", "AX_YAW"])
  # plt.show()

  # Well, I don't see a huge difference between the two. It's particularly
  # strange that a construction that ought to have kept pitch = 0 for all
  # time... is not doing so. It's also weird that pitch is slooooowly
  # increasing in magnitude. I think I have a bug here.


  # Knowing that you started with O_0, plot the body axes over the time of this simulation.

if __name__ == "__main__":
  Main()

