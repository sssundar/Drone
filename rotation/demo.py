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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

global Jb
global Jb_inv
global wb_0
global O_0
AX_ROLL, AX_PITCH, AX_YAW = (0, 1, 2)

def setup():
  global Jb, Jb_inv, wb_0, O_0

  Jb = np.zeros((3,3))
  Jb[0,0] = 3.25
  Jb[1,1] = 3.25
  Jb[2,2] = 6.25

  Jb_inv = inverse(J_b)

  O_0 = np.zeros(3)

  wb_0 = np.zeros(3)
  wb_0[AX_ROLL] = pi/4
  wb_0[AX_PITCH] = 0
  wb_0[AX_YAW] = pi/4

def ddt_wb(wb, t):
  global Jb
  global Jb_inv
  return dot(Jb_inv, cross(dot(Jb, wb), wb))

def HIB(roll, pitch, yaw):
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

def HBI(roll, pitch, yaw):
  return transpose(HIB(roll,pitch,yaw))

def Lbi(roll, pitch, yaw):
  L = np.zeros((3,3))
  L[0,0] = 1
  L[1,1] = c(roll)
  L[1,2] = -s(roll)
  L[2,1] = s(roll)
  L[2,2] = c(roll)
  return L

def Main():
  setup()

  # Sampling period of body angular velocity
  T_wb = 0.02 # s

  # Integrate d/dt wb = ddt_wb(wb, t). Sample it at T_wb

  # For each w_b(t), compute dO/dt and integrate it over time-steps of T_o << T_wb. Assume wb is constant over the T_wb window.

  # At the start of each T_wb, save the rotation matrix HBI.

  # Knowing that you started with O_0, plot the body axes over the time of this simulation with aircraft convention (z-down).

if __name__ == "__main__":
  Main()

