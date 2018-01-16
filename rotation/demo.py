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
from numpy import tan as tan
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

radians_to_degrees = lambda (rad): 180*rad/pi

def setup():
  global Jb, Jb_inv, wb_0, O_0

  Jb = np.zeros((3,3))
  Jb[0,0] = 3.25
  Jb[1,1] = 3.25
  Jb[2,2] = 6.25

  Jb_inv = inverse(Jb)

  O_0 = np.zeros(3)

  wb_0 = np.zeros(3)
  wb_0[AX_ROLL] = pi/15
  wb_0[AX_PITCH] = 0
  wb_0[AX_YAW] = pi/15

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
  return transpose(HIB(O))

# O % (2*pi) required
def Lbi(O):
  roll, pitch, yaw = O

  if (radians_to_degrees(abs(pitch - 0.5*pi)) < 10) or (radians_to_degrees(abs(pitch - 1.5*pi)) < 10):
    print "Error, too close to pitch singularity."
    sys.exit(1)

  L = np.zeros((3,3))
  L[0,0] = 1
  L[0,1] = s(roll) * tan(pitch)
  L[0,2] = c(roll) * tan(pitch)
  L[1,0] = 0
  L[1,1] = c(roll)
  L[1,2] = -s(roll)
  L[2,0] = 0
  L[2,1] = s(roll) / c(pitch)
  L[2,2] = c(roll) / c(pitch)
  return L

def prettify(euler_angles):
  result = [v / pi for v in euler_angles]
  for k in xrange(len(result)):
    for j in xrange(3):
      if result[k][j] > 1:
        result[k][j] -= 2
  return result

def aircraft_axes(vectors, decimator):
  N_decimation = decimator
  X = [0 for k in vectors[::N_decimation]]
  Y = [0 for k in vectors[::N_decimation]]
  Z = [0 for k in vectors[::N_decimation]]
  U = [k[0] for k in vectors[::N_decimation]]
  V = [k[1] for k in vectors[::N_decimation]]
  W = [k[2] for k in vectors[::N_decimation]]

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.quiver(Y,X,Z,V,U,W, length=0.1, arrow_length_ratio=0.05, pivot='tail')
  ax.set_xlabel("y")
  ax.set_ylabel("x")
  ax.set_zlabel("z")
  ax.invert_zaxis()
  plt.show()

def draw_body(axes, vectors, colors):
  line_collections = []

  for m in xrange(len(vectors)):
    vector = vectors[m]

    X = Y = Z = 0
    U = vector[0]
    V = vector[1]
    W = vector[2]

    line_collections.append(axes.quiver(Y,X,Z,V,U,W, length=0.05, arrow_length_ratio=0.05, pivot='tail', color=colors[m]))

  return line_collections

def animate(time_series, colors, decimator):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.set_xlabel("y")
  ax.set_ylabel("x")
  ax.set_zlabel("z")
  ax.invert_zaxis()

  line_collections = None
  count = 0

  for n in xrange(len(time_series[0])):
    if (n % decimator) != 0:
      continue

    if (line_collections is not None) and (count > 1):
      # The first set of axes drawn stays as a reference in all images.
      # The rest are wiped right after they are drawn/saved to give the illusion of motion.
      for col in line_collections:
        ax.collections.remove(col)

    line_collections = draw_body(ax, [vectors[n] for vectors in time_series], colors)

    plt.savefig('images/Frame%08i.png' % n)
    plt.draw()

    count += 1

def Main():
  global wb_0, O_0

  setup()

  # Sampling period of body angular velocity
  T_wb_s = 0.02
  F_wb_hz = 50
  T_final_s = 100.0

  # Integrate d/dt wb = ddt_wb(wb, t). Sample it at T_wb.
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
  # aircraft_axes(wb, F_wb_hz)

  # For each w_b(t), compute dO/dt and integrate it over time-steps of:
  # T_o << T_wb (infinitesimal case)
  # To = T_wb (discrete case)

  # Falsely assume wb is constant as in practice, it's the only sample we'd get. In discrete case, just step forward.
  # In infinitesimal case, rotate wb to wi, then step forward and rotate into wb as body frame changes.
  # You know it's wrong, but you want to find out how wrong.

  dT_discrete = T_wb_s
  O_discrete = [O_0]
  for k in xrange(len(wb[:-1])):
    dO_discrete = dot(Lbi(O_discrete[k]), wb[k])
    O_discrete.append(O_discrete[k] + (dO_discrete * dT_discrete))
    O_discrete[k+1] = O_discrete[k+1] % (2*pi)

  # plt.plot(time_s, prettify(O_discrete))
  # plt.legend(["AX_ROLL", "AX_PITCH", "AX_YAW"])
  # plt.show()

  # Knowing that you started with O_0 s.t. body = inertial frame, plot the body axes over the time of this simulation.
  body_x0 = np.zeros(3)
  body_y0 = np.zeros(3)
  body_z0 = np.zeros(3)

  body_x0[AX_ROLL] = 1.0
  body_y0[AX_PITCH] = 1.0
  body_z0[AX_YAW] = 1.0

  inertial_body_x = [dot(HBI(o), body_x0) for o in O_discrete]
  inertial_body_y = [dot(HBI(o), body_y0) for o in O_discrete]
  inertial_body_z = [dot(HBI(o), body_z0) for o in O_discrete]
  time_series = [inertial_body_x, inertial_body_y, inertial_body_z]
  colors = ["r", "b", "k"]

  animate(time_series, colors, F_wb_hz/2)

  ###############################################################################
  # Error #1: O_infinitesimal is constant. That's just wrong. PDB is your friend.
  ###############################################################################

  # N_infinitesimal = 10
  # dT_infinitesimal = T_wb_s / N_infinitesimal
  # O_infinitesimal = [O_0]
  # for k in xrange(len(wb[:-1])):
  #   O_temp = O_infinitesimal[k]
  #   wi = dot(HBI(O_temp), wb[k])
  #   for j in xrange(N_infinitesimal):
  #     dO_temp = dot(Lbi(O_temp), dot(HIB(O_temp), wi))
  #     O_temp += dO_temp * dT_infinitesimal
  #   O_infinitesimal.append(O_temp)

  # plt.plot(time_s, prettify(O_infinitesimal))
  # plt.legend(["AX_ROLL", "AX_PITCH", "AX_YAW"])
  # plt.show()

if __name__ == "__main__":
  Main()

