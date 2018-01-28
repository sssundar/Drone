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

import sys, os, glob

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
orientation_strings = ["roll", "pitch", "yaw"]

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

def dHBI_dt(O, dOdt):
  roll, pitch, yaw = O
  droll, dpitch, dyaw = dOdt

  dHBI = np.zeros((3,3))

  dHBI[2,2] = (-s(roll)*droll)*c(pitch)
  dHBI[2,2] += c(roll)*(-s(pitch)*dpitch)

  dHBI[2,1] = (c(roll)*droll)*c(pitch)
  dHBI[2,1] += s(roll)*(-s(pitch)*dpitch)

  dHBI[2,0] = -c(pitch)*dpitch

  dHBI[1,2] = (-c(roll)*droll)*c(yaw)
  dHBI[1,2] += -s(roll)*(-s(yaw)*dyaw)
  dHBI[1,2] += (-s(roll)*droll)*s(pitch)*s(yaw)
  dHBI[1,2] += c(roll)*(c(pitch)*dpitch)*s(yaw)
  dHBI[1,2] += c(roll)*s(pitch)*(c(yaw)*dyaw)

  dHBI[1,1] = (-s(roll)*droll)*c(yaw)
  dHBI[1,1] += c(roll)*(-s(yaw)*dyaw)
  dHBI[1,1] += (c(roll)*droll)*s(pitch)*s(yaw)
  dHBI[1,1] += s(roll)*(c(pitch)*dpitch)*s(yaw)
  dHBI[1,1] += s(roll)*s(pitch)*(c(yaw)*dyaw)

  dHBI[1,0] = (-s(pitch)*dpitch)*s(yaw)
  dHBI[1,0] += c(pitch)*(c(yaw)*dyaw)

  dHBI[0,2] = (c(roll)*droll)*s(yaw)
  dHBI[0,2] += s(roll)*(c(yaw)*dyaw)
  dHBI[0,2] += (-s(roll)*droll)*s(pitch)*c(yaw)
  dHBI[0,2] += c(roll)*(c(pitch)*dpitch)*c(yaw)
  dHBI[0,2] += c(roll)*s(pitch)*(-s(yaw)*dyaw)

  dHBI[0,1] = (s(roll)*droll)*s(yaw)
  dHBI[0,1] += -c(roll)*(c(yaw)*dyaw)
  dHBI[0,1] += (c(roll)*droll)*s(pitch)*c(yaw)
  dHBI[0,1] += s(roll)*(c(pitch)*dpitch)*c(yaw)
  dHBI[0,1] += s(roll)*s(pitch)*(-s(yaw)*dyaw)

  dHBI[0,0] = (-s(pitch)*dpitch)*c(yaw)
  dHBI[0,0] += c(pitch)*(-s(yaw)*dyaw)

  return dHBI

# Convert the matrix representation of differential rotation into the vector representation of angular velocity
def W_to_w(W):
  w = np.zeros(3)
  w[0] = W[2,1]
  w[1] = W[0,2]
  w[2] = W[1,0]
  return w

# Sets values lower than epsilon to zero.
# Prints the result with precision 0.3f.
def sanitize_matrix(A):
  print ""
  epsilon = 0.001
  for r in xrange(3):
    text = ""
    for c in xrange(3):
      if abs(A[r, c]) < epsilon:
        A[r,c] = 0
      text += "%6.2f,\t" % A[r,c]
    print text[:-2]
  print ""

def sanitize_vector(a):
  print ""
  epsilon = 0.001
  text = ""
  for r in xrange(3):
    if abs(a[r]) < epsilon:
      a[r] = 0
    text += "%6.2f,\t" % a[r]
  print text[:-2]
  print ""

# wi (angular velocity in inertial frame) can be calculated as:
# wi = dHBI/dt (O, dO/dt) * HIB(O)     (1)
# wi = HBI(O) wb                       (2)
# This gives us a nice way of checking for numerical consistency in the LBI transformation, and
# without a known solution for wi, it's the best correctness check we have.
def wi_1(O, dOdt):
  W = dot(dHBI_dt(O,dOdt), HIB(O))
  w = W_to_w(W)
  return w

def wi_2(O, wb):
  return dot(HBI(O), wb)

# This is just a sanity check that if we take wi_1 and rotate it, we get back the wb that gave us O in the first place.
def wb_2(O, wi):
  return dot(HIB(O), wi)

# Angular momentum in the inertial frame is computed as:
# hi = Ji wi = HBI Jb HIB wi = HBI Jb wb = HBI hb
def hi(O, wb):
  global Jb
  return dot(HBI(O), dot(Jb, wb))

# O % (2*pi) required
def LBI(O):
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

def clear_animation():
  files = glob.glob('./images/*')
  for f in files:
    os.remove(f)

def animate(time_series, colors, decimator):
  # Wipe the images/ directory
  clear_animation()

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
  global wb_0, O_0, orientation_strings

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
  dOdt_discrete = []
  for k in xrange(len(wb[:-1])):
    dOdt_discrete.append(dot(LBI(O_discrete[k]), wb[k]))
    O_discrete.append(O_discrete[k] + (dOdt_discrete[-1] * dT_discrete))
    O_discrete[k+1] = O_discrete[k+1] % (2*pi)
  dOdt_discrete.append(dot(LBI(O_discrete[-1]), wb[-1]))

  wi_est_1 = [wi_1(O_discrete[k], dOdt_discrete[k]) for k in xrange(len(O_discrete))]
  wi_est_2 = [wi_2(O_discrete[k], wb[k]) for k in xrange(len(O_discrete))]
  wb_est_2 = [wb_2(O_discrete[k], wi_est_1[k]) for k in xrange(len(O_discrete))]

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
  time_series_body = [inertial_body_x, inertial_body_y, inertial_body_z]
  colors_body = ["r", "b", "k"]
  # animate(time_series_body, colors_body, F_wb_hz/2)

  #####################################################################################################
  # Note that wi_est_1 is equivalent to wi_est_2, and wb is equivalent to wb_est_2, which is fantastic.
  # It's super weird that even though wi <-> wb is a rotation, they visibly rotate different rates.
  #####################################################################################################
  time_series_w = [wi_est_1, wi_est_2, wb, wb_est_2]
  colors_w = ["r", "b", "k", "c"]
  # animate(time_series_w, colors_w, F_wb_hz/2)

  # Now, compute the dot products of all the body axes to verify they remain orthogonal
  dot_xy = []
  dot_xz = []
  dot_yz = []
  for k in xrange(len(inertial_body_x)):
    dot_xy.append(dot(inertial_body_x[k], inertial_body_y[k]))
    dot_xz.append(dot(inertial_body_x[k], inertial_body_z[k]))
    dot_yz.append(dot(inertial_body_y[k], inertial_body_z[k]))

  # Visibly, yeah, they're orthogonal up to numerical precision.
  # plt.plot(time_s, dot_xy, 'r-')
  # plt.plot(time_s, dot_xz, 'g-')
  # plt.plot(time_s, dot_yz, 'b-')
  # plt.show()

  # One final sanity check is to compute the angular momentum hi in the inertial frame.
  # You integrated wb assuming zero external torque => hi should be constant.
  hi_est_1 = []
  for k in xrange(len(wb)):
    hi_est_1.append(hi(O_discrete[k], wb[k]))

  # Visibly, yeah, there's drift. Let's take a closer look.
  # aircraft_axes(hi_est_1, 1)
  # OR
  # plt.plot(time_s, hi_est_1)
  # plt.show()

  # Dump the error in inertial angular momentum to the console.
  print ""
  print "This was a %ds simulation of a rigid body rotating under zero external torque." % T_final_s
  print "The inertial angular momentum should have been constant."
  print ""
  print "The initial inertial angular velocity, with body and inertial frames"
  print "instantaneously aligned, was wb(%s, %s, %s) = [%6.2f, %6.2f, %6.2f] radians/s." % (orientation_strings[0], orientation_strings[1], orientation_strings[2], wb_0[0], wb_0[1], wb_0[2])
  print ""
  for k in xrange(3):
    if (hi_est_1[0][k] == 0):
      print "Inertial angular momentum along the original %s axis drifted from %1.5f to %1.5f" % (orientation_strings[k], 0, hi_est_1[-1][k])
    else:
      print "Inertial angular momentum along the original %s axis drifted from %1.5f by %1.5f%s" % (orientation_strings[k], hi_est_1[0][k], 100 * (hi_est_1[-1][k] - hi_est_1[0][k]) / (hi_est_1[0][k] + 0.000000001), "%")
  print ""
  print "This is less than 0.06%s error per second, which will be dwarfed by gyro measurement error" % "%"
  print "and corrected for by Kalman filters against our noisy magnetometer."
  print ""
if __name__ == "__main__":
  Main()

