# Python script to visualize rotation about a non-body axis.

# Let the lab frame be the inertial frame S.
# Let the origin of the rigid body be O, in the inertial frame S'.
# Let r_ss' be the vector from S to S'.

# Let the body frame relative to O be S''.
# Consider a fixed point on the body, r_s' in S', and r_s'' in S''.
# Assume the body is subject to zero external torques.
# It must be rotating about a fixed axis, n, by Euler's rotation theorem.
# It must have a constant angular velocity about that axis by d/dt L = sum(T_external) = 0 and L = Jw about the rotation axis.

# Let R be the rotation matrix mapping a vector in S'' to S', with inverse R^T
# We know r_s' = R r_s''
# We know d/dt r_s' = (dR/dt R^T) * (R r_s'') = (dR/dt R^T) r_s'
# Therefore we expect (dR/dt R^T) to be the operator (w x) in the S' frame.
# The goal of this script is to visualize this.

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

import numpy as np
from numpy import pi as pi
from numpy import cos as c
from numpy import sin as s
from numpy import dot as dot
from numpy import transpose as transpose

# The axis phi is a rotation about the z axis in the body frame (yaw)
# The axis theta is a rotation about the y axis in the phi-rotated body frame (pitch)
# The axis psi is a rotation about the x axis in the phi, theta-rotated body frame (roll)
def R(phi, theta, psi):
  R = np.zeros((3,3))

  R[0,0] = c(phi)*c(theta)
  R[1,0] = s(phi)*c(theta)
  R[2,0] = -s(theta)

  R[0,1] = -s(phi)*c(psi) + c(phi)*s(theta)*s(psi)
  R[1,1] = c(phi)*c(psi) + s(phi)*s(theta)*s(psi)
  R[2,1] = c(theta)*s(psi)

  R[0,2] = s(phi)*s(psi) + c(phi)*s(theta)*c(psi)
  R[1,2] = -c(phi)*s(psi) + s(phi)*s(theta)*c(psi)
  R[2,2] = c(theta)*c(psi)

  return R

# Rotate z-axis (0,0,1) by pi radians about x-axis. Should end up at (0,0,-1) cutting across y.
# Rotate (0,0,-1) by pi radians about y-axis. Should end up at (0,0,1) again, cutting across x.
# Try both at the same time. Should still end up at (0,0,1).
def test_R():
  e3_spp = np.array((0,0,1))
  vectors = []
  for k in np.linspace(0,pi,100):
    vectors.append(dot(R(0,0,k), e3_spp))
  e3_spp = vectors[-1]
  for k in np.linspace(0,pi,100):
    vectors.append(dot(R(0,k,0), e3_spp))
  e3_spp = vectors[-1]
  for k in np.linspace(0,pi,100):
    vectors.append(dot(R(0,k,k), e3_spp))

  xs = [k[0] for k in vectors]
  ys = [k[1] for k in vectors]
  zs = [k[2] for k in vectors]

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.plot(xs=xs,ys=ys,zs=zs)
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")
  plt.show()

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

def vectorize(W):
  v = np.zeros(3)
  v[0] = W[1,0]
  v[1] = W[0,2]
  v[2] = W[2,1]
  return v

# This is the (w x) operator, W, with respect to changing body yaw, pitch, and roll.
# It is dR/dt R^T. The arguments are the current Euler angles and their time derivatives.
def W(phi, theta, psi, dphi, dtheta, dpsi):
  Rp = np.zeros((3,3))

  Rp[0,0] = (-s(phi)*dphi)*c(theta)
  Rp[0,0] += c(phi)*(-s(theta)*dtheta)

  Rp[1,0] = (c(phi)*dphi)*c(theta)
  Rp[1,0] += s(phi)*(-s(theta)*dtheta)

  Rp[2,0] = -c(theta)*dtheta

  Rp[0,1] = (-c(phi)*dphi)*c(psi)
  Rp[0,1] += -s(phi)*(-s(psi)*dpsi)
  Rp[0,1] += (-s(phi)*dphi)*s(theta)*s(psi)
  Rp[0,1] += c(phi)*(c(theta)*dtheta)*s(psi)
  Rp[0,1] += c(phi)*s(theta)*(c(psi)*dpsi)

  Rp[1,1] = (-s(phi)*dphi)*c(psi)
  Rp[1,1] += c(phi)*(-s(psi)*dpsi)
  Rp[1,1] += (c(phi)*dphi)*s(theta)*s(psi)
  Rp[1,1] += s(phi)*(c(theta)*dtheta)*s(psi)
  Rp[1,1] += s(phi)*s(theta)*(c(psi)*dpsi)

  Rp[2,1] = (-s(theta)*dtheta)*s(psi)
  Rp[2,1] += c(theta)*(c(psi)*dpsi)

  Rp[0,2] = (c(phi)*dphi)*s(psi)
  Rp[0,2] += s(phi)*(c(psi)*dpsi)
  Rp[0,2] += (-s(phi)*dphi)*s(theta)*c(psi)
  Rp[0,2] += c(phi)*(c(theta)*dtheta)*c(psi)
  Rp[0,2] += c(phi)*s(theta)*(-s(psi)*dpsi)

  Rp[1,2] = (s(phi)*dphi)*s(psi)
  Rp[1,2] += -c(phi)*(c(psi)*dpsi)
  Rp[1,2] += (c(phi)*dphi)*s(theta)*c(psi)
  Rp[1,2] += s(phi)*(c(theta)*dtheta)*c(psi)
  Rp[1,2] += s(phi)*s(theta)*(-s(psi)*dpsi)

  Rp[2,2] = (-s(theta)*dtheta)*c(psi)
  Rp[2,2] += c(theta)*(-s(psi)*dpsi)

  w_i = vectorize(dot(Rp, transpose(R(phi,theta,psi))))
  w_b = dot(transpose(R(phi,theta,psi)), w_i)

  return (w_i, w_b)


def test_W():
  # Is the effective w for a rotation of x rad/s about ek just.. ek*x,
  # regardless of the angle about axis ek? We expect W = -W^T as well.
  # sanitize_matrix(W(3*pi/12,0,0,2*pi,0,0)[0])
  # sanitize_matrix(W(0,3*pi/12,0,0,2*pi,0)[0])
  # sanitize_matrix(W(0,0,3*pi/12,0,0,2*pi)[0])

  # Let's see what it looks like once we've rotated a bit.
  # It's still skew antisymmetric with zero trace! This looks like the operation (w x)!!!!
  phi, theta, psi = (pi/4, 3*pi/12, -pi)
  w_i, w_b = W(phi, theta, psi, pi, 2*pi, 3*pi)


def Main():
  test_W()

if __name__ == "__main__":
  Main()

