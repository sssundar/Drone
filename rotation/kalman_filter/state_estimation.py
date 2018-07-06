#############################################################################################
# A playground in which to test integrators, DC-killers, Kalman filters, synchronizers, ... #
#############################################################################################

import sys
import numpy as np
from matplotlib import pyplot as plt
from free_body import simulate
from animate import generate_body_frames, animate, compare
from quaternions import *

def f_g(q, a): 
  r, v = q
  q = [r] + list(v)
  line1 = 2*(q[1] * q[3] - q[0] * q[2]) - a[0]
  line2 = 2*(q[0] * q[1] + q[2] * q[3]) - a[1]
  line3 = 2*(1/2 - q[1]**2 - q[2]**2) - a[2]
  return np.array([line1, line2, line3])

def J_g(q):
  r, v = q
  q = [r] + list(v)
  line1 = [-2*q[2], 2*q[3], -2*q[0], 2*q[1]]
  line2 = [2*q[1], 2*q[0], 2*q[3], 2*q[2]]
  line3 = [0, -4*q[1], -4*q[2], 0]
  return np.matrix([line1, line2, line3])

def f_b(q, b, m): 
  r, v = q
  q = [r] + list(v)
  line1 = 2*b[1]*(0.5 - q[2]**2 - q[3]**2) + 2*b[3]*(q[1] * q[3] - q[0]* q[2]) - m[1]
  line2 = 2 * b[1] *( q[1] * q[2] - q[0] * q[3]) + 2 * b[3] * ( q[0] * q[1] + q[2] * q[3]) - m[2]
  line3 = 2 * b[1] *(q[0] * q[2] + q[1] * q[3]) + b[3]*(0.5 - q[1]**2 - q[2]**2) - m[3]
  return np.matrix([line1, line2, line3])

def J_b(q, b):
  r, v = q
  q = [r] + list[v]
  q1, q2, q3, q4 = q
  _, bx, _, bz = b
  
  line1 = [-2*bz*q3, 2*bz*q4, -4*bx*q3 - 2*bz*q1, -4*bx*q4+2*bz*q2]
  line2 = [-2*bx*q4+2*bz*q2, 2*bx*q3+2*bz*q1, 2*bx*q2+2*bz*q4, -2*bx*q1+2*bz*q3]
  line3 = [2*bx*q3 , 2*bx*q4-4*bz*q2, 2*bx*q1 - 4*bz*q3, 2*bx*q2]
  return np.matrix([line1, line2, line3])

def f_gb(q, a, b, m):
  return np.matrix([f_g(q, a);f_b(q, b, m)])

def J_gb(q, b):
  return np.matrix([J_g(q); J_b(q, b)])




def naive(hpf):
  # Simulate a simple two-axis rotation.
  inputs = {}
  inputs["r_i_bp"] = [np.asarray([1,0,0]), np.pi/2] # Principal body frame is initially rotated 90 degrees about the inertial x axis.
  inputs["r_bp_b"] = [np.asarray([0,1,0]), np.pi/4] # Actual body frame is rotated 45 degrees about the principal y axis
  inputs["J_bp"] = (1.0/6) * 0.05 * np.eye(3) # A uniform cubic 50g mass of length 1 m has J = M/6 I where M is the total mass.
  inputs["w_bp"] = 2*np.pi*np.asarray([0,0,1]) # 1 Hz CCW rotation about the principal body z-axis, initially.
  inputs["f_s"] = 100.0 # Hz
  inputs["t_f"] = 1.0 # seconds
  inputs["m_i"] = np.asarray([0,1,0]) # Normalized magnetic field points in Y-direction when the body is aligned with the inertial frame.
  inputs["a_i"] = np.asarray([0,0,1]) # Normalized gravitational field points straight up when the body is aligned with the inertial frame.

  alpha = 2



  outputs = simulate(inputs)
  sensor_stream = outputs         # TODO Do not yet apply realism (bias, noise, time jitter, ...)

  # Simplest integrator
  # Handling bias with a high-pass filter (nearly-overlapping pole zero pair at z = 1)
  # TODO Does not yet correct for orientation error using compass readings.
  # Assume (incorrectly) that we start out aligned with the inertial frame.

  r_i = [[1,np.asarray([0,0,0])]] # Our initial estimate of the rotation of the body frame relative to the inertial frame.

  x_nm1 = np.asarray([0,0,0])     # IIR Filter Memory
  y_nm1 = np.asarray([0,0,0])     # IIR Filter Memory
  beta = 0.99                     # IIR Filter Parameter

  N = len(sensor_stream["t_s"])
  for idx in xrange(N-1):
    dt = sensor_stream["t_s"][idx+1]-sensor_stream["t_s"][idx]

    q_dot = quaternion_times_scalar(scalar=.5, quaternion=quaternion_product(r_i[-1], (0, sensor_stream["w_b"][idx+1]), True))

    mu = alpha * quaternion_norm(q_dot) * dt

    haha = f_g(r_i[-1], sensor_stream["a_b"][idx+1])
    hahaha = J_g(r_i[-1])

    #q = r_i[end] - mu * 
    
    #r_i.append(quaternion_product(p=w_dt_to_quaternion(w_i, dt), q=r_i[-1], normalize=True))

  #################
  # Sanity Checks #
  #################

  e0_act, e1_act, e2_act = generate_body_frames(outputs["q_b"])
  e0_est, e1_est, e2_est = generate_body_frames(r_i)

  # What we're saying here is: take my initial body frame (unknown) as an inertial frame.
  # Track the net rotation relative to this frame. Do a coordinate transformation of the result to the 'standard' inertial frame.
  # Note in this case we don't actually know q_b[0] and we will eventually have noise in w_b.
  # Still, if we choose, we could separate integration from correction.
  e0_cor, e1_cor, e2_cor = generate_body_frames([quaternion_product(p=outputs["q_b"][0], q=r_i[idx], normalize=True) for idx in xrange(len(r_i))])

  # First, single principal axis rotations:
    # With zero initial offset, do they track exactly? Yes.

    # With 90 degree principal rotation, do they track with constant offset? Yes!
    # Visually, they are always a 90 degree rotation about the inertial x axis apart.
    # Since it's a rotation about a single principal axis it's not difficult to see this algebraically as well.

  # animate(N, e0_act, e1_act, e2_act, 5, "actual_rotation")
  # animate(N, e0_est, e1_est, e2_est, 5, "estimated_rotation")
  # animate(N, e0_cor, e1_cor, e2_cor, 5, "images")

  # Next, off-axis rotations:
    # Well, we can see from e*_ofs that q_b[0] * r_i[n] = q_b[n].
    # This means we expect q_b[n] r_i[n]^-1 to be a constant quaternion.
    # We can see that over the course of 1s we have a few degrees of drift in this 'constant' quaternion.
    # That's certainly something that can be corrected for.
  e0_ofs, e1_ofs, e2_ofs = generate_body_frames([quaternion_product(p=outputs["q_b"][idx], q=quaternion_inverse(r_i[idx]), normalize=True) for idx in xrange(len(r_i))])
  animate(N, e0_ofs, e1_ofs, e2_ofs, 5, "images")

# Ok - green light. Proceed with your filter design, followed by the realism module.

if __name__ == "__main__":
  if len(sys.argv) == 3:
    if sys.argv[1] == "naive":
      naive(sys.argv[2] == "hpf")
  else:
    print ""
    print "Usage: python state_estimation.py naive hpf|none"
    print ""
