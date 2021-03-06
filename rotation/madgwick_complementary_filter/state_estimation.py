#############################################################################################
# A playground in which to test integrators, DC-killers, Kalman filters, synchronizers, ... #
#############################################################################################

import sys
import numpy as np
from matplotlib import pyplot as plt
from free_body import simulate
from animate import generate_body_frames, animate, compare
from quaternions import *
from realism import *
import pdb

def f_g(q, a):
  r, v = q
  q = [r] + list(v)
  q1, q2, q3, q4 = q
  ax, ay, az = list(a)
  line1 = 2*(q2*q4 - q1*q3) - ax
  line2 = 2*(q1*q2 + q3*q4) - ay
  line3 = 2*(0.5 - q2**2 - q3**2) - az
  return np.array([line1, line2, line3])

def J_g(q):
  r, v = q
  q = [r] + list(v)
  q1, q2, q3, q4 = q
  line1 = [-2*q3, 2*q4, -2*q1, 2*q2]
  line2 = [2*q2, 2*q1, 2*q4, 2*q3]
  line3 = [0, -4*q2, -4*q3, 0]
  return np.matrix([line1, line2, line3])

def f_b(q, b, m):
  r, v = q
  q = [r] + list(v)
  q1, q2, q3, q4 = q
  bx, _, bz = list(b)
  mx, my, mz = list(m)
  line1 = 2*bx*(0.5 - q3**2 - q4**2) + 2*bz*(q2*q4 - q1*q3) - mx
  line2 = 2*bx*(q2*q3 - q1*q4) + 2*bz*(q1*q2 + q3*q4) - my
  line3 = 2*bx*(q1*q3 + q2*q4) + 2*bz*(0.5 - q2**2 - q3**2) - mz
  return np.array([line1, line2, line3])

def J_b(q, b):
  r, v = q
  q = [r] + list(v)
  q1, q2, q3, q4 = q
  bx, _, bz = b

  line1 = [-2*bz*q3, 2*bz*q4, -4*bx*q3 - 2*bz*q1, -4*bx*q4 + 2*bz*q2]
  line2 = [-2*bx*q4 + 2*bz*q2, 2*bx*q3 + 2*bz*q1, 2*bx*q2 + 2*bz*q4, -2*bx*q1 + 2*bz*q3]
  line3 = [2*bx*q3 , 2*bx*q4 - 4*bz*q2, 2*bx*q1 - 4*bz*q3, 2*bx*q2]

  return np.matrix([line1, line2, line3])

def f_gb(q, a, b, m):
  # print f_g(q, a)
  # print f_b(q, b, m)
  # pdb.set_trace()

  # return np.matrix([f_g(q, a);f_b(q, b, m)])
  return np.array(list(f_g(q, a)) + list(f_b(q, b, m)))

def J_gb(q, b):
  # pdb.set_trace()
  # print J_b(q, b)
  # return np.matrix([J_g(q); J_b(q, b)])
  return np.concatenate((J_g(q), J_b(q, b)), axis=0)

def gradient_f(q, a, b, m):
  # print np.transpose(J_gb(q, b))
  # print f_gb(q, a, b, m)
  # print np.transpose(f_gb(q, a, b, m))
  # print np.dot(np.transpose(J_gb(q, b)), f_gb(q, a, b, m))
  # print "----"
  return np.array(np.dot(np.transpose(J_gb(q, b)), f_gb(q, a, b, m)))[0]



def naive():
  # Simulate a simple two-axis rotation.
  inputs = {}
  inputs["r_i_bp"] = [np.asarray([0,0,1]), np.pi/10] # Principal body frame is initially rotated 18 degrees about the inertial x axis.
  inputs["r_bp_b"] = [np.asarray([0,1,0]), np.pi/4] # Actual body frame is rotated 45 degrees about the principal y axis
  inputs["J_bp"] = (1.0/6) * 0.05 * np.eye(3) # A uniform cubic 50g mass of length 1 m has J = M/6 I where M is the total mass.
  inputs["w_bp"] = 2*np.pi*np.asarray([0,0,2]) # 2 Hz CCW rotation about the principal body z-axis, initially.
  inputs["f_s"] = 100.0 # Hz
  inputs["t_f"] = 3.0 # seconds
  # Normalized magnetic field points in X/Z-direction when the body is aligned with the inertial frame.
  # Normalized gravitational field points straight up when the body is aligned with the inertial frame.
  # These were a crucial part of the derivation/simplification.
  inputs["m_i"] = np.asarray([0.5,0,np.sqrt(3)/2])
  inputs["a_i"] = np.asarray([0,0,1])

  outputs = simulate(inputs)
  sensor_stream = fuzz_accel(fuzz_compass(fuzz_gyro(outputs)))

  r_i = [[1, np.asarray([0,0,0])]] # Our initial estimate of the rotation of the body frame relative to the inertial frame.

  # Bias Accumulator
  int_we = np.asarray([0.0,0.0,0.0])

  N = len(sensor_stream["t_s"])
  for idx in xrange(N-1):
    dt = sensor_stream["t_s"][idx+1]-sensor_stream["t_s"][idx]

    # # Without Magnetic Distortion Compensation
    # b = sensor_stream["m_i"]
    # m = sensor_stream["m_b"][idx]

    # With Magnetic Distortion Compensation
    m = sensor_stream["m_b"][idx]
    b = quaternion_rotation(qv=[0,m], qr=r_i[-1])[1]
    b = np.asarray([np.sqrt(b[0]**2 + b[1]**2), 0, b[2]])

    grad_f = gradient_f(r_i[-1], sensor_stream["a_b"][idx], b, m) #q, a, b, m

    # Scale based on the realism spread, bias.
    # Note zeta cannot compensate for beta.
    # Note beta can compensate for bias if it is large enough.
    #   Then... you'll jitter if there's mag/acc noise. You don't want to make beta unnecessarily large.
    beta = np.pi/2   # ~18 dps spread iid in all axes. If this is zero we're just integrating without any orientation compensation
    zeta = np.pi/3    # ~60 dps bias error iid in all axes. If this is zero we don't use gyro bias compensation.

    if vector_norm(grad_f) > 1E-9:
      qe_dot = grad_f / vector_norm(grad_f)
      new_part = beta * qe_dot
    else:
      qe_dot = np.asarray([0,0,0,0])
      new_part = [0,0,0,0]
    qe_dot = [qe_dot[0], np.asarray([qe_dot[1], qe_dot[2], qe_dot[3]])]

    w_e = quaternion_times_scalar(scalar=2, quaternion=quaternion_product(quaternion_inverse(r_i[-1]), qe_dot, False))[1]
    int_we += w_e*dt

    q_dot = quaternion_times_scalar(scalar=.5, quaternion=quaternion_product(r_i[-1], [0, sensor_stream["w_b"][idx] - zeta*int_we], False))

    dq = [new_part[0], np.array([new_part[1], new_part[2], new_part[3]])]
    dq = [q_dot[0] - dq[0], q_dot[1] - dq[1]]
    dq = [dq[0]*dt, dq[1]*dt]
    q = [r_i[-1][0] + dq[0], r_i[-1][1] + dq[1]]

    r_i.append([e / quaternion_norm(q)  for e in q])


  #################
  # Sanity Checks #
  #################

  e0_act, e1_act, e2_act = generate_body_frames(outputs["q_b"])
  e0_est, e1_est, e2_est = generate_body_frames(r_i)

  #animate(N, e0_act, e1_act, e2_act, 5, "actual_rotation")
  #animate(N, e0_est, e1_est, e2_est, 5, "estimated_rotation")
  compare(N, e0_est, e1_est, e2_est, e0_act, e1_act, e2_act, 5, "estimated_rotation")

if __name__ == "__main__":
  naive()
