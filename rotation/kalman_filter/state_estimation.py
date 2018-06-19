#############################################################################################
# A playground in which to test integrators, DC-killers, Kalman filters, synchronizers, ... #
#############################################################################################

import sys
from matplotlib import pyplot as plt
from free_body import simulate
from animate import generate_body_frames, animate, compare
from quaternions import *

def naive(hpf):
  # Simulate a simple two-axis rotation.
  inputs = {}
  inputs["r_i_bp"] = [np.asarray([1,0,0]), np.pi/2] # Principal body frame is initially rotated 90 degrees about the inertial x axis.
  inputs["r_bp_b"] = [np.asarray([0,1,0]), np.pi/4] # Actual body frame is rotated 45 degrees about the principal y axis
  inputs["J_bp"] = (1.0/6) * 0.05 * np.eye(3) # A uniform cubic 50g mass of length 1 m has J = M/6 I where M is the total mass.
  inputs["w_bp"] = 2*np.pi*np.asarray([0,0,1]) # 1 Hz CCW rotation about the principal body z-axis, initially.
  inputs["f_s"] = 100.0 # Hz
  inputs["t_f"] = 1.0 # seconds
  inputs["m_i"] = np.asarray([0,0,1]) # Normalized magnetic field points straight up when the body is aligned with the inertial frame.

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

    # Coordinate transformation of latest angular velocity as measured in the body frame
    # into an angular velocity as measured from the inertial frame (our estimate of the inertial frame, anyways)
    q_wi = quaternion_rotation([0, sensor_stream["w_b"][idx]], r_i[-1])

    if hpf:
      # Apply IIR Filter to kill DC signal but otherwise leave gain, phase untouched.
      # This works out as y(n) = x(n) - x(n-1) + beta*y(n-1)
      #                   w_i(n) = [(1+beta)/2] y(n)
      x_n = q_wi[1] # Take the vector part
      y_n = x_n - x_nm1 + beta*y_nm1
      w_i = ((1+beta)/2)*y_n
      x_nm1 = x_n
      y_nm1 = y_n
    else:
      w_i = q_wi[1] # Take the vector part

    r_i.append(quaternion_product(p=w_dt_to_quaternion(w_i, dt), q=r_i[-1], normalize=True))

  e0_act, e1_act, e2_act = generate_body_frames(outputs["q_i"])
  e0_est, e1_est, e2_est = generate_body_frames(r_i)

  # TODO These look funny. First things first: with zero initial offset, do they track exactly?
  # animate(N, e0_act, e1_act, e2_act, 5)
  animate(N, e0_est, e1_est, e2_est, 5)
  # compare(N, e0_est, e1_est, e2_est, e0_act, e1_act, e2_act, 5)

if __name__ == "__main__":
  if len(sys.argv) == 3:
    if sys.argv[1] == "naive":
      naive(sys.argv[2] == "hpf")
  else:
    print ""
    print "Usage: python state_estimation.py naive hpf|none"
    print ""
