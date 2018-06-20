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

  #################
  # Sanity Checks #
  #################

  e0_act, e1_act, e2_act = generate_body_frames(outputs["q_i"])
  e0_est, e1_est, e2_est = generate_body_frames(r_i)

  # What we're saying here is: take my initial body frame (unknown) as an inertial frame.
  # Track the net rotation relative to this frame. Do a coordinate transformation of the result to the 'standard' inertial frame.
  # Note in this case we don't actually know q_i[0] and we will eventually have noise in w_b.
  # Still, if we choose, we could separate integration from correction.
  e0_cor, e1_cor, e2_cor = generate_body_frames([quaternion_product(p=outputs["q_i"][0], q=r_i[idx], normalize=True) for idx in xrange(len(r_i))])

  # First, single principal axis rotations:
    # With zero initial offset, do they track exactly? Yes.

    # With 90 degree principal rotation, do they track with constant offset? Yes!
    # Visually, they are always a 90 degree rotation about the inertial x axis apart.
    # Since it's a rotation about a single principal axis it's not difficult to see this algebraically as well.

  # animate(N, e0_act, e1_act, e2_act, 5, "actual_rotation")
  # animate(N, e0_est, e1_est, e2_est, 5, "estimated_rotation")
  # animate(N, e0_cor, e1_cor, e2_cor, 5, "images")

  # Next, off-axis rotations:
    # Well, we can see from e*_ofs that q_i[0] * r_i[n] = q_i[n].
    # This means we expect q_i[n] r_i[n]^-1 to be a constant quaternion.
    # We can see that over the course of 1s we have a few degrees of drift in this 'constant' quaternion.
    # That's certainly something that can be corrected for.
  e0_ofs, e1_ofs, e2_ofs = generate_body_frames([quaternion_product(p=outputs["q_i"][idx], q=quaternion_inverse(r_i[idx]), normalize=True) for idx in xrange(len(r_i))])
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
