import sys
import numpy as np
from scipy.integrate import odeint
from quaternions import *
#
# @brief      Initializes a horizontal quad oriented with the Earth's magnetic field, motors halted.
#
def plant_init():
  pass

#
# @brief      Simulates the plant dynamics of a quadcopter without external
#             disturbances.
#               - Reproduces the measured linear duty-cycle to thrust relation
#                 for DC brushed coreless motors
#               - Accounts for gyroscopic effects from the propellors
#               - Accounts for IMU-frame vs. body-frame mismatch
#               - Accounts for motor drive asymmetry
#               - Assumes a rectangular symmetric chassis
#
# @param[in]  duty  - a dictionary
#                   - keys {m1p2, p1p2, p1m2, m1m2} representing (m)inus and
#                     (p)lus body axes 1,2
#                   - values representing motor-drive PWM duty-cycles between
#                     [0,1]
# @param[in]  dt    the time-step to simulate, in seconds
#
# @return     outputs, a dictionary with the following fields, computed without
#             noise, bias, jitter, or delay:
#             - w: a numpy 3-vector [rad/s] representing a sample from a 3D
#               gyroscope on the quad
#             - m: a numpy 3-vector [unit norm, unitless] representing a sample
#               of the direction of the Earth's magnetic field from a 3D compass
#               on the quad
#             - a: a numpy 3-vector [unit norm, unitless] representing a sample
#               of the direction of the Earth's gravitational field from a 3D
#               accelerometer on the quad
#             - q: a quaternion [r, v], where v is a numpy 3-vector,
#               representing the coordinate transformation from the quad
#               body-frame to the space frame. rotating a vector from the quad
#               body frame by q_b yields a space-frame representation. note that
#               the IMU-frame is not necessarily concident with the quad body
#               frame.
#             - r: a numpy 3-vector representing the center of mass of the quad
#               in the space frame.
#
def plant_evolve(duty, dt):
  # Simulate in the quad frame, then rotate samples to the IMU frame.

  # Get the time step for this simulation
  dt = 1.0 / inputs["f_s"] # seconds

  # Get the rotation matrix from the principal body frame to the body frame, R
  q = axis_angle_to_quaternion(-inputs["r_bp_b"][0], inputs["r_bp_b"][1])
  R = np.zeros([3,3])
  R[:,0] = quaternion_rotation([0, np.asarray([1,0,0])], q)[1]
  R[:,1] = quaternion_rotation([0, np.asarray([0,1,0])], q)[1]
  R[:,2] = quaternion_rotation([0, np.asarray([0,0,1])], q)[1]

  # Get the moment of inertia matrix with respect to the body frame, J_b = R J_bp R^T
  J_b = np.dot(np.dot(R, inputs["J_bp"]), np.transpose(R))
  J_b_inv = np.linalg.inv(J_b)

  # Use q to compute the initial angular velocity as seen from the body frame
  w_b0 = quaternion_rotation([0, inputs["w_bp"]], q)[1]

  # Get the quaternion which represents the rotation from the standard inertial frame to the initial body frame
  q_i = axis_angle_to_quaternion(-inputs["r_i_bp"][0], inputs["r_i_bp"][1])
  q_b = axis_angle_to_quaternion(-inputs["r_bp_b"][0], inputs["r_bp_b"][1])
  # Represents the coordinate transformation we would apply to represent a vector from I in B, initially.
  q_ib = quaternion_product(p=q_b, q=q_i, normalize=True)

  # Adaptive integration of the equations of motion in the body frame for free body rotation under zero torque
  def ddt_wb(w, t):
    return np.dot(J_b_inv, np.cross(np.dot(J_b, w), w))

  N_samples = int( (inputs["t_f"]*1.0) / dt )
  time_s = np.linspace(0, inputs["t_f"], N_samples)
  w_b = odeint(ddt_wb, w_b0, t = time_s)

  # Collect outputs so far!
  outputs = {}
  outputs["m_i"] = inputs["m_i"]
  outputs["a_i"] = inputs["a_i"]
  outputs["t_s"] = time_s
  outputs["w_b"] = w_b

  # Compute the time series of:
  # - ground truth coordinate transformation
  # - the magnetic, gravitational fields as seen from the body frame
  outputs["q_b"] = [quaternion_inverse(q_ib)]
  outputs["m_b"] = [quaternion_rotation([0, inputs["m_i"]], q_ib)[1]]
  outputs["a_b"] = [quaternion_rotation([0, inputs["a_i"]], q_ib)[1]]
  for idx in xrange(len(w_b)-1):
    q_ib = quaternion_product(p=w_dt_to_quaternion(-w_b[idx], dt), q=q_ib, normalize=True)
    outputs["q_b"].append(quaternion_inverse(q_ib))
    outputs["m_b"].append(quaternion_rotation([0, inputs["m_i"]], q_ib)[1])
    outputs["a_b"].append(quaternion_rotation([0, inputs["a_i"]], q_ib)[1])

  return outputs

