import numpy as np
from quaternions import *

# @brief Simulates gyrometer and 3D compass readings during a rigid free-body rotation subject to zero external torque in a time-invariant local magnetic field.
#
# @param[in] inputs, a dictionary with the following fields:
#             - r_i_bp: [axis (unit numpy 3-vector), angle (radians)]
#                 an initial condition. how to rotate the standard inertial frame to the principal body frame
#             - r_bp_b: [axis (unit numpy 3-vector), angle (radians)]
#                 an initial condition. how to rotate the principal body frame to the actual body frame.
#             - J_bp: a numpy 3x3 matrix with x,y,z as indices 0,1,2, respectively. units kg m^2
#                 the 3x3 moment of inertia matrix as seen from the principal body frame
#             - w_i: a numpy 3-vector [rad/s]
#                 an initial condition. the angular velocity as seen from the standard inertial frame.
#             - f_s: a scalar. units Hz
#                 the sampling frequency for simulation
#             - t_f: a scalar. units seconds.
#                 the final simulation time, from zero.
#             - m_i: a numpy 3-vector. unitless.
#                 a unit vector representing the direction of the Earth's local magnetic field when the body frame is coincident with the standard inertial frame.
#                 note we could input a time-average while the device is booting (aka require horizontal calibration)
#                                     a hard-coded lookup of the local declination...
#
# @return outputs, a dictionary with the following fields:
#             - m_i: a numpy 3-vector. unitless.
#                 a unit vector representing the direction of the Earth's local magnetic field when the body frame is coincident with the standard inertial frame.
#                 intended as an input to the realism module
#             - t: a list of sample times (seconds) of length (t_f - 0) * f_s
#             - q_i: a list of quaternions [r, v] where v is a numpy 3-vector
#                    of the same length as t
#                 represents the 'ground truth' rotation of the body frame relative to the standard inertial frame
#                 as computed without sampling noise, bias, etc. slightly inaccurate due to numerical error but it's the best truth we've got by several orders of magnitude (see rotation/demo.py)
#             - w_b: a list of numpy 3-vectors [rad/s] representing ideal samples from a gyroscope coincident with the body frame
#                 intended as an input to the realism module
#             - m_b: a list of numpy 3-vectors [unit norm, unitless] representing ideal samples of the direction of the Earth's magnetic field from a 3D compass coincident with the body frame
#                 intended as an input to the realism module
def simulate(inputs):
  # Get the time step for this simulation
  dt = 1.0 / inputs["f_s"] # seconds

  # Get the moment of inertia matrix with respect to the body frame


  # Get the quaternion which represents the rotation from the standard inertial frame to the initial body frame

  # Use this to compute the initial angular velocity as seen from the body frame

  # Adaptive integration of the equations of motion in the body frame for free body rotation under zero torque

  # Compute the time series of:
  # - the quaternion representing rotation relative to the standard inertial frame
  # - the magnetic field as seen from the body frame

  # Collect outputs and return them!
