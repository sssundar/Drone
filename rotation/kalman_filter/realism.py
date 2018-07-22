# Will add realism to the ideal-measurement time series created by free_body.py.
# For instance, adding bias, noise, time jitter, or sensor streams with differing sampling rates.

import numpy as np
import sys

# @param[in] inputs, a dictionary with the following fields:
#             - m_i: a numpy 3-vector. unitless.
#                 a unit vector representing the direction of the Earth's local magnetic field when the body frame is coincident with the standard inertial frame.
#             - a_i: a numpy 3-vector. unitless.
#                 a unit vector representing the direction of the Earth's local gravitational field when the body frame is coincident with the standard inertial frame.
#             - t_s: a list of sample times (seconds) of length (t_f - 0) * f_s
#             - q_b: a list of quaternions [r, v] where v is a numpy 3-vector
#                    of the same length as t
#                 represents the 'ground truth' coordinate transformations. rotating a vector v_b by q_b yields v_i. computed without sampling noise, bias, etc.
#             - w_b: a list of numpy 3-vectors [rad/s] representing ideal samples from a gyroscope coincident with the body frame
#             - m_b: a list of numpy 3-vectors [unit norm, unitless] representing ideal samples of the direction of the Earth's magnetic field from a 3D compass coincident with the body frame
#             - a_b: a list of numpy 3-vectors [unit norm, unitless] representing ideal samples of the direction of the Earth's gravitational field from a 3D accelerometer coincident with the body frame
# @returns outputs, a dictionary with the following fields:
#             - m_i: a numpy 3-vector. unitless.
#                 a unit vector representing the direction of the Earth's local magnetic field when the body frame is coincident with the standard inertial frame.
#             - a_i: a numpy 3-vector. unitless.
#                 a unit vector representing the direction of the Earth's local gravitational field when the body frame is coincident with the standard inertial frame.
#             - t_s: a list of sample times (seconds) of length (t_f - 0) * f_s
#             - q_b: a list of quaternions [r, v] where v is a numpy 3-vector
#                    of the same length as t
#                 represents the 'ground truth' coordinate transformations. rotating a vector v_b by q_b yields v_i. computed without sampling noise, bias, etc.
#             - w_b: a list of numpy 3-vectors [rad/s] representing realistic samples from a gyroscope coincident with the body frame
#             - m_b: a list of numpy 3-vectors [unit norm, unitless] representing realistic samples of the direction of the Earth's magnetic field from a 3D compass coincident with the body frame
#             - a_b: a list of numpy 3-vectors [unit norm, unitless] representing realistic samples of the direction of the Earth's gravitational field from a 3D accelerometer coincident with the body frame
def fuzz_gyro(inputs):
  outputs = inputs

  sigma = np.eye(3) * ((np.pi/18)**2)     # 10 dps iid noise (appropriate for our beta)
  mu = np.zeros([1,3])[0]
  mu += np.pi/3                           # Gyro bias of 180 dps in each axis (appropriate for our zeta)
  for k in xrange(len(outputs["w_b"])):
    noise = np.random.multivariate_normal(mean=mu, cov=sigma)
    w = outputs["w_b"][k]
    outputs["w_b"][k] = w + noise

  return outputs

def fuzz_compass(inputs):
  pass

def fuzz_accel(inputs):
  pass
