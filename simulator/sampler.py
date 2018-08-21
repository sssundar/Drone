import numpy as np
import sys, copy
from quaternions import vector_norm

# TODO The plant will need to sample 10x whatever dt is, and output its sampling frequency.
# TODO This module should really take an Estimator as an input so it can ping the Estimator
#       as each sensor stream becomes asynchronously available.
# If you do this with callbacks at the top level, then you can put a limit on how long the loop goes.
#
class Sampler(object):

  #
  # @brief      Initializes an object which mimics realistic sampling behavior
  #             given an ideal sensor stream from the Plant.
  #
  #                 - Multivariate Gaussian Noise Injection (incl. bias)
  #                 - Sensor-System Clock Drift
  #
  # @param[in]  self       A Disturbance object
  # @param[in]  noise      A dictionary with keys {"gyro", "compass", and
  #                        "accel"} specifying values [mu, sigma] which
  #                        represent the mean (absolute bias) and standard
  #                        deviation (as a percentage of magnitude) of an i.i.d.
  #                        Gaussian over R3.
  # @param[in]  input_hz   The sampling frequency for sensor data fed into this
  #                        module. Should be O(10)x the expected output
  #                        frequency.
  # @param[in]  output_hz  A dictionary with keys {"gyro", "compass", and
  #                        "accel"} specifying the sampling frequency in Hz for
  #                        sensor data output by this module.
  #
  def __init__(self, noise, input_hz, output_hz):
    return

# def fuzz_gyro(inputs):
#   outputs = copy.deepcopy(inputs)

#   sigma = np.eye(3) * ((np.pi/18)**2)     # 10 dps iid noise (appropriate for our beta)
#   mu = np.zeros([1,3])[0]
#   mu += np.pi/3                           # Gyro bias of 180 dps in each axis (appropriate for our zeta)
#   for k in xrange(len(outputs["w_b"])):
#     noise = np.random.multivariate_normal(mean=mu, cov=sigma)
#     w = outputs["w_b"][k]
#     outputs["w_b"][k] = w + noise

#   return outputs

# def fuzz_compass(inputs):
#   outputs = copy.deepcopy(inputs)

#   sigma = np.eye(3) * (0.1**2)     # 30% iid noise
#   mu = np.zeros([1,3])[0]
#   mu += 0                          # Compass bias is assumed 0 ATM. Would lead to heading error.
#   for k in xrange(len(outputs["m_b"])):
#     noise = np.random.multivariate_normal(mean=mu, cov=sigma)
#     outputs["m_b"][k] += noise
#     outputs["m_b"][k] /= vector_norm(outputs["m_b"][k])

#   return outputs

# def fuzz_accel(inputs):
#   outputs = copy.deepcopy(inputs)

#   sigma = np.eye(3) * (0.05**2)     # 10% iid noise
#   mu = np.zeros([1,3])[0]
#   mu += 0                          # Accel bias is assumed 0 ATM. Would lead to heading error.
#   for k in xrange(len(outputs["a_b"])):
#     noise = np.random.multivariate_normal(mean=mu, cov=sigma)
#     outputs["a_b"][k] += noise
#     outputs["a_b"][k] /= vector_norm(outputs["a_b"][k])

#   return outputs
