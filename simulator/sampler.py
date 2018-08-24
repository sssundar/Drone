import numpy as np
import sys, copy
from quaternions import vector_norm

class Sampler(object):

  #
  # @brief      Initializes an object which mimics realistic sampling behavior
  #             given an ideal sensor stream from the Plant.
  #
  #                 - Multivariate Gaussian Noise Injection (incl. bias)
  #                 - Sensor-System Clock Drift
  #
  # @param[in]  self       A Sampler object
  # @param[in]  output_hz  A dictionary with keys {"gyro", "compass", and
  #                        "accel"} specifying the sampling frequency in Hz for
  #                        sensor data output by this module.
  # @param[in]  noise      A dictionary with keys {"gyro", "compass", and
  #                        "accel"} specifying values [mu, sigma] which
  #                        represent the mean (absolute bias) and standard
  #                        deviation (as a percentage of magnitude) of an i.i.d.
  #                        Gaussian over R3.
  # @param[in]  estimator  An Estimator object fed by this Sampler
  #
  def __init__(self, output_hz, noise, estimator=None):
    self.estimator = estimator
    self.output_hz = output_hz
    self.noise = noise

    # TODO For now, we just feed samples through perfectly. Later, we will need
    # to track time streams for each channel and resample to match the expected
    # output_hz. This kludge just takes perfect 100Hz samples as we know we're
    # feeding in 1kHz.
    self.t_target = 0.01
    return

  def process_samples(self, t_s, gyro, compass, accel):
    if t_s >= self.t_target:
      self.t_target += 0.01
      if self.estimator is not None:
        self.estimator.process_gyro(t_s, gyro)
        self.estimator.process_compass(t_s, compass)
        self.estimator.process_accel(t_s, accel)
