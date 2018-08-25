import sys, copy
import numpy as np
from animate import *
from matplotlib import pyplot as plt
from quaternions import *
from plant import Plant
from sampler import Sampler
from estimation import Estimator
from controller import Controller
import pdb

class Wiring(object):

  #
  # @brief      Wires up a closed-loop control system and logs the trajectory.
  #
  # @param[in]  self  A Wiring object
  #
  def __init__(self, iterations=100):
    # Sampling Configuration
    base_output_hz = 100.0
    n_oversampling = 10
    input_hz = base_output_hz*n_oversampling
    self.dt = 1.0/base_output_hz
    output_hz = {
      "gyro" : 1.0*base_output_hz,
      "compass" : 1.0*base_output_hz,
      "accel" : 1.0*base_output_hz
    }
    noise = {
      "gyro" : [0.0, 0.0],
      "compass" : [0.0, 0.0],
      "accel" : [0.0, 0.0]
    }

    # Estimator Configuration
    # Assume the IMU-to-quad frame offset is known perfectly, through offline calibration.
    q_offset = axis_angle_to_quaternion(np.asarray([1,0,0]), -np.pi/180)

    self.controller = Controller()
    self.estimator = Estimator(q_offset=q_offset, controller=self.controller)
    self.sampler = Sampler(output_hz=output_hz, noise=noise, estimator=self.estimator)
    self.plant = Plant(dt=self.dt, hz=input_hz, sampler=self.sampler, symmetric=True)

    # Trajectory
    self.t_s = 0
    self.t = []
    self.r = []
    self.q = []
    self.r_est = []

    # Loop Iterations
    self.iterations = iterations
    return

  def simulate(self):
    for idx in xrange(self.iterations):
      u = self.controller.get_duty_cycles()
      (q, r) = self.plant.evolve(self.t_s, u)
      self.t_s += self.dt
      self.t.append(self.t_s)
      self.r.append(r)
      self.q.append(q)
      self.r_est.append(copy.deepcopy(self.estimator.r))

  def visualize(self):
    plt.plot(self.t, self.r)
    plt.plot(self.t, self.r_est)
    plt.legend(["x", "y", "z", "x_est", "y_est", "z_est"])
    plt.show()

if __name__ == "__main__":
  wiring = Wiring()
  wiring.simulate()
  wiring.visualize()
