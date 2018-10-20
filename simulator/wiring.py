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
  def __init__(self, iterations=100, decimation=1):
    # Sampling Configuration
    base_output_hz = 100.0
    n_oversampling = 1
    input_hz = base_output_hz*n_oversampling
    self.dt = 1.0/base_output_hz
    output_hz = {
      "gyro" : 1.0*base_output_hz,
      "compass" : 1.0*base_output_hz,
      "accel" : 1.0*base_output_hz
    }
    noise = {
      "gyro" : [np.pi/20, 0.05],
      "compass" : [0.0, 0.01],
      "accel" : [0.0, 0.05]
    }

    # Initial Conditions
    q0 = np.array([1,1,-1])
    q0 = q0 / vector_norm(q0)
    q0 = axis_angle_to_quaternion(q0, np.pi/30)

    # Estimator Configuration
    # Assume the IMU-to-quad frame offset and magnetic field are known perfectly, through offline calibration.
    q_offset = axis_angle_to_quaternion(np.asarray([1,0,0]), -np.pi/180)
    H = np.asarray([0.5,0,np.sqrt(3)/2])

    self.controller = Controller()
    self.estimator = Estimator(m=H, q_offset=q_offset, controller=self.controller)
    self.sampler = Sampler(output_hz=output_hz, noise=noise, estimator=self.estimator)
    self.plant = Plant(dt=self.dt, hz=input_hz, q0=q0, H=H, sampler=self.sampler, symmetric=True)

    # Simulation Time
    self.t_s = 0

    # Trajectory - Truth
    self.t = []
    self.r = []
    self.q = []

    # Trajectory - Estimated
    self.r_est = []
    self.q_est = []

    # Controller Output
    self.u = []

    # Loop Iterations
    self.iterations = iterations
    self.decimation = decimation
    return

  def simulate(self):
    u = self.controller.get_duty_cycles()
    for idx in xrange(self.iterations):
      if idx % 30 == 0:
        percent = (100.0*idx)/self.iterations
        complete = "%0.1f%%" % percent
        sys.stdout.write('\rSimulation ' + complete + " complete.")
        sys.stdout.flush()

      (q, r) = self.plant.evolve(self.t_s, u)
      u = self.controller.get_duty_cycles()
      self.t_s += self.dt

      if idx % self.decimation == 0:
        self.t.append(self.t_s)
        self.r.append(r)
        self.q.append(q)
        self.r_est.append(copy.deepcopy(self.estimator.r))
        self.q_est.append(copy.deepcopy(self.estimator.q))
        self.u.append(copy.deepcopy(u))

  def visualize_chassis(self):
    gt0, gt1, gt2 = generate_body_frames(self.q)
    est0, est1, est2 = generate_body_frames(self.q_est)
    compare(len(self.t), est0, est1, est2, gt0, gt1, gt2, 1)

  def visualize_cm(self):
    x = lambda series: [v[0] for v in series]
    y = lambda series: [v[1] for v in series]
    z = lambda series: [v[2] for v in series]

    plt.subplot(311)
    plt.plot(self.t, x(self.r), 'k-')
    plt.plot(self.t, x(self.r_est), 'k--')
    plt.legend(["x", "x_est"])
    plt.xlabel("s")
    plt.ylabel("m")

    plt.subplot(312)
    plt.plot(self.t, y(self.r), 'k-')
    plt.plot(self.t, y(self.r_est), 'k--')
    plt.legend(["y", "y_est"])
    plt.xlabel("s")
    plt.ylabel("m")

    plt.subplot(313)
    plt.plot(self.t, z(self.r), 'k-')
    plt.plot(self.t, z(self.r_est), 'k--')
    plt.legend(["z", "z_est"])
    plt.xlabel("s")
    plt.ylabel("m")

    plt.show()

  def visualize_control(self):
    series = {}
    for k in self.u[0].keys():
      series[k] = []
      for u in self.u:
        series[k].append(u[k])

    legend = series.keys()
    for k in legend:
      plt.plot(self.t, series[k])
    plt.xlabel("s")
    plt.ylabel("duty")
    plt.legend(legend)
    plt.show()

if __name__ == "__main__":
  wiring = Wiring(iterations=1200, decimation=10)
  wiring.simulate()
  if (len(sys.argv) >= 2) and (sys.argv[1] == 'chassis'):
    # Install ImageMagick on Ubuntu then, after running this script, go to 'images' and run
    # convert -delay 0.05 -loop 0 *png stabilization.gif
    wiring.visualize_chassis()
  elif (len(sys.argv) >= 2) and (sys.argv[1] == 'control'):
    wiring.visualize_control()
  else:
    wiring.visualize_cm()
