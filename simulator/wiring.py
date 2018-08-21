import sys
import numpy as np
from animate import *
from plant import Plant
from sampler import Sampler
from estimation import Estimator
from controller import Controller

class Wiring(object):

  #
  # @brief      Wires up a closed-loop control system and logs the trajectory.
  #             Acts as a callback context to allow objects to talk to each
  #             other.
  #
  # @param[in]  self  A Wiring object
  #
  def __init__(self, max_iterations=100):
    self.dt = 0.01
    self.plant = Plant(ctx=self, dt=self.dt, hz=1000.0)
    self.sampler = Sampler(ctx=self)
    self.estimator = Estimator(ctx=self)
    self.controller = Controller(ctx=self)

    # Trajectory
    self.t_s = 0
    self.t = []
    self.r = []
    self.q = []

    # Loop Iterations
    self.iterations = 0
    self.max_iterations = max_iterations
    return

  def process_evolution(self, w, m, a, q, r, dr, ddr):
    self.t_s += self.dt
    self.t.append(self.t_s)
    self.r.append(r)
    self.q.append(q)

    HOW DO YOU AVOID AN INFINITE LOOP?
    pass

  def process_reference():
    pass
