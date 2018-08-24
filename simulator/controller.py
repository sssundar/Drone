import numpy as np

class Controller(object):
  def __init__(self):
    self.duty_cycles = {
      "m1p2m3" : 0.0,
      "p1p2p3" : 0.0,
      "p1m2m3" : 0.0,
      "m1m2p3" : 0.0
    }
    self.reference = {
      "r" : np.asarray([0, 0, 0]),
      "q" : [1, np.asarray([0,0,0])]
    }
    return

  def get_duty_cycles(self):
    return self.duty_cycles

  def process_state(self, q, w, r, dr):
    pass
