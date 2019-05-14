import numpy as np
from quaternions import *

# See notes from 8/30/18 to 9/4/18 for derivation.
class Controller(object):
  def __init__(self):
    # State Configuration
    self.duty_cycles = {
      "m1p2m3" : 0.0,
      "p1p2p3" : 0.0,
      "p1m2m3" : 0.0,
      "m1m2p3" : 0.0
    }

    # Input Configuration
    self.reference = {
      "r" : np.asarray([0.0, 0.0, 0.0]),
      "q" : [1, np.asarray([0.0,0.0,0.0])]
    }

    # Chassis Dynamics Configuration
    J1 = 2.383e-05
    J2 = 2.383e-05
    J3 = 4.583e-05
    self.J = np.eye(3)
    self.J[0,0] = J1
    self.J[1,1] = J2
    self.J[2,2] = J3
    self.J_inv = np.linalg.inv(self.J)

    # Motor Drive Dynamics Configuration
    D1 = [1.0, 1.0, -1.0, -1.0]
    D2 = [1.0, -1.0, -1.0, 1.0]
    D3 = [1.0, -1.0, 1.0, -1.0]
    D4 = [1.0, 1.0, 1.0, 1.0]
    self.D = np.matrix([D1, D2, D3, D4])
    self.D_inv = np.linalg.inv(self.D)

    # Scaling Factors
    self.MAX_DRIVE_TORQUE = 0.002
    self.e2_THRUST_MOMENT = 0.01
    self.e1_THRUST_MOMENT = 0.01
    self.BASE_DUTY = 2.0

    # PID Configuration
    self.kp = 20.0
    self.kd = 2.0

    return

  def get_duty_cycles(self):
    return self.duty_cycles

  def update_reference(self, r, q):
    self.reference["r"] = r
    self.reference["q"] = q

  # At the moment, this is purely orientation control. We ignore the CM's position.
  def process_state(self, t_s, q, w, r, dr):
    q_error = quaternion_product(p=self.reference["q"], q=quaternion_inverse(q), normalize=True)

    theta = np.arccos(q_error[0])
    sin_theta = np.sin(theta)
    if abs(sin_theta) < 1e-4:
      w_error = np.array([0.0, 0.0, 0.0])
      theta = 0.0
    else:
      w_error = q_error[1] / np.sin(theta)

    proportional = self.kp*theta*np.dot(self.J, w_error)
    derivative = -self.kd*np.dot(self.J, w)

    torque = proportional + derivative

    u = np.array([0.0, 0.0, 0.0, 0.0])
    u[0] = torque[0] / self.e1_THRUST_MOMENT
    u[1] = torque[1] / self.e2_THRUST_MOMENT
    u[2] = torque[2] / self.MAX_DRIVE_TORQUE
    u[3] = self.BASE_DUTY
    d = np.dot(self.D_inv, u)[0]

    for k in range(4):
      if d[0,k] > 1.0:
        d[0,k] = 1.0
      elif d[0,k] < 0.0:
        d[0,k] = 0.0

    self.duty_cycles["m1p2m3"] = d[0,0]
    self.duty_cycles["p1p2p3"] = d[0,1]
    self.duty_cycles["p1m2m3"] = d[0,2]
    self.duty_cycles["m1m2p3"] = d[0,3]
