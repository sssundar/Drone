import sys
import numpy as np
from quaternions import *

def f_g(q, a):
    r, v = q
    q = [r] + list(v)
    q1, q2, q3, q4 = q
    ax, ay, az = list(a)
    line1 = 2*(q2*q4 - q1*q3) - ax
    line2 = 2*(q1*q2 + q3*q4) - ay
    line3 = 2*(0.5 - q2**2 - q3**2) - az
    return np.array([line1, line2, line3])

def J_g(q):
  r, v = q
  q = [r] + list(v)
  q1, q2, q3, q4 = q
  line1 = [-2*q3, 2*q4, -2*q1, 2*q2]
  line2 = [2*q2, 2*q1, 2*q4, 2*q3]
  line3 = [0, -4*q2, -4*q3, 0]
  return np.matrix([line1, line2, line3])

def f_b(q, b, m):
  r, v = q
  q = [r] + list(v)
  q1, q2, q3, q4 = q
  bx, _, bz = list(b)
  mx, my, mz = list(m)
  line1 = 2*bx*(0.5 - q3**2 - q4**2) + 2*bz*(q2*q4 - q1*q3) - mx
  line2 = 2*bx*(q2*q3 - q1*q4) + 2*bz*(q1*q2 + q3*q4) - my
  line3 = 2*bx*(q1*q3 + q2*q4) + 2*bz*(0.5 - q2**2 - q3**2) - mz
  return np.array([line1, line2, line3])

def J_b(q, b):
  r, v = q
  q = [r] + list(v)
  q1, q2, q3, q4 = q
  bx, _, bz = b

  line1 = [-2*bz*q3, 2*bz*q4, -4*bx*q3 - 2*bz*q1, -4*bx*q4 + 2*bz*q2]
  line2 = [-2*bx*q4 + 2*bz*q2, 2*bx*q3 + 2*bz*q1, 2*bx*q2 + 2*bz*q4, -2*bx*q1 + 2*bz*q3]
  line3 = [2*bx*q3 , 2*bx*q4 - 4*bz*q2, 2*bx*q1 - 4*bz*q3, 2*bx*q2]

  return np.matrix([line1, line2, line3])

def f_gb(q, a, b, m):
  return np.array(list(f_g(q, a)) + list(f_b(q, b, m)))

def J_gb(q, b):
  return np.concatenate((J_g(q), J_b(q, b)), axis=0)

def gradient_f(q, a, b, m):
  return np.array(np.dot(np.transpose(J_gb(q, b)), f_gb(q, a, b, m)))[0]


class Estimator(object):
  def __init__(self, q_offset=[1, np.asarray([0,0,0])], controller=None):
    self.controller = controller
    self.q_offset = q_offset
    self.offset_correction = lambda v: quaternion_rotation(qv=[0,v], qr=quaternion_inverse(self.q_offset))[1]

    # Scale should match the sampling noise configured in wiring.py
    # Note zeta cannot compensate for beta.
    # Note beta can compensate for bias if it is large enough.
    #   Then... you'll jitter if there's mag/acc noise. You don't want to make beta unnecessarily large.
    self.beta = np.pi/2   # ~18 dps spread iid in all axes. If this is zero we're just integrating without any orientation compensation
    self.zeta = np.pi/3    # ~60 dps bias error iid in all axes. If this is zero we don't use gyro bias compensation.

    # Bias Accumulator
    self.int_we = np.asarray([0.0,0.0,0.0])

    # Our estimate of the rotation of the body frame relative to the inertial frame.
    self.q = [1, np.asarray([0,0,0])]
    self.ddt_q = [0, np.asarray([0,0,0])]
    self.r = np.asarray([0.0, 0.0, 0.0])
    self.ddt_r = np.asarray([0.0,0.0,0.0])

    # A buffer of samples that we synchronize and resample at 100Hz
    # TODO For now, we are just assuming we get perfect 100Hz data.
    self.have_received_gyro = False
    self.have_received_compass = False
    self.have_received_accel = False
    self.gyro = None
    self.compass = None
    self.accel = None
    self.t_previous_s = 0.0
    self.t_s = 0.0
    return

  def passthrough(self, t_s):
    if (self.have_received_gyro and self.have_received_accel and self.have_received_compass):
      self.t_s = t_s
      self.update(self.gyro, self.compass, self.accel)
      self.have_received_gyro = False
      self.have_received_compass = False
      self.have_received_accel = False

  def process_gyro(self, t_s, sample):
    # Make sure you copy the sample (when you do this for real) so the reference is clean.
    self.have_received_gyro = True
    self.gyro = self.offset_correction(sample)
    self.passthrough(t_s)

  def process_compass(self, t_s, sample):
    # Make sure you copy the sample (when you do this for real) so the reference is clean.
    self.have_received_compass = True
    self.compass = self.offset_correction(sample)
    self.passthrough(t_s)

  def process_accel(self, t_s, sample):
    # Make sure you copy the sample (when you do this for real) so the reference is clean.
    self.have_received_accel = True
    self.accel = self.offset_correction(sample)
    self.passthrough(t_s)

  def update(self, w_b, m_b, a_b):
    # Ideally the synchronizer that replaces passthrough() guarantees a fixed tick, e.g. 10ms.
    dt = self.t_s - self.t_previous_s
    self.t_previous_s = self.t_s

    # Estimate our orientation and rate of change of orientation
    m = m_b
    b = quaternion_rotation(qv=[0,m], qr=self.q)[1]
    b = np.asarray([np.sqrt(b[0]**2 + b[1]**2), 0, b[2]])

    grad_f = gradient_f(self.q, a_b, b, m)

    if vector_norm(grad_f) > 1E-9:
      qe_dot = grad_f / vector_norm(grad_f)
      new_part = self.beta * qe_dot
    else:
      qe_dot = np.asarray([0,0,0,0])
      new_part = [0,0,0,0]
    qe_dot = [qe_dot[0], np.asarray([qe_dot[1], qe_dot[2], qe_dot[3]])]

    w_e = quaternion_times_scalar(scalar=2, quaternion=quaternion_product(quaternion_inverse(self.q), qe_dot, False))[1]
    self.int_we += w_e*dt

    q_dot = quaternion_times_scalar(scalar=.5, quaternion=quaternion_product(self.q, [0, w_b - self.zeta*self.int_we], False))

    dq = [new_part[0], np.array([new_part[1], new_part[2], new_part[3]])]
    dq = [q_dot[0] - dq[0], q_dot[1] - dq[1]]
    self.ddt_q = dq
    dq = [dq[0]*dt, dq[1]*dt]

    q = [self.q[0] + (self.ddt_q[0]*dt), self.q[1] + (self.ddt_q[1] * dt)]
    self.q = [e / quaternion_norm(q)  for e in q]

    # Estimate our position and velocity. Bear in mind we currently have no
    # positional estimate whatsoever, so our goal here is to use heuristics here
    # to keep drift reasonable. We have no expectation of accuracy. The first
    # controller goal is attitude stabilization.

    d2dt2_r = quaternion_rotation(qv=[0, a_b], qr=self.q)[1]
    d2dt2_r += np.asarray([0,0,-9.8])
    self.r += self.ddt_r*dt
    self.ddt_r += d2dt2_r*dt

    # Tell the controller we have an updated state estimate.\
    if self.controller is not None:
      w_eff = quaternion_times_scalar(scalar=2, quaternion=quaternion_product(quaternion_inverse(self.q), self.ddt_q, False))[1]
      self.controller.process_state(self.t_s, self.q, w_eff, self.r, self.ddt_r)
