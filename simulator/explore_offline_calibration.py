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
  # return np.array(np.dot(np.transpose(J_b(q, b)), f_b(q, b, m)))[0]
  return np.array(np.dot(np.transpose(J_gb(q, b)), f_gb(q, a, b, m)))[0]

def get_sensor_to_quad_frame_quaternion(w, a):  
  beta = np.pi/100
  q = [1, np.asarray([0,0,0])] # Initially we assume no rotation is required.

  z_hat = a / vector_norm(a) # Since the quad was horizontal and still, acceleration tells us about the z-axis (i.e. gravity).
  x_hat = w / vector_norm(w) # Since the quad was rotated about its x-axis, angular velocity tells us about the x-axis.
  x_expected = np.asarray([1., 0., 0.])

  epoch = 0
  last_vnorm = None
  while True:
    epoch += 1
    if epoch % 100 == 0:
      beta /= 10

    grad_f = gradient_f(q, z_hat, x_expected, x_hat)
    vnorm = vector_norm(grad_f)
    
    print([epoch, vector_norm(grad_f)])

    if last_vnorm is None:
      last_vnorm = vnorm
    else:
      if (100.*(abs(last_vnorm - vnorm)/vnorm)) < 0.1:
        # We appear to have converged!
        break
      last_vnorm = vnorm

    if vnorm > 1E-9:  
      qe_dot = grad_f / vnorm
      dq = [-beta*qe_dot[0], -beta*np.array([qe_dot[1], qe_dot[2], qe_dot[3]])]
    else:
      # Even if we're still making great strides, this is good enough!
      break  

    q = [q[0] + dq[0], q[1] + dq[1]]
    q = [e / quaternion_norm(q)  for e in q]

  return q
  
if __name__ == "__main__":
  # Assume the gyro and accelerometer are already aligned in x,y,z 
  # Assume the IMU as a whole is offset from the quad frame.
  w_act = [0, np.array([1.,0., 0.])]
  a_act = [0, np.array([0., 0., 1.])]
  q_offset_w = axis_angle_to_quaternion(np.asarray([0.5,np.sqrt(3)/2,0.]), -np.pi/2)
  q_offset_a = axis_angle_to_quaternion(np.asarray([0.5,np.sqrt(3)/2,0.]), -0.95*np.pi/2)
  w_meas = quaternion_rotation(qv=w_act, qr=q_offset_w)[1]
  a_meas = quaternion_rotation(qv=a_act, qr=q_offset_a)[1]
  q_offset_est = get_sensor_to_quad_frame_quaternion(w_meas, a_meas)
  w_est = quaternion_rotation(qv=[0, w_meas], qr=q_offset_est)[1]
  a_est = quaternion_rotation(qv=[0, a_meas], qr=q_offset_est)[1]

  # This is doing quite well. The gradient descent (taken from the MCF) is fine.
  # I notice if I have measurement error, say misalignment in test jig, so w_meas 
  # and a_meas are not quite x,z by O(10^-2) degrees, then this propagates 
  # to an O(10^-2) error in my final angle estimate as well.
  # 
  # In the long run we'd probably want to do multiple independent measurements
  # including, for instance, axes like y as well, to try and get noise
  # to cancel itself out, but ... for now this is ok.
  # 
  # You can approach this time average where you normalize each time point to
  # a unit vector for w, a and then average them all to get a magnitude invariant 'which direction'
  # vector. O(10^-2 degree error, I can live with for Rev A).
  print(w_act[1], w_meas, w_est)
  print(a_act[1], a_meas, a_est)
  print(q_offset_w)
  print(q_offset_a)
  print(q_offset_est)
