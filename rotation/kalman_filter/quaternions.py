import numpy as np

# @param[in] w A numpy 3-vector
# @return The L2 norm of w
def vector_norm(w):
  return np.sqrt(sum(w*w))

# @param[in] w A numpy 3-vector representing the angular velocity
# @param[in] dt A time, in seconds, to rotate at this angular velocity
# @return Let w_hat = w/|w| be a unit axis of rotation, and theta = |w|dt be the angle of rotation about that axis.
#         This function returns a quaternion of the form exp(0.5*theta*w_hat)
def quaternion_exponential(w,dt):
  w_norm = vector_norm(w)
  if (w_norm > 1E-9):
    unit_w = w/w_norm
  else:
    w_norm = 0
    unit_w = np.asarray([0,0,0])
  c = np.cos(w_norm*dt/2)
  s = np.sin(w_norm*dt/2)
  return [c, unit_w * s]

# @param[in] v1 A numpy 3-vector
# @param[in] v2 A numpy 3-vector
# @return The dot product, <v1, v2>
def vector_dot(v1,v2):
  return sum(v1 * v2)

# @param[in] v1 A numpy 3-vector
# @param[in] v2 A numpy 3-vector
# @return A numpy 3-vector, the cross product, v1 x v2
def vector_cross(v1,v2):
  return np.asarray([ v1[1]*v2[2] - v1[2]*v2[1],
                      v1[2]*v2[0] - v1[0]*v2[2],
                      v1[0]*v2[1] - v1[1]*v2[0] ])

# @param[in] q A quaternion of the form [r, v] where v is a numpy 3-vector
# @return The L2 norm of the quaternion
def quaternion_norm(q):
  return np.sqrt(q[0]*q[0] + sum(q[1]*q[1]))

# @param[in] p, q A quaternion of the form [r, v] where v is a numpy 3-vector
# @param[in] normalize A boolean indicating whether the quaternion product is to be normalized (useful for rotation).
# @return - pq, possibly normalized by its magnitude
def quaternion_product(p, q, normalize):
  r1, v1 = p
  r2, v2 = q
  pq = [r1*r2 - vector_dot(v1,v2), r1*v2 + r2*v1 + vector_cross(v1,v2)]
  pquaternion_norm = quaternion_norm(pq)
  if normalize:
    return [e / pquaternion_norm for e in pq]
  else:
    return pq

# @param[in] q A quaternion of the form [r, v] where v is a numpy 3-vector
# @return The conjugate of q, [r, -v]
def quaternion_conjugate(q):
  return [q[0], -q[1]]

# @param[in] q A quaternion of the form [r, v] where v is a numpy 3-vector
# @return The multiplicative inverse of q
def quaternion_inverse(q):
  p = quaternion_conjugate(q)
  norm_sq = quaternion_norm(q)**2
  return [e / norm_sq  for e in p]

# @param[in] qv A 'pure vector' quaternion of the form [0, v] where v is a numpy 3-vector
# @param[in] qr A rotation quaternion of the form [r, v] where v is a numpy 3-vector
# @return The quaternion product qr qv qr^-1, a 'pure vector'
def quaternion_rotation(qv,qr):
  return quaternion_product(quaternion_product(qr,qv, False), quaternion_inverse(qr), False)
