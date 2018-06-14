import sys, glob, os
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def draw_body(axes, vectors, colors):
  line_collections = []

  for m in xrange(len(vectors)):
    vector = vectors[m]

    X = Y = Z = 0
    U = vector[1][0]
    V = vector[1][1]
    W = vector[1][2]

    line_collections.append(axes.quiver(Y,X,Z,V,U,W, length=0.05, arrow_length_ratio=0.05, pivot='tail', color=colors[m]))

  return line_collections

def clear_animation():
  files = glob.glob('./images/*')
  for f in files:
    os.remove(f)

def animate(n_vectors, q_e0, q_e1, q_e2, decimator):
  # Wipe the images/ directory
  clear_animation()

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")
  # Aircraft Axes
  # ax.set_xlabel("y")
  # ax.set_ylabel("x")
  # ax.set_zlabel("z")  
  # ax.invert_zaxis()

  line_collections = None
  count = 0

  for n in xrange(n_vectors):
    if (n % decimator) != 0:
      continue

    if (line_collections is not None) and (count > 1):
      # The first set of axes drawn stays as a reference in all images.
      # The rest are wiped right after they are drawn/saved to give the illusion of motion.
      for col in line_collections:
        ax.collections.remove(col)

    line_collections = draw_body(ax, [q_e0[n], q_e1[n], q_e2[n]], ["r", "g", "b"])

    plt.savefig('images/Frame%08i.png' % n)
    plt.draw()

    count += 1

def parse(log, show_it):
  # Container
  trace = {"s": [], "w" : []}
    
  # Measurement Settings
  dps_per_lsb = 2000.0/(2**15-1)
  with open(log, "r") as f:
    first_ms = None
    for line in f:
      ms, x, y, z = [int(x.strip()) for x in line.strip().split(",")]
      if first_ms is None:
        trace["s"].append(0)
        first_ms = ms
      else:
        trace["s"].append(1.0*(ms-first_ms)/1000)
      v = np.asarray([x,y,z])
      v = v*dps_per_lsb*(np.pi/180)
      trace["w"].append(v)      

  return trace

# We know:
#   the unit q(w, dt) = exp(2^-1 |w|dt [w/|w|]) = cos(|w|dt/2) + (w/|w|)sin(|w|dt/2) 
# along with 
#   p' = qpq^-1 
# represents a rotation of p by |w|dt radians about w in the inertial frame.
#
# Treat our original orientation as coincident with an inertial frame. 
# We make measurements in the body frame.
# 
# We build up the rotation of the body frame at time t relative to the original inertial frame (t=0)
# by accumulating r_i(t) = q(w_i(t-dt), dt)r_i(t-dt) then rotating w_b(t) -> w_i(t) using r_i(t)^-1,
# that is, w_i(t) = r_i(t)^-1 w_b(t) r_i(t)
#
# We therefore need routines to generate q(w,dt), multiply quaternions (post-normalizing), 
# and compute quaternion inverses. We also need to store r_i(t-dt) and w_i(t-dt).
def v_norm(w):
  return np.sqrt(sum(w*w))

def q_exp(w,dt):
  w_norm = v_norm(w)
  if (w_norm > 1E-9):
    unit_w = w/w_norm
  else:
    w_norm = 0
    unit_w = np.asarray([0,0,0])
  c = np.cos(w_norm*dt/2)
  s = np.sin(w_norm*dt/2) 
  return [c, unit_w * s]

def dot(v1,v2):
  return sum(v1 * v2)

def cross(v1,v2):
  return np.asarray([ v1[1]*v2[2] - v1[2]*v2[1], 
            v1[2]*v2[0] - v1[0]*v2[2],
            v1[0]*v2[1] - v1[1]*v2[0] ])  

def q_norm(q):
  return np.sqrt(q[0]*q[0] + sum(q[1]*q[1]))

def product(p, q, normalize):
  r1, v1 = p
  r2, v2 = q
  pq = [r1*r2 - dot(v1,v2), r1*v2 + r2*v1 + cross(v1,v2)]
  pq_norm = q_norm(pq)
  if normalize:
    return [e / pq_norm for e in pq]
  else:
    return pq

def conjugate(q):
  return [q[0], -q[1]]

def q_inv(q): 
  p = conjugate(q)
  norm_sq = q_norm(q)**2
  return [e / norm_sq  for e in p]

def rotate(qv,qr):
  return product(product(qr,qv, False), q_inv(qr), False)

# Assume a constant inertial-frame rotation about the x and z-axes 
#   (w_i = (2pi,0,2pi), where + is clockwise)
# Graph the evolution of the body axes in the inertial frame.
def rotation_example():
  w_i = np.asarray([2*np.pi,0,2*np.pi])
  r_i =  [[1,np.asarray([0,0,0])]]
  dt = 1.0/800 # 800 Hz
  N = 800

  # r_i(t) = q(w_i(t-dt), dt)r_i(t-dt) 
  for n in xrange(N):
    r_i.append(product(q_exp(w_i, dt), r_i[-1], True))

  e0_b = [[0,np.asarray([1,0,0])]]
  e1_b = [[0,np.asarray([0,1,0])]]
  e2_b = [[0,np.asarray([0,0,1])]]
  for rotation in r_i:
    e0_b.append(rotate(e0_b[0], rotation))
    e1_b.append(rotate(e1_b[0], rotation))
    e2_b.append(rotate(e2_b[0], rotation))  

  animate(len(r_i), e0_b, e1_b, e2_b, 4)

# See above.
#  We build up the rotation of the body frame at time t relative to the original inertial frame (t=0)
#  by accumulating r_i(t) = q(w_i(t-dt), dt)r_i(t-dt) then rotating w_b(t) -> w_i(t) using r_i(t)^-1,
#  that is, w_i(t) = r_i(t)^-1 w_b(t) r_i(t)
def trace_rotation(trace, time_it, sanity_checks, hpf): 
  r_i = [[1,np.asarray([0,0,0])]]
  q_wi = [[0,np.asarray([0,0,0])]]
  x_nm1 = np.asarray([0,0,0])
  M = 4
  y_nmM = [np.asarray([0,0,0])] * M
  beta = 0.5  
  for idx in xrange(len(trace["s"])-1):
    dt = trace["s"][idx+1]-trace["s"][idx]    
    q_wi.append(rotate([0, trace["w"][idx]], q_inv(r_i[-1])))
    if hpf:         
      # This works out as y(n) = x(n) - x(n-1) - B*y(n-M)
      #           out(n) = [(1-B)/2] y(n)     
      x_n = q_wi[-1][1]
      y_n = x_n - x_nm1 - beta*y_nmM[-1]
      w_i = ((1-beta)/2)*y_n
      x_nm1 = x_n
      y_nmM.pop(-1)
      y_nmM = [y_n] + y_nmM
    else:
      w_i = q_wi[-1][1] # Take the vector part    
    r_i.append(product(q_exp(w_i, dt), r_i[-1], True))

  e0_b = [[0,np.asarray([1,0,0])]]
  e1_b = [[0,np.asarray([0,1,0])]]
  e2_b = [[0,np.asarray([0,0,1])]]
  for rotation in r_i[1::]:
    e0_b.append(rotate(e0_b[0], rotation))
    e1_b.append(rotate(e1_b[0], rotation))
    e2_b.append(rotate(e2_b[0], rotation))  
  
  if time_it:
    # Plot a time series
    plt.subplot(411)
    plt.plot(trace["s"], [v[0] for v in trace["w"]])
    plt.plot(trace["s"], [v[1] for v in trace["w"]])
    plt.plot(trace["s"], [v[2] for v in trace["w"]])
    plt.legend(["w_x", "w_y", "w_z"])
    plt.ylabel("radians per second (body frame)")
    plt.title("Shaking Gyro Back and Forth")

    plt.subplot(412)
    plt.plot(trace["s"], [q[1][0] for q in e0_b])
    plt.plot(trace["s"], [q[1][1] for q in e0_b])
    plt.plot(trace["s"], [q[1][2] for q in e0_b])
    plt.legend(["e0_x", "e0_y", "e0_z"])
    plt.ylabel("Direction Cosines\nfor e0")

    plt.subplot(413)
    plt.plot(trace["s"], [q[1][0] for q in e1_b])
    plt.plot(trace["s"], [q[1][1] for q in e1_b])
    plt.plot(trace["s"], [q[1][2] for q in e1_b])
    plt.legend(["e1_x", "e1_y", "e1_z"])
    plt.ylabel("Direction Cosines\nfor e1")

    plt.subplot(414)
    plt.plot(trace["s"], [q[1][0] for q in e2_b])
    plt.plot(trace["s"], [q[1][1] for q in e2_b])
    plt.plot(trace["s"], [q[1][2] for q in e2_b])
    plt.legend(["e2_x", "e2_y", "e2_z"])
    plt.ylabel("Direction Cosines\nfor e2")
    plt.xlabel("seconds")
    plt.show()
  elif sanity_checks:
    plt.subplot(211)
    plt.plot(trace["s"], [v[0] for v in q_wi])    
    plt.plot(trace["s"], [v[0] for v in e0_b])    
    plt.plot(trace["s"], [v[0] for v in e1_b])    
    plt.plot(trace["s"], [v[0] for v in e2_b])    
    plt.ylabel("real part of q-vectors")
    plt.subplot(212)
    plt.plot(trace["s"], [v_norm(q[1]) for q in e0_b])
    plt.plot(trace["s"], [v_norm(q[1]) for q in e1_b])
    plt.plot(trace["s"], [v_norm(q[1]) for q in e2_b])
    plt.ylabel("vector norm of unit body axes")
    plt.xlabel("seconds")
    plt.show()
  else:
    # Dump a series of stills to ./images/ so it's easier to visualize the rotation
    animate(len(r_i), e0_b, e1_b, e2_b, 100)

if __name__ == "__main__":  
  # A visual test of our arithmetic
  # rotation_example()

  # The actual test. Show me the motion we measured!
  w_b = parse(sys.argv[1], False)

  # Looks quite reasonable, actually.   
  # trace_rotation(w_b, False, False, False) # Animation
  # trace_rotation(w_b, True, False, False)  # Time Series

  # Let's sanity check the 'rotation' part of this.
  # 1. Are your body vectors staying unit norm?
    # Yes.
  # 2. How large is the real part of quaternions representing vectors? Should be zero. 
    # It is zero, within numerical tolerance
  # trace_rotation(w_b, False, True, False)  # Sanity Checks

  # We can immediately see that bias kills us. We've rotated at least 10 degrees
  # during two seconds of stillness. 
  
  # A simple two-tap filter O(d_n - d_n-1) would (1) not kill DC and 
  # (2) see Fs/2 as 'fast' and everything else as 'slow' which is NOT what we want. 
  # Mechanical timescales are pretty slow relative to our sampling frequency. 

  # We need to kill the DC term and keep the rest near-level gain.
  # Phase... well, mechanical timescales are slow. Let's hope phase doesn't matter (incl. correction).
  # Instead, we'll use a normalized IIR filter with the constructed response:
  # H(z) = [(1-B)/2] (1-z^-1) / (1+Bz^-1)  to start with 0 <= B < 1.
  # This works out as y(n) = x(n) - x(n-1) - B*y(n-1)
  #           out(n) = [(1-B)/2] y(n)
  # We can add poles as needed to flatten the response.
  trace_rotation(w_b, True, False, True)
