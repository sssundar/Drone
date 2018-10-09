import sys, glob, os
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from quaternions import *

def draw_body(axes, vectors, colors):
  line_collections = []

  for m in xrange(len(vectors)):
    vector = vectors[m]

    X = Y = Z = 0
    U = vector[1][0]
    V = vector[1][1]
    W = vector[1][2]

    line_collections.append(axes.quiver(X,Y,Z,U,V,W, length=0.05, arrow_length_ratio=0.05, pivot='tail', color=colors[m]))

  return line_collections

def clear_animation(where):
  files = glob.glob('./%s/*' % where)
  for f in files:
    os.remove(f)

# @brief  Takes a time series of quaternions representing the body-frame relative to the inertial frame from the POV of the inertial frame.
#         Generates a series of 3-vectors representing the body axes, for animation.
def generate_body_frames(quats):
  e0 = [0,np.asarray([1,0,0])]
  e1 = [0,np.asarray([0,1,0])]
  e2 = [0,np.asarray([0,0,1])]
  e0_b = []
  e1_b = []
  e2_b = []
  for q in quats:
    e0_b.append(quaternion_rotation(e0, q))
    e1_b.append(quaternion_rotation(e1, q))
    e2_b.append(quaternion_rotation(e2, q))
  return (e0_b, e1_b, e2_b)

# @brief Takes a time series of unit body-axes (quaternion form [scalar r, 3-vector v]) relative to an inertial frame.
#        Decimates this time series and creates a series of still images which can be strung together as an animation of the rotation.
#        Theses images are written to the ./images folder
# @param[in] n_vectors The length of q_e0,1,2
# @param[in] q_e0 The 'x' body axis, of unit norm
# @param[in] q_e1 The 'y' body axis, of unit norm
# @param[in] q_e2 The 'z' body axis, of unit norm
# @param[in] decimator The factor by which to decimate the time series.
def animate(n_vectors, q_e0, q_e1, q_e2, decimator, where="images"):
  # Wipe the images/ directory
  clear_animation(where)

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")

  line_collections = None
  count = 0

  for n in xrange(n_vectors):
    percent = (100.0*n)/n_vectors
    complete = "%0.3f%%" % percent
    sys.stdout.write('\rAnimation ' + complete + " complete.")
    sys.stdout.flush()

    if (n % decimator) != 0:
      continue

    if (line_collections is not None) and (count > 1):
      # The first set of axes drawn stays as a reference in all images.
      # The rest are wiped right after they are drawn/saved to give the illusion of motion.
      for col in line_collections:
        ax.collections.remove(col)

    line_collections = draw_body(ax, [q_e0[n], q_e1[n], q_e2[n]], ["r", "g", "b"])

    plt.savefig('%s/Frame%08i.png' % (where, n))
    plt.draw()

    count += 1

# @brief Takes a time series of unit body-axes (quaternion form [scalar r, 3-vector v]) relative to an inertial frame.
#        Decimates this time series and creates a series of still images which can be strung together as an animation of the rotation.
#        Theses images are written to the ./images folder
# @param[in] n_vectors The length of q_e0,1,2
# @param[in] r_e0 The 'x' body axis, of unit norm, estimated
# @param[in] r_e1 The 'y' body axis, of unit norm, estimated
# @param[in] r_e2 The 'z' body axis, of unit norm, estimated
# @param[in] q_e0 The 'x' body axis, of unit norm, actual
# @param[in] q_e1 The 'y' body axis, of unit norm, actual
# @param[in] q_e2 The 'z' body axis, of unit norm, actual
# @param[in] decimator The factor by which to decimate the time series.
def compare(n_vectors, r_e0, r_e1, r_e2, q_e0, q_e1, q_e2, decimator, where="images"):
  # Wipe the images/ directory
  clear_animation(where)

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  ax.set_xlabel("x")
  ax.set_ylabel("y")
  ax.set_zlabel("z")

  line_collections = None

  for n in xrange(n_vectors):
    percent = (100.0*n)/n_vectors
    complete = "%0.3f%%" % percent
    sys.stdout.write('\rAnimation ' + complete + " complete.")
    sys.stdout.flush()

    if (n % decimator) != 0:
      continue

    if (line_collections is not None):
      # The axes are wiped right after they are drawn to give the illusion of motion.
      for col in line_collections:
        ax.collections.remove(col)

    line_collections = draw_body(ax, [r_e0[n], r_e1[n], r_e2[n], q_e0[n], q_e1[n], q_e2[n]], ["r", "g", "b", "c", "y", "k"])

    plt.savefig('%s/Frame%08i.png' % (where, n))
    plt.draw()
