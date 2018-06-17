import sys, glob, os
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

# @brief Takes a time series of unit body-axes (quaternion form [scalar r, 3-vector v]) relative to an inertial frame.
#        Decimates this time series and creates a series of still images which can be strung together as an animation of the rotation.
#        Theses images are written to the ./images folder
# @param[in] n_vectors The length of q_e0,1,2
# @param[in] q_e0 The 'x' body axis, of unit norm
# @param[in] q_e1 The 'y' body axis, of unit norm
# @param[in] q_e2 The 'z' body axis, of unit norm
# @param[in] decimator The factor by which to decimate the time series.
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
