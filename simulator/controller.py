from plant import Plant
from animate import *
from matplotlib import pyplot as plt

def FreeFall():
  dt=0.01
  t_s = np.asarray(range(100))*dt
  quad = Plant(dt=dt)
  duty_cycles = {
    "m1p2m3" : 0.0,
    "p1p2p3" : 0.0,
    "p1m2m3" : 0.0,
    "m1m2p3" : 0.0
    }
  orientations = []
  cm = []
  for k in xrange(100):
    (_, _, _, orientation, center_of_mass) = quad.evolve(duty_cycles)
    orientations.append(orientation)
    cm.append(center_of_mass)
  plt.plot(t_s, cm)
  plt.show()

if __name__ == "__main__":
  FreeFall()
