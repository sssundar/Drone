from plant import Plant
from animate import *
from matplotlib import pyplot as plt

def FreeFall():
  dt=0.01
  t_s = np.asarray(range(100))*dt
  quad = Plant(dt=dt)
  u = {
    "m1p2m3" : 0.0,
    "p1p2p3" : 0.0,
    "p1m2m3" : 0.0,
    "m1m2p3" : 0.0
    }
  gyro = []
  compass = []
  accel = []
  orientations = []
  cm = []
  for k in xrange(100):
    (w, m, a, q, r) = quad.evolve(u)
    gyro.append(w)
    compass.append(m)
    accel.append(a)
    orientations.append(q)
    cm.append(r)

  plt.subplot(211)
  plt.plot(t_s, cm)
  plt.legend(["x", "y", "z"])
  plt.ylabel("m")
  plt.title("An xy-stationary quad in free fall drops 4.9 meters in one second.\n")

  # Assume the IMU-to-quad frame offset is known perfectly, through offline calibration.
  offset = lambda v: [quaternion_rotation(qv=[0,x], qr=quaternion_inverse(quad.config["q_offset"]))[1] for x in v]

  plt.subplot(212)
  plt.plot(t_s, offset(gyro))
  plt.plot(t_s, offset(compass))
  plt.plot(t_s, offset(accel))
  plt.legend(["g_x", "g_y", "g_z", "m_x", "m_y", "m_z", "a_x", "a_y", "a_z"])
  plt.ylabel("rad/s, unitless, m/s^2")
  plt.xlabel("s")
  plt.title("An IMU cannot observe free-fall.\n")

  plt.show()

if __name__ == "__main__":
  FreeFall()
