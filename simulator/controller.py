from plant import Plant
from animate import *
from matplotlib import pyplot as plt

# From playing with symmetry of the duty cycles (left vs right,
# only one on, cw vs ccw), we learn the following:
# - Our device has a tiny moment of inertia. We'll have O(100ms) to react, or
#   O(10) controller time steps at 100Hz.
# - A perfectly symmetric quad falls straight down. Any growth of velocity
#   should be considered real. Numerical error is negligible on up to <10s timescales.
# - The scale of time (motors) and acceleration matches our expectation (O(200ms), O(g)).
# - We may consider the Plant usable.
def FreeFall(visual=False):
  dt=0.01
  t_s = np.asarray(range(200))*dt
  quad = Plant(dt=dt)
  u = {
    "m1p2m3" : 0.2,
    "p1p2p3" : 0.0,
    "p1m2m3" : 0.0,
    "m1m2p3" : 0.0
    }
  gyro = []
  compass = []
  accel = []
  orientations = []
  cm = []
  ddt_cm = []
  d2dt2_cm = []
  for k in xrange(len(t_s)):
    (w, m, a, q, r, dr, ddr) = quad.evolve(u)
    gyro.append(w)
    compass.append(m)
    accel.append(a)
    orientations.append(q)
    cm.append(r)
    ddt_cm.append(dr)
    d2dt2_cm.append(ddr)

  if visual:
    e0, e1, e2 = generate_body_frames(orientations)
    animate(len(t_s), e0, e1, e2, 1)
  else:
    plt.figure()
    plt.subplot(311)
    plt.plot(t_s, cm)
    plt.legend(["x", "y", "z"])
    plt.ylabel("m")
    plt.title("Center of Mass, Ground Truth")
    plt.subplot(312)
    plt.plot(t_s, ddt_cm)
    plt.legend(["x", "y", "z"])
    plt.ylabel("m/s")
    plt.subplot(313)
    plt.plot(t_s, d2dt2_cm)
    plt.legend(["x", "y", "z"])
    plt.ylabel("m/s^2")

    # Assume the IMU-to-quad frame offset is known perfectly, through offline calibration.
    offset = lambda v: [quaternion_rotation(qv=[0,x], qr=quaternion_inverse(quad.config["q_offset"]))[1] for x in v]

    plt.figure()
    plt.subplot(311)
    plt.plot(t_s, offset(gyro))
    plt.legend(["g_x", "g_y", "g_z"])
    plt.ylabel("rad/s")
    plt.title("Measurements")

    plt.subplot(312)
    plt.plot(t_s, offset(compass))
    plt.legend(["m_x", "m_y", "m_z"])
    plt.ylabel("unitless")

    plt.subplot(313)
    plt.plot(t_s, offset(accel))
    plt.legend(["a_x", "a_y", "a_z"])
    plt.ylabel("m/s^2")
    plt.xlabel("s")

    plt.show()

if __name__ == "__main__":
  FreeFall(visual=True)
