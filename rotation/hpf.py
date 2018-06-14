from scipy import signal
from matplotlib import pyplot as plt
import numpy as np

# A normalized IIR filter with the constructed response:
# H(z) = [(1-B)/2] (1-z^-1) / (1+Bz^-M)  to start with 0 <= B < 1 and M > 0
# This works out as y(n) = x(n) - x(n-1) - B*y(n-M)
#           out(n) = [(1-B)/2] y(n)
# Remember that at Fs = 800Hz = 2pi, a mechanical timescale of 100ms turns into 10Hz or 2pi/80
# which is 1/40 of pi (the fastest digital frequency). So... that's 0.025, which is DESTROYED by the DC killer.
# beta = 0.5
# M = 10
# c = (1-beta)/2
# b = c * np.asarray([1, -1])
# a = np.asarray([1] + [0] * (M-1) + [beta])

# What we need to do is place a pole very close to the zero to up that low-frequency gain.
# Wow, that totally works. It screws up the low-frequency phase but... who cares?
# Keep in mind, with a beta of 0.99, the decay timescale will be massive. Think...
# half a second (400 samples at 800Hz) to drop a disturbance to 1% of its original amplitude.
# Let's try it!
# Looks fine (see gyro.py)... but hard to tell about accuracy without ground truth.
beta = 0.99
c = (1+beta)/2
b = c * np.asarray([1, -1])
a = np.asarray([1, -beta])

w, h = signal.freqz(b=b, a=a)
w /= np.pi

fig = plt.figure()

ax1 = fig.add_subplot(111)
h[0] = 1E-16
plt.plot(w, 20*np.log10(abs(h)), 'b')
plt.ylabel("Amplitude [dB]", color='b')
plt.xlabel("Frequency [rad/pi per sample]")

ax2 = ax1.twinx()
angles = np.unwrap(np.angle(h)) * (180.0/np.pi)
plt.plot(w,  angles, 'g')
plt.ylabel("Angle (degrees)", color='g')

plt.grid()
plt.axis('tight')
plt.show()
