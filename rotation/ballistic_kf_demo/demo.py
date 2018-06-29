# This is a demo of a ballistic particle subject to drag forces and gravity.
# Measurement error and time-varying dynamics are simulated, but time jitter is not.
# The goal is to estimate the trajectory of the particle in real-time.

# For starters, I'm using a filter derived from Bayes' rule, including feedback from the model-measurement error.
# This is just for fun, to see where the Kalman Filter comes from. See notes from 6/24/18-6/28/18 for derivations.

import sys
import numpy as np
from scipy.integrate import odeint
from numpy import dot as dot
from numpy import transpose as transpose
from numpy.linalg import inv as inverse
from numpy import linspace as linspace
from numpy import zeros as zeros
from numpy import eye as eye
from numpy import log as ln
from matplotlib import pyplot as plt

def dynamics(t):
  gamma = 0.7 if (t > 10) else 0.3
  m = zeros([4,4])
  m[0,0] = -gamma
  m[1,1] = -gamma
  m[2,0] = 1
  m[3,1] = 1

  b = zeros([1,4])[0]
  b[1] = -9.8
  return m, b

def Main():
  # Simulation
  def ddt_x(x, t):
    m, b = dynamics(t)
    return dot(m, x) + b
  dt = 0.01
  T_final = 100
  N_samples = int( T_final / dt )
  time_s = linspace(0, T_final, N_samples)
  x_truth_0 = zeros([1,4])[0]
  x_truth_0[0] = 100
  x_truth_0[3] = 200
  x_truth = odeint(ddt_x, x_truth_0, t = time_s)

  # Realism
  sigma_n = eye(4) * (100**2)
  sigma_n_inv = inverse(sigma_n)
  mu_n = zeros([1,4])[0]
  # Add noise to x_truth drawn from the Normal Distribution (sigma_n, mu_n)
  x_measured = []
  for x in x_truth:
    noise = np.random.multivariate_normal(mean=mu_n, cov=sigma_n)
    x_measured.append(x + noise)

  # Estimation
  A, g = dynamics(0) # Model Dynamics are Time-Invariant (aka incorrect)
  g = g*dt
  A = A*dt + eye(4)
  A_inv = inverse(A)

  prior_sigma = eye(4) * (1000**2)
  prior_mu = zeros([1,4])[0]
  x_estimated = []
  for x in x_measured:
    # Step A: Use measurement in posterior estimate. Save the mean for plots.
    prior_sigma_inv = inverse(prior_sigma)
    posterior_sigma_inv = prior_sigma_inv + sigma_n_inv
    posterior_sigma = inverse(posterior_sigma_inv)
    posterior_mu = dot(posterior_sigma, dot(prior_sigma_inv, prior_mu) + dot(sigma_n_inv, x))

    x_estimated.append(posterior_mu)

    # Step B: Predict future state as a naive prior
    prior_sigma_inv = dot(transpose(A_inv), dot(posterior_sigma_inv, A_inv))
    prior_sigma = inverse(prior_sigma_inv)
    prior_mu = dot(A, posterior_mu) + g

    # Step C: Sample future state and feed back model prediction error to adjust the covariance of the prior
    gain = 1 # + (1*(ln() - ln())) # TODO compute ln of gaussian probabilities
    prior_sigma = gain * prior_sigma

  plt.plot(time_s, x_truth, 'k-')
  plt.plot(time_s, x_measured, 'rx')
  plt.plot(time_s, x_estimated, 'b--')
  plt.show()

if __name__ == "__main__":
  Main()
