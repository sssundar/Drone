# This is a demo of a ballistic particle subject to drag forces and gravity.
# Measurement error and time-varying dynamics are simulated, but time jitter is not.
# The goal is to estimate the trajectory of the particle in real-time.

# For starters, I'm using a filter derived from Bayes' rule, including feedback from the model-measurement error.
# This is just for fun, to see where the Kalman Filter comes from. See notes from 6/24/18-6/28/18 for derivations.

import sys
import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt

def Main():


if __name__ == "__main__":
  Main()
