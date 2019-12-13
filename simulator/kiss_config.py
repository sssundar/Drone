import numpy as np

dt_s = 0.05
n = 100

alpha = 1.
theta_ref = 0.
theta_initial = +np.pi/6

beta = 3.87e+02
J = 0.834
gamma_f = 4.
gamma_g = beta*0.1
phi = 0.08
tau = 1./3
asymmetry = 0.05
beta1 = beta * (1-asymmetry)
beta2 = beta * (1+asymmetry)

CLIP = {"integrator": beta/6, "total": beta/4}