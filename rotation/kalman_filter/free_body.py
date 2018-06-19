import sys
import numpy as np
from scipy.integrate import odeint
from quaternions import *
from matplotlib import pyplot as plt
from animate import generate_body_frames, animate

# @brief Simulates gyrometer and 3D compass readings during a rigid free-body rotation subject to zero external torque in a time-invariant local magnetic field.
#
# @param[in] inputs, a dictionary with the following fields:
#             - r_i_bp: [axis (unit numpy 3-vector), angle (radians)]
#                 an initial condition. how to rotate the standard inertial frame to the principal body frame
#                 specified from the standard inertial frame!
#             - r_bp_b: [axis (unit numpy 3-vector), angle (radians)]
#                 an initial condition. how to rotate the principal body frame to the actual body frame.
#                 specified from the principal body frame!
#             - J_bp: a numpy 3x3 matrix with x,y,z as indices 0,1,2, respectively. units kg m^2
#                 the diagonal 3x3 moment of inertia matrix seen from the principal body frame
#             - w_bp: a numpy 3-vector [rad/s]
#                 an initial condition. the angular velocity seen from the principal body frame
#             - f_s: a scalar. units Hz
#                 the sampling frequency for simulation
#             - t_f: a scalar. units seconds.
#                 the final simulation time, from zero.
#             - m_i: a numpy 3-vector. unitless.
#                 a unit vector representing the direction of the Earth's local magnetic field when the body frame is coincident with the standard inertial frame.
#                 note we could input a time-average while the device is booting (aka require horizontal calibration)
#                                     a hard-coded lookup of the local declination...
#                 note even if we're wrong in this... our controller will just adjust the reference attitude until we come to rest at our desired position.
#                 it will think it is tilted, but really the mechanics will require it to be horizontal to be at rest.
#
# @return outputs, a dictionary with the following fields:
#             - m_i: a numpy 3-vector. unitless.
#                 a unit vector representing the direction of the Earth's local magnetic field when the body frame is coincident with the standard inertial frame.
#                 intended as an input to the realism module
#             - t_s: a list of sample times (seconds) of length (t_f - 0) * f_s
#             - q_i: a list of quaternions [r, v] where v is a numpy 3-vector
#                    of the same length as t
#                 represents the 'ground truth' rotation of the body frame relative to the standard inertial frame
#                 as computed without sampling noise, bias, etc. slightly inaccurate due to numerical error but it's the best truth we've got by several orders of magnitude (see rotation/demo.py)
#             - w_b: a list of numpy 3-vectors [rad/s] representing ideal samples from a gyroscope coincident with the body frame
#                 intended as an input to the realism module
#             - m_b: a list of numpy 3-vectors [unit norm, unitless] representing ideal samples of the direction of the Earth's magnetic field from a 3D compass coincident with the body frame
#                 intended as an input to the realism module
def simulate(inputs):
  # Get the time step for this simulation
  dt = 1.0 / inputs["f_s"] # seconds

  # Get the rotation matrix from the principal body frame to the body frame, R
  q = axis_angle_to_quaternion(-inputs["r_bp_b"][0], inputs["r_bp_b"][1])
  R = np.zeros([3,3])
  R[:,0] = quaternion_rotation([0, np.asarray([1,0,0])], q)[1]
  R[:,1] = quaternion_rotation([0, np.asarray([0,1,0])], q)[1]
  R[:,2] = quaternion_rotation([0, np.asarray([0,0,1])], q)[1]

  # Get the moment of inertia matrix with respect to the body frame, J_b = R J_bp R^T
  J_b = np.dot(np.dot(R, inputs["J_bp"]), np.transpose(R))
  J_b_inv = np.linalg.inv(J_b)

  print inputs["J_bp"]
  print J_b

  # Use q to compute the initial angular velocity as seen from the body frame
  w_b0 = quaternion_rotation([0, inputs["w_bp"]], q)[1]

  # Get the quaternion which represents the rotation from the standard inertial frame to the initial body frame
  # First, recall that we specified the rotation from the principal body frame to the body frame from the POV of the principal body frame.
  # We need to convert this axis of rotation to the standard inertial frame then combine it with the known rotation from
  # the inertial frame to the principal body frame.
  q_i = axis_angle_to_quaternion(inputs["r_i_bp"][0], inputs["r_i_bp"][1])
  n = quaternion_rotation([0, inputs["r_bp_b"][0]], q_i)[1] # Inertial representation of principal body - to - body rotation axis
  q_b = axis_angle_to_quaternion(n, inputs["r_bp_b"][1])
  q_ib = quaternion_product(p=q_b, q=q_i, normalize=True) # Represents the initial inertial rotation from I to B.
                                                          # The coordinate transoformation would use the inverse.

  # Adaptive integration of the equations of motion in the body frame for free body rotation under zero torque
  def ddt_wb(w, t):
    return np.dot(J_b_inv, np.cross(np.dot(J_b, w), w))

  N_samples = int( (inputs["t_f"]*1.0) / dt )
  time_s = np.linspace(0, inputs["t_f"], N_samples)
  w_b = odeint(ddt_wb, w_b0, t = time_s)

  # Collect outputs so far!
  outputs = {}
  outputs["m_i"] = inputs["m_i"]
  outputs["t_s"] = time_s
  outputs["w_b"] = w_b

  # Compute the time series of:
  # - the quaternion representing rotation of the body frame relative to the standard inertial frame
  # - the magnetic field as seen from the body frame
  outputs["q_i"] = [q_ib]
  outputs["m_b"] = [quaternion_rotation([0, inputs["m_i"]], quaternion_inverse(q_ib))[1]]
  for idx in xrange(len(w_b)-1):
    q_wi = quaternion_rotation([0, w_b[idx]], outputs["q_i"][-1])
    w_i = q_wi[1] # Take the vector part

    outputs["q_i"].append(quaternion_product(w_dt_to_quaternion(w_i, dt), outputs["q_i"][-1], True))
    outputs["m_b"].append(quaternion_rotation([0, inputs["m_i"]], quaternion_inverse(outputs["q_i"][-1]))[1])

  return outputs

def check_principal(axis):
  # For instance, check a single axis 1 Hz rotation where the principal body axis IS the body axis,
  # and make sure time, orientations, ... are correct. This will also help you flesh out the animation code.
  inputs = {}
  inputs["r_i_bp"] = [np.asarray([0,0,1]), 0] # Standard inertial frame IS the principal body frame (initially)
  inputs["r_bp_b"] = [np.asarray([0,0,1]), 0] # Principal body frame IS the actual body frame
  inputs["J_bp"] = (1.0/12) * 0.05 * np.eye(3) # A uniform cubic 50g mass of length 1 m has J = M/12 I where M is the total mass.
  if axis == 'x':
    inputs["w_bp"] = 2*np.pi*np.asarray([1,0,0]) # 1 Hz CCW rotation about the body x-axis is a 1 Hz CCW rotation about the inertial x axis.
  elif axis == 'y':
    inputs["w_bp"] = 2*np.pi*np.asarray([0,1,0]) # 1 Hz CCW rotation about the body y-axis is a 1 Hz CCW rotation about the inertial y axis.
  else:
    inputs["w_bp"] = 2*np.pi*np.asarray([0,0,1]) # 1 Hz CCW rotation about the body z-axis is a 1 Hz CCW rotation about the inertial z axis.
  inputs["f_s"] = 100.0 # Hz
  inputs["t_f"] = 1.0 # seconds
  inputs["m_i"] = np.asarray([0,0,1]) # Normalized magnetic field points straight up when the body is aligned with the inertial frame.

  outputs = simulate(inputs)

  # Check: Animate it. Orientations look right? Rotating about chosen inertial axis CCW (+)?
  # Status: Pass.
  e0, e1, e2 = generate_body_frames(outputs["q_i"])
  N_frames = len(outputs["t_s"])
  animate(N_frames, e0, e1, e2, 5)

def check_misalignment():
  # Do the initial rotations of the principal and body axes relative to the inertial frame look right?
  inputs = {}
  inputs["r_i_bp"] = [np.asarray([1,0,0]), np.pi/2] # Principal body frame is initially rotated 90 degrees about the inertial x axis.
  inputs["r_bp_b"] = [np.asarray([0,1,0]), np.pi/4] # Actual body frame is rotated 45 degrees about the principal y axis
  inputs["J_bp"] = (1.0/12) * 0.05 * np.eye(3) # A uniform cubic 50g mass of length 1 m has J = M/12 I where M is the total mass.
  inputs["w_bp"] = 2*np.pi*np.asarray([0,0,1]) # 1 Hz CCW rotation about the principal body z-axis, initially.
  inputs["f_s"] = 100.0 # Hz
  inputs["t_f"] = 1.0 # seconds
  inputs["m_i"] = np.asarray([0,0,1]) # Normalized magnetic field points straight up when the body is aligned with the inertial frame.

  outputs = simulate(inputs)

  # Check: Animate it. Initial orientation looks right?
  # Status: Pass.
  e0, e1, e2 = generate_body_frames(outputs["q_i"])
  N_frames = len(outputs["t_s"])
  animate(N_frames, e0, e1, e2, 100)

def check_precession():
  # Let's misalign our body and principal body axes. Do we precess?
  inputs = {}
  inputs["r_i_bp"] = [np.asarray([1,0,0]), np.pi/2] # Principal body frame is initially rotated 90 degrees about the inertial x axis.
  inputs["r_bp_b"] = [np.asarray([0,1,0]), np.pi/10] # Actual body frame is rotated 18 degrees about the principal y axis
  inputs["J_bp"] = np.zeros([3,3])
  inputs["J_bp"][0,0] = 1
  inputs["J_bp"][1,1] = 2
  inputs["J_bp"][2,2] = 3
  inputs["w_bp"] = 2*np.pi*np.asarray([0,0,1]) # 1 Hz CCW rotation about the principal body z-axis, initially.
  inputs["f_s"] = 800.0 # Hz
  inputs["t_f"] = 5.0 # seconds
  inputs["m_i"] = np.asarray([0,0,1]) # Normalized magnetic field points straight up when the body is aligned with the inertial frame.

  outputs = simulate(inputs)

  # Check: Animate it. Does it look reasonable?
  # Status: Pass.
  e0, e1, e2 = generate_body_frames(outputs["q_i"])
  N_frames = len(outputs["t_s"])
  animate(N_frames, e0, e1, e2, 80)

if __name__ == "__main__":
  # Run some sanity checks!
  if len(sys.argv) == 3:
    if sys.argv[1] == 'check_principal':
      check_principal(sys.argv[2])
  elif len(sys.argv) == 2:
    if sys.argv[1] == 'check_misalignment':
      check_misalignment()
    elif sys.argv[1] == 'check_precession':
      check_precession()
  else:
    print ""
    print "Usage: python free_body.py check_principal x|y|z"
    print "Usage: python free_body.py check_misalignment"
    print "Usage: python free_body.py check_precession"
    print ""
    sys.exit(0)
