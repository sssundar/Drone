import sys
import numpy as np
from scipy.integrate import odeint
from quaternions import *

# See Quad notes from 7/31/2018-8/13/2018 for derivations.
class Plant(object):

  #
  # @brief      Initializes a horizontal quad oriented with the Earth's magnetic
  #             field, motors halted.
  #               - Accounts for gyroscopic effects from the propellors
  #               - Accounts for IMU-frame vs. body-frame mismatch
  #               - Accounts for motor drive asymmetry
  #
  #               - Assertion (1a): a rigid rectangular symmetric chassis.
  #               - Assertion (1b): a rigid propellor-shaft system fixed in the quad z axis.
  #               - Assertion (2): negligible chassis air resistance.
  #               - Measurement (3a): linear duty-cycle to thrust relation for small DC brushed motors.
  #               - Assertion (3b): quadratic w to thrust relation by a momentum transfer accounting
  #               - Assertion (3c): quadratic w to drag torque relation
  #               - Assertion (3d): very weak drag torque relative to peak motor torque
  #               - Measurement (4): Electrical timescales <<< Mechanical timescales (winding inductance, for instance).
  #
  # @param[in]  self  A Plant object
  # @param[in]  dt    The time-step to simulate, in seconds
  # @param[in]  symmetric Is this meant to be a fully symmetric quadcopter?
  #
  def __init__(self, dt, symmetric=True):
    # Quad Frame Definition
    #
    # Bodies 1,2,3,4 are the four propellor-shaft systems rigidly attached
    # to the frame, tracking +e3 of the chassis.
    #
    #  m1m2p3  4---------1 m1p2m3
    #           -       -
    #           -       -
    #           -   .   -    x = chassis CM, +e3, tip of arrow, out of page
    #           -       -    m/p = -/+; 1/e1/x, 2/e2/y, 3/e3/z (m/p = cw/ccw)
    #           -       -
    #  p1m2m3  3---------2 p1p2p3
    #
    #
    #              +e3 = z (height)
    #              ^
    #              -
    #              -
    #              -----> +e2 = y (width)
    #             -
    #            -
    #           v +e1 = x (length)
    #
    #
    #    J_c = [  z^2 + y^2     -xy         -xz
    #               -xy       z^2 + x^2     -yz
    #               -xz         -yz        x^2 + y^2   ]
    #
    #

    # Time Configuration
    self.dt = dt

    # Chassis Mechanical Configuration
    self.config = {}
    self.config["Propellor Orientation"] = {
      "m1p2m3" : -1,
      "p1p2p3" : 1,
      "p1m2m3" : -1,
      "m1m2p3" : 1 } # CW (-1) or CCW (1) spin direction of each propellor.

    self.config["l_chassis"] = 0.05 # Length of chassis in meters
    self.config["w_chassis"] = 0.05 # Width of chassis in meters
    self.config["h_chassis"] = 0.01 # Height of chassis in meters
    self.config["m_chassis"] = 0.08 # Mass of chassis in kg
    self.config["R_prop"] = { "m1p2m3" : 0.5*np.asarray([-self.config["l_chassis"], self.config["w_chassis"], self.config["h_chassis"]]),
                              "p1p2p3" : 0.5*np.asarray([self.config["l_chassis"], self.config["w_chassis"], self.config["h_chassis"]]),
                              "p1m2m3" : 0.5*np.asarray([self.config["l_chassis"], -self.config["w_chassis"], self.config["h_chassis"]]),
                              "m1m2p3" : 0.5*np.asarray([-self.config["l_chassis"], -self.config["w_chassis"], self.config["h_chassis"]])
                              } # Quad-frame vector (m) to CM of propellor-shaft system.

    self.config["r_shaft"] = 0.0025 # Radius of the propellor shaft in meters
    self.config["m_shaft"] = 0.002 # Mass of the propellor shaft in kg
    self.config["l_blade"] = 0.0254 # Length of propellor blade in meters
    self.config["w_blade"] = 0.005 # Width of propellor blade in meters
    self.config["m_blade"] = 0.00025 # Mass of propellor blade in kg
    self.config["m_prop"] = 2*self.config["m_blade"] + self.config["m_shaft"] # Mass of propellor-shaft system in kg

    self.config["J_prop"] = 0.5*self.config["m_shaft"]*(self.config["r_shaft"]**2) # Principal moment (kg m^2) about e3 axis of propellor-shaft system (symmetric across motors)
    self.config["J_prop"] += 0.1*self.config["m_blade"]*(9*(self.config["l_blade"]**2) + 4*(self.config["w_blade"]**2))
    self.config["J_prop_inverse"] = 1.0/self.config["J_prop"]

    self.config["J_chassis"] = np.eye(3) # Moment of inertia tensor (kg m^2) for chassis + propellor-shaft CM. Diagonal by assumption (1).
    self.config["J_chassis"][0,0] = self.config["h_chassis"]**2 + self.config["w_chassis"]**2
    self.config["J_chassis"][1,1] = self.config["h_chassis"]**2 + self.config["l_chassis"]**2
    self.config["J_chassis"][2,2] = self.config["l_chassis"]**2 + self.config["w_chassis"]**2
    self.config["J_chassis"] *= ((self.config["m_chassis"]/12) + self.config["m_prop"])
    self.config["J_chassis_inverse"] = np.linalg.inv(self.config["J_chassis"])

    # Field Configuration
    # Note:
    #   At hover, there is no acceleration but the IMU will read -G (+9.8 +z) due to the test mass resting on the bottom wall.
    #   This is equivalent to a-G with a = 0.
    #   When accelerating upwards, the IMU will read a-G, for the same reason.
    #   When accelerating downwards, the IMU will read a-G, for the same reason.
    #   When in free-fall, the IMU will read 0, which is equal to a-G if a = G.
    #   So, at all times, the IMU reads the net acceleration minus gravity (-9.8z).
    self.config["G"] = np.asarray([0,0,-9.8]) # m/s^2
    self.config["H"] = np.asarray([0.5,0,np.sqrt(3)/2]) # Normalized, unitless.

    # Motor Drive Configuration. See Drone/motor/quadratic_drag.py.
    # Configured to match measurements: O(300ms) to w steady-state at duty cycles in [0.1,1].
    # Configured to match measurements: ~40g peak thrust
    # Configured for a 12,000 RPM limit; this was guessed and seems reasonable.
    # These configurations ensure T_prop >>> B_motor >> B_drag, which keeps w_ss ~ duty^0.5,
    # which satisfies the measured linear duty-thrust relationship assuming quadratic w-thrust.
    self.config["Max RPM Base"] = 12000.0 # rotations per minute
    if not symmetric:
      self.config["Max RPM"] = {"m1p2m3" : 0.97*self.config["Max RPM Base"],
                                "p1p2p3" : 1.03*self.config["Max RPM Base"],
                                "p1m2m3" : 1.02*self.config["Max RPM Base"],
                                "m1m2p3" : 0.98*self.config["Max RPM Base"],
                                }
    else:
      self.config["Max RPM"] = {"m1p2m3" : 1.00*self.config["Max RPM Base"],
                                "p1p2p3" : 1.00*self.config["Max RPM Base"],
                                "p1m2m3" : 1.00*self.config["Max RPM Base"],
                                "m1m2p3" : 1.00*self.config["Max RPM Base"],
                                }
    self.rpm_to_w = lambda rpm: (2*np.pi*rpm)/60 # rad/s
    self.config["B_drag"] = self.config["J_prop"]/120 # w^2 to drag (Nm) coefficient
    self.config["B_motor"] = 10*self.config["B_drag"] # w to effective EMF drive counter-torque (Nm) coefficient
    self.config["T_prop"] = {} # Maximum drive torque (Nm), by motor (asymmetry possible)
    for k in self.config["Max RPM"].keys():
      self.config["T_prop"][k] = self.config["B_drag"]*(self.rpm_to_w(self.config["Max RPM"][k])**2)
    self.config["B_thrust"] = (0.04*(-self.config["G"]))/(self.rpm_to_w(self.config["Max RPM Base"])**2) # w^2 to thrust (N) coefficient
    self.thrust_force = lambda w: ((self.config["B_thrust"] * (w**2)) * np.asarray([0,0,1]))
    self.drag_torque = lambda w: (self.config["B_drag"] * (w**2))

    # IMU Misalignment Configuration
    # Note:
    #   q_offset represents a transformation from a vector in the quad frame to the IMU frame
    #   so a 1 degree offset in the quad frame would be a -1 degree transformation.
    self.config["q_offset"] = axis_angle_to_quaternion(np.asarray([1,0,0]), -np.pi/180)

    # State Variables
    # We start out immobile and perfectly aligned with the space frame.
    self.state = {}
    self.state["Omega"] = np.asarray([0,0,0]) # rad/s quad body angular velocity
    self.state["w"] = { "m1p2m3" : 0.0,
                        "p1p2p3" : 0.0,
                        "p1m2m3" : 0.0,
                        "m1m2p3" : 0.0
                        } # rad/s propellor angular velocity
    self.state["q"] = [1.0, np.asarray([0,0,0])] # quaternion ([r, v] with v a numpy 3-vector) representing quad-to-space transformation
    self.state["R"] = np.asarray([0,0,0]) # CM of quad, meters
    self.state["ddt_R"] = np.asarray([0,0,0]) # d/dt of CM of quad, meters/s
    self.state["d2dt2_R"] = np.asarray([0,0,0]) # d2/dt2 of CM of quad, meters/s^2

    return

  #
  # @brief      A helper function which returns the space frame acceleration due
  #             to gravity and thrust
  #
  # @param[in]  self  A Plant object
  # @param[in]  w     A dictionary of motor angular velocities (keys: {m1p2m3,
  #                   p1p2p3, p1m2m3, m1m2p3}) in rad/s
  # @param[in]  q     A quaternion of the form [r,v] with representing the
  #                   coordinate transformation from the quad body frame to the
  #                   space frame.
  #
  # @return     A numpy 3-vector representing the net acceleration of the quad
  #             as seen from the space frame.
  #
  def space_frame_acceleration(self, w, q):
    thrust = quaternion_rotation(qv=[0, sum([self.thrust_force(w[k]) for k in w.keys()])], qr=q)[1]
    return self.config["G"] + (thrust/(self.config["m_chassis"] + 4*self.config["m_prop"]))

  #
  # @brief      Simulates the plant dynamics of a quadcopter without external
  #             disturbances.
  #
  # @param[in]  self  A Plant object
  # @param[in]  duty  - a dictionary
  #                   - keys {{m1p2m3, p1p2p3, p1m2m3, m1m2p3} representing
  #                     (m)inus and (p)lus body axes 1,2 and the direction of
  #                     rotation of the motor
  #                   - values representing motor-drive PWM duty-cycles between
  #                     [0,1]
  #
  # @return     computed without noise (excepting numerical), bias, jitter, or
  #             delay:
  #             - Measurements
  #              - n: a numpy 3-vector [rad/s] representing a sample from a 3D
  #                gyroscope on the quad
  #              - m: a numpy 3-vector [unit norm, unitless] representing a
  #                sample of the direction of the Earth's magnetic field from a
  #                3D compass on the quad
  #              - a: a numpy 3-vector in [Newtons] representing a sample of the
  #                acceleration measured by a 3D accelerometer on the quad
  #             - Ground Truth
  #              - q: a quaternion [r, v], where v is a numpy 3-vector,
  #                representing the coordinate transformation from the quad
  #                body-frame to the space frame. rotating a vector from the quad
  #                body frame by q_b yields a space-frame representation. note
  #                that the IMU-frame is not necessarily concident with the quad
  #                body frame.
  #              - r: a numpy 3-vector representing the center of mass of the
  #                quad in the space frame.
  #              - ddt_r: a numpy 3-vector representing the derivative of r in the
  #                space-frame.
  #              - d2dt2_r: a numpy 3-vector representing the 2nd derivative of r in
  #                the space-frame.
  #
  # @note       The simulator operates in the quad frame, then rotates samples
  #             to the IMU frame.
  #
  def evolve(self, duty):
    state_0 = np.asarray([
                  self.state["Omega"][0],          # 0
                  self.state["Omega"][1],          # 1
                  self.state["Omega"][2],          # 2
                  self.state["w"]["m1p2m3"],       # 3
                  self.state["w"]["p1p2p3"],       # 4
                  self.state["w"]["p1m2m3"],       # 5
                  self.state["w"]["m1m2p3"],       # 6
                  self.state["q"][0],              # 7
                  self.state["q"][1][0],           # 8
                  self.state["q"][1][1],           # 9
                  self.state["q"][1][2],           # 10
                  self.state["R"][0],              # 11
                  self.state["R"][1],              # 12
                  self.state["R"][2],              # 13
                  self.state["ddt_R"][0],          # 14
                  self.state["ddt_R"][1],          # 15
                  self.state["ddt_R"][2],          # 16
                  ])

    def ddt_state(state, t):
      omega = np.asarray([state[0], state[1], state[2]])
      w = {
        "m1p2m3" : state[3],
        "p1p2p3" : state[4],
        "p1m2m3" : state[5],
        "m1m2p3" : state[6]
      }
      q = [state[7], np.asarray([state[8], state[9], state[10]])]
      r = np.asarray([state[11], state[12], state[13]])
      ddt_r = np.asarray([state[14], state[15], state[16]])

      ddt_w = {}
      ddt_omega = np.cross(-omega, np.dot(self.config["J_chassis"], omega))
      for k in w.keys():
        internal_torque_k = duty[k] * ((self.config["Propellor Orientation"][k] * self.config["T_prop"][k]) - (self.config["B_motor"] * w[k]))

        ddt_w[k] = internal_torque_k - (self.config["Propellor Orientation"][k] * self.drag_torque(w[k]))
        ddt_w[k] *= self.config["J_prop_inverse"]

        ddt_omega += np.cross(self.config["R_prop"][k], self.thrust_force(w[k]))
        ddt_omega -= (internal_torque_k * np.asarray([0,0,1]))
        ddt_omega -= np.cross(omega, (self.config["J_prop"] * w[k] * np.asarray([0,0,1])))
      ddt_omega = np.dot(self.config["J_chassis_inverse"], ddt_omega)

      ddt_q = quaternion_times_scalar(scalar=0.5, quaternion=quaternion_product(p=q, q=[0, omega], normalize=False))

      # Net Force in Space Frame
      q_n = quaternion_product(p=[1, np.asarray([0,0,0])], q=q, normalize=True)
      d2dt2_r = self.space_frame_acceleration(w, q_n)

      return np.asarray([ ddt_omega[0],
                          ddt_omega[1],
                          ddt_omega[2],
                          ddt_w["m1p2m3"],
                          ddt_w["p1p2p3"],
                          ddt_w["p1m2m3"],
                          ddt_w["m1m2p3"],
                          ddt_q[0],
                          ddt_q[1][0],
                          ddt_q[1][1],
                          ddt_q[1][2],
                          ddt_r[0],
                          ddt_r[1],
                          ddt_r[2],
                          d2dt2_r[0],
                          d2dt2_r[1],
                          d2dt2_r[2]
                          ])

    state_dt = odeint(ddt_state, state_0, t = [0, self.dt])[1]

    self.state["Omega"] = np.asarray([state_dt[0], state_dt[1], state_dt[2]])
    self.state["w"]["m1p2m3"] = state_dt[3]
    self.state["w"]["p1p2p3"] = state_dt[4]
    self.state["w"]["p1m2m3"] = state_dt[5]
    self.state["w"]["m1m2p3"] = state_dt[6]
    self.state["q"] = [state_dt[7], np.asarray([state_dt[8], state_dt[9], state_dt[10]])]
    self.state["q"] = quaternion_product(p=[1, np.asarray([0,0,0])], q=self.state["q"], normalize=True)
    self.state["R"] = np.asarray([state_dt[11], state_dt[12], state_dt[13]])
    self.state["ddt_R"] = np.asarray([state_dt[14], state_dt[15], state_dt[16]])
    self.state["d2dt2_R"] = self.space_frame_acceleration(self.state["w"], self.state["q"])

    # This is already in the quad-frame. We simply rotate to the IMU frame.
    n = self.state["Omega"]
    n = quaternion_rotation(qv=[0, n],qr=self.config["q_offset"])[1]

    # This is specified in the space-frame, so we rotate from space->quad->IMU frame.
    m = self.config["H"]
    m = quaternion_rotation(qv=[0, m], qr=quaternion_inverse(self.state["q"]))[1]
    m = quaternion_rotation(qv=[0, m], qr=self.config["q_offset"])[1]

    # This is calculated in the space-frame, so we rotate from space->quad->IMU frame.
    # Aside: A prediction made here is that the IMU will always see quad-frame z-acceleration (no x,y)
    # because it only measures acceleration relative to free-fall.
    a = self.state["d2dt2_R"] - self.config["G"]
    a = quaternion_rotation(qv=[0, a], qr=quaternion_inverse(self.state["q"]))[1]
    a = quaternion_rotation(qv=[0, a], qr=self.config["q_offset"])[1]

    return (n, m, a, self.state["q"], self.state["R"], self.state["ddt_R"], self.state["d2dt2_R"])

