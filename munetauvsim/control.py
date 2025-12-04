"""
Control algorithms and autopilot functions for AUV motion regulation.

Implements the Control block of GNC design for generating actuator commands to
track guidance references while maintaining vehicle stability and rejecting
disturbances. Provides autopilot implementations for heading, depth, and pitch
regulation with anti-windup protection.

Functions
---------
**Heading Control**
    headingPID(vehicle) : PID autopilot for yaw angle regulation via rudder.

**Depth Control**
    depthPID(vehicle) : Cascade PI-PID autopilot for depth regulation via stern
    plane.
    pitchPID(vehicle) : PID autopilot for pitch angle regulation via stern
    plane.

**Propulsion Control**
    constProp(vehicle) : Constant propeller speed command (baseline propulsion).

Notes
-----
**GNC Architecture Context:**

The Control block operates as the third component in the GNC (Guidance,
Navigation, Control) design pattern:

1. **Guidance** : Computes desired trajectories and state change commands
2. **Navigation** : Estimates vehicle state from sensors and kinematics
3. **Control** : Generates actuator commands to track guidance references

**Control Block Scope:**

- **Inputs:** 

  - State change commands from Guidance block (desired heading, depth, pitch)
  - State vectors from Navigation block (position, attitude, velocities)
  
- **Outputs:**

  - Actuator commands: rudder angle (delta_r), stern plane angle (delta_s),
    propeller RPM (n)

**Typical Control Components:**

1. **Controller:**

   Generate control forces/moments (desired control action) from tracking
   errors. Implemented as feedback control laws in this module (e.g., PID,
   cascade control).

2. **Control Allocation:**

   Translate force/moment commands into physical actuator commands (angles, RPM,
   power). Combined with controller in current implementation via direct
   actuator command generation.

**Design Philosophy:**

This module provides a library of control algorithms that interface with vehicle
objects through standardized attributes (gains, states, setpoints). The modular
design allows:

- Drop-in replacement of control laws
- Vehicle-specific tuning via gain parameters
- Cascade and multi-loop architectures
- Extension to additional control axes (roll, speed, etc.)

Main control functions are assigned to vehicles as callable methods, enabling
different control strategies without modifying vehicle dynamics or guidance
systems.

**Current Implementation:**

The present module implements classical PID-based autopilots suitable for:

- Waypoint-based path following (headingPID + depthPID cascade)
- Target tracking and formation control (headingPID + pitchPID direct)
- Constant-speed operations (constProp baseline propulsion)

**Future Extensions:**

The control library architecture supports addition of:

- Advanced control laws (LQR, LQG, H-infinity, adaptive control)
- Multi-input multi-output (MIMO) controllers
- Speed regulation autopilots (speedPID, thrustPID)
- Model predictive control (MPC) for constrained optimization
- Nonlinear and robust control methods


References
----------
[1] Fossen, T.I. (2021). Handbook of Marine Craft Hydrodynamics and Motion
Control. 2nd Edition, Wiley. https://www.fossen.biz/wiley

[2] Fossen, T.I. Python Vehicle Simulator. GitHub repository.
https://github.com/cybergalactic/PythonVehicleSimulator

[3] Fossen, T. I. and Perez, T. (2004). Marine Systems Simulator (MSS).
https://github.com/cybergalactic/MSS

[4] Astrom, K.J. and Murray, R.M. (2008). Feedback Systems: An Introduction for
Scientists and Engineers. Princeton University Press.
http://www.cds.caltech.edu/~murray/amwiki
"""

from __future__ import annotations
from typing import TYPE_CHECKING
if (TYPE_CHECKING):
    from munetauvsim.vehicles import Vehicle
import math
from munetauvsim import logger
from munetauvsim import gnc

#-----------------------------------------------------------------------------#

# Global variable
log = logger.addLog('ctrl')

###############################################################################

def headingPID(vehicle:Vehicle)->float:
    """
    PID controller for heading (yaw) angle regulation via rudder commands.
    
    Computes rudder deflection angle to track desired heading setpoint while
    rejecting disturbances. Uses smallest signed angle error to ensure shortest
    rotation path. Includes integral action for zero steady-state error and
    anti-windup to prevent integrator saturation.
    
    Parameters
    ----------
    vehicle : Vehicle
        Vehicle object with control parameters and state. Must have attributes:
        
        **Gains:**

        - Kp_psi : float
            Proportional gain for yaw error.
        - Ki_psi : float
            Integral gain for yaw error accumulation.
        - Kd_psi : float
            Derivative gain on yaw rate (damping).
        
        **State Variables:**

        - eta : ndarray, shape (6,)
            Position/attitude [x, y, z, phi, theta, psi]. psi = eta[5].
        - nu : ndarray, shape (6,)
            Body-frame velocities [u, v, w, p, q, r]. r = nu[5] (yaw rate).
        - psi_d : float
            Desired heading angle in radians (setpoint).
        - psi_int : float
            Integral term state (accumulated error). Updated by this function.
        
        **Parameters:**

        - sampleTime : float
            Euler integration time step in seconds.
        - deltaMax_r : float
            Maximum rudder deflection in radians.
            
    Returns
    -------
    delta_r : float
        Rudder angle command in radians, saturated to [-deltaMax_r, +deltaMax_r].
        Positive rudder deflection produces positive yaw moment (turn right).
    
    Notes
    -----
    **Side Effects**

    Updates vehicle.psi_int with new integral state via anti-windup logic.

    **Control Law**

    - **PID Equation:**
    
        delta_r = -K_p * ssa(psi - psi_d) - K_i * I - K_d * r
    
      where:

        - psi: Current yaw angle (eta[5])
        - psi_d: Desired yaw angle (psi_d)
        - r: Yaw rate (nu[5])
        - I: Integral term (psi_int)
        - ssa(): Smallest signed angle function
    
    - **Integral Update:**
    
      Without saturation:
    
        I_(k+1) = I_k + err_psi * h
    
      With saturation (anti-windup):
    
        I_(k+1) = I_k + (delta_r_sat - delta_r) / K_i
    
      where:

        - h: sampleTime.
        - err_psi: difference in yaw actual from yaw setpoint
        - delat_r_sat: delta_r clamped to saturation limit
    
    - **Error Calculation:**
    
      Uses smallest signed angle to wrap heading error to [-pi, pi]:
    
        err_psi = ssa(psi - psi_d)
    
      Ensures controller takes shortest rotation path:

      - Current: 350 deg (6.11 rad), Desired: 10 deg (0.17 rad)
      - Naive error: 350 deg - 10 deg = 340 deg (turn left 340 deg)
      - SSA error: ssa(340 deg) = -20 deg (turn right 20 deg) <- Preferred

    **Sign Convention:**
    
    Gains typically negative because:

    - Positive heading error -> need negative rudder to correct
    - Standard marine convention:
      positive rudder -> positive yaw moment -> turn right
    - Negative gains provide negative feedback for stability
    
    **Anti-Windup Mechanism:**
    
    When rudder saturates (delta_r =/= delta_r):

    1. Compute difference: d = delta_r_sat - delta_r
    2. Back-calculate integral: psi_int += d / Ki_psi
    3. Prevents accumulating integral effects during saturation
    
    Without anti-windup:

    - Integral continues growing while saturated
    - Causes large overshoot when error reverses
    - Recovery time increases significantly
    
    **Derivative Term:**
    
    Uses measured yaw rate r (not error derivative) because:

    - Cleaner signal (gyro measurement vs. numerical derivative)
    - Avoids noise amplification from differentiating error
    - Provides damping proportional to rotation speed
    
    **Zero Ki Handling:**
    
    If Ki_psi = 0 (pure PD control):

    - Anti-windup condition never triggers
    - Integral updated normally (but with zero weight)
    - No steady-state error elimination (acceptable for some applications)
    
    **Tuning Impact:**
    
    Increasing absoluate value of Kp_psi:

    - Faster response to heading errors
    - Risk: Oscillation if too high
    
    Increasing absoluate value of Ki_psi:

    - Better disturbance rejection
    - Eliminates steady-state heading bias
    - Risk: Overshoot, slower settling
    
    Increasing absoluate value of Kd_psi:

    - More damping, reduces overshoot
    - Faster settling time
    - Risk: Noise sensitivity
    

    See Also
    --------
    gnc.ssa : Smallest signed angle wrapping
    gnc.saturation : Control output limiting
    navigation.headingFilterLOS : Heading observer with LOS guidance
    guidance.ALOSlaw : Provides desired heading psi_d
    

    References
    ----------
    [1] Fossen, T. I., "An Adaptive Line-of-Sight (ALOS) Guidance Law for Path
    Following of Aircraft and Marine Craft," in IEEE Transactions on Control
    Systems Technology, 31(6), 2887-2894, Nov. 2023, doi:
    10.1109/TCST.2023.3259819.
    
        
    Examples
    --------
    ### Basic usage:
    
    >>> import munetauvsim.vehicles as veh
    >>> auv = veh.Remus100s()
    >>> auv.psi_d = np.pi / 4  # Desired heading: 45 deg
    >>> auv.eta[5] = 0.0       # Current heading: 0 deg
    >>> auv.nu[5] = 0.0        # Yaw rate: 0 rad/s
    >>> auv.psi_int = 0.0      # Reset integral
    >>> 
    >>> import munetauvsim.control as ctrl
    >>> delta_r = ctrl.headingPID(auv)
    >>> print(f"Rudder command: {np.degrees(delta_r):.2f} deg")
    Rudder command: -7.85 deg  # Proportional response to 45 deg error
    
    ### Simulation loop:
    
    >>> import munetauvsim.guidance as guid
    >>> for i in range(1000):
    ...     # Update desired heading from guidance
    ...     auv.psi_d = guid.ALOSlaw(auv, pt1, pt2)
    ...     
    ...     # Compute control
    ...     delta_r = ctrl.headingPID(auv)
    ...     
    ...     # Apply to dynamics
    ...     u_control = np.array([delta_r, 0, 1200])
    ...     auv.nu, _ = auv.dynamics(u_control)
    ...     
    ...     # Update position
    ...     auv.eta, _ = auv.Attitude(auv)
    
    ### Verify anti-windup:
    
    >>> auv = veh.Remus100s()
    >>> auv.psi_d = np.pi  # 180 deg turn
    >>> auv.eta[5] = 0.0
    >>> auv.deltaMax_r = np.radians(30)  # +/- 30 deg limit
    >>> 
    >>> for i in range(100):
    ...     delta_r = ctrl.headingPID(auv)
    ...     if abs(delta_r) >= auv.deltaMax_r:
    ...         print(f"Saturated at iteration {i}")
    ...         # Verify psi_int doesn't grow unbounded
    ...         print(f"Integral state: {auv.psi_int:.3f}")
    """

    #Input Parameters Loaded on Vehicle
    h = vehicle.sampleTime
    Kp_psi = vehicle.Kp_psi
    Ki_psi = vehicle.Ki_psi
    Kd_psi = vehicle.Kd_psi
    psi = vehicle.eta[5]                # yaw angle
    psi_d = vehicle.psi_d               # desired yaw angle
    psi_int_p = vehicle.psi_int         # previous integral term
    r = vehicle.nu[5]                   # yaw rate
    
    # PID
    err_psi = gnc.ssa(psi - psi_d)      # error
    delta_r_raw = (Kp_psi*err_psi) + (Ki_psi*psi_int_p) + (Kd_psi*r)

    # Clamp to saturation limits
    delta_r = gnc.saturation(delta_r_raw, vehicle.deltaMax_r)

    # Anti-integrator wind-up
    if ((delta_r != delta_r_raw) and (Ki_psi != 0)):
        vehicle.psi_int = psi_int_p + ((delta_r - delta_r_raw)/Ki_psi)
    else:
        vehicle.psi_int = psi_int_p + (err_psi * h)

    return delta_r

###############################################################################

def depthPID(vehicle:Vehicle)->float:
    """
    Cascade PI-PID controller for depth regulation via stern plane commands.
    
    Two-loop control architecture with outer depth PI controller generating
    desired pitch angle, and inner pitch PID controller computing stern plane
    deflection. Cascade structure provides improved disturbance rejection and
    prevents depth overshoot from aggressive pitch commands.
    

    Parameters
    ----------
    vehicle : Vehicle
        Vehicle object with control parameters and state. Must have attributes:
        
        **Outer Loop Gains (Depth PI):**

        - Kp_z : float
            Proportional gain for depth error.
        - Ki_z : float
            Integral gain for depth error.
        
        **Inner Loop Gains (Pitch PID):**

        - Kp_theta, Ki_theta, Kd_theta : float
            Gains for pitch control (see pitchPID documentation).
        
        **State Variables:**

        - eta : ndarray, shape (6,)
            Position/attitude [x, y, z, ...]. z = eta[2] (depth).
        - z_d : float
            Desired depth in meters (setpoint).
        - z_int : float
            Depth integral term state. Updated by this function.
        - theta_d : float
            Desired pitch angle (intermediate setpoint). Set by this function.
        - theta_int : float
            Pitch integral term state. Updated by pitchPID.
        
        **Parameters:**

        - sampleTime : float
            Integration time step.
        - deltaMax_s : float
            Maximum stern plane deflection in radians.
            

    Returns
    -------
    delta_s : float
        Stern plane angle command in radians, saturated to [-deltaMax_s, +deltaMax_s]. 
        Positive stern plane produces negative pitch moment (nose down).
    
        
    Notes
    -----
    **Side Effects**

    - Updates vehicle.z_int with new depth integral state
    - Sets vehicle.theta_d to desired pitch angle for inner loop
    - Updates vehicle.theta_int via pitchPID call
    
    **Control Architecture**

    - **Cascade Structure:**
    
        z_d, z -> [Depth PI] -> theta_d
        theta_d, theta, q  -> [Pitch PID] -> delta_s
    
    - **Outer Loop (Depth PI):**
    
        theta_d = Kp_z * (z - z_d) + Ki_z * integral[(z - z_d) dt]
    
      Generates desired pitch angle from depth error.
      Saturated to +/- deltaMax_s to prevent excessive pitch commands.
    
    - **Inner Loop (Pitch PID):**
    
        delta_s = Kp_theta * (theta - theta_d) + 
                  Ki_theta * integral[(theta - theta_d) dt] + Kd_theta * q
    
      Generates stern plane command from pitch error (see pitchPID for details).
    
    - **Anti-Windup (Outer Loop):**
    
      When theta_d saturates:
    
        I_z = I_z + (theta_d_sat - theta_d) / Ki_z
    
      Prevents depth integral accumulation during pitch saturation.

    **Why Cascade Control:**
    
    Advantages over single-loop depth control:

    1. **Improved Dynamics:** Inner loop responds faster than single depth loop
    2. **Disturbance Rejection:** Pitch disturbances handled by inner loop
    3. **Overshoot Prevention:** Pitch saturation limits depth rate of change
    4. **Decoupling:** Separates slow depth response from fast pitch response
    
    **Design Rationale:**
    
    Cascade allows:

    - Slow outer loop for smooth depth tracking
    - Fast inner loop for tight pitch regulation
    - Better overall performance than single loop
    
    **Gain Relationships:**
    
    Typical cascade tuning puts the inner loop bandwidth approximately 5-10x
    faster than outer loop to ensure loops don't fight each other.
    
    **Saturation Interaction:**
    
    Two saturation points:

    1. theta_d saturated to +/- deltaMax_s
    2. delta_s saturated to +/- deltaMax_s (in pitchPID)
    
    Double anti-windup:

    - Outer loop: Prevents z_int growth when theta_d saturates
    - Inner loop: Prevents theta_int growth when delta_s saturates
    
    **Positive Depth Convention:**
    
    END frame: Positive z is down (depth increases with z).
    Error calculation: err_z = z - z_d

    - If deeper than desired (z > z_d): Positive error -> pitch up
    - If shallower than desired (z < z_d): Negative error -> pitch down
    
    **Integral Action:**
    
    Outer loop integral eliminates steady-state depth error from:

    - Buoyancy errors
    - Trim angle offsets
    - Constant vertical currents
    
    Inner loop integral (in pitchPID) eliminates pitch bias.
    
    **Tuning Guidelines:**
    
    1. Tune inner loop (pitch) first with theta_d fixed
    2. Then tune outer loop (depth) with inner loop active
    3. Outer loop gains much smaller than inner loop
    4. Ki_z very small (0.001-0.01) to avoid depth overshoot
    

    See Also
    --------
    pitchPID : Inner loop pitch controller
    gnc.saturation : Control output limiting
    navigation.depthFilter : Depth setpoint filter
    guidance.pathFollow : Provides desired depth z_d
    

    References
    ----------
    [1] Fossen, T.I. (2021). Handbook of Marine Craft Hydrodynamics and Motion
    Control. 2nd Edition, Wiley. https://www.fossen.biz/wiley
    
        
    Examples
    --------
    ### Basic depth change:
    
    >>> import munetauvsim.vehicles as veh
    >>> auv = veh.Remus100s()
    >>> auv.z_d = 25.0     # Desired depth: 25m
    >>> auv.eta[2] = 15.0  # Current depth: 15m
    >>> auv.z_int = 0.0    # Reset integral
    >>> 
    >>> import munetauvsim.control as ctrl
    >>> delta_s = ctrl.depthPID(auv)
    >>> print(f"Stern plane: {np.degrees(delta_s):.2f} deg")
    >>> print(f"Intermediate pitch cmd: {np.degrees(auv.theta_d):.2f} deg")
    Stern plane: -2.85 deg  # Nose down to increase depth
    Intermediate pitch cmd: -1.00 deg
    
    ### Simulation loop:
    
    >>> for i in range(2000):
    ...     # Update desired depth from guidance
    ...     auv.z_d = waypoint[2]
    ...     
    ...     # Compute control
    ...     delta_s = ctrl.depthPID(auv)
    ...     
    ...     # Apply to dynamics
    ...     u_control = np.array([0, delta_s, 1200])
    ...     auv.nu, _ = auv.dynamics(u_control)
    ...     
    ...     # Update position
    ...     auv.eta, _ = auv.Attitude(auv)
    """

    # Input Parameters Loaded on Vehicle
    h = vehicle.sampleTime
    Kp_z = vehicle.Kp_z
    Ki_z = vehicle.Ki_z
    z = vehicle.eta[2]                                  # depth
    z_d = vehicle.z_d                                   # desired depth
    z_int_p = vehicle.z_int                             # previous integral term

    # PID
    err_z = z - z_d                                     # error
    theta_d_raw = (Kp_z * err_z) + (Ki_z * z_int_p)

    # Clamp to saturation limit
    theta_d = gnc.saturation(theta_d_raw, vehicle.deltaMax_s)

    # Anti-integrator wind-up
    if ((theta_d != theta_d_raw) and (Ki_z != 0)):
        vehicle.z_int = z_int_p + ((theta_d - theta_d_raw)/Ki_z)
    else:
        vehicle.z_int = z_int_p + (err_z * h)
    
    # Depth Inner Loop PID
    vehicle.theta_d = theta_d
    delta_s = pitchPID(vehicle)
    
    return delta_s

###############################################################################

def pitchPID(vehicle:Vehicle)->float:
    """
    PID controller for pitch angle regulation via stern plane commands.
    
    Computes stern plane deflection to track desired pitch angle setpoint. Used
    as inner loop in cascade depth control or standalone for direct pitch
    control in target tracking scenarios. Includes integral action and
    anti-windup.
    

    Parameters
    ----------
    vehicle : Vehicle
        Vehicle object with control parameters and state. Must have attributes:
        
        **Gains:**

        - Kp_theta : float
            Proportional gain for pitch error.
        - Ki_theta : float
            Integral gain for pitch error.
        - Kd_theta : float
            Derivative gain on pitch rate.
        
        **State Variables:**

        - eta : ndarray, shape (6,)
            Position/attitude [..., phi, theta, psi]. theta = eta[4].
        - nu : ndarray, shape (6,)
            Body velocities [..., p, q, r]. q = nu[4] (pitch rate).
        - theta_d : float
            Desired pitch angle in radians (setpoint).
        - theta_int : float
            Integral term state. Updated by this function.
        
        **Parameters:**

        - sampleTime : float
            Integration time step.
        - deltaMax_s : float
            Maximum stern plane deflection in radians.
            

    Returns
    -------
    delta_s : float
        Stern plane angle command in radians, saturated to [-deltaMax_s, +deltaMax_s]. 
        Positive deflection produces negative pitch moment (nose down).
    
        
    Notes
    -----
    **Side Effects**

    Updates vehicle.theta_int with new integral state via anti-windup logic.
    
    **Control Law**

    - **PID Equation:**
    
        delta_s = Kp_theta * ssa(theta - theta_d) + Ki_theta * I + Kd_theta * q
    
      where:

        - theta: Current pitch angle (eta[4])
        - theta_d: Desired pitch angle (theta_d)
        - q: Pitch rate (nu[4])
        - I: Integral term (theta_int)
        - ssa(): Smallest signed angle
    
    - **Integral Update:**
    
      Without saturation:
    
        I_(k+1) = I_k + err_theta * h
    
      With saturation (anti-windup):
    
        I_(k+1) = I_k + (delta_s_sat delta_s) / Ki_theta
    
    - **Error Wrapping:**
    
      Uses smallest signed angle for pitch error to handle wraparound:
    
        err_theta = ssa(theta - theta_d)
    
      Though pitch typically +/- 30 deg, ssa ensures robustness for extreme
      maneuvers.

    **Usage Contexts:**
    
    1. **Cascade Depth Control (Common):**

       Called by depthPID as inner loop. theta_d comes from outer depth
       controller.
       
    2. **Direct Pitch Control (Target Tracking):**

       Called directly for APF velocity guidance. theta_d computed from desired
       vertical velocity.
    
    **Sign Convention:**
    
    Stern plane and pitch angle relationship:

    - Positive delta_s (stern plane down): Nose down moment -> pitch decreases
    - Negative delta_s (stern plane up): Nose up moment -> pitch increases
    
    Gains negative to provide correct sign:

    - Pitch too high (theta > theta_d): 
      Positive error -> need negative delta_s (nose up)
    - Negative Kp produces negative delta_s from positive error
    
    **Anti-Windup Importance:**
    
    Stern plane saturates frequently during:

    - Aggressive depth changes
    - Steep dive/climb maneuvers
    - Disturbance rejection
    
    Without anti-windup:

    - Integral grows during saturation
    - Large overshoot when error reverses
    - Oscillatory depth response in cascade
    
    **Derivative Term:**
    
    Uses measured pitch rate q (not error derivative):

    - Gyro provides clean q measurement
    - Avoids numerical differentiation noise
    - Provides proportional damping
    
    **Tuning Impact:**
    
    Increasing absoluate value of Kp_theta:

    - Faster pitch response
    - Risk: Oscillation, actuator wear
    
    Increasing absoluate value of Ki_theta:

    - Eliminates pitch bias from trim/buoyancy
    - Risk: Overshoot in cascade depth control
    
    Increasing absoluate value of Kd_theta:
    
    - More damping, smoother response
    - Reduces control effort
    - Risk: Noise amplification
    

    See Also
    --------
    depthPID : Outer loop calling this controller
    gnc.ssa : Angle wrapping function
    gnc.saturation : Control output limiting
    guidance.targetTrack : Direct pitch control usage
    

    References
    ----------
    [1] Fossen, T.I. (2021). Handbook of Marine Craft Hydrodynamics and Motion
    Control. 2nd Edition, Wiley. https://www.fossen.biz/wiley
    
        
    Examples
    --------
    ### Direct pitch control:
    
    >>> import munetauvsim.vehicles as veh
    >>> auv = veh.Remus100s()
    >>> auv.theta_d = np.radians(-5)  # 5 deg nose down
    >>> auv.eta[4] = 0.0              # Level
    >>> auv.nu[4] = 0.0               # No pitch rate
    >>> auv.theta_int = 0.0           # Reset integral
    >>> 
    >>> import munetauvsim.control as ctrl
    >>> delta_s = ctrl.pitchPID(auv)
    >>> print(f"Stern plane: {np.degrees(delta_s):.2f} deg")
    Stern plane: 2.5 deg  # Positive deflection for nose down
    
    ### Cascade usage (called by depthPID):
    
    >>> # Outer loop sets theta_d
    >>> auv.theta_d = np.radians(-3)
    >>> # Inner loop tracks pitch
    >>> delta_s = ctrl.pitchPID(auv)
    >>> # Used in depthPID return value
    """

    # Input Parameters Loaded on Vehicle
    h = vehicle.sampleTime
    Kp_th = vehicle.Kp_theta
    Ki_th = vehicle.Ki_theta
    Kd_th = vehicle.Kd_theta
    theta = vehicle.eta[4]                          # pitch angle
    theta_d = vehicle.theta_d                       # desired pitch angle
    theta_int_p = vehicle.theta_int                 # previous integral term
    q = vehicle.nu[4]                               # pitch rate

    # PID
    err_th = gnc.ssa(theta - theta_d)               # error
    delta_s_raw = (Kp_th*err_th) + (Ki_th*theta_int_p) + (Kd_th*q)
    
    # Clamp to saturation limits
    delta_s = gnc.saturation(delta_s_raw, vehicle.deltaMax_s)

    # Anti-integrator wind-up
    if ((delta_s != delta_s_raw) and (Ki_th != 0)): 
        vehicle.theta_int = theta_int_p + ((delta_s - delta_s_raw)/Ki_th)
    else:
        vehicle.theta_int = theta_int_p + (err_th * h)

    return delta_s

###############################################################################

def constProp(vehicle:Vehicle)->float:
    """
    Constant propeller speed command (no active speed control).
    
    Simple propeller allocation that returns fixed RPM setpoint. Provides baseline
    propulsion for path following and tracking scenarios where speed regulation is
    not critical. Vehicle maintains approximate constant speed through propeller
    thrust characteristics.
    
    Parameters
    ----------
    vehicle : Vehicle
        Vehicle object with propeller setpoint. Must have attribute:
        
        - n_setpt : float
            Propeller speed setpoint in RPM (revolutions per minute).
            
    Returns
    -------
    n : float
        Propeller RPM command (same as vehicle.n_setpt).
        No saturation or transformation applied.
        
    Notes
    -----
    **Design Philosophy:**
    
    AUV speed control often unnecessary because:

    - Path following cares about trajectory, not speed
    - Propeller thrust roughly balances drag at equilibrium
    - Speed variations minor for typical maneuvers
    - Simplifies control architecture
    
    **When Speed Control Needed:**
    
    Active speed control beneficial for:

    - Time-critical missions (rendezvous, docking)
    - Formation keeping (multi-vehicle coordination)
    - Energy optimization (variable speed profiles)
    - Strong current compensation
    
    Future extension: Implement speedPID() for thrust regulation.
    
    **Load Assignment:**
    
    n_setpt typically assigned in vehicle initialization:
    
    >>> auv = Remus100s()
    >>> auv.loadConstantProp(n_setpt=1200)  # Sets vehicle.n_setpt
    
    Or modified dynamically during mission:
    
    >>> if mission_phase == 'transit':
    ...     auv.n_setpt = 1400  # Fast
    >>> elif mission_phase == 'survey':
    ...     auv.n_setpt = 1000  # Slow, stable
    
    **Propeller Dynamics:**
    
    Even with constant command, actual propeller speed varies due to:

    - Actuator dynamics (1st order lag with time constant ~1-2s)
    - Load variations (drag changes with attitude, speed)
    - Thrust allocation (propeller shares power with control surfaces)
    
    These effects modeled in vehicle.dynamics().
    
    **Integration in Control Loop:**
    
    Typical usage as PropCmd method:
    
    >>> auv.PropCmd = constProp  # Assign function pointer
    >>> # Later in guidance/control:
    >>> n = auv.PropCmd(auv)  # Call assigned function
    
    Or direct call:
    
    >>> n = constProp(auv)
    

    See Also
    --------
    vehicles.Vehicle.loadConstantProp : Assigns n_setpt and PropCmd
    vehicles.Vehicle.dynamics : Models propeller dynamics
    guidance.pathFollow : Uses constProp for propulsion
    

    Examples
    --------
    ### Basic usage:
    
    >>> import munetauvsim.vehicles as veh
    >>> import munetauvsim.control as ctrl
    >>> auv = veh.Remus100s()
    >>> auv.n_setpt = 1200  # Set cruise RPM
    >>> n_cmd = ctrl.constProp(auv)
    >>> print(f"Propeller command: {n_cmd} RPM")
    Propeller command: 1200 RPM
    
    ### Load during path following:
    
    >>> auv.loadPathFollowing()
    >>> auv.loadConstantProp(n_setpt=1300)
    >>> # Now PropCmd assigned to constProp
    >>> 
    >>> for i in range(N):
    ...     # Guidance and control
    ...     delta_r = control.headingPID(auv)
    ...     delta_s = control.depthPID(auv)
    ...     n = auv.PropCmd(auv)  # Calls constProp
    ...     
    ...     # Package control vector
    ...     u_control = np.array([delta_r, delta_s, n])
    
    ### Speed profile mission:
    
    >>> auv.loadConstantProp(n_setpt=1000)  # Initial slow speed
    >>> 
    >>> for i in range(N):
    ...     # Change speed based on mission phase
    ...     if auv.eta[0] > 500:  # Past 500m East
    ...         auv.n_setpt = 1500  # Speed up
    ...     
    ...     n = constProp(auv)
    ...     # ... rest of control loop ...
    """
    
    return vehicle.n_setpt

###############################################################################