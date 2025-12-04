"""
Navigation functions and sensor classes for AUV state estimation.

Implements the Navigation block of GNC design for determining vehicle position,
attitude, velocity, course, and distance traveled. Provides sensor abstractions,
coordinate transformations, state observers, and filtering algorithms.


Classes
-------
Sensor
    Abstract base class for sensor implementations.
OceanCurrentSensor
    Measures ocean current speed and direction.
OceanDepthSensor
    Measures ocean floor depth at vehicle position.

    
Functions
---------
**Coordinate Transformations:**

    - attitudeEuler(vehicle) : Integrate vehicle attitude using Euler angles.
    - Rzyx(phi, theta, psi) : Rotation matrix in SO(3) using zyx convention.
    - Tzyx(phi, theta) : Attitude transformation matrix using zyx convention.

**State Computation:**

    - statePT(vehicle, pt1, pt2) : Path-tangential angle and track errors.
    - stateSpeed(vehicle) : Vehicle speed magnitude.

**Observers and Filters:**

    - headingFilterLOS(vehicle, psi_ref) : LOS heading observer with yaw rate
      estimation.
    - depthFilter(vehicle, pt) : Exponential moving average depth filter.
    - maxDepthLimit(vehicle, z) : Enforce depth safety limits.

    
Notes
-----
- Navigation block inputs: Sensors, vehicle motion
- Navigation block outputs: State vectors to Guidance and Control blocks


References
----------
[1] Fossen, T.I. (2021). Handbook of Marine Craft Hydrodynamics and Motion
Control. 2nd Edition, Wiley. https://www.fossen.biz/wiley

[2] Fossen, T.I. Python Vehicle Simulator. GitHub repository.
https://github.com/cybergalactic/PythonVehicleSimulator

[3] Fossen, T. I. and Perez, T. (2004). Marine Systems Simulator (MSS).
https://github.com/cybergalactic/MSS
"""

from __future__ import annotations
from typing import Any, List, Tuple, TYPE_CHECKING
from numpy.typing import NDArray
from abc import ABC, abstractmethod
if (TYPE_CHECKING):
    from munetauvsim.vehicles import Vehicle
    from munetauvsim.environment import Ocean
import numpy as np
import math
from munetauvsim import gnc
from munetauvsim import logger

#-----------------------------------------------------------------------------#

# Type Aliases
NPFltArr = NDArray[np.float64]

# Globarl Variables
log = logger.addLog('nav')

###############################################################################

class Sensor(ABC):
    """
    Abstract base class for sensor implementations.
    
    Defines interface for sensor objects that collect data from the simulation
    environment. Subclasses must implement collectData() method.
    

    Notes
    -----
    - Sensors as objects supports ability to create a data log or to cache data
      on the sensor itself.
    - Consider writing an __init__ method for class-wide attributes, such as a
      'name' string that provides a default for the AUV.sensor dictionary to use
      as a reference.
    """

    @abstractmethod
    def collectData(self)->Any:
        """
        Collect sensor measurement data.
        
        Returns
        -------
        data : Any
            Sensor-specific measurement data.
        """

###############################################################################

class OceanCurrentSensor(Sensor):
    """
    Sensor for measuring ocean current speed and direction.
    
    Reads current data from Ocean object at specified simulation iteration.
    """

    def collectData(self, 
                    ocean:Ocean=None, 
                    i:int=None,
                    **kwargs)->List[float]:
        """
        Measure ocean current at simulation iteration i.
        

        Parameters
        ----------
        ocean : Ocean
            Ocean object with current.speed and current.angle arrays.
        i : int
            Simulation iteration counter.
        **kwargs
            Unused. Required for AUV sensor interface compatibility.
            

        Returns
        -------
        speed : float
            Current speed in m/s.
        direction : float
            Current direction in radians.
            

        Notes
        -----
        Returns [-1.0, -1.0] on error with log message.
        """
        
        if (ocean is None) or (i is None):
            log.error("%s requires 'ocean' and 'i' arguments.",
                      self.__class__.__name__)
            return [-1.0, -1.0]
        
        return [ocean.current.speed[i], ocean.current.angle[i]]

###############################################################################

class OceanDepthSensor(Sensor):
    """
    Sensor for measuring ocean floor depth at vehicle position.
    
    Queries Ocean.floor() method at vehicle's (x, y) coordinates.
    """

    def collectData(self, 
                    ocean:Ocean=None, 
                    eta:NPFltArr=None,
                    **kwargs)->float:
        """
        Measure ocean depth at vehicle position.
        

        Parameters
        ----------
        ocean : Ocean
            Ocean object with floor(x, y) method.
        eta : ndarray, shape (6,)
            Vehicle position/attitude [x, y, z, phi, theta, psi].
        **kwargs
            Unused. Required for AUV sensor interface compatibility.

               
        Returns
        -------
        depth : float
            Ocean floor depth in meters at position (eta[0], eta[1]).

             
        Notes
        -----
        Returns -1.0 on error with log message.
        """
        
        if (ocean is None) or (eta is None):
            log.error("%s requires 'ocean' and 'eta' arguments.",
                      self.__class__.__name__)
            return -1.0
        
        return ocean.floor(eta[0],eta[1])

###############################################################################

def attitudeEuler(vehicle:Vehicle)->Tuple[NPFltArr,NPFltArr]:
    """
    Integrate the generalized position/Euler angles vector (eta[k+1]), and the
    velocity vector in END reference frame (p_dot).

    
    Parameters
    ----------
    vehicle : Vehicle
        Vehicle with sampleTime, eta, nu attributes.

        - sampleTime: Simulation time step.
        - eta : [x, y, z, phi, theta, psi], vehicle position / attitude vector.
        - nu : [u, v, w, p, q, r], vehicle linear / angular velocity vector in
          BODY frame.
    
          
    Returns
    -------
    eta : ndarray, shape (6,)
        Updated position/attitude [x, y, z, phi, theta, psi].
    p_dot : ndarray, shape (3,)
        Velocity in END frame [x_dot, y_dot, z_dot].

         
    Notes
    -----
    - Uses forward Euler integration.
    - Position integrated from END velocities, attitude from body angular rates
      via transformation matrices.
    - Based on Fossens Python Vehicle Simulator.

    
    References
    ----------
    [1] Fossen, T.I. Python Vehicle Simulator. GitHub repository.
    https://github.com/cybergalactic/PythonVehicleSimulator
    """
   
    # Input Parameters Loaded on Vehicle
    h = vehicle.sampleTime
    eta = np.copy(vehicle.eta)
    nu = vehicle.nu

    p_dot   = np.matmul(Rzyx(eta[3], eta[4], eta[5]), nu[0:3])
    v_dot   = np.matmul(Tzyx(eta[3], eta[4]), nu[3:6])

    # Forward Euler integration
    eta[0:3] = eta[0:3] + h * p_dot
    eta[3:6] = eta[3:6] + h * v_dot

    return eta, p_dot

###############################################################################

def Rzyx(phi:float,
         theta:float,
         psi:float,
         )->NPFltArr:
    """
    Compute the 3x3 Euler angle rotation matrix R in SO(3) using the zyx
    convention.

    
    Parameters
    ----------
    phi : float
        Roll angle in radians.
    theta : float
        Pitch angle in radians.
    psi : float
        Yaw angle in radians.
        

    Returns
    -------
    R : ndarray, shape (3, 3)
        Rotation matrix from BODY to END frame.
    
        
    References
    ----------
    [1] Fossen, T.I. Python Vehicle Simulator. GitHub repository.
    https://github.com/cybergalactic/PythonVehicleSimulator
    """
    
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth  = math.cos(theta)
    sth  = math.sin(theta)
    cpsi = math.cos(psi)
    spsi = math.sin(psi)
    
    R = np.array([
        [ cpsi*cth, -spsi*cphi+cpsi*sth*sphi,  spsi*sphi+cpsi*cphi*sth ],
        [ spsi*cth,  cpsi*cphi+sphi*sth*spsi, -cpsi*sphi+sth*spsi*cphi ],
        [ -sth,      cth*sphi,                 cth*cphi ] ])

    return R

###############################################################################

def Tzyx(phi:float,theta:float)->NPFltArr:
    """
    Compute the 3x3 Euler angle attitude transformation matrix T using the zyx
    convention.

    
    Parameters
    ----------
    phi : float
        Roll angle in radians.
    theta : float
        Pitch angle in radians.
        

    Returns
    -------
    T : ndarray, shape (3, 3)
        Transformation matrix mapping body angular rates to Euler angle rates.
        

    Notes
    -----
    - Singular at theta = +/-90 degrees. Logs error on singularity.

    
    References
    ----------
    [1] Fossen, T.I. Python Vehicle Simulator. GitHub repository.
    https://github.com/cybergalactic/PythonVehicleSimulator
    """
    
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth  = math.cos(theta)
    sth  = math.sin(theta)    

    try: 
        T = np.array([
            [ 1,  sphi*sth/cth,  cphi*sth/cth ],
            [ 0,  cphi,          -sphi],
            [ 0,  sphi/cth,      cphi/cth] ])
        
    except (ZeroDivisionError):  
        log.error("Tzyx is singular for theta = +-90 degrees." )
        
    return T

###############################################################################

def statePT(vehicle:Vehicle,
            pt1:List[float],
            pt2:List[float],
            )->List[float]:
    """
    Compute path-tangential angle and track errors for line segment following.

    Computes the path-tangential (azimuth) angle (pi_h) with respect to the
    East axis, and the along-track (x_e) and cross-track (y_e) errors of a
    vehicle on a path between two points. 

    
    Parameters
    ----------
    vehicle : Vehicle
        Vehicle with eta attribute.
        eta : [x, y, z, phi, theta, psi], vehicle position/attitude vector
    pt1 : list of float, [x, y, z]
        Start point in NED coordinates (m).
    pt2 : list of float, [x, y, z]
        End point in NED coordinates (m).

        
    Returns
    -------
    x_e : float
        Along-track error from start point pt1 (m).
    y_e:
        Cross-track error from path (m).
    pi_h:
        Path-tangential (azimuth) angle w.r.t. East axis (rad).
    
        
    Notes
    -----
    Based on a section in Fossens ALOSpsi.m function.

    
    References
    ----------
    [1] Fossen, T. I. and Perez, T. (2004). Marine Systems Simulator (MSS). 
    https://github.com/cybergalactic/MSS
    """

    # Input Parameters Loaded on Vehicle
    eta = vehicle.eta

    # Guidance Positions
    x, y = eta[0:2]
    x1, y1 = pt1[0:2]
    x2, y2 = pt2[0:2]

    # Compute the Path-Tangential Angle w.r.t. East
    pi_h = math.atan2((y2 - y1), (x2 - x1))

    # Compute the Along-Track and Cross-Track Errors
    x_e =  (x - x1) * math.cos(pi_h) + (y - y1) * math.sin(pi_h)
    y_e = -(x - x1) * math.sin(pi_h) + (y - y1) * math.cos(pi_h)

    return [x_e, y_e, pi_h]

###############################################################################

def stateSpeed(vehicle:Vehicle)->float:
    """
    Compute vehicle speed magnitude in END frame.
    

    Parameters
    ----------
    vehicle : Vehicle
        Vehicle with velocity attribute.
        velocity : [vx, vy, vz], vehicle linear velocity vector in END frame.

          
    Returns
    -------
    speed : float
        Speed magnitude in m/s.
    """

    return np.linalg.norm(vehicle.velocity)

###############################################################################

def headingFilterLOS(vehicle:Vehicle,
                     psi_ref:float,
                     )->None:
    """
    Update the vehicle heading command and yaw rate using LOS observer.

    Propagates heading estimate with feedback from reference angle and
    estimates yaw rate via numerical differentiation.
    

    Parameters
    ----------
    vehicle : Vehicle
        Vehicle with sampleTime, psi_d, r_d, K_f attributes.
        
        - sampletime: Simulation time step (s).
        - psi_d: Desired heading angle (rad).
        - r_d: Desired yaw rate (rad/s).
        - K_f: Observer gain for desired yaw angle (typically 0.1-0-5).

    psi_ref : float
        Reference LOS angle (rad) computed from guidance system.

        
    Notes
    -----
    Based on Fossen's LOSobserver.m function. The observer propagates the
    estimate of the LOS angle according to

        psi_d = psi_d + h * (r_d + K_f * ssa(psi_ref - psi_d))
    
    where the yaw rate estimate (r_d) is computed by numerical differentiation

        r_d = T_f * s / (T_f * s + 1) * psi_d

    where T_f is the differentiator time constant, which can be determined by
    pole-placement and inspection of the closed-loop system

        psi_d / psi_ref = w_n^2 * (T_f*s + 1) / (s^2 + 2*w_n*s + w_n^2)

    If K_f > 0, it follows that T_f = 1 / (K_f + 2*sqrt(K_f) + 1) and that the
    natural frequency is w_n = K_f + sqrt(K_f). Exact discretization of the
    observer gives

        r_d = psi_d - xi
        xi = exp(-h/T_f) * xi + (1 - exp(-h/T_f)) * psi_d

        
    References
    ----------
    [1] Fossen, T. I. and Perez, T. (2004). Marine Systems Simulator (MSS).
    https://github.com/cybergalactic/MSS
    """

    # Input Parameters Loaded on Vehicle
    h = vehicle.sampleTime      # sample time (s)
    psi_d = vehicle.psi_d       # desired heading angle (rad)
    r_d = vehicle.r_d           # desired yaw rate (rad/s)
    K_f = vehicle.K_f           # observer gain for desired yaw angle

    # Differentiator Time Constant
    T_f = 1 / (K_f + 2 * math.sqrt(K_f) + 1)

    # Internal differentiator state
    xi = psi_d - r_d

    # Observer for the LOS angle
    psi_d += h * (r_d + K_f * gnc.ssa(psi_ref - psi_d))
    vehicle.psi_d = psi_d

    # Propagate the differentiator state
    phi = math.exp(-h / T_f)
    xi = (phi * xi) + ((1 - phi) * psi_d)
    vehicle.r_d = psi_d - xi

###############################################################################

def depthFilter(vehicle:Vehicle,pt:List[float])->None:
    """
    Update vehicle depth command using exponential moving average filter.
    
    Smooths desired depth command with an EMA low-pass filter to reduce control
    chattering and enforce safety limits.
    

    Parameters
    ----------
    vehicle : Vehicle
        Vehicle with sampleTime, z_d, wn_d_z, z_max, z_bed, z_safe attributes.

        - sampletime: Simulation time step (s).  
        - z_d: Desired depth command (m)
        - wn_d_z: Desired natural frequency (Hz), depth
        - z_max: Maximum operating depth (m).
        - z_bed: Ocean floor depth (m).
        - z_safe: Safety distance from ocean floor (m).

    pt : list of float, [x, y, z]
        Target waypoint coordinates in END (m).
    
        
    Notes
    -----
    - Based on Fossens Remus100 autopilot function.
    - Filter: 

        z_d = alpha * z_d_prev + (1 - alpha) * z_target

      where:

        alpha = exp(-h * wn_d_z).

    - Calls maxDepthLimit() to enforce maximum depth limit before filtering.
    - Updates vehicle.z_d in place.

    
    References
    ----------
    [1] Fossen, T.I. Python Vehicle Simulator. GitHub repository.
    https://github.com/cybergalactic/PythonVehicleSimulator
    """

    # Input Parameters
    h = vehicle.sampleTime
    z = pt[2]
    z_d = vehicle.z_d
    wn_d_z = vehicle.wn_d_z

    # Enforce Depth Limit
    z = maxDepthLimit(vehicle,z)
        
    # Filter
    alpha = math.exp(-h * wn_d_z)
    vehicle.z_d = (alpha * z_d) + ((1 - alpha) * z)

###############################################################################

def maxDepthLimit(vehicle:Vehicle,z:float)->float:
    """
    Enforce vehicle depth limits based on operating limit and ocean floor depth.

    
    Parameters
    ----------
    vehicle : Vehicle
        Vehicle with z_max, z_bed, z_safe attributes.

        - z_max: Maximum operating depth (m).
        - z_bed: Ocean floor depth (m).
        - z_safe: Safety distance from ocean floor (m).

    z : float
        Intended depth in meters.

         
    Returns
    -------
    z_limited : float
        Depth bounded by min(z_max, z_bed - z_safe).
    """

    # Input Parameters
    z_max = vehicle.z_max
    z_bed = vehicle.z_bed
    z_safe = vehicle.z_safe

    # Enforce Maximum Depth Limit
    max_depth = min(z_max, z_bed-z_safe)
    z = min(z, max_depth)

    return z

###############################################################################