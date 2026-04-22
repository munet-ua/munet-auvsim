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
ForwardLookingSonar
    Forward-looking Sonar (FLS) for terrain and obstacle detection.

    
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
    def collectData(self, **kwargs)->Any:
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
                    i:int=None,
                    ocean:Ocean=None, 
                    **kwargs)->List[float]:
        """
        Measure ocean current at simulation iteration i.
        

        Parameters
        ----------
        i : int
            Simulation iteration counter.
        ocean : Ocean
            Ocean object with current.speed and current.angle arrays.
        **kwargs
            Unused. Required for AUV sensor interface compatibility.
            

        Returns
        -------
        [speed, direction] : list of float
            where:

            - speed : float -- Current speed in m/s
            - direction : float -- Current direction in radians.

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
        Returns np.inf when ocean.floor is None (no bathymetry modeled),
        signalling an unbounded depth below the vehicle.
        """

        if (ocean is None) or (eta is None):
            log.error("%s requires 'ocean' and 'eta' arguments.",
                      self.__class__.__name__)
            return -1.0

        if (ocean.floor is None):
            return np.inf

        return ocean.floor(eta[0],eta[1])

###############################################################################

class ForwardLookingSonar(Sensor):
    """
    Forward-Looking Sonar (FLS) for terrain and obstacle detection.

    Models the functional behavior of a BlueView ProViewer 450-15 FLS. Detects
    terrain and vehicle obstacles by sampling the environment along a 2D grid of
    beams (horizontal azimuths x elevation angles) radiating forward from the
    vehicle, using body-fixed ray-floor intersection and ray-sphere vehicle
    intersection.

    The sensor produces a 2D range image indexed by [azimuth, elevation] and a
    matching 3D beam-direction grid. Per-beam range is the distance to the
    nearest detected object (floor or vehicle). Each beam direction is a unit
    vector in the Earth frame, pairing one-to-one with the range image so
    consumers can reconstruct full 3D obstacle vectors from range[k, m] *
    beam_dirs[k, m, :].


    Parameters
    ----------
    nBeamsAz : int, default=11
        Number of horizontal azimuthal beams across hFov. Should be odd
        so the center beam aligns with vehicle heading.
    nBeamsEl : int, default=5
        Number of vertical elevation beams across vFov. Should be odd so the
        center beam aligns with tiltAngle.
    maxRange : float, default=135.0
        Maximum detection range in meters (~450 ft per BlueView spec).
    nStepsRange : int, default=27
        Number of sample points along range per beam. 
        Spacing = maxRange / nStepsRange (5 m at default settings).
    tiltAngle : float, default=2.5
        Angle the sensor is tilted in degrees below horizontal, relative to the
        vehicle body frame. Positive tilt angle points the sensor downward.
    hFov : float, default=50.0
        Horizontal field of view in degrees, centered on vehicle heading.
        Beams span [-hFov/2, +hFov/2] relative to heading.
    vFov : float, default=15.0
        Vertical field of view in degrees, centered on tiltAngle.
        Beams span [-vFov/2, +vFov/2] relative to tiltAngle.
    sampleRate : float, default=10.0
        Sensor update rate in Hz. The host vehicle converts this to a
        step interval _sampleInterval using its sampleTime when
        the sensor is installed via addSensor, and re-syncs whenever
        sampleTime changes. On intermediate steps, collectData
        returns cached results.


    Notes
    -----
    **Body-Fixed Mounting:**

    Beam directions are fixed in the vehicle body frame and rotated to the Earth
    frame using the vehicle's full attitude (roll, pitch, yaw) using the Rzyx
    function. This models a sensor mounted on the hull: when the vehicle pitches
    or rolls, the beam fan moves with it.

    **Detectable Objects:**

    The sensor detects any object that reflects acoustic energy. In the
    simulation this includes:

    - Ocean floor: detected with ray-floor intersection using batched
      samplePoints() queries.
    - Other vehicles: detected with ray-sphere intersection using vehicle
      length L as the detection radius.

    The sensor does not distinguish between floor and vehicle returns;
    beam_ranges reports the nearest object of either type per beam. When
    ocean.floor is None, floor detection is skipped and beam_ranges reflects
    vehicle detections only.

    **Ray-Floor Intersection:**

      - Each beam is parameterized as `p(r) = origin + r * d`. 
      - Evenly-spaced ranges `nStepsRange` are sampled and
        `ocean.floor.samplePoints` is queried for the seabed depth at each
        sample's `(x, y)`.
      - A hit is any sample where `floor_z <= z_beam` (seabed at or above the
        beam's `z`, since `z` is positive down).
      - The nearest hit along the beam is the reported `range`.
      - This is a discrete approximation, where range resolution is 
        `maxRange / nStepsRange`.

    **Ray-Sphere Intersection:**

      - Each other vehicle is a sphere at position `C` with radius `L`. 
      - For beam origin `O` and direction `d`, let `w = C - O` and 
        `t = dot(d,w)` (distance along the beam to closest approach). 
      - The perpendicular distance squared is `perp_sq = ||w||^2 - t^2`.
      - A hit exists if and only if `perp_sq < L^2` and `0 < t < maxRange`. 
      - The range to the sphere entry point is 
        `t_hit = t - sqrt(L^2 - perp_sq)`.
      - The minimum `t_hit` across vehicles is then min-reduced with the floor
        range per beam.

    **Vectorized Implementation:**

    All per-beam computations use numpy broadcasting. Floor queries are batched
    into a single samplePoints() call and vehicle detection uses broadcasted
    ray-sphere intersection.

    
    References
    ----------
    [1] Healey, A.J. (2004). Obstacle Avoidance While Bottom Following for
    the REMUS Autonomous Underwater Vehicle. NPS Technical Report.

    [2] Furukawa, T. (2006). Reactive Obstacle Avoidance Using a
    Forward-Looking Sonar for the REMUS Autonomous Underwater Vehicle.
    NPS Master's Thesis.


    See Also
    --------
    Rzyx : Computes 3x3 Euler angle rotation.
    OceanDepthSensor : Downward-looking depth sensor (DVL analogue).
    """
    
    ## Constructor ===========================================================#
    def __init__(self,
                 nBeamsAz:int=11,
                 nBeamsEl:int=5,
                 maxRange:float=135.0,
                 nStepsRange:int=27,
                 tiltAngle:float=2.5,
                 hFov:float=50.0,
                 vFov:float=15.0,
                 sampleRate:float=10.0)->None:

        self.nBeamsAz = nBeamsAz            # Num Beams in Azimuth
        self.nBeamsEl = nBeamsEl            # Num Beams in Elevation
        self.maxRange = maxRange            # Max Beam Range (m)
        self.nStepsRange = nStepsRange      # Num Sample Points along Beam Range
        self.tiltAngle = tiltAngle          # Angle of Sensor Tilt (deg)
        self.hFov = hFov                    # Horizontal FOV (deg)
        self.vFov = vFov                    # Vertical FOV (deg)
        self.sampleRate = sampleRate        # Sensor Sample Rate (Hz)

        # Precompute range steps (near to far, excluding zero)
        self._ranges = np.linspace(maxRange/nStepsRange, maxRange, nStepsRange)

        # Precompute body-frame beam unit directions (fixed to hull)
        # Body frame: x = forward, y = starboard, z = down
        # Positive elevation = below horizontal
        tilt = np.radians(tiltAngle)
        v_fov = np.radians(vFov)
        h_fov = np.radians(hFov)
        elevations = np.linspace(tilt - v_fov/2, tilt + v_fov/2, nBeamsEl) #(M,)
        azimuths = np.linspace(-h_fov/2, h_fov/2, nBeamsAz)                #(N,)

        # _body_beams[k, m, :] = unit vector for azimuth k, elevation m
        cos_el = np.cos(elevations)[np.newaxis, :]              # (1, M)
        sin_el = np.sin(elevations)[np.newaxis, :]              # (1, M)
        cos_az = np.cos(azimuths)[:, np.newaxis]                # (N, 1)
        sin_az = np.sin(azimuths)[:, np.newaxis]                # (N, 1)

        self._body_beams = np.empty((nBeamsAz, nBeamsEl, 3))
        self._body_beams[:, :, 0] = cos_az * cos_el             # forward
        self._body_beams[:, :, 1] = sin_az * cos_el             # starboard
        self._body_beams[:, :, 2] = sin_el                      # down

        # Default safe result (no detections; body-frame beam directions)
        self._emptyResult = [
            np.full((nBeamsAz, nBeamsEl), maxRange),
            self._body_beams.copy(),
        ]

        # Sample rate state
        self._sampleInterval = 1                # default; set by AUV.addSensor
        self._cachedResult = self._emptyResult

    ## Methods ===============================================================#
    def collectData(self,
                    i:int=None,
                    ocean:Ocean=None,
                    vehicles:list=None,
                    eta:NPFltArr=None,
                    **kwargs)->List[NPFltArr]:
        """
        Inspect environment along beam ranges across the 2D beam grid.

        Detects both ocean floor terrain and vehicles. Per-beam range is the
        nearest object of either type. Beam directions are body-fixed and
        rotated to the Earth frame using the vehicle's full attitude (roll,
        pitch, yaw).

        Recomputes every _sampleInterval steps (set by host vehicle from
        sampleRate) and returns cached results on intermediate steps.


        Parameters
        ----------
        i : int, optional
            Simulation iteration counter. Recomputes when
            i % _sampleInterval == 0, otherwise returns cached result.
        ocean : Ocean, optional
            Ocean environment with floor for terrain detection.
        vehicles : list of Vehicle, optional
            All vehicles in the simulation. The sensor excludes the vehicle
            whose eta matches to prevent self-detection.
        eta : ndarray, shape (6,), optional
            Vehicle position/attitude [x, y, z, phi, theta, psi].
            Full attitude is used to rotate body-fixed beams to Earth frame.
        **kwargs
            Unused. Required for AUV sensor interface compatibility.


        Returns
        -------
        [beam_ranges, beam_dirs] : list of ndarray
            where:

            - beam_ranges : ndarray, shape (nBeamsAz, nBeamsEl) --
              range to nearest object per beam; maxRange if no detection.
            - beam_dirs : ndarray, shape (nBeamsAz, nBeamsEl, 3) --
              unit 3D direction vectors per beam in the Earth frame.
              Each beam_dirs[k, m, :] pairs with beam_ranges[k, m].
        """

        # Return cached result if not at a sample interval
        if ((i is not None) and (i % self._sampleInterval != 0)):
            return self._cachedResult

        # Default safe output
        if (ocean is None) or (eta is None):
            return self._emptyResult

        # Beam geometry parameters
        M = self.nBeamsEl
        N = self.nBeamsAz
        maxR = self.maxRange

        # Extract position and attitude
        x, y, z, phi, theta, psi = eta

        # --- Rotate body-fixed beams to Earth frame ------------------------ #
        R = Rzyx(phi, theta, psi)
        # einsum does: earth_beams[k, m, :] = R @ _body_beams[k, m, :]
        earth_beams = np.einsum('ij,kmj->kmi', R, self._body_beams) # (N, M, 3)

        # Export the full 3D beam direction grid. 
        # Each beam_dirs[k, m, :] pairs with beam_ranges[k, m].
        # Rotation preserves unit length so no renormalization is required.
        beam_dirs = earth_beams                                     # (N, M, 3)

        # --- Floor detection ----------------------------------------------- #
        if (ocean.floor is None):
            floor_ranges = np.full((N, M), maxR)
        else:
            # Sample points along each beam at each range step
            # pts[k, m, j] = origin + ranges[j] * earth_beams[k, m]
            ranges = self._ranges[np.newaxis,np.newaxis,:] # (1, 1, nStepsRange)
            eb_x = earth_beams[:, :, 0, None]              # (N, M, 1)
            eb_y = earth_beams[:, :, 1, None]              # (N, M, 1)
            eb_z = earth_beams[:, :, 2, None]              # (N, M, 1)
            x_s = x + ranges * eb_x                        # (N, M, nStepsRange)
            y_s = y + ranges * eb_y                        # (N, M, nStepsRange)
            z_s = z + ranges * eb_z                        # (N, M, nStepsRange)

            # Batch floor query
            # FLS beam fan is irregular shape so can't use floor grid sampling.
            # Vectorize the position arrays with ravel, then reshape after
            floor_z = ocean.floor.samplePoints(x_s.ravel(), y_s.ravel())
            floor_z = floor_z.reshape(N, M, self.nStepsRange)

            # Dectect obstacle: floor at or above beam (z positive downward)
            hits = floor_z <= z_s                          # (N, M, nStepsRange)

            # First hit per beam: argmax on bool gives index of first True
            has_hit = hits.any(axis=2)                                  # (N, M)
            first_idx = np.argmax(hits, axis=2)                         # (N, M)
            floor_ranges = np.where(has_hit,ranges[0,0,first_idx],maxR) # (N, M)

        # --- Vehicle detection --------------------------------------------- #
        beam_ranges = floor_ranges

        if ((vehicles is not None) and (len(vehicles) > 0)):
            others = [v for v in vehicles if v.eta is not eta]

            if (others):
                C = np.array([v.eta[:3] for v in others])         # (nVeh, 3)
                r_det = np.array([v.L for v in others])           # (nVeh,)
                origin = np.array([x, y, z])
                w = C - origin                                    # (nVeh, 3)

                # Project w onto each Earth-frame beam direction
                t = np.einsum('ijk,vk->ijv', earth_beams, w)      # (N, M, nVeh)

                # Perpendicular distance squared
                w_sq = np.sum(w**2, axis=1)[np.newaxis,np.newaxis,:] #(1,1,nVeh)
                perp_sq = w_sq - t**2                             # (N, M, nVeh)

                # Valid hits
                r_det_sq = (r_det ** 2)[np.newaxis, np.newaxis, :]     #(1,1,nV)
                valid = ((perp_sq < r_det_sq) & (t > 0) & (t < maxR))  #(N,M,nV)

                # Range to sphere entry point
                perp_sq_safe = np.clip(perp_sq, 0, None)
                t_hit = t-np.sqrt(np.maximum(r_det_sq-perp_sq_safe,0)) #(N,M,nV)

                veh_ranges = np.where(valid, t_hit, maxR)
                veh_min = veh_ranges.min(axis=2)                  # (N, M)

                # Take nearest hit from vehicles and terrain
                beam_ranges = np.minimum(floor_ranges, veh_min)

        self._cachedResult = [beam_ranges, beam_dirs]

        return self._cachedResult

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
        Start point in END coordinates (m).
    pt2 : list of float, [x, y, z]
        End point in END coordinates (m).

        
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
        - seabed_z: Sensed ocean floor depth (m).
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
        Vehicle with z_max, seabed_z, z_safe attributes.

        - z_max: Maximum operating depth (m).
        - seabed_z: Sensed ocean floor depth (m).
        - z_safe: Safety distance from ocean floor (m).

    z : float
        Intended depth in meters.

         
    Returns
    -------
    z_limited : float
        Depth bounded by min(z_max, seabed_z - z_safe).
    """

    # Input Parameters
    z_max = vehicle.z_max
    seabed_z = vehicle.seabed_z
    z_safe = vehicle.z_safe

    # Enforce Maximum Depth Limit
    max_depth = min(z_max, seabed_z - z_safe)
    z = min(z, max_depth)

    return z

###############################################################################