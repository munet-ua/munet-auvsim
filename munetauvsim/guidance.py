"""
Guidance algorithms and waypoint management for AUV path planning.

Implements the Guidance block of GNC design for computing desired trajectories
(position, velocity, acceleration) and state change commands from current state.


Classes
-------
Position
    Data structure for 3D coordinates.
Waypoint
    Waypoint database with path calculation utilities.

    
Functions
---------
**Path Following:**

    - pathFollow(vehicle) : Waypoint-based path following guidance system.
    - ALOSlaw(vehicle, pt1, pt2) : Adaptive Line-of-Sight guidance law.

**Target Tracking:**

    - targetTrack(vehicle) : Target-tracking guidance system.
    - velAPF(vehicle) : APF-based velocity guidance law.
    - velCB(vehicle) : Constant bearing velocity guidance law.

**APF Attraction Functions:**

    - variableCubicAttractionAPF(vehicle) : Cubic attraction with inflection point.
    - variableLinearAttractionAPF(vehicle) : Linear attraction.
    - variableLinearZoneAttractionAPF(vehicle) : Linear with constant bearing zone.

**APF Repulsion Functions:**

    - exponentialRepulsionAPF(vehicle) : Exponential repulsion from neighbors.
    - variableExpRepulsionAPF(vehicle) : Exponential with variable maximum.
    - depthAPF(vehicle, vel) : Depth constraint repulsion.

**Waypoint Utilities:**

    - updateWpt(vehicle) : Update active waypoint based on position.
    - getNextWpt(vehicle) : Get next waypoint coordinates.
    - addEtaWpt(vehicle) : Add current position to waypoint database.
    - generateRandomPath(num, start, psiLims, rLims, zLims) : Generate random path.

**Prediction and State Estimation:**

    - predictNextEtaVel(vehicle) : Predict vehicle's next position and velocity.
    - predictSwarmState(vehicle) : Update swarm member position predictions.

    
Notes
-----
- Guidance block inputs: State vectors from Navigation, operator commands,
  sensors
- Guidance block outputs: State change commands to Control, operator feedback


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
from dataclasses import dataclass, fields
from typing import List, Optional, Tuple, TYPE_CHECKING, Union
from typing_extensions import Self
from numpy.typing import NDArray
if (TYPE_CHECKING):
    from munetauvsim.vehicles import Vehicle
import numpy as np
from munetauvsim import navigation as nav
from munetauvsim import logger

#-----------------------------------------------------------------------------#

# Type Aliases
NPFltArr = NDArray[np.float64]

# Global Variables
log = logger.addLog('guid')

###############################################################################

@dataclass
class Position:
    """
    Coordinate data structure holding 3D positions in END coordinates.
    

    Attributes
    ----------
    x : list of float
        East coordinates in meters.
    y : list of float
        North coordinates in meters.
    z : list of float
        Down coordinates in meters (positive downward).
        

    Notes
    -----
    - Uses __slots__ for memory efficiency. 
    - Supports indexing and slicing.
    - Automatically converts scalar and array-like inputs to lists.

    
    Examples
    --------
    ### Basic creation and access:

    >>> import munetauvsim.guidance as guid
    >>> pos = guid.Position([0, 100, 200], [50, 150, 250], [10, 15, 20])
    >>> print(f"Number of points: {len(pos.x)}")
    Number of points: 3

    ### Index to get coordinates at specific point:

    >>> coords = pos[0]  # Get first point
    >>> print(coords)
    [0, 50, 10]
    >>> x, y, z = pos[1]  # Unpack second point
    >>> print(f"Point 1: East={x}m, North={y}m, Depth={z}m")
    Point 1: East=100m, North=150m, Depth=15m

    ### Slice to get multiple points:

    >>> first_two = pos[0:2]  # Get first two points
    >>> print(first_two)
    [[0, 100], [50, 150], [10, 15]]

    ### Access individual coordinate arrays:

    >>> east_coords = pos.x
    >>> print(f"East coordinates: {east_coords}")
    East coordinates: [0, 100, 200]

    ### Convert scalar inputs to lists automatically:

    >>> single_point = guid.Position(100, 200, 30)
    >>> print(f"Single point: {single_point[0]}")
    Single point: [100, 200, 30]

    ### Convert numpy arrays to lists:

    >>> import numpy as np
    >>> x_arr = np.array([0, 100, 200])
    >>> y_arr = np.array([0, 50, 100])  
    >>> z_arr = np.array([5, 10, 15])
    >>> pos_from_arrays = guid.Position(x_arr, y_arr, z_arr)
    >>> print(f"From arrays: {pos_from_arrays[1]}")
    From arrays: [100, 50, 10]
    """

    __slots__ = ['x', 'y', 'z']
    x: List[float]
    y: List[float]
    z: List[float]

    def __post_init__(self):
        """Enforce list type for all coordinate fields."""
        for field in fields(self):
            item = getattr(self,field.name)
            if (not isinstance(item, list)):
                if (hasattr(item,'__getitem__') and hasattr(item,'__len__')):
                    # Input is not a list but has sequence attributes, unpack
                    setattr(self,field.name,[*item])
                else:
                    # Otherwise input is a single value, make into a list
                    setattr(self,field.name,[item])

    def __getitem__(self, key:Union[int,slice])->List[float]:
        """Return [x, y, z] at index or slice."""
        return [self.x[key], self.y[key], self.z[key]]

###############################################################################

class Waypoint:
    """
    Waypoint database for path planning and trajectory analysis.
    
    Stores a sequence of (x, y, z) positions in END frame. Supports indexing,
    slicing, and path analysis (distances, headings, time estimates).
    

    Parameters
    ----------
    xPos : float or list of float, default=0
        East coordinates in meters.
    yPos : float or list of float, default=0
        North coordinates in meters.
    zPos : float or list of float, default=0
        Down coordinates (depth) in meters.
    
        
    Attributes
    ----------
    pos : Position
        Position object containing x, y, z coordinate lists.

        
    Methods
    -------
    insert(index, point)
        Insert a waypoint before index.
    calcWptDistance(index)
        Compute distances between consecutive waypoints.
    calcTotalDistance()
        Sum total linear distance through all waypoints.
    calcWptHeading(index)
        Compute heading angles between consecutive waypoints.
    estimateTime(speed, coeff)
        Estimate travel time through all waypoints.
    estimateAreaRoot(padding)
        Determine bounding square containing all waypoints.

        
    Notes
    -----
    - Possible extensions could add additional waypoint attributes such as
      speed, heading, orientation (eta / pose), guidance directives, etc.
    - Structure is based on the design used in Fossen's MSS.

    
    References
    ----------
    [1] Fossen, T. I. and Perez, T. (2004). Marine Systems Simulator (MSS).
    https://github.com/cybergalactic/MSS

        
    Examples
    --------
    ### Basic waypoint creation:

    >>> import munetauvsim.guidance as guid
    >>> wpt = guid.Waypoint([0, 500, 1000], [500, 500, 500], [20, 20, 20])
    >>> print(f"Number of waypoints: {len(wpt)}")
    Number of waypoints: 3
    >>> print(wpt)
    (     0.0,    500.0,     20.0)
    (   500.0,    500.0,     20.0)
    (  1000.0,    500.0,     20.0)

    ### Access waypoint coordinates:

    >>> coords = wpt.pos[0]  # Get first waypoint as [x, y, z]
    >>> x, y, z = wpt.pos[0]  # Unpack coordinates
    >>> print(f"Waypoint 0: East={x}m, North={y}m, Depth={z}m")
    Waypoint 0: East=0m, North=500m, Depth=20m

    ### Access coordinate arrays directly:

    >>> east_coords = wpt.pos.x
    >>> print(f"All East coordinates: {east_coords}")
    All East coordinates: [0, 500, 1000]

    ### Slice waypoints (returns new Waypoint object):

    >>> first_two = wpt[0:2]
    >>> print(f"First two waypoints: {len(first_two)} points")
    First two waypoints: 2 points
    >>> print(first_two.pos.x)
    [0, 500]

    ### Index single waypoint (returns new Waypoint object):

    >>> single_wpt = wpt[1]
    >>> print(f"Single waypoint: {single_wpt.pos[0]}")
    Single waypoint: [500, 500, 20]

    ### Add waypoints dynamically:

    >>> wpt.insert(1, [250, 400, 25])  # Insert before index 1
    >>> print(f"After insert: {len(wpt)} waypoints")
    After insert: 4 waypoints
    >>> print(wpt.pos[1])
    [250, 400, 25]

    ### Concatenate waypoint databases:

    >>> wpt1 = guid.Waypoint([0, 100], [0, 100], [10, 15])
    >>> wpt2 = guid.Waypoint([200, 300], [200, 300], [20, 25])
    >>> combined = wpt1 + wpt2
    >>> print(f"Combined length: {len(combined)}")
    Combined length: 4

    ### Path analysis:

    >>> distances = wpt.calcWptDistance()
    >>> print(f"Leg distances: {distances}")
    Leg distances: [269.30466019 269.30466019 500.        ]
    >>> total_dist = wpt.calcTotalDistance()
    >>> print(f"Total path length: {total_dist:.1f}m")
    Total path length: 1038.6m

    ### Heading calculations:

    >>> headings = wpt.calcWptHeading()
    >>> print(f"Leg headings (rad): {headings}")
    Leg headings (rad): [-0.38050638  0.38050638  0.        ]

    ### Time estimates:

    >>> travel_time = wpt.estimateTime(speed=2.0)
    >>> print(f"Estimated travel time: {travel_time:.1f} seconds")
    Estimated travel time: 623.2 seconds
    """
    
    __slots__ = ['pos']

    ## Constructor ===========================================================#
    def __init__(self,
                 xPos:Union[float, List[float]] = 0,
                 yPos:Union[float, List[float]] = 0,
                 zPos:Union[float, List[float]] = 0,
                 ):
        """Initialize waypoint database with coordinate lists."""
        self.pos = Position(xPos, yPos, zPos)
    
    ## Special Methods =======================================================#
    def __getitem__(self, key:Union[int,slice])->Self:
        """Index or slice waypoints, returns new Waypoint object."""
        return self.__class__(*self.pos[key])
    
    #-------------------------------------------------------------------------#
    def __len__(self)->int:
        """Return number of waypoints in database."""
        return len(self.pos.x)
    
    #-------------------------------------------------------------------------#
    def __repr__(self)->str:
        """Detailed description of Waypoint"""
        fmt = f'.1f'
        out = ''
        # Length
        out += f'len={len(self)}, '
        # Position
        out += 'pos=('
        for i in range(len(self)):
            out += (f'[{i}]:(' + 
                    f'{self.pos.x[i]:{fmt}}, ' +
                    f'{self.pos.y[i]:{fmt}}, ' +
                    f'{self.pos.z[i]:{fmt}}' +
                    f')')
            out += ', ' if (i < len(self)-1) else ')' 
        return f'{self.__class__.__name__}(' + out + ')'

    #-------------------------------------------------------------------------#
    def __str__(self)->str:
        """Return user-friendly string representation of Waypoint"""
        fmt = f'8.1f'
        out = ''
        # Position
        for i in range(len(self)):
            out += (f'(' + 
                    f'{self.pos.x[i]:{fmt}}, ' +
                    f'{self.pos.y[i]:{fmt}}, ' +
                    f'{self.pos.z[i]:{fmt}}' +
                    f')\n')
        return out.rstrip('\n')

    #-------------------------------------------------------------------------#
    def __add__(self, other:Self)->Self:
        """Concatenate two waypoint databases."""
        x = self.pos.x + other.pos.x
        y = self.pos.y + other.pos.y
        z = self.pos.z + other.pos.z
        return self.__class__(x, y, z)

    ## Methods ===============================================================#
    def insert(self, index:int, point:List[float])->None:
        """
        Insert point into the database before the position at index.
        
        Parameters
        ----------
        index : int
            Insert point before this index. Use len(wpt) to append to end.
        point : list of float, [x, y, z]
            Position coordinates to insert.
        """

        self.pos.x.insert(index,point[0])
        self.pos.y.insert(index,point[1])
        self.pos.z.insert(index,point[2])

    #-------------------------------------------------------------------------#
    def calcWptDistance(self, index:Optional[int]=None)->NPFltArr:
        """
        Return array of distances between consecutive waypoints.
        
        Parameters
        ----------
        index : int, optional
            If specified, return only distance at that index.
            
        Returns
        -------
        distances : ndarray, shape (len-1,)
            Euclidean distances between waypoints in meters.
        
        Notes
        -----
        - Output will have length(self)-1. 
        - Using index = -1 will still return the final distance measurement.
        """

        if (len(self) <= 1):
            return 0
        if (index is not None):
            return self.calcWptDistance()[index]
        else:
            return np.sqrt(np.diff(self.pos.x)**2 +
                           np.diff(self.pos.y)**2 + 
                           np.diff(self.pos.z)**2)

    #-------------------------------------------------------------------------#
    def calcTotalDistance(self)->np.float64:
        """
        Calculate total path length through all waypoints.
        
        Returns
        -------
        distance : float
            Sum of distances between consecutive waypoints in meters.
        """

        return sum(self.calcWptDistance())
    
    #-------------------------------------------------------------------------#
    def calcWptHeading(self, index:Optional[int]=None)->NPFltArr:
        """
        Return heading angles (rad) between consecutive waypoints.
        
        Parameters
        ----------
        index : int, optional
            If specified, return only heading from index to index+1.
            
        Returns
        -------
        headings : ndarray, shape (len-1,)
            Azimuth angles in radians. East = 0 rad, North = pi/2 rad.

        Notes
        -----
        - Output will have length(self)-1.
        - Using index = -1 will still return the final heading angle.
        """

        if (len(self) <= 1):
            return 0
        if (index is not None):
            return self.calcWptHeading()[index]
        else:
            return np.arctan2(np.diff(self.pos.y), np.diff(self.pos.x))
        
    #-------------------------------------------------------------------------#
    def estimateTime(self, speed:float=2.0, coeff:float=1.2)->np.float64:
        """
        Estimate travel time to traverse full path through all waypoints.
        
        Parameters
        ----------
        speed : float, default=2.0
            Vehicle speed in m/s.
        coeff : float, default=1.2
            Scaling coefficient. Adds buffer to account for variations in how
            particular guidance functions generate paths or from environmental
            influences.
            
        Returns
        -------
        time : float
            Estimated travel time in seconds.
        """

        return coeff * self.calcTotalDistance()/speed
    
    #-------------------------------------------------------------------------#
    def estimateAreaRoot(self, padding:int=300)->Tuple[int,int]:
        """
        Compute bounding square dimensions for path area.
        
        Parameters
        ----------
        padding : int, default=300
            Buffer zone in meters added to all sides around path area.
            
        Returns
        -------
        length : int
            Side length of padded square containing path.
        offset : list of int
            Offset coordinates of origin (0,0) relative to lower-left corner.

        Notes
        -----
        Computes the largest distance in the x or y direction and adds a buffer
        zone of padding. The padding is added on all sides, extending the
        calculated length by 2*padding. Root length is rounded up to the
        nearest integer.
        """

        xmin = min(self.pos.x)
        ymin = min(self.pos.y)
        l = max((max(self.pos.x) - xmin), (max(self.pos.y) - ymin))
        p = int(padding)
        return int(np.ceil(l)) + 2*p, [abs(xmin)+p, abs(ymin)+p]
    
###############################################################################

def generateRandomPath(num:int=1,
                       start:List[float]=[0,0,0],
                       psiLims:List[float]=[-np.pi/2,np.pi/2],
                       rLims:List[float]=[100,200],
                       zLims:List[float]=[0,100],
                       )->Waypoint:
    """
    Generate random waypoint path with specified constraints.

    Parameters
    ----------
    num : int
        Number of waypoints to generate.
    start : [x, y, z]
        Starting position in END.
    psiLims : [min, max]
        Heading change bounds in radians.
    rLims : [min, max]
        Distance change bounds in meters.
    zLims : [min, max]
        Depth change bounds in meters.
        
    Returns
    -------
    wpt : Waypoint
        Generated waypoint database.

    Notes
    -----
    - Intended for quick testing and demo purposes when no specific path or
      mission scenario is required.
    - If the need arises for repeating these paths, incorporate a seed value
      that can be used by the RNG.
    """

    wpt = Waypoint(*start)
    for point in range(num):
        psi = wpt.calcWptHeading(-1) + np.random.uniform(*psiLims)
        r = np.random.randint(*rLims)
        x = wpt.pos.x[-1] + (r * np.cos(psi))
        y = wpt.pos.y[-1] + (r * np.sin(psi))
        z = (wpt.pos.z[-1] + np.random.randint(*zLims)) % zLims[1]
        wpt.insert(len(wpt), [x,y,z])
    return wpt

###############################################################################

def addEtaWpt(vehicle:Vehicle)->None:
    """
    Add current vehicle position to waypoint database.

    Parameters
    ----------
    vehicle : Vehicle
        Vehicle with eta, wpt, wpt_k, R_switch attributes.
        eta: [x, y, z, phi, theta, psi], vehicle position / attitude vector
        wpt: Waypoint database
        wpt_k: Index of previous active waypoint
        R_switch: Acceptance radius (m) for reaching a waypoint
        
    Notes
    -----
    - Inserts position before current active waypoint.
    - Only adds position if distance > R_switch between waypoints.
    """

    # Input Parameters Loaded on Vehicle
    eta = vehicle.eta
    wpt = vehicle.wpt
    k_p = vehicle.wpt_k
    R_switch = vehicle.R_switch

    # Add Current Vehicle Position if it is Not Next Waypoint
    d = np.sqrt((wpt.pos.x[k_p]-eta[0])**2 + (wpt.pos.y[k_p]-eta[1])**2 +
                (wpt.pos.z[k_p]-eta[2])**2)
    if (d > R_switch):
        vehicle.wpt.insert(k_p, [eta[0],eta[1],eta[2]])

###############################################################################

def getNextWpt(vehicle:Vehicle)->List[float]:
    """
    Get next waypoint from database, or continue on last bearing if complete.
    
    Parameters
    ----------
    vehicle : Vehicle
        Vehicle with wpt, wpt_k attributes.
        wpt: Waypoint database
        wpt_k: Index of previous active waypoint
        
    Returns
    -------
    pt : [x, y, z]
        Next waypoint coordinates. If at end of database, continues on last
        bearing to distant extrapolated point.

    Notes
    -----
    Handling the vehicle behavior when reaching the end of the waypoint
    database should not be the responsibility of this function, but it's here
    for now or until there is a need to develop an alternative. Present
    development is to simply continue on last bearing. Other behaviors to
    consider may be to return to the first waypoint, maintain the position
    (station keeping, or a holding pattern), return to the surface and power
    down, or end the simulation.
    """
    
    # Input Parameters Loaded on Vehicle
    wpt = vehicle.wpt
    k_p = vehicle.wpt_k

    # Determine Next Waypoint
    ## More Waypoints Available
    if (k_p < len(wpt)-1):                 
        x_n, y_n, z_n = wpt.pos[k_p+1]
    ## No More Available: Continue Last Bearing
    else:
        x_p, y_p, z_p = wpt.pos[k_p]
        bearing = wpt.calcWptHeading(k_p-1)
        R = 1e10
        x_n = x_p + R * np.cos(bearing)
        y_n = y_p + R * np.sin(bearing)
        z_n = z_p
    
    return [x_n, y_n, z_n]

###############################################################################

def updateWpt(vehicle:Vehicle)->List[List[float]]:
    """
    Update active waypoint index when vehicle reaches acceptance radius.
    
    Parameters
    ----------
    vehicle : Vehicle
        Vehicle with wpt, wpt_k, R_switch, and eta attributes.
        wpt: Waypoint database
        wpt_k: Index of previous active waypoint
        R_switch: Acceptance radius (m) for reaching a waypoint
        eta : [x, y, z, phi, theta, psi], vehicle position/attitude vector
    
    Returns
    -------
    prevPt : [x, y, z]
        Previous active waypoint coordinates.
    nextPt : [x, y, z]
        Next active waypoint coordinates.
        
    Notes
    -----
    - Currently only uses 2D (x,y) values to determine successful arrival at
      waypoint.
    - Updates vehicle.wpt_k index when vehicle enters acceptance radius.
    - Logs waypoint arrival events.
    """

    # Input Parameters Loaded on Vehicle
    wpt = vehicle.wpt
    k_p = vehicle.wpt_k
    R_switch = vehicle.R_switch

    # Validate Input Values
    if (R_switch > 
        min(np.sqrt(np.diff(wpt.pos.x)**2 + np.diff(wpt.pos.y)**2))):
        raise ValueError(f'Distances between waypoints must be larger than '+
                         f'R_switch: {R_switch}')
    if (R_switch < 0):
        raise ValueError('R_switch must be larger than zero')

    # Determine Two Reference Points
    x_p, y_p, z_p = wpt.pos[k_p]
    x_n, y_n, z_n = getNextWpt(vehicle)

    # Update Active Waypoint Index
    x_e, y_e, pi_h = nav.statePT(vehicle,[x_p,y_p],[x_n,y_n])
    d = np.sqrt((x_n-x_p)**2 + (y_n-y_p)**2)
    n = len(wpt)
    if ((d-x_e < R_switch) and (k_p < n-1)):
        k_p += 1
        log.info("%s @ WP%02d %s",vehicle.callSign, k_p, wpt[k_p])
    
    prevPt = [x_p, y_p, z_p]
    nextPt = [x_n, y_n, z_n]
    vehicle.wpt_k = k_p

    return [prevPt, nextPt]

###############################################################################

def predictNextEtaVel(vehicle:Vehicle)->List[NPFltArr]:
    """
    Predict vehicle's own next position and velocity for communication messages.
    
    Generates forward-looking state estimates used in swarm coordination
    messages to help followers anticipate leader motion and compensate for
    communication delays.

    
    Parameters
    ----------
    vehicle : Vehicle
        Vehicle object. Must have the following attributes defined:

        - eta : ndarray, shape (6,)
            Current position/attitude [x, y, z, phi, theta, psi] in END frame.
        - wpt : Waypoint
            Waypoint database for path planning.
        - n_rate : float or None
            Propeller acceleration rate in rpm/s. If None, assumes constant
            speed.
        - n_setpt : float
            Propeller setpoint RPM for constant speed scenarios.
        - u_actual : ndarray, shape (3,)
            Actual control state [rudder, stern, propeller_rpm].
        - xferN2U : callable
            Method to convert propeller RPM to vehicle speed (m/s).

             
    Returns
    -------
    nextPoint : ndarray, shape (3,)
        Predicted next position [x, y, z] in meters.
        Determined from next waypoint in database via getNextWpt().
    nextVel : ndarray, shape (3,)
        Predicted next velocity [vx, vy, vz] in m/s.
        Computed from constant acceleration model.

        
    Notes
    -----
    **Position Prediction:**

    Next position is taken directly from the waypoint database via getNextWpt().
    If at the end of waypoints, extrapolates along last bearing to distant point.
    
    **Velocity Prediction:**

    Uses kinematic constant acceleration model:
    
        v_f^2 = v_i^2 + 2 a x
    
    where:

        - v_i: Current speed from xferN2U(u_actual[2])
        - a: Acceleration from xferN2U(n_rate)
        - x: Distance to next waypoint
        - v_f: Predicted speed at next waypoint
    
    Direction is along the vector from current position to next waypoint.
    
    **Usage Context:**

    This function is called by communication message generation routines to populate
    nextEta and nextVel fields in broadcast messages. Followers use these predictions
    with predictSwarmState() to extrapolate leader position during message delays.
    
    **Limitations:**

    - Assumes straight-line motion to next waypoint (no turning dynamics)
    - Ignores environmental effects (currents, disturbances)
    - Only valid for target using waypoint-based guidance
    - If n_rate is None, assumes constant speed (no acceleration term)

    
    See Also
    --------
    getNextWpt : Retrieves next waypoint coordinates
    predictSwarmState : Uses predictions to extrapolate follower group states
    """

    # Determine 'Next Position' from Waypoint Database
    nextPoint = np.array(getNextWpt(vehicle))

    # Determine 'Next Velocity'
    course = nextPoint - vehicle.eta[:3]
    dist = np.linalg.norm(course)
    if (vehicle.n_rate):
        speed = vehicle.xferN2U(vehicle.u_actual[2])
        accel = vehicle.xferN2U(vehicle.n_rate)
        nextSpeed = np.sqrt((speed)**2 + (2*accel*dist))
        nextVel = nextSpeed * (course/dist)
    else:
        nextVel = vehicle.xferN2U(vehicle.n_setpt) * (course/dist)

    return [nextPoint, nextVel]

###############################################################################

def ALOSlaw(vehicle:Vehicle,
            pt1:List[float],
            pt2:List[float],
            )->float:
    """
    Compute desired heading using Adaptive Line-of-Sight (ALOS) guidance law.
    
    Implements Fossen's ALOS algorithm for path following with automatic
    compensation for ocean current drift via adaptive crab angle estimation.
    Provides robust tracking of straight-line path segments between waypoints.

    
    Parameters
    ----------
    vehicle : Vehicle
        Vehicle object. Must have the following attributes defined:
        
        - sampleTime : float
            Euler integration time step in seconds.
        - eta : ndarray, shape (6,)
            Position/attitude [x, y, z, phi, theta, psi] in END frame.
        - beta_c_hat : float
            Current estimate of crab angle in radians. Updated by this function.
        - Delta : float
            Look-ahead distance in meters (typically 10-50 m). Larger values
            give smoother but less responsive tracking. Should be >> cross-track
            error.
        - gamma : float
            Adaptive gain constant (positive, typically 0.0001-0.001). Larger
            values give faster crab angle adaptation but may cause oscillation.
            
    pt1 : list of float, [x, y, z]
        Previous waypoint in END coordinates (meters).
    pt2 : list of float, [x, y, z]
        Next waypoint in END coordinates (meters).
        

    Returns
    -------
    psi_ref : float
        Desired heading angle in radians. Range: [-pi, pi].
        Feed to heading observer/autopilot for closed-loop tracking.
    
        
    Notes
    -----
    **Side Effects:**

    Updates vehicle.beta_c_hat via forward Euler integration.
    
    **ALOS Guidance Law:**

    The complete guidance equation is:

        psi_ref = pi_h - beta_c_hat - atan(y_e / Delta)

    where:
    
        - psi_ref: Desired heading angle (output)
        - pi_h: Path-tangential angle (azimuth) bearing from pt1 to pt2 wrt East
        - beta_c_hat: Estimated crab angle (adapted online)
        - y_e: Cross-track error (perpendicular distance to path)
        - Delta: Look-ahead distance (design parameter)
    
    **Adaptive Crab Angle:**

    The crab angle beta_c compensates for sideslip caused by ocean currents or
    other environmental disturbances. The adaptation law:

        d/dt beta_c_hat = gamma * Delta * y_e / sqrt(Delta^2 + y_e^2)

    has the following properties:

        - Converges to steady-state crab angle for constant current
        - Integral action drives cross-track error to zero
        - Normalized by sqrt term for bounded adaptation rate
        - Gamma * Delta product determines convergence speed

    **Tuning Guidelines:**

    Look-ahead distance Delta:

        - Larger: Smoother path, slower response, tolerates larger y_e
        - Smaller: Tighter tracking, more sensitive to disturbances
        - Rule of thumb: Delta = 2-5x vehicle length
        - Must satisfy: Delta >> max expected y_e for stability
    
    Adaptive gain gamma:

        - Larger: Faster convergence, potential oscillation
        - Smaller: Slower convergence, smoother behavior
        - Typical range: 0.0001-0.001 for AUV applications
        - Scale with 1/speed for speed-invariant adaptation

    **State Calculation Dependencies:**

    Internally calls nav.statePT(vehicle, pt1, pt2) which computes:

        - x_e: Along-track error (progress along path)
        - y_e: Cross-track error (perpendicular to path)
        - pi_h: Path-tangential azimuth angle


    Warnings
    --------
    - Do not set Delta too small (< 2 m) or division by small numbers causes
      numerical issues
    - Gamma too large causes oscillation or instability
    - beta_c_hat should be initialized to 0 or small value
    - Singularity at Delta = 0 (avoided by proper parameter selection)
    

    See Also
    --------
    navigation.statePT : Computes path-tangential state (x_e, y_e, pi_h)
    navigation.headingFilterLOS : Heading observer for ALOS reference tracking
    pathFollow : Complete guidance system using ALOS
        
    
    References
    ----------
    [1] Fossen, T. I., "An Adaptive Line-of-Sight (ALOS) Guidance Law for Path
    Following of Aircraft and Marine Craft," in IEEE Transactions on Control
    Systems Technology, 31(6), 2887-2894, Nov. 2023, doi:
    10.1109/TCST.2023.3259819.
    
    [2] Fossen, T. I. and Perez, T. (2004). Marine Systems Simulator (MSS).
    https://github.com/cybergalactic/MSS
    """

    # Input Parameters Loaded on Vehicle
    h = vehicle.sampleTime
    beta_c_hat = vehicle.beta_c_hat
    Delta = vehicle.Delta
    gamma = vehicle.gamma

    # State Calculations
    _, y_e, pi_h = nav.statePT(vehicle,pt1,pt2)

    # ALOS guidance law
    psi_ref = pi_h - beta_c_hat - np.arctan(y_e / Delta)

    # Update Crab Angle Estimate
    Dbeta_c_hat = gamma * Delta * y_e / np.sqrt(Delta**2 + y_e**2)
    vehicle.beta_c_hat = beta_c_hat + (h * Dbeta_c_hat)

    return psi_ref

###############################################################################

def pathFollow(vehicle:Vehicle)->NPFltArr:
    """
    Waypoint-based path following guidance system.
    
    Implements a complete guidance system that coordinates waypoint tracking,
    heading observation, depth filtering, and autopilot control. Manages
    waypoint transitions and coordinates all guidance/control components. This
    function is designed to be assigned to a vehicle as the GuidSystem callable
    attribute.
    

    Parameters
    ----------
    vehicle : Vehicle
        Vehicle object. Must have the following assigned methods and attributes:

        - GuidLaw : callable
            Core guidance algorithm implementing specific mathematical
            approaches for path generation.
        - HeadingObs : callable
            Heading state estimator and reference generator.
        - HeadingAP : callable
            Heading autopilot implementing control commands for rudder
            defleciton.
        - DepthObs : callable
            Depth command filter and state estimator.
        - DepthAP : callable
            Depth autopilot implementing control commands for stern plane
            deflection.
        - PropCmd : callable
            Propeller command generator for thrust management. 
        - wpt : Waypoint
            Waypoint database.
        - eta : ndarray, shape (6,)
            Position/attitude state.
        
            
    Returns
    -------
    u_control : ndarray, shape (3,)
        Control command vector [delta_r, delta_s, n] where:

        - delta_r : float
            Rudder angle in radians from HeadingAP output.
        - delta_s : float
            Stern plane angle in radians from DepthAP output.
        - n : float
            Propeller RPM command from PropCmd output.

            
    Notes
    -----
    **Workflow:**

    The function implements a three-stage control pipeline with configurable
    components:

    1. **Waypoint Management**

    - Updates waypoint index when vehicle enters acceptance radius
    - Provides previous and next waypoint coordinates to guidance law
    - Logs waypoint arrival events for mission tracking

    2. **Guidance Law Computation**

    - Processes waypoint coordinates to compute desired trajectory
    - Generates heading reference command (psi_ref) for path following
    - Updates any internal guidance states (e.g., adaptive parameters)

    3. **Navigation and Control Coordination**:

    - **Heading Channel**:

        - HeadingObs(psi_ref): Processes heading reference through observer/
          filter to generate heading commands and rate references
        - HeadingAP(): Converts heading error to rudder deflection command

    - **Depth Channel**:

        - DepthObs(nextPt): Processes depth reference from next waypoint 
          through observer/filter for depth command generation
        - DepthAP(): Converts depth error to stern plane deflection command

    - **Speed Control**:

        - PropCmd(): Generates propeller command (RPM or thrust) for
          speed regulation or constant speed operation

    Each component (GuidLaw, HeadingObs, HeadingAP, DepthObs, DepthAP, PropCmd)
    is assigned to the vehicle as a callable function, allowing different
    implementations to be plugged in without modifying this coordination
    workflow. This design supports various guidance algorithms (ALOS, LOS,
    etc.), observer types (Kalman filters, complementary filters), and autopilot
    configurations (PID, LQR, etc.) through a common interface.

    
    See Also
    --------
    vehicles.Remus100s.loadPathFollowing() : Assigns use of pathFollow()
    ALOSlaw : Computes desired heading
    updateWpt : Manages waypoint transitions
    navigation.headingFilterLOS : Heading observer
    navigation.depthFilter : Depth filter
    control.headingPID : Heading autopilot
    control.depthPID : Depth autopilot
    control.constProp : Constant propeller command
    """

    # Get Waypoint Positions
    prevPt, nextPt = updateWpt(vehicle)

    # Heading Guidance Law
    psi_ref = vehicle.GuidLaw(vehicle,prevPt,nextPt)

    # Heading Command
    vehicle.HeadingObs(vehicle,psi_ref)
    delta_r = vehicle.HeadingAP(vehicle)

    # Depth Command
    vehicle.DepthObs(vehicle,nextPt)
    delta_s = vehicle.DepthAP(vehicle)

    # Propeller Command
    n = vehicle.PropCmd(vehicle)

    return np.array([delta_r, delta_s, n], float)

###############################################################################

def velCB(vehicle:Vehicle)->NPFltArr:
    """
    Constant bearing velocity guidance law for target tracking.
    
    Implements parallel navigation (constant bearing) where the follower
    maintains a constant bearing angle toward the target while matching target
    velocity. This function is designed to be assigned to a vehicle as the
    GuidLaw callable attribute as part of a target tracking guidance system.

    
    Parameters
    ----------
    vehicle : Vehicle
        Vehicle object (follower). Must have:
        
        - Delta : float
            Look-ahead/approach distance in meters (typically 50-200 m).
        - r_safe : float
            Minimum safe separation distance in meters.
        - u_max : float
            Maximum vehicle speed in m/s (saturates output velocity).
        - eta : ndarray, shape (6,)
            Follower position [x, y, z, phi, theta, psi] in END frame.
        - velocity : ndarray, shape (3,)
            Follower velocity [vx, vy, vz] in m/s (END frame).
        - target : Model or Vehicle
            Leader/target vehicle. Must have:

            - eta : ndarray, shape (6,)
                Target position in END frame.
            - velocity : ndarray, shape (3,)
                Target velocity in m/s.
            - nextEta : ndarray, shape (3,)
                Predicted next position (from communication).

                    
    Returns
    -------
    v_d : ndarray, shape (3,)
        Desired velocity vector [vx, vy, vz] in END frame (m/s).
        Saturated to magnitude u_max and depth-limited by network.

        
    Notes
    -----
    **Constant Bearing:**

    Computes the desired velocity vector (v_d) using a constant bearing
    guidance law

    .. code-block:: none

        v_d = v_t + v_los
        v_los = -Kappa * p_los / norm(p_los)
        Kappa = U_amax * norm(p_los) / sqrt(Delta^2 + (p_los*p_los^T))
        p_los = p - p_t - r_safe
    
    The follower uses parallel navigation to match the target velocity vector 
    (v_t) and aligns the follower-target velocity (v_los) along the Line of
    Sight (LOS) vector between the follower and the target.
    
    The LOS vector is constructed from the follower position (p) and the target
    position (p_t), modified by a minimum safe distance between the vehicles
    (r_safe). Delta is the look ahead distance, U_amax is the maximum speed 
    along the LOS vector, and Kappa is the approach velocity regulation 
    coefficient.

    **Velocity Triangle Geometry:**

    Applies law of sines/cosines to solve for the desired velocity that:

    1. Maintains constant bearing to target
    2. Matches target velocity (avoids separation/collision)
    3. Results in closure (approaches target) or parallel motion
    
    **Design Parameter Delta (Look-ahead distance):**

    - Controls how aggressively to approach target
    - Larger Delta: Gentle, smooth approach
    - Smaller Delta: Tight, aggressive interception
    - Typical values: 50-200 m for AUV swarms
    

    See Also
    --------
    vehicles.Remus100s.loadTargetTracking : Assigns guidance system
    velAPF : APF-based guidance (more complex, handles multiple targets)
    targetTrack : Complete tracking guidance system using velCB or velAPF
    

    References
    ----------
    [1] Fossen, T.I. (2021). Handbook of Marine Craft Hydrodynamics and Motion
    Control. 2nd Edition, Wiley. https://www.fossen.biz/wiley
    """

    # Input Parameters Loaded on Vehicle
    eta_t = vehicle.target.eta
    vel_t = vehicle.target.velocity

    # Guidance Parameter Constants
    Delta = vehicle.Delta
    r_safe = vehicle.r_safe
    U_max = vehicle.u_max

    # Determine LOS Vector
    v_t = vel_t[0:3]                            # Target Velocity Vector
    p_t = eta_t[0:3]                            # Target Position Vector
    p = vehicle.eta[0:3]                        # Follower Position Vector
    if (np.linalg.norm(p-p_t) == 0):
        return [0., 0., 0.]
    los_hat = (p-p_t) / np.linalg.norm(p-p_t)   # LOS Unit Vector
    p_los = p - p_t - (r_safe * los_hat)        # LOS Vec. Corrected for r_safe

    # Determine U_amax
    """
    Since v_d = v_t + v_los, the three vectors form a triangle. 
    1) The angle between LOS and v_t is determined with the Vector Dot Product
    2) Then the angle between LOS and v_d is determined by the Law of Sines
    3) The last angle is found by the Triangle Angle Sum Theorem
    4) Finally the maximum magnitude of v_los is found by the Law of Cosines
    """
    norm_vt = np.linalg.norm(v_t)
    if (norm_vt == 0):
        return [0., 0., 0.]
    theta_t_los = np.arccos(np.dot(los_hat,v_t) / (1.0 * norm_vt))
    theta_d_los = np.arcsin((norm_vt / U_max) * np.sin(theta_t_los))
    theta_t_d = np.pi - theta_t_los - theta_d_los
    U_amax = np.sqrt(norm_vt**2 + U_max**2 - 2*norm_vt*U_max*np.cos(theta_t_d))

    # Determine Desired Velocity Vector
    Kappa = ((U_amax * np.linalg.norm(p_los)) /
             (np.sqrt(Delta**2 + np.dot(p_los,p_los))))
    v_los = -Kappa * los_hat
    v_d = v_t + v_los

    return v_d

###############################################################################

def predictSwarmState(vehicle:Vehicle)->None:
    """
    Update predicted postion/velocity for swarm group members using data models.

    Extrapolates each group member's state based on last received message and
    elapsed communication delay. Enables followers to maintain accurate
    estimated positions of neighbors even during communication gaps. For
    simulations that do not use Direct Accessing.

    Parameters
    ----------
    vehicle : Vehicle
        Follower vehicle whose swarm group state should be updated. Must have:
        
        - sampleTime : float
            Simulation iteration time step in seconds.
        - id : int
            Vehicle ID (excludes self from group predictions).
        - target : Model or Vehicle
            Leader vehicle. Must have:

            - eta : ndarray, shape (6,)
                Last known position.
            - velocity : ndarray, shape (3,)
                Last known velocity.
            - nextEta : ndarray, shape (3,)
                Predicted next position from communication.
            - nextVel : ndarray, shape (3,)
                Predicted next velocity from communication.
            - delayLastMsg : float
                Time elapsed since last message in seconds.

        - group : list of Model
            Other swarm members. Each must have:

            - id : int
                Vehicle ID.
            - eta : ndarray, shape (6,)
                Last known position.
            - velocity : ndarray, shape (3,)
                Last known velocity.
            - delayLastMsg : float
                Time elapsed since message in seconds.

                
    Returns
    -------
    None
    

    Notes
    -----
    **Side Effects:**

    Updates for target vehicle:

    - target.eta : Propagated via constant acceleration model
    - target.delayLastMsg : Incremented by sampleTime
    - target.nextEta, target.nextVel : Reused for next prediction
    
    Updates for group members:

    - member.eta : Propagated via constant velocity model
    - member.delayLastMsg : Incremented by sampleTime

    Data on swarm group member states is not logged. Predictions overwrite
    previous held values.

    **Target Predictions:**

    Projects target.velocity and target.nextVel onto distance vector from
    target.eta to target.nextEta (d_l) and, assuming constant acceleration,
    estimates position and velocity along d_l.

        x = x_0 + v_0*t + 0.5*a*t^2
        v = v_0 + a*t
    
    **Group Predictions:**

    Updates position using simple assumption of constant velocity.

        x = x_0 +v_0*t

    
    See Also
    --------
    predictNextEtaVel : Generates predictive states for communication messages
    updateWpt : Updates target waypoint position
    velAPF : Uses current group.eta for repulsion forces
    communication.writeEtaVelLogs : Logs reported position and velocities
    """

    def timeStep(vehicle:Vehicle, sampleTime:float)->float:
        """Return time since last position data point"""
        if (vehicle.delayLastMsg > sampleTime):
            h = vehicle.delayLastMsg
            vehicle.delayLastMsg = 0.0
        else:
            h = sampleTime
        return h

    # Target
    if (vehicle.target):
        d_l = vehicle.target.nextEta - vehicle.target.eta
        dl_mag = np.linalg.norm(d_l)
        if (dl_mag != 0):
            # Estimate Velocity and Acceleration Along Target Direction Vector
            dl_hat = d_l / dl_mag
            vi_dl = np.dot(vehicle.target.velocity,dl_hat) * dl_hat
            vf_dl = np.dot(vehicle.target.nextVel,dl_hat) * dl_hat 
            a_dl = (np.dot(vf_dl,vf_dl)-np.dot(vi_dl,vi_dl))/(2*dl_mag)*dl_hat
            # Determine Time Step
            h = timeStep(vehicle.target, vehicle.sampleTime)
            # Position
            vehicle.target.eta += (vi_dl * h) + (0.5 * a_dl * h**2)
            # Velocity
            vehicle.target.velocity += (a_dl * h)
        
    # Group  
    for member in vehicle.group:
        if (member.id != vehicle.id):
            # Determine Time Step
            h = timeStep(member, vehicle.sampleTime)
            # Position
            member.eta += (member.velocity * h)

###############################################################################

def variableCubicAttractionAPF(vehicle:Vehicle)->NPFltArr:
    """
    Compute cubic Artificial Potential Field (APF) attraction velocity vector.

    Generates attractive velocity vector toward target vehicle with cubic
    nonlinearity, providing smooth but responsive guidance. Cubic function
    constructed such that the inflection point provides a neutral or stationary
    point. Velocity is zero at maximum avoidance radius and approaches target
    velocity at preferred following distance. This function is designed to be
    assigned to a vehicle as the GuidLaw.attraction callable attribute as part
    of a target tracking guidance system that uses the velAPF Guidance Law.
    

    Parameters
    ----------
    vehicle : Vehicle
        Follower vehicle. Must have:
        
        - eta : ndarray, shape (6,)
            Follower position [x, y, z, ...] in END frame.
        - velocity : ndarray, shape (3,)
            Current velocity [vx, vy, vz] in m/s.
        - u_max : float
            Maximum velocity magnitude in m/s.
        - r_avoid : float
            Radius of avoidance (m), the minimum distance for APF repulsion.
        - r_follow : float
            Preferred following distance in meters.
        - target : Vehicle or Model
            Target/leader vehicle with:

            - eta : ndarray, shape (6,)
                Target position.
            - velocity : ndarray, shape (3,)
                Target velocity.

                          
    Returns
    -------
    v_a : ndarray, shape (3,)
        Attractive velocity vector in END frame (m/s).
        Magnitude ranges from 0 at r_avoid to target velocity at r_follow.
    
        
    Notes
    -----
    The attraction function has only the cubic term, creating a monotonically
    increasing function with a flat inflection point,

        v = (alpha * distance^3) + v_target
        
    where alpha and distance are such that the function goes to zero at the
    Avoidance Radius, goes to the target velocity at the Following Radius, and
    increases at larger distances,

        alpha = -v_target / (r_avoid - r_follow)^3,
        distance = (dist_to_target - r_follow)

    The resulting velocity will be along the vector from the follower to the
    target.

    
    See Also
    --------
    vehicles.Remus100s.loadTargetTracking : Assigns guidance system
    variableLinearAttractionAPF : Simple linear attraction
    variableLinearZoneAttractionAPF : Linear with uniform zone
    variableExpRepulsionAPF : Repulsion complement to attraction
    velAPF : Main guidance law combining attraction and repulsion
    """

    # Vehicle Parameters
    p = vehicle.eta[0:3]            # Vehicle Position Vector
    leader = vehicle.target         # Tracking Target / Swarm Leader
    r_a = vehicle.r_avoid           # Avoidance Radius
    r_f = vehicle.r_follow          # Following Distance

    # APF Parameters
    u_l = nav.stateSpeed(leader)
    alpha = -u_l / (r_a - r_f)**3

    # Determine Attraction Velocity to Leader
    p_l = leader.eta[0:3]       # Leader Position Vector
    d_l = p_l - p               # Distance Vector from Vehicle to Leader
    dl_mag = np.linalg.norm(d_l)
    dl_hat = d_l / dl_mag
    if (dl_mag <= r_a):
        v_a = np.zeros(3)
    else:
        u_v = (alpha*(dl_mag - r_f)**3) + u_l
        v_a = u_v * dl_hat
    
    return v_a

###############################################################################

def variableLinearAttractionAPF(vehicle:Vehicle)->NPFltArr:
    """
    Compute linear Artificial Potential Field (APF) attraction velocity vector.
    
    Generates attractive velocity toward target proportional to distance.
    Simpler and more computationally efficient than cubic, with linear scaling
    from zero force at r_avoid to maximum at r_follow. This function is designed
    to be assigned to a vehicle as the GuidLaw.attraction callable attribute as
    part of a target tracking guidance system that uses the velAPF Guidance Law.

    
    Parameters
    ----------
    vehicle : Vehicle
        Follower vehicle. Must have:
        
        - eta : ndarray, shape (6,)
            Follower position [x, y, z, ...] in END frame.
        - velocity : ndarray, shape (3,)
            Current velocity [vx, vy, vz] in m/s.
        - u_max : float
            Maximum velocity magnitude in m/s.
        - r_avoid : float
            Radius of avoidance (m), the minimum distance for APF repulsion.
        - r_follow : float
            Preferred following distance in meters.
        - target : Vehicle or Model
            Target/leader vehicle with:

            - eta : ndarray, shape (6,)
                Target position.
            - velocity : ndarray, shape (3,)
                Target velocity.

                       
    Returns
    -------
    v_a : ndarray, shape (3,)
        Attractive velocity vector in END frame (m/s).

        
    Notes
    -----
    The attraction function is a simple linear scaling with distance, with the
    slope based on the speed of the target,
    
        v = alpha * distance

    where alpha and distance are such that the function goes to zero at the
    Avoidance Radius and equals the target speed at the Following Radius.

        alpha = v_target / (r_follow - r_avoid),
        distance = (dist_to_target - r_avoid)
    
    The resulting velocity will be along the vector from the follower to the
    target.
    

    See Also
    --------
    vehicles.Remus100s.loadTargetTracking : Assigns guidance system
    variableCubicAttractionAPF : More responsive cubic variant
    variableLinearZoneAttractionAPF : Linear with uniform zone
    variableExpRepulsionAPF : Repulsion complement to attraction
    velAPF : Main guidance law combining attraction and repulsion
    """

    # Vehicle Parameters
    p = vehicle.eta[0:3]            # Vehicle Position Vector
    leader = vehicle.target         # Tracking Target / Swarm Leader
    r_a = vehicle.r_avoid           # Avoidance Radius
    r_f = vehicle.r_follow          # Following Distance

    # APF Parameters
    alpha = nav.stateSpeed(leader) / (r_f - r_a)    # Attraction Slope

    # Determine Attraction Velocity to Leader
    p_l = leader.eta[0:3]       # Leader Position Vector
    d_l = p_l - p               # Distance Vector from Vehicle to Leader
    dl_mag = np.linalg.norm(d_l)
    dl_hat = d_l / dl_mag
    if (dl_mag <= r_a):
        v_a = np.zeros(3)
    else:
        u_l = alpha * (dl_mag - r_a)
        v_a =  u_l * dl_hat   # Attraction Vector

    return v_a

###############################################################################

def variableLinearZoneAttractionAPF(vehicle:Vehicle)->NPFltArr:
    """
    Compute linear APF attraction velocity vector with three zones.

    Creates three regions centered spherically on the target: inner, center, and
    outer. The inner region has linear scaling less than target velocity, center
    region equals target velocity, and outer region has linear scaling greater
    than target velocity. This function is designed to be assigned to a vehicle
    as the GuidLaw.attraction callable attribute as part of a target tracking
    guidance system that uses the velAPF Guidance Law.

    
    Parameters
    ----------
    vehicle : Vehicle
        Follower vehicle. Must have:
        
        - eta : ndarray, shape (6,)
            Follower position [x, y, z, ...] in END frame.
        - velocity : ndarray, shape (3,)
            Current velocity [vx, vy, vz] in m/s.
        - u_max : float
            Maximum velocity magnitude in m/s.
        - r_avoid : float
            Radius of avoidance (m), the minimum distance for APF repulsion.
        - r_follow : float
            Preferred following distance in meters.
        - target : Vehicle or Model
            Target/leader vehicle with:

            - eta : ndarray, shape (6,)
                Target position.
            - velocity : ndarray, shape (3,)
                Target velocity.

                       
    Returns
    -------
    v_a : ndarray, shape (3,)
        Attractive velocity vector in END frame (m/s).

        
    Notes
    -----
    **Attraction Algorithm:**

    The attraction function implements a three-zone piecewise linear attraction
    model that creates smooth transitions between different behavioral regions
    around the target vehicle. Zones are defined by radial distance so the
    regions are spherical.

    **Zone Definitions:**

    The boundaries of the Central zone are computed as:

        central_inner = r_follow - (cbz_w * r_avoid)
        central_outer = r_follow + (cbz_w * r_avoid)

    where cbz_w (constant bearing zone width) determines the size of the central
    region as a fraction of the avoidance radius, and the central zone is
    centered at the preferred following distance.

    **Piecewise Linear Function Regions:**

    The slope of the function in the two linear regions changes dynamically to
    the speed of the:

        alpha = u_target / (r_inner - r_avoid)

    1. **Avoidance Zone** (distance < r_avoid)

        - v_a = 0
        - No attraction, leaving APF repulsion as only forces
        - Prevent collision with target when very close

    2. **Inner Linear Zone** (r_avoid <= distance < central_inner)

        - v_a = alpha * (distance - r_avoid) * direction vector
        - Linear increase from zero to target velocity
        - Direction is along vector towards target

    3. **Central Uniform Zone** (central_inner <= distance <= central_outer)

        - v_a = u_a * (v_target / abs(v_target))
        - u_a is speed of target or a minimum value of 0.5 m/s
        - Maintains formation by matching target speed and direction
        - Creates a 'constant bearing' region for stable following
    
    4. **Outer Linear Zone** (central_outer < distance)

        - v_a = alpha * (distance - r_avoid - center_width) * direction vector
        - Linear increase beyond target velocity for catch-up behavior
        - Enables faster approach from distant positions
        - Direction is along vector towards target
        - Upper bound is naturally set by limit of vehicle capabilities
        
    **Key Parameters:**

    - u_min (0.5 m/s): Minimum speed in uniform zone prevents stagnation
    - cbz_w (0.5): Zone width ratio - larger values create wider stable region
    - alpha: Linear slope ensuring continuity at zone boundaries

    
    See Also
    --------
    vehicles.Remus100s.loadTargetTracking : Assigns guidance system
    variableCubicAttractionAPF : More responsive cubic variant
    variableLinearAttractionAPF : Simple linear attraction
    variableExpRepulsionAPF : Repulsion complement to attraction
    velAPF : Main guidance law combining attraction and repulsion
    """

    # Vehicle Parameters
    p = vehicle.eta[0:3]            # Vehicle Position Vector
    leader = vehicle.target         # Tracking Target / Swarm Leader
    r_f = vehicle.r_follow          # Following Distance
    r_a = vehicle.r_avoid           # Avoidance Radius

    # APF Parameters
    u_min = 0.5                     # Minimum Speed in Uniform Zone
    cbz_w = 0.5                     # Ratio of r_a to use as 1/2 CB Zone width

    # Calculate Distance
    p_l = leader.eta[0:3]           # Leader Position Vector
    d_l = p_l - p                   # Distance Vector from Vehicle to Leader
    dl_mag = np.linalg.norm(d_l)

    # Calculate APF Terms
    u_l = nav.stateSpeed(leader)    # Target Speed
    u_a = max(u_min, u_l)           # Attraction Speed
    inner = r_f - (cbz_w * r_a)     # Inner CB Zone Boundary Distance
    outer = r_f + (cbz_w * r_a)     # Outer CB Zone Boundary Distance

    # Central Uniform Zone
    if ((dl_mag <= outer) and (dl_mag >= inner)):
        if (u_l == 0):
            v_a = np.zeros(3)
        else:
            v_l = leader.velocity[0:3]
            v_a = u_a * (v_l / u_l)
    # Avoidance Zone
    elif (dl_mag < r_a):
        v_a = np.zeros(3)
    # Attraction Zone
    else:
        alpha = u_a / (inner - r_a)                 # Linear Attraction Slope
        dl_hat = d_l / dl_mag
        # Outer
        if (dl_mag > outer):
            v_a = alpha * (dl_mag - r_a - (outer-inner)) * dl_hat
        # Inner
        else:
            v_a = alpha * (dl_mag - r_a) * dl_hat

    return v_a

###############################################################################

def exponentialRepulsionAPF(vehicle:Vehicle)->NPFltArr:
    """
    Compute exponential APF repulsion vector from swarm neighbors and target.

    Generates repulsive velocity that avoids collisions with other vehicles.
    Uses exponential decay function for smooth, gradual avoidance with
    short-range hard constraint at r_safe. This function is designed to be
    assigned to a vehicle as the GuidLaw.repulsion callable attribute as part of
    a target tracking guidance system that uses the velAPF Guidance Law.

    
    Parameters
    ----------
    vehicle : Vehicle
        Vehicle computing repulsion. Must have:
        
        - eta : ndarray, shape (6,)
            Own position [x, y, z, ...] in END frame.
        - velocity : ndarray, shape (3,)
            Own velocity [vx, vy, vz] in m/s.
        - id : int
            Own vehicle ID.
        - u_max : float
            Maximum vehicle speed in m/s.
        - r_safe : float
            Minimum safe distance in meters.
        - r_avoid : float
            Maximum repulsion influence radius in meters.
        - group : list of Model
            Swarm neighbor vehicles with:

            - eta : ndarray, shape (6,)
                Neighbor position.
            - id : int
                Neighbor vehicle ID.

        - target : Vehicle or Model
            Target/leader vehicle with eta and id attributes.
            

    Returns
    -------
    v_r : ndarray, shape (3,)
        Total repulsive velocity (END frame, m/s). Magnitude ≤ u_max.
    
        
    Notes
    -----
    **Repulsion Function:**

    The total repulsion velocity is the sum of all repulsion velocities to each
    swarm group member.

    The repulsion function has a maximum value inside the Safety Radius and
    decays exponentially toward zero at the Avoidance Radius. 

        v = v_max * e^((r_safety - distance_to_member)/beta)
    
    where the maximum velocity is determined as a factor of the maximum vehicle
    speed, and beta is used to decay the function within the Avoidance Radus.

        v_max = gamma * u_max,
        beta = (r_safety - r_avoid) / log_e(zero_value / v_max)
    
    Because the function cannot decay to exactly zero, a 'zero_value' is used,
    which is a value small enough to produce an acceptably negligible result.
    The resulting velocity is along the vector from the swarm group member
    toward the follower.

    **Properties:**

    1. Zero repulsion for r_avoid < d_i (no influence beyond avoidance radius)
    2. Rapid increase for r_safe < d_i < r_avoid (collision avoidance)
    3. Maximum repulsion at d_i <= r_safe (hard constraint)

    
    See Also
    --------
    vehicles.Remus100s.loadTargetTracking : Assigns guidance system
    variableLinearZoneAttractionAPF : Attraction complement to repulsion
    variableExpRepulsionAPF : Velocity-dependent repulsion
    depthAPF : Depth safety constraints
    velAPF : Main guidance law combining attraction and repulsion
    """

    # Vehicle Parameters
    p = vehicle.eta[0:3]                # Vehicle Position Vector
    leader = vehicle.target             # Tracking Target / Swarm Leader
    vehList = vehicle.group             # List of Followers in Group / Swarm
    r_s = vehicle.r_safe                # Minimum Safe Vehicle Distance
    r_a = vehicle.r_avoid               # Avoidance Radius
    u_max = vehicle.u_max               # Vehicle Maximum Speed

    # APF Repulsion Parameters
    gamma = 1.2                         # Repulsion Max Scaling Factor
    vr_zero = 0.0001                    # Repulsion Velocity at r_avoid
    vr_max = gamma * u_max              # Maximum Repulsion Velocity
    beta = (r_s - r_a) / np.log(vr_zero / vr_max)   #Repulsion Decay

    # Determine Repulsion Potential from Swarm Group Vehicles
    v_r = np.zeros(3)                   # Repulsion Potential Vector
    for member in [leader, *vehList]:
        if (member.id != vehicle.id):
            p_s = member.eta[:3]        # Swarm Group Vehicle Position Vector
            d_s = p - p_s               # Distance Vector, Member to Vehicle
            ds_mag = np.linalg.norm(d_s)
            ds_hat = d_s / ds_mag
            if (ds_mag < r_a):
                # Safety Radius
                if (ds_mag <= r_s):
                    v_r += vr_max * ds_hat
                # Avoidance Radius
                else:
                    v_r += vr_max * np.exp(r_s-ds_mag / beta) * ds_hat

    return v_r

###############################################################################

def variableExpRepulsionAPF(vehicle:Vehicle)->NPFltArr:
    """
    Compute exponential APF repulsion with velocity-dependent maximum.

    Adaptive repulsion that scales based on relative velocity. Higher closing
    velocity generates stronger repulsion. This function is designed to be
    assigned to a vehicle as the GuidLaw.repulsion callable attribute as part of
    a target tracking guidance system that uses the velAPF Guidance Law.

    Parameters
    ----------
    vehicle : Vehicle
        Vehicle computing repulsion. Must have:
        
        - eta : ndarray, shape (6,)
            Own position [x, y, z, ...] in END frame.
        - velocity : ndarray, shape (3,)
            Own velocity [vx, vy, vz] in m/s.
        - id : int
            Own vehicle ID.
        - u_max : float
            Maximum vehicle speed in m/s.
        - r_safe : float
            Minimum safe distance in meters.
        - r_avoid : float
            Maximum repulsion influence radius in meters.
        - group : list of Model
            Swarm neighbor vehicles with:

            - eta : ndarray, shape (6,)
                Neighbor position.
            - id : int
                Neighbor vehicle ID.

        - target : Vehicle or Model
            Target/leader vehicle with eta and id attributes.

               
    Returns
    -------
    v_r : ndarray, shape (3,)
        Total repulsive velocity (END frame, m/s).
    
        
    Notes
    -----
    The total repulsion velocity is the sum of all repulsion velocities to each
    swarm group member.
    
    The repulsion function has a maximum value inside the Safety Radius and
    decays exponentially toward zero at the Avoidance Radius.

        v = v_max * e^((r_safety - distance_to_member)/beta)
    
    where maximum velocity is equal to the component of the vehicle velocity
    along the vector toward the group member vehicle, and beta is used to decay
    the function within the Avoidance Radius.

        v_max = v_vehicle * r_hat (dot product),
        beta = (r_safety - r_avoid) / log_e(zero_value / v_max)
    
    Because the function cannot decay to exactly zero, a 'zero_value' is used,
    which is a value small enough to produce an acceptably negligible result.
    The resulting velocity is along the vector from the swarm group member
    toward the follower.

    
    See Also
    --------
    vehicles.Remus100s.loadTargetTracking : Assigns guidance system
    variableLinearZoneAttractionAPF : Attraction complement to repulsion
    variableExpRepulsionAPF : Fixed repulsion version
    depthAPF : Depth safety constraints
    velAPF : Main guidance law combining attraction and repulsion
    """

    # Vehicle Parameters
    p = vehicle.eta[0:3]            # Vehicle Position Vector
    v = vehicle.velocity[0:3]       # Vehicle Velocity Vector
    leader = vehicle.target         # Tracking Target / Swarm Leader
    vehList = vehicle.group         # List of Followers in Group / Swarm
    r_s = vehicle.r_safe            # Minimum Safe Vehicle Distance
    r_a = vehicle.r_avoid           # Avoidance Radius

    # APF Repulsion Parameters
    v_r = np.zeros(3)               # Repulsion Potential Vector
    vr_zero = 0.0001                # Repulsion Velocity at r_avoid
    gamma = 0.1                     # Minimum Value of Repulsion Max (m/s)

    # Determine Repulsion Potential from Swarm Group Vehicles
    for member in [leader, *vehList]:
        if (member.id != vehicle.id):
            p_s = member.eta[:3]    # Swarm Group Vehicle Position Vector
            d_s = p - p_s           # Distance Vector, Member to Vehicle
            ds_mag = np.linalg.norm(d_s)
            ds_hat = d_s / ds_mag
            if (ds_mag < r_a):
                vr_max = -min(np.dot(v,d_s) / ds_mag, -gamma)
                # Safety Radius
                if (ds_mag <= r_s):
                    v_r += vr_max * ds_hat
                # Avoidance Radius
                else:
                    beta = (r_s - r_a) / np.log(vr_zero / vr_max)
                    v_r += vr_max * np.exp(r_s-ds_mag / beta) * ds_hat

    return v_r

###############################################################################

def depthAPF(vehicle:Vehicle, vel:NPFltArr)->NPFltArr:
    """
    Apply depth safety constraint via artificial potential field.
    
    Modifies desired velocity to enforce minimum distance from ocean floor
    (z=z_bed - z_safe) using repulsive APF forces.
    

    Parameters
    ----------
    vehicle : Vehicle
        Vehicle with z_max, z_bed, z_safe, eta, and velocity attributes.

        - z_max: Maximum operating depth (m).
        - z_bed: Ocean floor depth (m).
        - z_safe: Safety distance from ocean floor (m).
        - eta : [x, y, z, phi, theta, psi], vehicle position / attitude vector
        - velocity : [vx, vy, vz], vehicle velocity vector (m/s)
        - z_safe: Safety Distance from maximum depth limit (m) 

    vel : ndarray, shape (3,)
        Desired velocity vector [vx, vy, vz] in END.

         
    Returns
    -------
    vel_constrained : ndarray, shape (3,)
        Modified velocity respecting depth limits.

         
    Notes
    -----
    - Applied as final stage in velAPF() pipeline.
    - Attempts to prevent ground collision (z > z_bed - z_safe).

    When the vehicle is below the Safety Distance threshold from the Maximum
    Depth Limit, the depth component of the velocity command is cancelled out
    and any velocity the vehicle has in the downward direction is opposed in
    proportion to how far past the Safety Distance threshold the vehicle has
    travelled.
    
        v_z = -velocity_command_z - (gamma * velocity_z),
        gamma = (z - safety_depth) / safety_distance
    """
    
    # Vehicle Parameters
    z = vehicle.eta[2]         # vehicle depth (m)
    v_z = vehicle.velocity[2]  # vehicle depth velocity (m/s)
    z_max = vehicle.z_max
    z_safe = vehicle.z_safe

    # APF
    z_max = nav.maxDepthLimit(vehicle,z_max)
    vel_copy = np.copy(vel)
    vel_z = vel_copy[2]
    if ((z > (z_max-z_safe)) and ((vel_z+v_z) > 0)):
        if (vel_z > 0):
            vel_copy[2] += -vel_z
        if (v_z > 0):
            gamma = ((z - (z_max - z_safe)) / z_safe)   # Scaling Factor
            vel_copy[2] += -gamma * v_z
    
    return vel_copy

###############################################################################

def velAPF(vehicle:Vehicle)->NPFltArr:
    """
    Artificial Potential Field (APF) velocity guidance law.
    
    Combines attraction to target, repulsion from neighbors, and depth
    constraints to compute desired velocity for swarm coordination.
    

    Parameters
    ----------
    vehicle : Vehicle
        Vehicle with GuidLaw.attraction, GuidLaw.repulsion, u_max, CommNetwork.

        - GuidLaw.attraction: Function used for APF attraction
        - GuidLaw.repulsion: Function used for APF repulsion
        - u_max: Maximum vehicle speed (m/s)        
        - CommNetwork: Communication network manager

         
    Returns
    -------
    v_d : ndarray, shape (3,)
        Desired velocity in END frame, limited to u_max.

           
    Notes
    -----
    - Calls predictSwarmState() if using communication network.
    - Uses depthAPF() to enforce depth safety constraints.
    """

    # Update Swarm State Data
    if (vehicle.CommNetwork is not None):
        predictSwarmState(vehicle)

    # APF Attraction Velocity
    v_a = vehicle.GuidLaw.attraction(vehicle)
    
    # APF Repulsion Velocity
    v_r = vehicle.GuidLaw.repulsion(vehicle)
    
    # Total APF
    v_tot = v_a + v_r

    # Limit to Maximum Vehicle Speed
    u_tot = np.linalg.norm(v_tot)
    u_max = vehicle.u_max
    if (u_tot > u_max):
        v_tot = u_max * (v_tot / u_tot)
    if (u_tot == 0):
        return np.array([0., 0., 0.])
    
    # APF Depth Constraints
    v_d = depthAPF(vehicle,v_tot)

    return v_d

###############################################################################

def targetTrack(vehicle:Vehicle)->NPFltArr:
    """
    Target tracking guidance system.

    Implements a complete guidance system that coordinates velocity guidance
    into control commands via heading and depth autopilots. Converts high-level
    desired velocity into low-level control surface commands for path execution.
    This function is designed to be assigned to a vehicle as the GuidSystem
    callable attribute.
    

    Parameters
    ----------
    vehicle : Vehicle
        Vehicle object. Must have the following assigned methods:

        - GuidLaw : callable
            Core guidance algorithm implementing specific mathematical
            approaches for path generation.
        - HeadingAP : callable
            Heading autopilot implementing control commands for rudder
            defleciton.
        - DepthAP : callable
            Depth autopilot implementing control commands for stern plane
            deflection.
        - xferU2N : callable
            Function to convert speed to propeller RPM

                
    Returns
    -------
    u_control : ndarray, shape (3,)
        Control command vector [delta_r, delta_s, n] where:

        - delta_r : float
            Rudder angle in radians from HeadingAP output.
        - delta_s : float
            Stern plane angle in radians from DepthAP output.
        - n : float
            Propeller RPM command from xferU2N output.

            
    Notes
    -----
    **Workflow:**

    The function implements a desired-velocity-to-control conversion pipeline
    with configurable guidance components:
    
    1. **Velocity Guidance Computation**

        - Processes target state and environmental constraints
        - Generates desired velocity vector in END frame
        - Handles swarm coordination and target tracking

    2. **Velocity-to-Attitude Conversion**

        - Computes desired course angle (chi) from horizontal velocity
          components
        - Computes desired pitch angle (theta) from vertical velocity component
        - Maps velocity commands to vehicle attitude references for autopilot
          control

    3. **Autopilot Control Coordinator**

    - **Heading Channel**:

        - Sets vehicle.psi_d = chi (course-to-heading mapping)
        - HeadingAP(): Converts heading error to rudder deflection command

    - **Depth/Pitch Channel**:

        - Sets vehicle.theta_d = theta (pitch reference)
        - DepthAP(): Converts pitch/depth error to stern plane deflection command

    - **Speed Control**:

        - xferU2N(): Converts desired speed magnitude to propeller RPM command
    
    Each component (GuidLaw, HeadingAP, DepthAP, xferU2N) is assigned to the 
    vehicle as a callable function, allowing different implementations to be 
    plugged in without modifying this coordination workflow. This design supports 
    various velocity guidance algorithms (APF-based, geometric, reactive), 
    autopilot configurations (PID, LQR, sliding mode), and propulsion models 
    through a common interface.

    **Key Design Feature:**

    Unlike waypoint-based pathFollow(), this system operates on instantaneous 
    velocity commands, making it suitable for reactive behaviors like target 
    tracking, obstacle avoidance, and formation control where desired trajectory 
    changes dynamically based on real-time conditions.

    **Current Limitations:**

    - Direct velocity-to-attitude mapping assumes no environmental filtering
    - Course angle (chi) used directly as heading reference without drift compensation
    - Pitch angle computed geometrically without considering depth reference
    - Future versions should incorporate state estimation and wave filtering

    
    See Also
    --------
    vehicles.Remus100s.loadTargetTracking() : Assigns use of targetTrack()
    velAPF : APF velocity guidance (typical choice)
    velCB : Constant bearing velocity guidance (alternative)
    variableLinearZoneAttractionAPF : APF attraction function
    variableExpRepulsionAPF : APF repulsion function
    control.headingPID : Heading autopilot
    control.depthPID : Depth autopilot
    """

    # Desired Velocity Vector
    v_d = vehicle.GuidLaw(vehicle)
    u_d = np.linalg.norm(v_d)

    # Heading Command
    """
    Chi is a course angle, not a heading angle (psi). I will try feeding it 
    into the existing algorithms, ie the heading autopilot.
    """
    #-------------------------------
    chi = np.arctan2(v_d[1],v_d[0])     # This part needs a state estimator to
    vehicle.psi_d  = chi                # filter environmental forces
    #-------------------------------
    delta_r = vehicle.HeadingAP(vehicle)

    # Pitch Command
    """
    I'll try a simple approach and let the pitch angle be determined by the
    angle from the vector components.
    """
    #-----------------------------------------------
    theta = np.arcsin(-v_d[2]/u_d) if u_d !=0 else 0    # Needs state estimator
    vehicle.theta_d = theta                             # included here
    #-----------------------------------------------
    delta_s = vehicle.DepthAP(vehicle)

    # Propeller Command
    n = vehicle.xferU2N(u_d)

    return np.array([delta_r, delta_s, n], float)

###############################################################################
