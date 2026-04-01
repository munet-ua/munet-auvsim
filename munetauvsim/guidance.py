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
    - velocitySubsumptionCascadeAPF(vehicle) : APF velocity guidance.
    - velCB(vehicle) : Constant bearing velocity guidance law.

**APF Mission Functions:**

    - missionTargetFeedForwardAPF : Target velocity feed-forward vector.

**APF Formation Keeping Functions:**

    - formationTargetNormPolyAPF : Attraction and repulsion to target vehicle.

**APF Survival Functions:**

    - survivalGroupNormPolyAPF : Repulsion from group vehicles.

**Depth Governing Functions**

    - depthSafetyLimit(vehicle, vel) : Depth constraint repulsion.

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

log = logger.addLog('guid')
wmax = 0.6 # Inertia Coefficient Max
wmin = 0.3 # Inertia Coefficient Min
wz = 0.2 # Inertia Coefficient for Z Axis
c1max = 1.5 # Max Value for PBest Influence
c1min = 0.5 # Min Value for PBest Infuence
c2max = 1.8 # Max Value for GBest Influence
c2min = 0.2 # Min value for GBest influence
r1max = 0.6 # Max for Randomized part of PBest
r1min = 0.4 # Min for Randomized part of PBest
r2max = 0.9 # Max for Randomized part of GBest
r2min = 0.4 # Min for Randomized part of GBest

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
    velocitySubsumptionCascadeAPF : APF-based guidance
    targetTrack : Target tracking guidance system
    

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
    velocitySubsumptionCascadeAPF : Uses group state for APF forces
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

def missionTargetFeedForwardAPF(vehicle:Vehicle)->NPFltArr:
    """
    Compute feed-forward control velocity vector from leader velocity.

    Returns the leader's current velocity vector directly as the contol velocity
    command. Provides a feed-forward term that drives the vehicle to match the
    target's instantaneous velocity, independent of relative position. Only 
    takes the target vehicle into consideration. Designed to be assigned to a 
    vehicle as the GuidLaw.mission callable attribute as part of a guidance 
    system.

    Parameters
    ----------
    vehicle : Vehicle
        Follower vehicle computing the control vector. Must have:

        - target : Model or Vehicle
            Leader vehicle. Must have:

            - velocity : ndarray, shape (3,) or greater
                Leader velocity [vx, vy, vz, ...] in END frame (m/s).
                Only the first three elements are used.

    Returns
    -------
    v_c : ndarray, shape (3,)
        Control velocity vector [vx, vy, vz] in END frame (m/s). Equal to
        the leader's current velocity vector.

    Notes
    -----
    **Feed-Forward Control:**

    Returns a copy of the leader's velocity vector directly:

        v_c = target.velocity[0:3]

    As the lowest-priority layer in the subsumption architecture, this vector is
    scaled by the residual speed budget remaining after collision avoidance and
    formation keeping. When higher-priority layers are inactive, this term
    drives the follower to match the leader's velocity, providing smooth
    coordinated motion without relying solely on the reactive APF layers.

    **Role in Subsumption Architecture:**

    In a stable formation where collision avoidance and formation keeping
    produce little or no output, this feed-forward term becomes the dominant
    velocity command. This is consistent with the formation stability
    assumption: when the swarm is in formation, the follower should simply
    match the leader's velocity to maintain relative position.

    **Modularity:**

    This function implements one specific control algorithm for the
    GuidLaw.mission role. Alternative algorithms can be assigned to
    GuidLaw.mission, provided they accept a vehicle argument and return an
    ndarray of shape (3,) in the END frame. Possible alternatives include
    waypoint-tracking control vectors, mission objective vectors, or
    operator-commanded velocity inputs.

    See Also
    --------
    velocitySubsumptionCascadeAPF : Subsumption guidance law
    survivalGroupNormPolyAPF : Collision avoidance from group vehicles
    formationTargetNormPolyAPF : Formation keeping with leader vehicle
    depthSafetyLimit : Depth safety constraint applied after control vector
    """

    return vehicle.target.velocity[0:3].copy()

###############################################################################

def formationTargetNormPolyAPF(vehicle:Vehicle)->NPFltArr:
    """
    Compute formation keeping APF vector using a normalized polynomial function.

    Generates a velocity vector that drives the vehicle to maintain a preferred
    following distance from the leader using zones of different behavior, with
    repulsion and attraction goverened by a normalized polynomial APF. Produces
    repulsion when too close, attraction when too far, and a deadband neutral
    zone around the preferred following distance. Only computes APF between
    vehicle and target. Designed to be assigned to a vehicle as the
    GuidLaw.formation callable attribute as part of a guidance system.

    Parameters
    ----------
    vehicle : Vehicle
        Follower vehicle computing formation keeping. Must have:

        - eta : ndarray, shape (6,)
            Own position [x, y, z, ...] in END frame.
        - u_max : float
            Maximum vehicle speed in m/s. Sets velocity ceiling.
        - r_safe : float
            Safety radius in meters. Hard maximum repulsion applied inside
            this distance.
        - r_avoid : float
            Avoidance zone width in meters. Used to compute width of neutral
            zone and attraction zone.
        - r_follow : float
            Preferred following distance in meters. Center of neutral zone.
        - target : Vehicle
            Leader vehicle. Must have:

            - eta : ndarray, shape (6,)
                Leader position [x, y, z, ...] in END frame.

    Returns
    -------
    v_fk : ndarray, shape (3,)
        Formation keeping velocity vector in END frame (m/s). Directed along
        the radial vector between the vehicle and the leader. Magnitude ranges
        from -u_max (maximum repulsion) to +u_max (maximum attraction).

    Notes
    -----
    **Zone Geometry:**

    Five zones are defined along the radial distance r from the vehicle to
    the leader. Zone boundaries are derived from r_safe, r_avoid, and r_follow:

        - outer_n = r_follow + (0.5 * r_avoid)   # Outer neutral boundary
        - inner_n = r_follow - (0.5 * r_avoid)   # Inner neutral boundary
        - r_max   = outer_n + r_avoid            # Maximum attraction radius

    **Piecewise APF Function:**

    The velocity magnitude v(r) along r_hat (unit vector toward leader) is:

    1. **Safety Zone** (r <= r_safe)

        - v(r) = -u_max
        - Hard maximum repulsion away from leader
        - Prevents collision when dangerously close

    2. **Repulsion Zone** (r_safe < r < inner_n)

        - v(r) = -u_max * ((inner_n - r) / (inner_n - r_safe))^p_rep
        - Polynomial decay from -u_max at r_safe to 0 at inner_n
        - Exponent p_rep = 2 produces a smooth quadratic onset

    3. **Neutral Zone / Deadband** (inner_n <= r <= outer_n)

        - v(r) = 0
        - No force applied; Formation keeping APF produces no velocity input
        - Deadband centered on r_follow prevents oscillation between repulsion 
          and attraction

    4. **Attraction Zone** (outer_n < r < r_max)

        - v(r) = +u_max * ((r - outer_n) / (r_max - outer_n))^p_att
        - Polynomial increase from 0 at outer_n to u_max at r_max
        - Exponent p_att = 3 produces a fast-onset cubic pull at r_max

    5. **Maximum Attraction Zone** (r >= r_max)

        - v(r) = +u_max
        - Hard maximum attraction toward leader
        - Enables fast recovery from large separation distances

    **Modularity:**

    This function implements one specific formation keeping algorithm for the
    GuidLaw.formation role. Alternative algorithms can be assigned to
    GuidLaw.formation, provided they accept a vehicle argument and return an
    ndarray of shape (3,) in the END frame.

    **Relationship to survivalGroupNormPolyAPF:**

    Both functions use the same normalized polynomial structure, but serve
    different roles in the subsumption hierarchy. This formation keeping
    function operates only on the leader and includes both repulsion and
    attraction zones. survivalGroupNormPolyAPF operates on all group members
    (excludes leader) and uses only repulsion.

    See Also
    --------
    velocitySubsumptionCascadeAPF : Subsumption guidance law
    survivalGroupNormPolyAPF : Collision avoidance from group vehicles
    missionTargetFeedForwardAPF : Control vector from target velocity
    depthSafetyLimit : Depth safety constraint applied after formation keeping
    """

    # Vehicle Parameters
    p = vehicle.eta[0:3].copy()             # Vehicle Position Vector
    leader = vehicle.target                 # Leader Vehicle
    p_l = leader.eta[0:3].copy()            # Leader Position Vector
    u_max = vehicle.u_max                   # Maximum Vehicle Speed

    # Distance Parameters
    r_vector = p_l - p                      # Radial Vector: Vehicle to Leader
    r = np.linalg.norm(r_vector)            # Radial distance to Leader
    if (r < 1e-6):                          # Guard against Div by Zero
        return np.zeros(3)
    r_hat = r_vector / r                    # Radial Unit Vector
    
    # Zone Parameters
    r_safe = vehicle.r_safe                 # Safety Radius
    r_avoid = vehicle.r_avoid               # Avoidance Radius
    r_follow = vehicle.r_follow             # Following Radius
    inner_n = r_follow - (0.5 * r_avoid)    # Inner Neutral Boundary Distance
    outer_n = r_follow + (0.5 * r_avoid)    # Outer Neutral Boundary Distance
    r_max = outer_n + r_avoid               # Maximum Attraction Radius

    # APF Parameters
    p_rep = 2                               # Repulsion polynomial exponent
    p_att = 3                               # Attraction polynomial exponent
    
    # Safety Zone: Maximum Repulsion
    if (r <= r_safe):
        return -u_max * r_hat
    
    # Repulsion Zone: Reduce Radial Velocity
    if (r < inner_n):
        return -u_max * ((inner_n - r) / (inner_n - r_safe))**p_rep * r_hat
    
    # Neutral Zone (Deadband): Zero Repulsion and Zero Attraction
    if (r <= outer_n):
        return np.zeros(3)
    
    # Attraction Zone: Increase Radial Velocity
    if (r < r_max):
        return u_max * ((r - outer_n) / (r_max - outer_n))**p_att * r_hat
    
    # Maximum Attraction Zone
    return u_max * r_hat

###############################################################################

def survivalGroupNormPolyAPF(vehicle:Vehicle)->NPFltArr:
    """
    Compute collision avoidance APF vector with a normalized polynomial function

    Generates a repulsive velocity vector away from swarm group members that are
    within the avoidance radius. Uses a normalized polynomial repulsion function
    with a maximum at the safety radius and zero at the avoidance radius. Only
    checks vehicles that are in the group list. Designed to be assigned to a
    vehicle as the GuidLaw.survival callable attribute as part of a guidance
    system.

    Parameters
    ----------
    vehicle : Vehicle
        Vehicle computing collision avoidance. Must have:

        - eta : ndarray, shape (6,)
            Own position [x, y, z, ...] in END frame.
        - u_max : float
            Maximum vehicle speed in m/s. Sets repulsion ceiling.
        - r_safe : float
            Safety radius in meters. Maximum repulsion applied inside this
            distance.
        - r_avoid : float
            Avoidance radius in meters. Outer boundary of repulsion influence.
            No repulsion applied beyond this distance.
        - group : list of Vehicle
            Swarm group members. Each must have:

            - eta : ndarray, shape (6,)
                Member position [x, y, z, ...] in END frame.

    Returns
    -------
    v_ca : ndarray, shape (3,)
        Total collision avoidance velocity vector in END frame (m/s).
        Magnitude clamped to u_max if multiple neighbors cause the sum to
        exceed vehicle capability.


    Notes
    -----
    **Polynomial Repulsion Function:**

    For each group member within the avoidance radius, a repulsion velocity is
    computed along the radial unit vector from the member toward the vehicle.
    The total collision avoidance vector is the sum over all contributing
    members:

        v_ca = sum_i( v(r_i) * r_hat_i )

    The per-member repulsion magnitude v(r) is a piecewise function of the
    radial distance r to each group member:

    1. **Safety Zone** (r <= r_safe)

        - v(r) = u_max
        - Hard maximum repulsion applied regardless of distance
        - Provides strong barrier against collision

    2. **Repulsion Zone** (r_safe < r < r_avoid)

        - v(r) = u_max * ((r_avoid - r) / (r_avoid - r_safe))^p_rep
        - Polynomial increase as distance decreases toward r_safe
        - Exponent p_rep = 3 produces a smooth cubic onset

    3. **Outside Avoidance Radius** (r >= r_avoid)

        - v(r) = 0
        - No collision avoidance influence

    **Output Clamping:**

    When multiple neighbors contribute simultaneously, the summed vector can
    exceed u_max. The total is clamped to u_max while preserving direction:

        if norm(v_ca) > u_max: v_ca = u_max * (v_ca / norm(v_ca))

    **Modularity:**

    This function implements one specific collision avoidance algorithm for the
    GuidLaw.survival role. Alternative algorithms can be assigned to
    GuidLaw.survival, provided they accept a vehicle argument and return an
    ndarray of shape (3,) in the END frame.

    **Relationship to formationTargetNormPolyAPF:**

    Both functions use the same normalized polynomial structure, but serve
    different roles in the subsumption hierarchy. This survival function
    operates on all group members and uses only repulsion.
    formationTargetNormPolyAPF operates only on the leader and includes both
    repulsion and attraction zones.

    See Also
    --------
    velocitySubsumptionCascadeAPF : Subsumption guidance law
    formationTargetNormPolyAPF : Formation keeping layer
    missionTargetFeedForwardAPF : Control layer
    depthSafetyLimit : Depth safety constraint applied after collision avoidance
    """

    # Vehicle Parameters
    p = vehicle.eta[0:3].copy()             # Vehicle Position Vector
    u_max = vehicle.u_max                   # Maximum Vehicle Speed
    
    # Zone Parameters
    r_safe = vehicle.r_safe                 # Safety Radius
    r_avoid = vehicle.r_avoid               # Avoidance Radius

    # APF Parameters
    p_rep = 3                               # Repulsion polynomial exponent

    # Sum Repulsion from Swarm Group
    v_ca = np.zeros(3)                      # Total Collision Avoidance Vector
    
    for member in vehicle.group:
        # Distance Parameters
        p_m = member.eta[0:3].copy()        # Member Position Vector
        r_vector = p - p_m                  # Radial Vector: Member to Vehicle
        r = np.linalg.norm(r_vector)        # Radial distance to Member
        if ((r < 1e-6) or (r >= r_avoid)):  # Outside CA Zone & Div by Zero
            continue
        r_hat = r_vector / r                # Radial Unit Vector

        # Safety Zone: Maximum Repulsion
        if (r <= r_safe):
            v_ca += u_max * r_hat
        
        # Repulsion Zone: Reduce Radial Velocity
        else:
            v_ca += u_max * ((r_avoid - r) / (r_avoid - r_safe))**p_rep * r_hat
    
    # Clamp Total Collision Avoidance to Maximum Vehicle Speed
    u_ca = np.linalg.norm(v_ca)
    if (u_ca > u_max):
        return u_max * (v_ca / u_ca)
    
    return v_ca

###############################################################################

def depthSafetyLimit(vehicle:Vehicle, vel:NPFltArr)->NPFltArr:
    """
    Apply depth safety constraint.
    
    Modifies desired velocity to enforce minimum distance from ocean floor
    (z=z_bed - z_safe).
    

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
    z = vehicle.eta[2]          # vehicle depth (m)
    v_z = vehicle.velocity[2]   # vehicle depth velocity (m/s)
    dz_safe = vehicle.z_safe    # width of depth safety zone
    vel_copy = np.copy(vel)     # proposed velocity command
    vel_z = vel_copy[2]         # z-component of proposed velocity

    # Depth control geometry
    hard_floor = nav.maxDepthLimit(vehicle, vehicle.z_max)
    safety_depth = hard_floor - dz_safe

    # Apply depth filtering if vehicle below safety depth and moving down
    if ((z > safety_depth) and ((vel_z+v_z) > 0)):
        gamma = min(1.0, ((z - safety_depth) / dz_safe))   # Scaling Factor
        # Reduce downward component of proposed command
        if (vel_z > 0):
            vel_copy[2] -= gamma * vel_z
        # Apply braking against vehicles downward motion
        if (v_z > 0):
            vel_copy[2] -= gamma * v_z
    
    return vel_copy

###############################################################################

def velocitySubsumptionCascadeAPF(vehicle:Vehicle)->NPFltArr:
    """
    APF velocity guidance law with subsumption-based saturation cascade.
    
    Combines three Artificial Potential Field (APF) behaviors -- collision
    avoidance, formation keeping, and a control vector -- into a single desired
    velocity vector. Behavioral priority follows the subsumption architecture of
    Brooks [1], where higher-priority layers inhibit lower-priority layers.
    Behaviors are fused by weighted vector summation following Arkin's motor
    schema formulation [2]. The weight computation uses saturation ratios that
    cascade multiplicatively through the priority hierarchy, analogous to the
    nested null-space projections in the NSB approach of Antonelli et al. [3]
    but reduced to scalar velocity-budget residuals. This function is designed
    to be assigned to a vehicle as the GuidLaw callable attribute as part of a
    guidance system.

    Parameters
    ----------
    vehicle : Vehicle
        Vehicle object. Must have:

        - GuidLaw.survival : callable
            Computes collision avoidance velocity vector. Highest priority
            layer.
        - GuidLaw.formation : callable
            Computes formation keeping velocity vector. Middle priority layer.
        - GuidLaw.mission : callable
            Computes control / misison velocity vector. Lowest priority layer.
        - u_max : float
            Maximum vehicle speed in m/s. Defines total velocity budget.
        - CommNetwork : object or None
            Communication network manager. If not None, swarm state is
            predicted via predictSwarmState() before guidance evaluation.

    Returns
    -------
    v_d : ndarray, shape (3,)
        Desired velocity vector [vx, vy, vz] in END frame (m/s).
        Magnitude limited to u_max. Depth-constrained by depthSafetyLimit().

    Notes
    -----
    **Subsumption Architecture:**

    Behaviors are organized into a strict priority hierarchy. Each layer can
    fully suppress all layers below it by consuming the shared velocity budget.
    The total available speed u_max is treated as the budget, and each layer
    draws down a fraction proportional to its output magnitude:

    1. **Collision Avoidance** (highest layer)

        - Saturation ratio: `s_ca = norm(v_ca) / u_max`
        - Scaling weight: `w_ca = 1.0` (always full weight)
        - Residual passed down: `w_fk = 1 - (w_ca * s_ca)`
        - When `s_ca = 1`: all lower layers are fully suppressed

    2. **Formation Keeping** (middle layer)

        - Saturation ratio: `s_fk = norm(v_fk) / u_max`
        - Scaling weight: `w_fk = 1 - (w_ca * s_ca)`
        - Residual passed down: `w_m = w_fk * (1 - s_fk)`
        - Fully suppressed when collision avoidance saturates the budget

    3. **Mission** (lowest layer)

        - Scaling weight: `w_m = w_fk * (1 - s_fk)`
        - Only active when neither collision nor formation consumes the budget
        - Fully suppressed when either higher layer saturates the budget

    **Total Velocity:**

        v_tot = (w_ca * v_ca) + (w_fk * v_fk) + (w_m * v_m)

    Clamped to u_max if the total magnitude exceeds the vehicle limit, then
    passed through depthSafetyLimit() for depth safety enforcement.

    The subsumption model allows higher-priority behaviors to dominate when
    needed: when collision avoidance is active it progressively suppresses
    formation keeping, and both suppress the mission vector. This prevents
    lower-priority behaviors from counteracting safety-critical responses.

    The general N-layer pattern:

    .. code-block:: none

        w[0] = alpha
        w[1] = 1.0 - (w[0] * s[0])
        w[2] = w[1] * (1.0 - s[1])
        w[3] = w[2] * (1.0 - s[2])
        ...
    

    **Full Suppression:**

    When `s_ca = 1` (collision avoidance saturates u_max), the formation keeping
    residual `w_fk = 0`, which in turn forces `w_m = 0`. Both lower layers are
    completely inhibited and the vehicle behaves purely as a collision avoidance
    agent, consistent with subsumption's hard behavioral priority guarantee.

    **Multiplicative Cascade:**

    The mission weight `w_m = w_fk * (1 - s_fk)` means the mission layer
    is doubly suppressed: first by how much collision avoidance consumes the
    budget (via `w_fk`), then by how much formation keeping consumes what
    remains (via `1 - s_fk`). This nested structure mirrors the cascaded 
    null-space projections of the NSB framework, but operating on scalar budget 
    residuals.
 
    **Communication:**

    If vehicle.CommNetwork is not None, predictSwarmState() is called at the
    start of each evaluation to propagate estimated member positions forward
    using a prediction model.

    References
    ----------
    [1] R. Brooks, "A robust layered control system for a mobile robot," 
    in IEEE Journal on Robotics and Automation, vol. 2, no. 1, pp. 14-23, 
    March 1986, doi: 10.1109/JRA.1986.1087032.

    [2] R. C. Arkin, "Motor schema-based mobile robot navigation," 
    The International Journal of Robotics Research. 1989;8(4):92-112. 
    doi:10.1177/027836498900800406

    [3] G. Antonelli, F. Arrichiello and S. Chiaverini, "Experiments of 
    Formation Control With Multirobot Systems Using the Null-Space-Based 
    Behavioral Control," in IEEE Transactions on Control Systems Technology, 
    vol. 17, no. 5, pp. 1173-1182, Sept. 2009, 
    doi: 10.1109/TCST.2008.2004447.

    See Also
    --------
    survivalGroupNormPolyAPF : Collision avoidance from group vehicles
    formationTargetNormPolyAPF : Formation keeping with leader vehicle
    missionTargetFeedForwardAPF : Control/tracking vector from leader velocity
    depthSafetyLimit : Depth safety constraint enforcement
    predictSwarmState : Swarm state prediction during communication gaps
    targetTrack : Target tracking guidance system
    """

    # Maximum Vehicle Speed
    u_max = vehicle.u_max

    # Update Swarm State Data
    if (vehicle.CommNetwork is not None):
        predictSwarmState(vehicle)

    # Evaluate Velocity Components
    v_ca = vehicle.GuidLaw.survival(vehicle)        # Collision Avoidance
    v_fk = vehicle.GuidLaw.formation(vehicle)       # Formation Keeping
    v_m = vehicle.GuidLaw.mission(vehicle)          # Mission

    # Evaluate Component Saturation Ratios
    s_ca = min(np.linalg.norm(v_ca) / u_max, 1.0)
    s_fk = min(np.linalg.norm(v_fk) / u_max, 1.0)

    # Evaluate Subsumption Weights
    w_ca = 1.0                                      # 1.0 = No scaling
    w_fk = 1.0 - (w_ca * s_ca)
    w_m = w_fk * (1.0 - s_fk)
    
    # Total Velocity
    v_tot = (w_ca * v_ca) + (w_fk * v_fk) + (w_m * v_m)

    # Limit to Maximum Vehicle Speed
    u_tot = np.linalg.norm(v_tot)
    if (u_tot > u_max):
        v_tot = u_max * (v_tot / u_tot)
    
    # APF Depth Constraints
    v_d = depthSafetyLimit(vehicle,v_tot)

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
    velocitySubsumptionCascadeAPF : APF velocity guidance
    velCB : Constant bearing velocity guidance
    missionTargetFeedForwardAPF : Target velocity feed-forward
    formationTargetNormPolyAPF : Attraction and repulsion to target vehicle
    survivalGroupNormPolyAPF : Repulsion from all group vehicles
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

#Adding PSO System from Scratch 1/13/26

def PSOAttraction(vehicle:Vehicle)->NPFltArr:
    """
    Returns attraction velocity vector of a vehicle using the Particle Swarm Optimization (PSO) algorithm.

    Parameters
    ----------
    vehicle:
        Vehicle object. Must have the listed vehicle attributes defined.
    
    Vehicle Attributes
    ------------------
    eta : [x, y, z, phi, theta, psi]
        Vehicle position / attitude vector

    Returns:
    --------
    NPFltArr
        A NumPy array representing the attraction velocity vector for the vehicle.

    Notes:
    ------
    - The PSO algorithm combines inertia, cognitive, and social components to determine the new velocity:
        1. **Inertia** (w * v): Maintains momentum from the previous velocity.
        2. **Cognitive Component** (c1 * r1 * (pbest - x)): Guides the vehicle towards its personal best position.
        3. **Social Component** (c2 * r2 * (gbest - x)): Attracts the vehicle towards the global best position in the swarm.
    - Coefficients:
        - c1 (personal learning coefficient): Controls the influence of the personal best position.
        - c2 (global learning coefficient): Controls the influence of the global best position.
        - w (inertia coefficient): Balances the trade-off between exploration (high w) and exploitation (low w).
    - Random coefficients (r1, r2) add stochasticity to prevent premature convergence.
    --------
    """

    x = np.copy(vehicle.eta[0:3])          # Vehicle Position Vector
    v = np.copy(vehicle.velocity[0:3])       # Vehicle Velocity Vector

    r1 = r1min + (r1max - r1min) * np.random.rand(2) # random coefficients (in range [rmin-rmax))
    r2 = r1min + (r1max - r2min) * np.random.rand(2) # random coefficients (in range [rmin-rmax))

    c1 = (c1max-c1min) * ((np.copy(vehicle.simTime)-vehicle.clock)/np.copy(vehicle.simTime)) + c1min #Calculate C1

    #if found source, or if in bad area compared to others, go to better location
    if vehicle.gbest[3] > .01 or vehicle.gbest[3] > (np.copy(vehicle.pbest[3]) * 5):
        c1 = 0 #This makes the vehicles approach more direct to the objective with 5 vehicles, 10 they all piled up

    c2 = (c2min-c2max) * ((np.copy(vehicle.simTime)-vehicle.clock)/(np.copy(vehicle.simTime))) + c2max #Calculate C2

    w = wmax - ((np.copy(vehicle.simTime)-vehicle.clock)/(np.copy(vehicle.simTime))) * (wmax - wmin) #Calculate w
    #If at the source, slow down motors to preserve energy

    if vehicle.pbest[3] == 0:
        w = 1
        c1 = 0

    v[:2] = w*np.copy(v[:2]) + r1*c1*(np.copy(vehicle.pbest[:2])-np.copy(x[:2])) + r2*c2*(np.copy(vehicle.gbest[:2])-np.copy(x[:2]))
   
    v[2] = wz*np.copy(v[2]) + c1*(np.copy(vehicle.pbest[2])-np.copy(x[2])) + c2*(np.copy(vehicle.gbest[2])-np.copy(x[2]))

    if vehicle.pbest[3] > 0.01 and vehicle.pbest[3] < 1:
        v[:3] = v[:3] / 10
    return v

def PSORep(vehicle:Vehicle, k_rep=80.0)->NPFltArr:
        """
        Compute the repulsive Artificial Potential Field (APF) force
        using inverse sqaure law to avoid collisions.

        Parameters:
        swarm  : List of SwarmAgent instances (all swarm members).
        r_s : Safe distance threshold (repulsion activates below this).
        k_rep  : Repulsion strength constant.
        Returns:
        v_r : The total repulsive force as a 3D numpy array.
        """
        # Vehicle Parameters
        p = np.copy(vehicle.eta[:3]) + np.copy(vehicle.velocity[:3])*0.02        # Vehicle Position Vector
        vehList = vehicle.group         # List of Followers in Group / Swarm
        r_s = vehicle.r_safe            # Minimum Safe Vehicle Distance

        v_r = np.zeros(3)  # Initialize repulsive force vector

        for member in vehList:
        # for member in vehicle.group:
            if (member.id != vehicle.id):
                p_s = member.eta[:3] + member.velocity[:3]*0.02    # Swarm Group Vehicle Position Vector
                # Compute distance vector and magnitude
                dist = p - p_s
                dist_mag = np.linalg.norm(dist)

                # Apply repulsive force if within the safe zone
                if 0 < dist_mag < r_s:
                    unit_vector = dist / dist_mag  # Direction of repulsion
                    force_magnitude = k_rep #/ (dist_mag ** 2)  # Inverse square law
                    v_r += force_magnitude * unit_vector  # Accumulate forces

        v_r[2] = 0

        return v_r

###############################################################################

def PSOLaw(vehicle:Vehicle)->NPFltArr:
    """
    Guidance law for a vhelicle using PSO. Generates attraction vector using PSO function
    and repulsion vector using variableExpRepulsionAPF, then adds them together.

    Parameters
    ----------
    vehicle:
        Vehicle object. Must have the following vehicle attributes defined.

    Vehicle Attributes
    ------------------
    GuidLaw.attraction:
        Function used for APF attraction
    GuidLaw.repulsion:
        Function used for APF repulsion
    u_max:
        Maximum vehicle speed (m/s)  
    """

    # APF Attraction Velocity
    v_a = vehicle.GuidLaw.attraction(vehicle)
    
    # APF Repulsion Velocity
    v_r = vehicle.GuidLaw.repulsion(vehicle)
    # if np.linalg.norm(v_a) == 0:
    #     print('zeroed')
    
    # Total APF
    v_tot = v_a + v_r
    if vehicle.pbest[3] > .4: #If high concentration, stop
        v_tot = [0,0,0]

    # Limit to Maximum Vehicle Speed
    u_tot = np.linalg.norm(v_tot)
    u_max = vehicle.u_max
    if (u_tot > u_max):
        v_tot = u_max * (v_tot / u_tot)
    if (u_tot == 0):
        return np.array([0., 0., 0.])
    
    # APF Depth Constraints
    v_d = depthAPF(vehicle,v_tot)

    # v_d[2] = 0 # limit movement to 2D

    return v_d

def PSOSystem(vehicle:Vehicle)->NPFltArr:
    """
    Guidance system for a vehicle utilizing PSO. Calculates
    desired velocity vectors using PSOLaw guidance law. Returns the
    stabilizing control commands for rudder, stern, and thrust.

    Parameters
    ----------
    vehicle:
        Vehicle object. Must have the following vehicle attributes defined.

    Vehicle Attributes
    ------------------
    GuidLaw:
        Guidance law  
    HeadingAP:
        Heading auto pilot
    DepthAP:
        Depth auto pilot
    xferU2N:
        Function to convert speed to propeller RPM
        
    Notes
    -----
    Current development does not include a wave filter, so the vehicle is
    subject to destabilizing forces due to ocean conditions. See Fossen
    Handbook Ch.10 & Ch.12 lecture notes.
    
    #TODO: Develop a wave filter (Heading/Depth Observer / State Estimator)!

    
    //JPC 07/2023
    """

    #Ask about this later 11/13
    # if vehicle.pbest[3] > 0.01:
    #     vehicle.loadPathFollowing()
    #     source = Waypoint(*vehicle.pbest[:3])
    #     vehicle.wpt = source
    #     v_d = vehicle.GuidLaw(vehicle, vehicle.eta, *source)
        

    # Desired Velocity Vector
    v_d = vehicle.GuidLaw(vehicle)
    u_d = np.linalg.norm(v_d)
    if u_d == 0:  # If zero, opposite of current velocity to stop
        return -vehicle.velocity

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
###################################################################
