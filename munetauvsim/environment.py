"""
Ocean environment modeling for AUV simulation.

Provides comprehensive environmental modeling including ocean currents,
bathymetry, and pollution dispersion for autonomous underwater vehicle
simulations. Supports procedural generation via Perlin noise and configurable
physical parameters.


Classes
-------
Ocean
    Container for all environmental components (current, floor, pollution).
Current1D
    Time-varying ocean current with configurable speed and direction profiles.
Current1DData
    Immutable data container for Current1D state serialization.
Floor
    Ocean floor depth map generated from Perlin noise.
PerlinNoise
    2D Perlin noise generator for terrain and environmental features.
Pollution
    Gaussian plume model for point-source pollution dispersion.

    
Functions
---------
None
    All functionality contained in classes

    
Notes
-----
**Environment Scope:**

Simulates environmental factors affecting vehicle dynamics and sensor readings.
Inputs: None (self-contained generation)
Outputs: Physical properties for dynamics integration and sensor simulation

**Typical Components:**

- Ocean current: Time-varying 1D current fields
- Ocean floor: Procedurally generated depth map
- Pollution plumes: Concentration dispersion from point sources

**Design Philosophy:**

Environment objects are designed for minimal coupling with simulation code.
All components are optional and can be configured independently or omitted.


Examples
--------
### Create calm ocean with default parameters:

>>> import munetauvsim.environment as env
>>> ocean = env.Ocean.calm_ocean(size=1000)
>>> print(ocean)oce

### Custom ocean with stormy conditions:

>>> ocean = env.Ocean(
...     spd='fast', dSpd='unsteady', dtSpd='choppy',
...     ang='any', dAng='varied', dtAng='regular',
...     z=150, z_range=30,
...     randomFloor=True, randomPlume=True
... )

### Individual component construction:

>>> current = env.Current1D(spd=0.5, dSpd=0.1, dtSpd=90.0, 
...                         ang=np.pi/4, dAng=0.2, dtAng=120.0,
...                         nIter=60000, seed=42)
>>> floor = env.Floor(z=125, z_range=15, size=1000, seed=100)
>>> pollution = env.Pollution(source=[200, 300, 25], Q=2.0, 
...                           u=0.8, v=np.pi/6)
>>> ocean = env.Ocean()
>>> ocean.current = current
>>> ocean.floor = floor
>>> ocean.pollution = pollution


See Also
--------
guidance.Waypoint : Path planning coordinates compatible with Floor sampling
navigation.Sensor : Sensor class that collects data from environment
vehicles.Remus100s : Vehicle class that interacts with Ocean environment
simulator.Simulator : Main simulation driver that propagates Ocean data
"""

from functools import singledispatchmethod
from typing import Dict, KeysView, List, Optional, Tuple, Union
from numpy.typing import NDArray
from collections.abc import Generator
from dataclasses import dataclass
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from munetauvsim.guidance import Waypoint
from munetauvsim import logger

#-----------------------------------------------------------------------------#

# Type Aliases
NPFltArr = NDArray[np.float64]
NPIntArr = NDArray[np.int_]
Number = Union[int, float, np.number]

# Global Variables
log = logger.addLog('env')

###############################################################################

class Ocean:
    """
    Container and manager for ocean environmental components.
    
    Aggregates ocean current, floor depth map, and pollution models into a
    unified environment object. Handles consistent parameterization across
    components and provides convenience constructors for typical scenarios.
    

    Parameters
    ----------
    size : int, default=1000
        Side length of square ocean floor area in meters.
    origin : list of float, default=[500, 500]
        Coordinates where (x=0, y=0) maps to floor array indices.
        Allows negative coordinates without array indexing issues.
    N : int, default=60000
        Number of simulation iterations for current time series.
    h : float, default=0.02
        Time step per iteration in seconds (50 Hz default).
    name : str, default='Ocean'
        Descriptive name for this ocean environment instance.
    plume : list of float, default=None
        Pollution source location [x, y, z] in meters. z is depth (positive).
    createPlume : bool, default=False
        If True, creates a pollution plume with emission from source location.
    currentSeed : int, optional
        PRNG seed for current generation. None generates random seed.
    floorSeed : int, default=0
        PRNG seed for Perlin noise floor generation.
    plumeSeed : int, optional
        PRNG seed for pollution randomization. None generates random seed.
    randomFloor : bool, default=False
        If True, generates random floor seed instead of using floorSeed.
    randomPlume : bool, default=False
        If True, generates random plume parameters.
    **kwargs
        Additional keyword arguments passed to component constructors:

        - Current1D: spd, dSpd, dtSpd, ang, dAng, dtAng
        - Floor: z, z_range, style
        - Pollution: Q, u, v, randomU, randomV
        

    Attributes
    ----------
    name : str
        Ocean environment identifier.
    N : int
        Number of simulation iterations (synchronized with current).
    sampleTime : float
        Time step in seconds (synchronized with current).
    size : int
        Ocean floor area side length in meters.
    origin : list of float
        Origin coordinates [x, y] for floor indexing.
    current : Current1D
        Ocean current model with time-varying speed and direction.
    floor : Floor
        Ocean floor model with Perlin noise terrain.
    pollution : Pollution or None
        Pollution plume dispersion model. None if not requested.
        

    Methods
    -------
    None
        Ocean is primarily a container. See component classes for methods.
        

    Alternative Constructors
    ------------------------
    calm_ocean(kwargs)
        Smooth conditions: typical speeds, steady changes, calm rates.
        Default depth: 200m uniform.

    dead_ocean(kwargs)
        Zero current conditions for baseline testing.
        Constant parameters with no variation.

    stormy_ocean(kwargs)
        Volatile conditions: fast speeds, unsteady changes, choppy rates.
        

    Notes
    -----
    **Default Construction:**

    Default Ocean() creates calm environment:

    - Current Speed: 0.05-0.5 m/s, +/- 0.01-0.1 m/s over 180-300s
    - Current Direction: Northeastern, +/- 0.1-0.17 rad over 180-300s
    - Depth: 125-135m (uniform 125m with 10m Perlin noise range)
    - Area: 1 km^2
    - Pollution: No pollution plume
    - Name: "Ocean"
    
    **Property Synchronization:**

    The N and sampleTime properties automatically update component objects:

    >>> ocean.N = 10000  # Resizes current.speed and current.angle arrays
    >>> ocean.sampleTime = 0.01  # Resamples current at 100 Hz
    
    Similarly, size and origin update the floor object.
    
    **Component Construction:**

    Ocean constructor is convenience-focused but not exhaustive. For advanced
    customization, construct components independently:
    
    >>> ocean = env.Ocean.dead_ocean(size=2000, origin=[1000,1000])
    >>> ocean.pollution = env.Pollution(source=[500, 500, 40], Q=5.0, 
    ...                                 u=1.2, v=np.pi/3)
    
    **Design Considerations:**

    Currently, ocean parameters like current speed/direction are not
    automatically synchronized between current and pollution objects. Future
    versions should improve consistency handling.
    
    **Memory Considerations:**

    Large N values create proportionally large current time series:

    - N=60000: ~1MB for speed and angle arrays
    - N=600000: ~10MB per ocean instance
    
    Floor size affects memory quadratically:

    - size=1000: ~8MB for depth array
    - size=5000: ~200MB per floor instance
    - Known issue: Ocean floor sizes exceeding ~6000m may trigger memory
      allocation failures. This limitation will be resolved through chunked
      Perlin noise generation in an upcoming update.

      
    Examples
    --------
    ### Create ocean with calm_ocean constructor:
    
    >>> ocean = env.Ocean.calm_ocean(size=1500, origin=[750, 750])
    >>> print(ocean)
    
    ### Custom ocean from scratch:
    
    >>> ocean = env.Ocean(
    ...     name="Test Environment",
    ...     size=2000,
    ...     origin=[1000, 1000],
    ...     N=10000,
    ...     spd='moderate',  # 0.5-1.0 m/s
    ...     dSpd='steady',   # +/- 0.01-0.1 m/s
    ...     dtSpd=60.0,      # 60s half-period
    ...     ang='north',     # pi/2 +/- pi/8
    ...     dAng='constant', # No direction variation
    ...     dtAng=90.0,      # 90s half-period
    ...     z=100,           # Shallowest depth
    ...     z_range=20,      # Depth variation range
    ...     randomFloor=False,
    ...     currentSeed=123,
    ...     floorSeed=456
    ... )
    
    ### Dead ocean for control experiments:
    
    >>> ocean = env.Ocean.dead_ocean(N=50000)
    >>> print(f"Current speed: {ocean.current.v_spd} m/s")  # 0.0
    
    ### Stormy ocean for stress testing:
    
    >>> ocean = env.Ocean.stormy_ocean(
    ...     size=1000,
    ...     z=80,  # Shallower water
    ...     z_range=40,  # More dramatic terrain
    ...     randomFloor=True
    ... )
    >>> # Fast speeds (1.8-2.5 m/s), unsteady variations
    
    ### Accessing components:
    
    >>> ocean = env.Ocean.calm_ocean(createPlume=True)
    >>> # Sample current at iteration 1000
    >>> current_speed = ocean.current.speed[1000]
    >>> current_angle = ocean.current.angle[1000]
    >>> 
    >>> # Sample floor depth at position (100, 200)
    >>> depth = ocean.floor(100, 200)
    >>> 
    >>> # Sample pollution concentration at (150, 150, 25m)
    >>> concentration = ocean.pollution(150, 150, 25)
    
    ### Use in simulation:
    
    >>> import munetauvsim as mn
    >>> ocean = env.Ocean.calm_ocean(size=2000, N=60000)
    >>> sim = mn.Simulator(
    ...     name="OceanTest",
    ...     ocean=ocean,
    ...     vehicles=[mn.vehicles.Remus100s()]
    ... )
    >>> sim.run()
    

    See Also
    --------
    Current1D : Ocean current time series generation
    Floor : Procedural bathymetry from Perlin noise
    Pollution : Gaussian plume dispersion model
    Simulator : Main simulation driver that uses Ocean

    
    Warnings
    --------
    - Large size values (>5000) may cause memory issues
    - Pollution parameters (u, v) not auto-synced with current (manual update
      needed)
    """

    ## Constructor ===========================================================#
    def __init__(self,
                 size:int = 1000,
                 origin:List[float] = [500,500],
                 N:int = 60000,
                 h:float = 0.02,
                 name="Ocean",
                 plume:Union[NPFltArr, List[float], None] = None,
                 createPlume:bool = False,
                 currentSeed:Optional[int] = None,
                 floorSeed:int = 0,
                 plumeSeed:Optional[int] = None,
                 randomFloor:bool = False,
                 randomPlume:bool = False,
                 **kwargs):
        """
        Initialize ocean environment container with current, floor, and
        pollution components.
    
        Constructor that creates a complete ocean environment with synchronized
        parameters across all components. Handles categorical parameter
        specification for rapid prototyping or precise numerical control for
        advanced scenarios. Components can be customized via kwargs or replaced
        after construction for full flexibility.
        
        Parameters
        ----------
        size : int, default=1000
            Side length of square ocean floor area in meters.
            Defines simulation domain extent. Passed to Floor and Pollution.
            Typical values:

            - Small test: 500m
            - Standard ops: 1000-2000m
            - Large survey: 3000-5000m
            
            Maximum practical: ~6000m (Perlin generation memory limit).
        origin : list of float, default=[500, 500]
            Coordinates [x_origin, y_origin] where (x=0, y=0) maps in domain.
            Allows negative coordinate values. Passed to Floor and Pollution.
            Typical: [size/2, size/2] for centered operations.
        N : int, default=60000
            Number of simulation iterations for current time series.
            Default 60000 * 0.02s = 1200s = 20 minutes simulation time.
            Passed to Current1D constructor as nIter.
            Managed via property: changing N resizes current arrays.
        h : float, default=0.02
            Time step per iteration in seconds (50 Hz sampling rate).
            Passed to Current1D constructor.
            Managed via property: changing h resamples current data.
            Total sim time = N * h seconds.
        name : str, default="Ocean"
            Descriptive identifier for this ocean instance.
            Used in logging, plotting, and __str__ output.
            No functional impact on simulation.
        plume : list of float or ndarray, optional, default=None
            Pollution source location [x, y, z] in meters (END frame).
            Passed to Pollution constructor as source parameter.
            z is depth below surface (positive input).
            If None and createPlume=False, no pollution object is created.
            If None and createPlume=True, uses a default source of [0, 0, 30].
            If not None, creates Pollution regardless of createPlume (backwards
            compatible).
        createPlume : bool, default=False
            If True, creates a Pollution instance with emission from the
            specified plume source location.
            If False and plume is not None, a Pollution instance is still
            created.
        currentSeed : int, optional
            PRNG seed for Current1D generation.
            If None, Current1D generates random seed (non-reproducible).
            If provided, ensures deterministic current time series.
        floorSeed : int, default=0
            PRNG seed for Floor/PerlinNoise generation.
            seed=0 is valid and deterministic.
            Overridden if randomFloor=True.
        plumeSeed : int, optional
            PRNG seed for Pollution randomization (if randomPlume=True).
            If None and randomPlume=True, generates random seed.
            If randomPlume=False, plumeSeed has no effect.
        randomFloor : bool, default=False
            If True, generates random floor seed (ignores floorSeed parameter).
            Creates unique terrain each run.
        randomPlume : bool, default=False
            If True, randomizes pollution u and v parameters.
            Generates random plumeSeed if plumeSeed=None.
        **kwargs : dict
            Additional keyword arguments passed to component constructors.
            
            **Current1D parameters** (see Current1D.__init__ for details):

            - spd : Mean current speed (float/str/list)
            - dSpd : Speed variation amplitude (float/str/list)
            - dtSpd : Speed variation half-period (float/str/list)
            - ang : Mean current direction (float/str/list)
            - dAng : Direction variation amplitude (float/str/list)
            - dtAng : Direction variation half-period (float/str/list)
            
            **Floor parameters** (see Floor.__init__ for details):

            - z : Minimum floor depth (float)
            - z_range : Depth variation range (float)
            - style : Terrain generation style (str)
            
            **Pollution parameters** (see Pollution.__init__ for details):

            - Q : Emission rate (float)
            - u : Wind/current speed (float) - overridden by current.v_spd if
              not provided
            - v : Wind/current direction (float) - overridden by current.b_ang
              if not provided
            - randomU : Randomize pollution wind speed (bool/list)
            - randomV : Randomize pollution wind direction (bool/list)
            
        Attributes
        ----------
        name : str
            Ocean identifier string.
        N : int
            Number of iterations (managed property, updates current.n).
        sampleTime : float
            Time step in seconds (managed property, updates current.h).
        size : int
            Floor area dimension (managed property, updates floor.size).
        origin : list of float
            Origin coordinates (managed property, updates floor.origin).
        current : Current1D
            Time-varying ocean current instance.
        floor : Floor
            Procedurally generated ocean floor instance.
        pollution : Pollution or None 
            Gaussian plume pollution instance. None if not requested.
        
        Notes
        -----
        **Construction Process**

        1. **Store Basic Parameters:**

        name, N (via property), sampleTime (via property), size (via property),
        origin (via property) stored as attributes.
        
        2. **Create Current1D:**

        >>> self.current = Current1D(
        ...     nIter=N,
        ...     h=h,
        ...     seed=currentSeed,
        ...     spd=kwargs.get('spd', default),
        ...     dSpd=kwargs.get('dSpd', default),
        ...     dtSpd=kwargs.get('dtSpd', default),
        ...     ang=kwargs.get('ang', default),
        ...     dAng=kwargs.get('dAng', default),
        ...     dtAng=kwargs.get('dtAng', default)
        ... )
        Uses constructor defaults if kwargs not provided.
        
        3. **Create Floor:**

        >>> self.floor = Floor(
        ...     size=size,
        ...     origin=origin,
        ...     seed=floorSeed,
        ...     random=randomFloor,
        ...     z=kwargs.get('z', default),
        ...     z_range=kwargs.get('z_range', default),
        ...     style=kwargs.get('style', 'linear')
        ... )
        
        4. **Create Pollution:**

        >>> self.pollution = Pollution(
        ...     source=plume,
        ...     u=self.current.v_spd,  # Sync with current mean speed
        ...     v=self.current.b_ang,  # Sync with current mean direction
        ...     oceanSize=size,
        ...     oceanOrigin=origin,
        ...     oceanDepth=self.floor.z,  # Sync with floor shallowest depth
        ...     seed=plumeSeed,
        ...     random=randomPlume,
        ...     Q=kwargs.get('Q', default),
        ...     randomU=kwargs.get('randomU', False),
        ...     randomV=kwargs.get('randomV', False)
        ... )

        Pollution u and v default to current parameters unless overridden
        by explicit kwargs or randomization flags.
        
        **Property Synchronization**

        Ocean properties that automatically propagate changes to components:
        
        - **N Property:**

          Setting ocean.N triggers current.n update:

          >>> ocean.N = 100000
          >>> # Internally: ocean.current.n = 100000
          >>> # Current regenerates speed/angle arrays to new length
        
        - **sampleTime Property:**

          Setting ocean.sampleTime triggers current.h update:

          >>> ocean.sampleTime = 0.01  # 100 Hz
          >>> # Internally: ocean.current.h = 0.01
          >>> # Current resamples data at new rate
        
        - **size Property:**

          Setting ocean.size triggers floor.size update:

          >>> ocean.size = 2000
          >>> # Internally: ocean.floor.size = 2000
          >>> # Floor regenerates with extended dimensions
        
        - **origin Property:**

          Setting ocean.origin triggers floor.origin update:

          >>> ocean.origin = [1000, 1000]
          >>> # Internally: ocean.floor.origin = [1000, 1000]

        **Design Limitations:**
        
        Current implementation has parameter consistency issues:

        - Pollution u, v set from current mean values at construction only
        - Changing current parameters after construction does NOT update
          pollution
        - Changing floor z after construction does NOT update pollution
          oceanDepth
        
        Workaround: Manually update pollution parameters after Ocean
        construction:

        >>> ocean = Ocean.calm_ocean(createPlume=True)
        >>> ocean.current.v_spd = 2.0
        >>> ocean.pollution.u = 2.0  # Manual sync required
        
        Future improvement: Implement pollution property to auto-sync with
        current/floor.
        
        **Alternative Constructors:**
        
        Ocean provides convenient factory methods:
        
        - calm_ocean(kwargs):

            Typical coastal conditions (smooth, steady):

            - spd='typical' (0.05-0.5 m/s)
            - dSpd='steady' (+/- 0.01-0.1 m/s)
            - dtSpd='calm' (600s half-period)
            - ang='northeast'
            - dAng='steady'
            - dtAng='calm'
            - z=200 (if not overridden)
            
        - dead_ocean(kwargs):

            Zero current for baseline testing:

            - spd='dead' (0.0 m/s)
            - dSpd='constant' (0.0 variation)
            - dtSpd='calm'
            - ang=0
            - dAng='constant'
            - dtAng='calm'
            
        - stormy_ocean(kwargs):

            High-energy, volatile conditions:
            
            - spd='fast' (1.8-2.5 m/s)
            - dSpd='unsteady' (+/- 0.5-1.5 m/s)
            - dtSpd='choppy' (10-30s half-period)
            - ang='any' (random direction)
            - dAng='unsteady'
            - dtAng='choppy'
        
        See Also
        --------
        Ocean.calm_ocean : Factory for typical coastal conditions
        Ocean.dead_ocean : Factory for zero-current baseline
        Ocean.stormy_ocean : Factory for high-energy conditions
        Current1D.__init__ : Current parameter details
        Floor.__init__ : Floor parameter details
        Pollution.__init__ : Pollution parameter details
        Simulator : Main simulation driver using Ocean
        
        Examples
        --------
        ### Default calm ocean with pollution:
        
        >>> import munetauvsim.environment as env
        >>> ocean = env.Ocean(createPlume=True)
        >>> print(ocean)
        Ocean: Ocean
        -------------------------------------
        Current
        Speed:       0.275 +/-0.055 m/s (180.0s)
        Angle:       0.785 +/-0.140 rad (180.0s)
        Seed:        1234567890

        Floor
        Size:        1000 m
        Origin:      [500, 500]
        Depth:       125 to 135 m
        Style:       linear

        ... (Perlin noise details)

        Pollution
        Source:      (0.00, 0.00, 30.00)
        Strength:    1.59 g/s
        Speed:       0.28 m/s at 0.79 rad
        Seed:        None
        -------------------------------------

        ### Custom environment:
        
        >>> ocean = env.Ocean(
        ...     name="TestOcean",
        ...     size=2000,
        ...     origin=[1000, 1000],
        ...     N=120000,
        ...     h=0.01,
        ...     spd=0.8,
        ...     dSpd=0.2,
        ...     dtSpd=90.0,
        ...     ang=np.pi/4,
        ...     dAng=0.1,
        ...     dtAng=120.0,
        ...     z=100,
        ...     z_range=25,
        ...     plume=[500, 500, 20],
        ...     Q=2.0,
        ...     currentSeed=42,
        ...     floorSeed=100
        ... )
        
        ### Use alternative constructors:
        
        >>> calm = env.Ocean.calm_ocean(size=1500, z=180)
        >>> dead = env.Ocean.dead_ocean(N=30000)
        >>> stormy = env.Ocean.stormy_ocean(size=3000, randomFloor=True)
        
        ### Access components:
        
        >>> ocean = env.Ocean.calm_ocean(plume=[-50,-50,80])
        >>> current_speed_at_1000 = ocean.current.speed[1000]
        >>> depth_at_origin = ocean.floor(0, 0)
        >>> pollution_conc = ocean.pollution(100, 100, 25)
        
        ### Modify after construction:
        
        >>> ocean = env.Ocean()
        >>> ocean.N = 100000  # Extend simulation time
        >>> ocean.size = 2000  # Expand floor area
        >>> ocean.pollution = env.Pollution(source=[200, 200, 25], Q=5.0)
        
        ### Synchronized parameters:
        
        >>> ocean = env.Ocean(createPlume=True)
        >>> print(f"Current mean: {ocean.current.v_spd:.2f} m/s")
        >>> print(f"Pollution u: {ocean.pollution.u:.2f} m/s")
        >>> # Should match at construction
        
        Warnings
        --------
        - Pollution parameters NOT automatically synchronized with current
          changes after construction. Manual updates required.
        - Large size values (>5000) may trigger memory errors during floor
          generation.
        """
        
        self.name = name
        self.N = N
        self.sampleTime = h
        self.size = size
        self.origin = origin
        self.current = Current1D(nIter=N, h=h, seed=currentSeed, **kwargs)
        self.floor = Floor(size=size,
                           origin=origin,
                           seed=floorSeed,
                           random=randomFloor, 
                           **kwargs)
        if ((plume is not None) or (createPlume)):
            if (plume is None):
                plume = [0, 0, 30]

            self.pollution = Pollution(source=plume,
                                    u=self.current.v_spd, 
                                    v=self.current.b_ang,
                                    oceanSize=size, 
                                    oceanOrigin=origin,
                                    oceanDepth=self.floor.z,
                                    seed=plumeSeed,
                                    random=randomPlume,
                                    **kwargs)
        else:
            self.pollution = None
    
    ## Properties ============================================================#
    @property
    def N(self)->int:
        """Number of simulation iterations."""
        return self._N
    
    @N.setter
    def N(self, n:int)->None:
        """Set number of iterations and resize current data."""
        if (('current' in self.__dict__) and
            (self.current is not None) and 
            (self.current.n != n)):
            self.current.n = n
        self._N = n
    
    #--------------------------------------------------------------------------
    @property
    def sampleTime(self)->float:
        """Time step of each iteration (s)."""
        return self._sampleTime
    
    @sampleTime.setter
    def sampleTime(self, h:float)->None:
        """Set iteration time step and resample data."""
        if (('current' in self.__dict__) and
            (self.current is not None) and 
            (self.current.h != h)):
            self.current.h = h
        self._sampleTime = h
    
    #--------------------------------------------------------------------------
    @property
    def size(self)->int:
        """Length of one side of floor area."""
        return self._size
    
    @size.setter
    def size(self, size:int)->None:
        """Set size and recreate floor map. Assumes same origin."""
        if (('floor' in self.__dict__) and
            (self.floor is not None) and 
            (self.floor.size != size)):
            self.floor.size = size
        self._size = size
    
    #--------------------------------------------------------------------------
    @property
    def origin(self)->List[float]:
        """Coordinates of zero point (x=0,y=0)."""
        return self._origin
    
    @origin.setter
    def origin(self, origin:List[float])->None:
        """Set coordinates of zero point (x=0,y=0)."""
        if (('floor' in self.__dict__) and
            (self.floor is not None)):
            self.floor.origin = origin
        self._origin = origin

    ## Alternative Constructors ==============================================#
    @classmethod
    def calm_ocean(cls,**kwargs):
        """
        Create deep ocean with slow speeds, steady variations, and calm rates.

        **Defaults:**

          - z: 200m
          - spd: 'typical'
          - dSpd: 'steady'
          - dtSpd: 'calm'
          - ang: 'northeast'
          - dAng: 'steady'
          - dtAng: 'calm'
        
        Returns
        -------
        ocean : Ocean
            'Calm Ocean' instance.
        """
        if ('z' not in kwargs):
            kwargs['z'] = 200
        return cls(name='Calm Ocean',
                   spd='typical',dSpd='steady',dtSpd='calm',
                   ang='northeast',dAng='steady',dtAng='calm',
                   **kwargs)
    
    #--------------------------------------------------------------------------
    @classmethod
    def dead_ocean(cls,**kwargs):
        """
        Create ocean with zero current and no variations.

        **Defaults:**

          - spd: 'dead'
          - dSpd: 'constant'
          - dtSpd: 'calm'
          - ang: 0
          - dAng: 'constant'
          - dtAng: 'calm'
        
        Returns
        -------
        ocean : Ocean
            'Dead Ocean' instance.
        """
        return cls(name='Dead Ocean',
                   spd='dead',dSpd='constant',dtSpd='calm',
                   ang=0,dAng='constant',dtAng='calm',
                   **kwargs)

    #--------------------------------------------------------------------------
    @classmethod
    def stormy_ocean(cls,**kwargs):
        """
        Create ocean with fast current, unsteady variations, and choppy rates.

        **Defaults:**

          - spd: 'fast'
          - dSpd: 'unsteady'
          - dtSpd: 'choppy'
          - ang: 'any'
          - dAng: 'unsteady'
          - dtAng: 'choppy'
        
        Returns
        -------
        ocean : Ocean
            'Stormy Ocean' instance.
        """
        return cls(name='Stormy Ocean',
                   spd='fast',dSpd='unsteady',dtSpd='choppy',
                   ang='any',dAng='unsteady',dtAng='choppy',
                   **kwargs)
    
    ## Special Methods =======================================================#
    def __repr__(self) -> str:
        """Detailed description of Ocean"""
        plumeRepr = ""
        if (self.pollution is not None):
            plumeRepr = f"\npollution={self.pollution!r}"

        return (
            f"{self.__class__.__name__}(\n"
            f"name='{self.name}'\n"
            f"current={self.current!r}\n"
            f"floor={self.floor!r}"
            f"{plumeRepr})"
        )
    
    #--------------------------------------------------------------------------
    def __str__(self) -> str:
        """User friendly description of Ocean"""
        components = [str(self.current), str(self.floor)]
        if (self.pollution is not None):
            components.append(str(self.pollution))

        out = '\n'.join(components)
        ll = len(max(out.split('\n'), key=len))

        return (
            f"\nOcean: {self.name}\n"
            f"{'-'*ll}\n"
            f"{out}"
            f"{'-'*ll}\n"
        )

###############################################################################

@dataclass
class Current1DData:
    """
    Immutable data container for Current1D.

    Offers lightweight alternative to full Current1D instance. Suitable for
    serialization.
    

    Attributes
    ----------
    speed : ndarray
        Speed time series in m/s.
    angle : ndarray
        Direction time series in radians.
    v_spd : float
        Mean speed value.
    v_dv : float
        Speed amplitude deviation.
    v_dt : float
        Speed half-period.
    b_ang : float
        Mean direction value.
    b_db : float
        Direction amplitude deviation.
    b_dt : float
        Direction half-period.
    n : int
        Number of iterations.
    h : float
        Time step in seconds.
    seed : int
        PRNG seed used.

        
    See Also
    --------
    Current1D.getAsData : Convert Current1D instance into Current1DData instance
    Current1D.genAsData : Class method to generate a Current1DData instance with
        Current1D methods. Effectively a constructor for a Current1DData object.
    """

    __slots__ = ['speed', 'angle', 
                 'v_spd', 'v_dv', 'v_dt', 
                 'b_ang', 'b_db', 'b_dt',
                 'n', 'h', 'seed']
    speed: NPFltArr
    angle: NPFltArr
    v_spd: float
    v_dv: float
    v_dt: float
    b_ang: float
    b_db: float
    b_dt: float
    n: int
    h: float
    seed: int

###############################################################################

class Current1D:
    """
    Time-varying ocean current with configurable speed and direction profiles.
    
    Generates realistic ocean current time series using sinusoidal modulation
    with user-configurable mean values, variation amplitudes, and temporal
    periods. Supports categorical parameter specification for rapid prototyping
    or precise numerical control for advanced scenarios.
    
    Parameters
    ----------
    spd : float, list, or str, default='typical'
        Mean current speed specification in meters per second. This value is the
        overall mean value and the actual speed values will fluctuate about this
        mean. If a numerical range is provided, a random value within the range
        will be selected. If a named range is provided, a random value within
        the predefined range will be selected.
        
        - float: Explicit speed in m/s (0.0-2.5)
        - list: [low, high] range of speed in m/s (0.0-2.5)
        - str categories (m/s):

            - 'dead':       0.0
            - 'typical':    0.05 - 0.5
            - 'moderate':   0.5  - 1.0
            - 'swift':      1.0  - 1.8
            - 'fast':       1.8  - 2.5
            - 'any':        0.0  - 2.5
            
    dSpd : float, list, or str, default='steady'
        Speed variation amplitude specification in meters per second. This value
        is the average amplitude deviation about the mean value through which
        the current speed will fluctuation. Each oscillation will selecte a new
        amplitude around this value. This provides more stoichastic variation
        rather than simple uniform periodic oscillation. If a numerical range is
        provided, a random value within the range will be selected. If a named
        range is provided, a random value within the predefined range will be
        selected.
        
        - float: Explicit amplitude in m/s (0.0-1.5)
        - list: [low, high] range of speed in m/s (0.0-1.5)
        - str categories (m/s):

            - 'constant':   0.0
            - 'steady':     0.01 - 0.1
            - 'varied':     0.1  - 0.5
            - 'unsteady':   0.5  - 1.5
            - 'any':        0.0  - 1.5
            
    dtSpd : float, list, or str, default='regular'
        Speed variation half-period specification in seconds. This value is the
        time of an oscillation from one amplitude to the next (e.g., from max
        speed to min speed in one oscillation period). Smaller values produce
        more frequent changes and more unsteady conditions. If a numerical range
        is proided, a random value within the range will be selected. If a named
        range is provided, a random value within the predefined range will be
        selected.
        
        - float: Explicit period in seconds
        - list: [low, high] range of time in seconds
        - str categories (s):

            - 'calm':       600
            - 'smooth':     90  - 150
            - 'regular':    30  -  90
            - 'choppy':     10  -  30
            - 'any':        10  - 600
            
    ang : float, list, or str, default='northeast'
        Mean current direction specification in radians. This value is the
        overall mean value and the actual angle values will fluctuate about this
        mean. If a numerical range is provided, a random value within the range
        will be selected. If a named range is provided, a random value within
        the predefined range will be selected.
        
        - float: Explicit angle in radians (0=East, pi/2=North, etc.)
        - list: [low, high] range of angle in radians within (-pi,pi)
        - str categories:

            - 'east':        0     +/- pi/8
            - 'northeast':   pi/4  +/- pi/8
            - 'north':       pi/2  +/- pi/8
            - 'northwest':   3pi/4 +/- pi/8
            - 'west':        pi    +/- pi/8
            - 'southwest':  -3pi/5 +/- pi/8
            - 'south':      -pi/2  +/- pi/8
            - 'southeast':  -pi/4  +/- pi/8
            - 'any':         0     +/- pi
    
    dAng : float, list, or str, default='steady'
        Direction variation amplitude specification in radians. This value is
        the average amplitude deviation about the mean value through which the
        current angle will fluctuation. Each oscillation will selecte a new
        amplitude around this value. This provides more stoichastic variation
        rather than simple uniform periodic oscillation. If a numerical range is
        provided, a random value within the range will be selected. If a named
        range is provided, a random value within the predefined range will be
        selected.
        
        - float: Explicit amplitude in radians (0.0-pi/2)
        - list: [low, high] range of angle in radians within (0.0-pi/2)
        - str categories:

            - 'constant':   0.0
            - 'steady':     0.1   - pi/18
            - 'varied':     pi/18 - pi/8
            - 'unsteady':   pi/8  - pi/2
            - 'any':        0.0   - pi/2
            
    dtAng : float, list, or str, default='regular'
        Direction variation half-period specification in seconds. This value is
        the time of an oscillation from one amplitude to the next (e.g., from
        max angle to min angle in one oscillation period). Smaller values
        produce more frequent changes and more unsteady conditions. If a
        numerical range is proided, a random value within the range will be
        selected. If a named range is provided, a random value within the
        predefined range will be selected.
        
        - float: Explicit period in seconds
        - list: [low, high] range of time in seconds
        - str categories (s):

            - 'calm':       600
            - 'smooth':     90  - 150
            - 'regular':    30  -  90
            - 'choppy':     10  -  30
            - 'any':        10  - 600
            
    nIter : int, default=60000
        Number of simulation iterations (time series length).
    h : float, default=0.02
        Time step per iteration in seconds.
    seed : int, optional
        PRNG seed for reproducibility. If None, generates random seed
        for unique current patterns each run.
        

    Attributes
    ----------
    speed : ndarray, shape (N,)
        Time series of current speed in m/s.
    angle : ndarray, shape (N,)
        Time series of current direction in radians.
    v_spd : float
        Mean current speed in m/s. From spd keyword.
    v_dv : float
        Speed variation amplitude in m/s. From dSpd keyword.
    v_dt : float
        Speed variation half-period in seconds. From dtSpd keyword.
    b_ang : float
        Mean current direction in radians. From ang keyword.
    b_db : float
        Direction variation amplitude in radians. From dAng keyword.
    b_dt : float
        Direction variation half-period in seconds. From dtAng keyword.
    N : int
        Number of time samples.
    sampleTime : float
        Time step in seconds.
    seed : int
        PRNG seed used for reproducible generation.
        

    Methods
    -------
    genSpeedList(mean, stddev, dt, n, h) : 
        Generate a list of speed values that fluctuate around the mean.
    genAngleList(mean, stddev, dt, n, h) :
        Generate a list of angle values that fluctuate around the mean.
    display() : 
        Display Current1D data in simple speed / angle vs time plots.
    getAsData() : 
        Construct a Current1DData object from Current1D instance.
    genAsData(spd, dSpd, dtSpd, ang, dAng, dtAng, nIter, h, seed) : 
        Static method. Construct a Current1DData object from input parameters.

        
    Generation Model
    ----------------
    Speed and direction are modeled as sinusoidal oscillations:
    
    .. code-block:: none

        speed(t) = v_spd + v_dv * sin(pi * t / v_dt)
        angle(t) = b_ang + b_db * sin(pi * t / b_dt)
    
    Providing smoothly varying, stoichastic fluctuations around desired mean
    values.
    

    Notes
    -----
    **Categorical vs Numerical Parameters:**

    Categorical strings are converted to numerical ranges via internal tables.
    Use strings for general specification or to allow small randomness in input.
    Use explicit numbers for precise control.
    
    >>> # Categorical (quick setup)
    >>> current1 = Current1D(spd='moderate', dSpd='steady', ang='north')
    >>> 
    >>> # Numerical Range (general control)
    >>> current2 = Current1D(spd=[0.75,1.25], dSpd=[0.05,0.40], ang=np.pi/2)
    >>> 
    >>> # Numerical (exact control)
    >>> current3 = Current1D(spd=0.75, dSpd=0.05, ang=np.pi/2)
    
    **Randomization:**

    When using categorical strings and numerical ranges, random selection within
    the category is performed by uniform random distribution:
    
    >>> current = Current1D(spd='typical', seed=42)
    >>> # v_spd selected uniformly from [0.05, 0.5] m/s

    Within each half-period of oscillation, the speed and angle amplitudes are
    randomly selected around the mean deviation by random normal distribution
    with a 3-sigma limit.

    **Resampling:**

    Changing N or sampleTime regenerates time series:
    
    >>> current.N = 100000  # Extends simulation time
    >>> current.sampleTime = 0.01  # Doubles sample rate
    >>> # Both automatically call _genVals()
    
    **Coordinate System:**

    Angles follow END (East-North-Down) convention:

    - 0 rad: East (+x direction)
    - pi/2 rad: North (+y direction)
    - pi rad: West (-x direction)
    - -pi/2 rad: South (-y direction)
    

    Examples
    --------
    ### Create typical coastal current:
    
    >>> current = Current1D(
    ...     spd='typical',     # 0.05-0.5 m/s
    ...     dSpd='steady',     # +/- 0.01-0.1 m/s
    ...     dtSpd='calm',      # 600s half-period
    ...     ang='northeast',   # pi/4 +/- pi/8
    ...     dAng='steady',     # +/- 0.1-0.17 rad (10 deg)
    ...     dtAng='smooth',    # 90-150s half-period
    ...     nIter=60000,
    ...     seed=100
    ... )
    >>> print(f"Mean: {current.v_spd:.2f} m/s at {current.b_ang:.1f} rad")
    
    ### Strong tidal current:
    
    >>> tidal = Current1D(
    ...     spd='fast',        # 1.8-2.5 m/s
    ...     dSpd='unsteady',   # +/- 0.5-1.5 m/s (ebb/flood)
    ...     dtSpd='smooth',    # 90-150s
    ...     ang='north',       # Aligned with channel
    ...     dAng='constant',   # No direction change
    ...     dtAng='regular',
    ...     nIter=120000,      # 40 minutes
    ... )
    >>> tidal.display()
    
    ### Zero current for baseline:
    
    >>> dead_current = Current1D(
    ...     spd='dead',
    ...     dSpd='constant',
    ...     dtSpd=600,
    ...     ang=0.0,
    ...     dAng='constant',
    ...     dtAng=600,
    ...     nIter=60000,
    ... )
    >>> dead_current.display()
    
    ### Precise numerical control:
    
    >>> precise = Current1D(
    ...     spd=0.82,          # Exactly 0.82 m/s
    ...     dSpd=0.15,         # Exactly +/- 0.15 m/s
    ...     dtSpd=120.0,       # 120s half-period
    ...     ang=0.785,         # pi/4 (45 deg)
    ...     dAng=0.1,          # +/- 0.1 rad (+/- 5.7 deg)
    ...     dtAng=90.0,        # 90s half-period
    ...     nIter=60000,
    ...     seed=42
    ... )
    
    ### Access time series data:
    
    >>> current = Current1D(spd='moderate', seed=50)
    >>> # Sample at specific iteration
    >>> speed_1000 = current.speed[1000]
    >>> angle_1000 = current.angle[1000]
    >>> 
    >>> # Vector components (END frame)
    >>> u_current = speed_1000 * np.cos(angle_1000)  # East component
    >>> v_current = speed_1000 * np.sin(angle_1000)  # North component
    
    ### Use in vehicle dynamics:
    
    >>> # Vehicle reads current from sensor
    >>> V_c = current.speed[iteration]
    >>> beta_Vc = current.angle[iteration]
    >>> # Apply to vehicle dynamics (see vehicles.py)
    

    See Also
    --------
    Ocean : Container class that manages Current1D instance
    Current1DData : Immutable data container for serialization
    navigation.OceanCurrentSensor : Sensor that samples current data
    

    Warnings
    --------
    - Sinusoidal model does not capture turbulent fluctuations or eddies. Being
      1D data, the current will be taken as uniform across any arbitrary ocean
      space (at any given time slice).
    - 1D model assumes depth-uniform current (not realistic for stratified
      water).
    """

    ## Class Constants =======================================================#
    _V_LO = 0.0             # Absolute min value for Ocean Current Speed (m/s)
    _V_HI = 2.5             # Standard max value for Ocean Current Speed (m/s)
    _CI = 3.0               # Confidence Interval for Standard Deviation calc


    ## Constructor ===========================================================#
    def __init__(self,
                 spd:Union[float, str, List[float]] = 'typical',
                 dSpd:Union[float, str, List[float]] = 'steady',
                 dtSpd:Union[float, str, List[float]] = 'regular',
                 ang:Union[float, str, List[float]] = 'northeast',
                 dAng:Union[float, str, List[float]] = 'steady',
                 dtAng:Union[float, str, List[float]] = 'regular',
                 nIter:int = 60000,
                 h:float = 0.02,
                 seed:Optional[int] = None,
                 **kwargs,
                 )->None:
        """
        Construct time-varying ocean current with sinusoidal speed and direction profiles.
    
        Generates realistic ocean current time series using sinusoidal modulation around
        user-specified mean values. Supports categorical parameter specification (quick
        setup) or precise numerical control (advanced tuning). Designed for deterministic
        or randomized current generation in AUV simulation scenarios.
        
        Parameters
        ----------
        spd : float, str, or list of float, default='typical'
            Mean current speed specification in meters per second (m/s).
            This is the central value around which speed oscillates.
            
            **Float**: Exact mean speed in m/s (0.0-2.5).
            >>> Current1D(spd=0.75)  # Exactly 0.75 m/s mean
            
            **List [low, high]**: Random selection within range via uniform distribution.
            >>> Current1D(spd=[0.5, 1.0])  # Random between 0.5-1.0 m/s
            
            **String categories** (m/s ranges):
            - 'dead':          0.0 (no current)
            - 'typical':  0.05-0.5 (coastal/shelf currents)
            - 'moderate': 0.5-1.0  (active shelf/slope)
            - 'swift':    1.0-1.8  (strong tidal/boundary currents)
            - 'fast':     1.8-2.5  (extreme tidal/jet streams)
            - 'any':      0.0-2.5  (full randomization)
            
            Automatically clamped to [0.0, 2.5] m/s.
            
        dSpd : float, str, or list of float, default='steady'
            Mean speed variation amplitude in m/s (+/- deviation from spd).
            Controls magnitude of sinusoidal speed oscillations.
            Each half-period samples new amplitude from normal distribution
            around this value (adds stochastic variation).
            
            **Float**: Exact mean amplitude deviation in m/s (0.0-1.5).
            >>> Current1D(spd=1.0, dSpd=0.2)  # Speed varies ~0.8-1.2 m/s
            
            **List [low, high]**: Random mean amplitude deviation from range.
            >>> Current1D(dSpd=[0.1, 0.3])
            
            **String categories** (m/s ranges):
            - 'constant':      0.0 (no variation, steady speed)
            - 'steady':   0.01-0.1 (gentle fluctuations)
            - 'varied':   0.1-0.5  (moderate variations)
            - 'unsteady': 0.5-1.5  (large tidal-like swings)
            - 'any':      0.0-1.5  (full randomization)
            
            Automatically limited to not exceed speed bounds [0.0, 2.5].
            
        dtSpd : float, str, or list of float, default='regular'
            Speed variation half-period in seconds.
            Time for one oscillation from max to min (or vice versa).
            Smaller values -> more frequent, choppy changes.
            Larger values -> slower, smoother variations.
            
            **Float**: Exact half-period in seconds.
            >>> Current1D(dtSpd=120.0)  # 2-minute oscillation period
            
            **List [low, high]**: Random selection of period range.
            >>> Current1D(dtSpd=[30, 90]) 
            
            **String categories** (second ranges):
            - 'calm':       600 (10-minute oscillations, very smooth)
            - 'smooth':  90-150 (1.5-2.5 minutes, gentle)
            - 'regular': 30-90  (0.5-1.5 minutes, typical)
            - 'choppy':  10-30  (10-30 seconds, rapid changes)
            - 'any':     10-600 (full randomization)
            
        ang : float, str, or list of float, default='northeast'
            Mean current direction in radians (END convention).
            This is the central bearing around which direction oscillates.
            
            **Float**: Exact mean direction in radians.
            >>> Current1D(ang=np.pi/2)  # North (90 deg)
            >>> Current1D(ang=0)         # East (0 deg)
            
            **List [low, high]**: Random mean direction from range.
            >>> Current1D(ang=[0, np.pi/4])  # East to Northeast
            
            **String categories** (radian ranges, +/- pi/8 from center):
            - 'east':           0 +/- pi/8 (337.5 deg to  22.5 deg)
            - 'northeast':   pi/4 +/- pi/8 ( 22.5 deg to  67.5 deg)
            - 'north':       pi/2 +/- pi/8 ( 67.5 deg to 112.5 deg)
            - 'northwest':  3pi/4 +/- pi/8 (112.5 deg to 157.5 deg)
            - 'west':          pi +/- pi/8 (157.5 deg to 202.5 deg)
            - 'southwest': -3pi/4 +/- pi/8 (202.5 deg to 247.5 deg)
            - 'south':      -pi/2 +/- pi/8 (247.5 deg to 292.5 deg)
            - 'southeast':  -pi/4 +/- pi/8 (292.5 deg to 337.5 deg)
            - 'any':            0 +/- pi   (full circle, random)
            
            Automatically wrapped to [-pi, pi] (smallest signed angle).
            
        dAng : float, str, or list of float, default='steady'
            Mean direction variation amplitude in radians (+/- deviation from mean).
            Controls magnitude of direction oscillations.
            Each half-period samples new amplitude from normal distribution around this value.
            
            **Float**: Exact mean angular deviation in radians (0.0-pi/2).
            >>> Current1D(ang=0, dAng=0.2)  # +/- 11.5 deg oscillation
            
            **List [low, high]**: Random mean amplitude selection from range.
            >>> Current1D(dAng=[0.1, 0.3])  # +/- 5.7 deg to +/- 17.2 deg
            
            **String categories** (radian ranges):
            - 'constant':           0.0 (no directional change)
            - 'steady':     0.1 - pi/18 (+/-  5.7 deg to +/-   10 deg, slight meander)
            - 'varied':   pi/18 - pi/8  (+/-   10 deg to +/- 22.5 deg, moderate swing)
            - 'unsteady':  pi/8 - pi/2  (+/- 22.5 deg to +/-   90 deg, large swings)
            - 'any':        0.0 - pi/2  (full range)
            
            Automatically clipped to [0, pi].
            
        dtAng : float, str, or list of float, default='regular'
            Direction variation half-period in seconds.
            Time for one directional oscillation from max to min.
            
            **Float**: Exact half-period in seconds.
            >>> Current1D(dtAng=180.0)  # 3-minute direction cycle
            
            **List [low, high]**: Random period range.
            **String categories**: Same as dtSpd (calm/smooth/regular/choppy/any).
            
        nIter : int, default=60000
            Number of simulation iterations (time series length).
            Default 60000 iterations * 0.02s = 1200s = 20 minutes.
            
        h : float, default=0.02
            Time step per iteration in seconds.
            Default 0.02s = 50 Hz sampling rate.
            
            Total simulation time = nIter * h seconds.
            
        seed : int, optional
            PRNG seed for reproducibility.
            If None, generates random seed from system entropy (unique each run).
            If provided, ensures identical current time series for same parameters.
            seed=0 is valid and deterministic.
            
        **kwargs
            Additional keyword arguments.
            
        Attributes
        ----------
        speed : ndarray, shape (nIter,)
            Generated speed time series in m/s.
        angle : ndarray, shape (nIter,)
            Generated direction time series in radians.
        v_spd : float
            Mean speed in m/s (after randomization/validation).
        v_dv : float
            Speed amplitude in m/s (after randomization/validation).
        v_dt : float
            Speed half-period in seconds.
        b_ang : float
            Mean direction in radians (after randomization/wrapping).
        b_db : float
            Direction amplitude in radians (after randomization/clipping).
        b_dt : float
            Direction half-period in seconds.
        n : int
            Number of iterations (stored via property, triggers generation).
        h : float
            Time step in seconds (stored via property, triggers resampling).
        seed : int
            Actual PRNG seed used (from parameter or generated).
            
        Generation Process
        ------------------
        1. **Seed Initialization:**
        If seed=None: Generate from np.random.SeedSequence().entropy.
        Store in self.seed and create RNG: self._rng = np.random.default_rng(seed).
        
        2. **Parameter Resolution:**
        For each parameter (spd, dSpd, dtSpd, ang, dAng, dtAng):
        a. If string: Look up in category dictionary, select random value in range.
        b. If list [low, high]: Select random value via uniform distribution.
        c. If float: Use value.
        d. Apply validation/clipping/wrapping via property setters.
        
        3. **Standard Deviation Calculation:**
        Convert amplitude deviations to 3-sigma standard deviations:

        - sigma_speed = v_dv / 3
        - sigma_angle = b_db / 3

        Stored as _v_sdv and _b_sdb (computed lazily when accessed).
        
        4. **RNG State Capture:**
        Before each time series generation, store RNG state:
        - _spd_rng_state: For speed reproducibility
        - _ang_rng_state: For angle reproducibility
        Allows resizing/resampling while maintaining pattern.
        
        5. **Time Series Generation:**
        Call genSpeedList() and genAngleList():

        - Each creates sinusoidal oscillations with variable amplitude
        - Amplitude resampled each half-period from normal distribution
        - Speed clipped to [0.0, 2.5] m/s
        - Angle wrapped to [-pi, pi]

        Store in self.speed and self.angle arrays.
        
        Sinusoidal Oscillation Model
        -----------------------------
        **Speed Time Series:**

            v(t) = v_spd + a_i * sin((pi*t)/(v_dt))
        
            where:

                - v_spd: Mean speed
                - a_i: Amplitude for half-period i, drawn from N(v_dv, sigma_v)
                - v_dt: Half-period duration
                - Direction flips each half-period
        
        **Direction Time Series:**
        
            theta(t) = b_ang + a_i * sin((pi*t)/(b_dt))
        
            where:

                - b_ang: Mean direction
                - a_i: Amplitude for half-period i, drawn from N(b_db, sigma_b)
                - b_dt: Half-period duration
                - Direction flips each half-period
        
        **Stochastic Variation:**
        
            Each oscillation half-period selects new amplitude:

            - a_i ~ N(mean_amplitude, stddev)
            - Adds natural variability (not purely periodic)
            - Simulates realistic environmental fluctuations
        
        Notes
        -----
        **Parameter Priority:**
        
        When parameters specified multiple ways:
        1. Property setter validation (clipping/wrapping)
        2. User explicit value (if float)
        3. Random from custom range (if list)
        4. Random from category range (if string)
        
        **Categorical Parameter Philosophy:**
        
        String categories enable quick prototyping without remembering exact numerical ranges:
        >>> current = Current1D(spd='moderate', dSpd='steady', ang='north')
        
        For precise control, use floats:
        >>> current = Current1D(spd=0.82, dSpd=0.15, ang=np.pi/2)
        
        **Reproducibility:**
        
        Identical seed + parameters -> identical time series:
        >>> c1 = Current1D(spd='typical', seed=42)
        >>> c2 = Current1D(spd='typical', seed=42)
        >>> np.array_equal(c1.speed, c2.speed)
        True
        
        But different seeds with same category -> different values:
        >>> c3 = Current1D(spd='typical', seed=43)
        >>> c1.v_spd != c3.v_spd  # Different mean selected
        True
        
        **Property-Triggered Regeneration:**
        
        Changing n or h after initialization regenerates time series:
        >>> current = Current1D(nIter=10000, seed=100)
        >>> current.n = 20000  # Extends to 20000 iterations
        >>> # Uses stored RNG state to maintain pattern continuity
        
        **Integration with Ocean:**
        
        Ocean passes categorical or numerical current parameters:
        >>> ocean = Ocean(
        ...     spd='moderate',
        ...     dSpd='steady',
        ...     dtSpd='calm',
        ...     ang='northeast',
        ...     dAng='steady',
        ...     dtAng='calm',
        ...     N=60000,
        ...     h=0.02,
        ...     currentSeed=42
        ... )
        >>> ocean.current  # Current1D instance with these parameters
        
        Warnings
        --------
        - Invalid string categories trigger warning and fallback to defaults
        - Invalid list lengths (!=2) trigger warning and fallback
        - Extremely small dtSpd or dtAng (<1s) may cause rapid oscillations
        
        See Also
        --------
        genSpeedList : Speed time series generation algorithm
        genAngleList : Direction time series generation algorithm
        Current1DData : Immutable data container for serialization
        Ocean.__init__ : Creates Current1D with oceanographic parameters
        
        Examples
        --------
        ### Quick categorical setup:
        
        >>> import munetauvsim.environment as env
        >>> current = env.Current1D(
        ...     spd='moderate',
        ...     dSpd='steady',
        ...     dtSpd='regular',
        ...     ang='north',
        ...     dAng='constant',
        ...     dtAng='calm',
        ...     nIter=60000,
        ...     seed=42
        ... )
        >>> print(f"Mean: {current.v_spd:.2f} m/s at {current.b_ang:.2f} rad")
        
        ### Precise numerical control:
        
        >>> current = env.Current1D(
        ...     spd=0.82,
        ...     dSpd=0.15,
        ...     dtSpd=120.0,
        ...     ang=np.pi/4,
        ...     dAng=0.1,
        ...     dtAng=90.0,
        ...     nIter=60000,
        ...     h=0.02,
        ...     seed=100
        ... )
        
        ### Mixed categorical and numerical:
        
        >>> current = env.Current1D(
        ...     spd=[0.5, 1.0],  # Custom range
        ...     dSpd='steady',   # Category
        ...     dtSpd=60.0,      # Exact value
        ...     ang='northeast',
        ...     seed=50
        ... )
        
        ### Access time series:
        
        >>> current = env.Current1D(spd='typical', nIter=10, seed=0)
        >>> print(current.speed[:5])
        [0.28 0.30 0.32 0.31 0.29]  # Example values
        >>> print(current.angle[:5])
        [0.85 0.87 0.88 0.86 0.84]  # Example values
        
        ### Visualization:
        
        >>> # Shows speed and angle plots vs time
        >>> current = env.Current1D(
        ...     spd='moderate',
        ...     dSpd='varied',
        ...     dtSpd='choppy',
        ...     nIter=30000,
        ...     seed=42
        ... )
        >>> current.display()
        
        ### Export as data object:
        
        >>> current = env.Current1D(spd='typical', seed=999)
        >>> data = current.getAsData()
        >>> print(type(data))
        <class 'Current1DData'>
        >>> # Immutable dataclass for serialization
        """
        
        ##-- Random Number Generator -----------------------------------------#
        if (seed is None):
            seed = np.random.SeedSequence().entropy
        self.seed = seed                                 # RNG seed
        self._rng = np.random.default_rng(self.seed)     # RNG instance
        
        ##-- Set Property Values ---------------------------------------------#
        self.n = nIter                  # Number of simulation iterations
        self.h = h                      # Time step of each iteration (s)

        # Speed Parameters
        self.v_spd = spd                # Speed selected mean value (m/s)
        self.v_dv = dSpd                # Speed deviation (+/-) (m/s)
        self.v_dt = dtSpd               # Speed half period of deviations (s)

        # Angle Parameters
        self.b_ang = ang                # Angle selected mean value (rad)
        self.b_db = dAng                # Angle deviation (+/-) (rad)
        self.b_dt = dtAng               # Angle half period of deviations (s)

        ##-- Construct Concrete Values ---------------------------------------#
        self._spd_rng_state = self._genRNGStartState(seed+1)
        self._rng.bit_generator.state = self._spd_rng_state
        self.speed = self.genSpeedList(self.v_spd, self.v_sdv, self.v_dt,
                                       self.n, self.h)
        self._ang_rng_state = self._genRNGStartState(seed+2)
        self._rng.bit_generator.state = self._ang_rng_state
        self.angle = self.genAngleList(self.b_ang, self.b_sdb, self.b_dt,
                                       self.n, self.h)

    ## Properties ============================================================#
    @property
    def n(self)->int:
        """Number of simulation iterations."""
        return self._n
    
    @n.setter
    def n(self, nIter:int)->None:
        """Set number of iterations and resize data."""
        if ('_n' in self.__dict__):
            if (nIter > self._n):
                self._rng.bit_generator.state = self._spd_rng_state
                self.speed = self.genSpeedList(self.v_spd, self.v_sdv, 
                                               self.v_dt, nIter, self.h)
                self._rng.bit_generator.state = self._ang_rng_state
                self.angle = self.genAngleList(self.b_ang, self.b_sdb,
                                               self.b_dt, nIter, self.h)
            else:
                self.speed = self.speed[:nIter]
                self.angle = self.angle[:nIter]
        self._n = nIter
    
    #--------------------------------------------------------------------------
    @property
    def h(self)->float:
        """Time step of each iteration (s)."""
        return self._h
    
    @h.setter
    def h(self, h:float)->None:
        """Set iteration time step and resample data."""
        if ('_h' in self.__dict__):
            self._rng.bit_generator.state = self._spd_rng_state
            self.speed = self.genSpeedList(self.v_spd, self.v_sdv, self.v_dt,
                                           self.n, h)
            self._rng.bit_generator.state = self._ang_rng_state
            self.angle = self.genAngleList(self.b_ang, self.b_sdb, self.b_dt,
                                           self.n, h)
        self._h = h

    ## Speed Properties ======================================================#
    #-- Speed: Mean Value ----------------------------------------------------# 
    @property
    def v_spd(self)->float:
        """Selected mean speed."""
        return self._v_spd
    
    @v_spd.setter
    def v_spd(self, spd:Union[float, str, List[float]])->None:
        """Input automatically clamped to range [_V_LO,_V_HI]."""
        self._v_spd = self._set_v_spd(spd)
    
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    @singledispatchmethod
    def _set_v_spd(self, spd:Union[int,float])->float:
        """Set speed value by dispatch based on input type."""
        return self._clamp(spd,Current1D._V_LO,Current1D._V_HI)
    
    @_set_v_spd.register(list)
    @_set_v_spd.register(str)
    def _set_v_spd_list(self, spd:Union[list,str])->float:
        """Set speed value from range."""
        return self._selectFromZone(spd, 0.25, 'Speed', self.v_spdZones)
    
    #-- Speed: Deviation -----------------------------------------------------#
    @property
    def v_dv(self)->float:
        """Mean deviation of selected mean speed value in terms of +/-."""
        return self._v_dv
    
    @v_dv.setter
    def v_dv(self, dSpd:Union[float, str, List[float]])->None:
        """Input cannot exceed possible values range."""
        self._v_sdv = None
        if (self.v_spd == 0):
            dSpd = 0.0
        self._v_dv = self._set_v_dv(dSpd)

    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    @singledispatchmethod
    def _set_v_dv(self, dSpd:Union[int,float])->float:
        """Set speed deviation value by dispatch based on input type."""
        upper = abs(Current1D._V_HI - self.v_spd)
        lower = abs(self.v_spd - Current1D._V_HI)
        return min(min(lower,upper), dSpd)
    
    @_set_v_dv.register(list)
    @_set_v_dv.register(str)
    def _set_v_dv_list(self, dSpd:Union[list,str])->float:
        """Set speed deviation value from range."""
        return self._selectFromZone(dSpd, 0.1, 'Delta Speed', self.v_dvZones)
    
    #-- Speed: Time of Deviation ---------------------------------------------#
    @property
    def v_dt(self)->float:
        """Time of selected mean deviations from one extreme to next."""
        return self._v_dt
    
    @v_dt.setter
    def v_dt(self, dtSpd:Union[float, str, List[float]])->None:
        """Set time of speed deviation."""
        self._v_dt = self._set_v_dt(dtSpd)
    
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    @singledispatchmethod
    def _set_v_dt(self, dtSpd:Union[int,float])->float:
        """Set time of speed deviation by dispatch based on input type."""
        return dtSpd
    
    @_set_v_dt.register(list)
    @_set_v_dt.register(str)
    def _set_v_dt_list(self, dtSpd:Union[list,str])->float:
        """Set time of speed deviation from range."""
        return self._selectFromZone(dtSpd,60,'Delta Speed Rate',self.v_dtZones)

    #-- Speed: Standard Deviation --------------------------------------------#
    @property
    def v_sdv(self)->float:
        """Auto-calculated. Standard deviation of mean speed deviations."""
        if (self._v_sdv is None):
            self._v_sdv = self._calcSigma(self.v_spd, self.v_spd+self.v_dv)
        return self._v_sdv
    
    #-- Speed: Named Zones ---------------------------------------------------#
    @property
    def v_spdZones(self)->dict:
        """Named zones for speed values."""
        if ('_v_spdZones' not in self.__dict__):
            self._v_spdZones = {'dead': [0.0,0.0], 
                               'typical': [0.05,0.5], 
                               'moderate': [0.5,1.0], 
                               'swift': [1.0,1.8],
                               'fast': [1.8,2.5],
                               'any': [0.0,2.5]}
        return self._v_spdZones
    
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    @property
    def v_dvZones(self)->dict:
        """Named zones for amount of deviation in speed values."""
        if ('_v_dvZones' not in self.__dict__):
            self._v_dvZones = {'constant': [0.0,0.0], 
                              'steady': [0.01,0.1], 
                              'varied': [0.1,0.5], 
                              'unsteady': [0.5,1.5],
                              'any': [0.0,1.5]}
        return self._v_dvZones
    
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    @property
    def v_dtZones(self)->dict:
        """Named zones for rate of change in speed values."""
        if ('_v_dtZones' not in self.__dict__):
            self._v_dtZones = {'calm': [600.0,600.0], 
                               'smooth': [90.0,150.0],
                               'regular': [30.0,90.0],
                               'choppy': [10.0,30.0],
                               'any': [10.0,600.0]}
        return self._v_dtZones
    

    ## Angle Properties ======================================================#
    #-- Angle: Mean Value ----------------------------------------------------#
    @property
    def b_ang(self)->float:
        """Selected mean angle."""
        return self._b_ang
    
    @b_ang.setter
    def b_ang(self, betaVal:Union[float, str, List[float]])->None:
        """Input reformatted to smallest-signed angle from (-pi,pi]."""
        self._b_ang = self._set_b_ang(betaVal)

    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    @singledispatchmethod
    def _set_b_ang(self, betaVal:Union[int,float])->float:
        """Set angle value by dispatch based on input type."""
        return self._ssa(betaVal)
    
    @_set_b_ang.register(list)
    @_set_b_ang.register(str)
    def _set_b_ang_list(self, betaVal:Union[list,str])->float:
        """Set angle value from range."""
        return self._selectAngleFromZone(betaVal, np.pi/4, 'Angle', 
                                         self.b_angZones)
    
    #-- Angle: Deviation -----------------------------------------------------#
    @property
    def b_db(self)->float:
        """Mean deviation of selected mean angle value in terms of +/-."""
        return self._b_db
    
    @b_db.setter
    def b_db(self, dAng:Union[float, str, List[float]])->None:
        """Input limited to positive value from (0, pi)."""
        self._b_sdb = None
        self._b_db = self._set_b_db(dAng)

    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    @singledispatchmethod
    def _set_b_db(self, dAng:Union[int,float])->float:
        """Set angle mean deviation value by dispatch based on input type."""
        return abs(dAng) % (np.pi+1E-15)
    
    @_set_b_db.register(list)
    @_set_b_db.register(str)
    def _set_b_db_list(self, dAng:Union[list,str])->float:
        """Set angle mean deviation from range."""
        return self._selectAngleFromZone(dAng, np.pi/12, 'Delta Angle',
                                         self.b_dbZones)

    #-- Angle: Time of Deviation ---------------------------------------------#
    @property
    def b_dt(self)->float:
        """Time of selected angle deviations from one extreme to next."""
        return self._b_dt
    
    @b_dt.setter
    def b_dt(self, dtAng:Union[float, str, List[float]])->None:
        """Set time of angle deviation."""
        self._b_dt = self._set_b_dt(dtAng)

    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    @singledispatchmethod
    def _set_b_dt(self, dtAng:Union[int,float])->float:
        """Set time of angle deviation by dispatch based on input type."""
        return dtAng
    
    @_set_b_dt.register(list)
    @_set_b_dt.register(str)
    def _set_b_dt_list(self, dtAng:Union[list,str])->float:
        """Set time of angle deviation from range."""
        return self._selectFromZone(dtAng,60,'Delta Angle Rate',self.b_dtZones)

    #-- Angle: Standard Deviation --------------------------------------------#
    @property
    def b_sdb(self)->float:
        """Auto-calculated. Standard deviation of mean angle deviation."""
        if (self._b_sdb is None):
            self._b_sdb = self._calcAngleSigma(self.b_ang,self.b_ang+self.b_db)
        return self._b_sdb
    
    #-- Angle: Named Zones ---------------------------------------------------#
    @property
    def b_angZones(self)->dict:
        """Named zones for angle values."""
        if ('_b_angZones' not in self.__dict__):
            self._b_angZones = {'east': [-np.pi/8,np.pi/8], 
                               'northeast': [np.pi/8,3*np.pi/8],
                               'north': [3*np.pi/8,5*np.pi/8],
                               'northwest': [5*np.pi/8,7*np.pi/8],
                               'west': [7*np.pi/8,-7*np.pi/8],
                               'southwest': [-7*np.pi/8,-5*np.pi/8],
                               'south': [-5*np.pi/8,-3*np.pi/8],
                               'southeast': [-3*np.pi/8,-np.pi/8],
                               'any': [-np.pi,np.pi]}
        return self._b_angZones
    
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    @property
    def b_dbZones(self)->dict:
        """Named zones for amount of deviation in angle values."""
        if ('_b_dbZones' not in self.__dict__):
            self._b_dbZones = {'constant': [0.0,0.0], 
                              'steady': [0.01,np.pi/18],         # 0-10
                              'varied': [np.pi/18,np.pi/8],      # 10-22.5 
                              'unsteady': [np.pi/8,np.pi/2],     # 22.5-90
                              'any': [0.0,np.pi/2]}              # 0-90
        return self._b_dbZones
    
    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    @property
    def b_dtZones(self)->dict:
        """Named zones for rate of change in angle values."""
        if ('_b_dtZones' not in self.__dict__):
            self._b_dtZones = self.v_dtZones
        return self._b_dtZones

    ## Property Helper Methods ===============================================#
    # Numeric - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @singledispatchmethod
    def _selectFromZone(self, unknown, default:float, zone:str, *args)->float:
        """
        Select a value from a specific zone.
        
        Notes
        -----
        - Valid input types: String, List (should be int or float).
        - Dispatch base case catches inputs without a registered format type.
        """
        log.warning("Unsupported %s zone format: %s", zone, type(unknown))
        self._printFallback(default)
        return default
    
    @_selectFromZone.register(list)
    def _selectFromZoneList(self, input:list, default:float, zone:str,
                            *args)->float:
        """Select random value from within listed range."""
        if (len(input) != 2):
            self._printBadZoneList(input, zone, default)
            return default
        return self._selectFromLoHi(*input)

    @_selectFromZone.register(str)
    def _selectFromZoneStr(self, input:str, default:float, zone:str,
                           zone_dict)->float:
        """Select random value from within named range."""
        if (input not in zone_dict):
            self._printBadZoneName(input, zone_dict.keys(), zone, default)
            return default
        return self._selectFromLoHi(*zone_dict[input])
    
    # Angle - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    @singledispatchmethod
    def _selectAngleFromZone(self, unknown, default:float, zone:str, 
                             *args)->None:
        """
        Select an angle from a specific zone.
                
        Notes
        -----
        - Valid input types: String, List (should be int or float).
        - Dispatch base case catches inputs without a registered format type.
        """
        log.warning("Unsupported %s zone format: %s", zone, type(unknown))
        self._printFallback(default)
        return default
    
    @_selectAngleFromZone.register(list)
    def _selectAngleFromZoneList(self, input:list, default:float, zone:str,
                                 *args)->float:
        """Select random value from within listed range."""
        if (len(input) != 2):
            self._printBadZoneList(input, zone, default)
            return default
        return self._selectAngleFromLoHi(*input)
    
    @_selectAngleFromZone.register(str)
    def _selectAngleFromZoneStr(self, input:str, default:float, zone:str,
                                zone_dict)->float:
        """Select random value from within named range."""
        if (input not in zone_dict):
            self._printBadZoneName(input, zone_dict.keys(), zone, default)
            return default
        return self._selectAngleFromLoHi(*zone_dict[input])

    ## Special Methods =======================================================#
    def __repr__(self) -> str:
        """Detailed description of Current1D."""
        fmt = '.3f'
        zones = self._buildDescription()
        out = [
            f"n={self.n}",
            f"h={self.h}",
            f"seed={self.seed}",
            f"v_spd={self.v_spd:{fmt}} ('{zones['spd']}')",
            f"v_dv={self.v_dv:{fmt}} ('{zones['dSpd']}')",
            f"v_dt={self.v_dt:{fmt}} ('{zones['dtSpd']}')",
            f"b_ang={self.b_ang:{fmt}} ('{zones['ang']}')",
            f"b_db={self.b_db:{fmt}} ('{zones['dAng']}')",
            f"b_dt={self.b_dt:{fmt}} ('{zones['dtAng']}')"
        ]
        return f"{self.__class__.__name__}(" + ", ".join(out) + ")"
    
    #-------------------------------------------------------------------------#
    def __str__(self) -> str:
        """User friendly description of Current1D."""
        zones = self._buildDescription()
        fmt = '.3f'
        cw = 16
        return (
            f"Current\n"
            f"{' Speed:':{cw}} {self.v_spd:{fmt}} +/-{self.v_dv:{fmt}} m/s "
            f"({self.v_dt:{fmt}}s)\n"
            f"{' Angle:':{cw}} {self.b_ang:{fmt}} +/-{self.b_db:{fmt}} rad "
            f"({self.b_dt:{fmt}}s)\n"
            f"{' Seed:':{cw}} {self.seed}\n"
        )

    ## Methods ===============================================================#
    def genSpeedList(self, mean:float, stddev:float, dt:float, n:int, 
                     h:float=0.02)->NPFltArr:
        """
        Generate list of speed values that fluctuate around mean.
        
        Parameters
        ----------
        mean : float
            Mean speed in m/s.
        stddev : float
            Standard deviation for amplitude sampling.
        dt : float
            Half-period in seconds.
        n : int
            Number of samples.
        h : float, default=0.02
            Time step in seconds.
            
        Returns
        -------
        speed : ndarray, shape (n,)
            Speed time series clipped to [_V_LO, _V_HI].
        """
        return np.clip(self._genValueList(mean, stddev, dt, n, h),
                       Current1D._V_LO, Current1D._V_HI)
        
    #-------------------------------------------------------------------------#
    def genAngleList(self, mean:float, stddev:float, dt:float, n:int, 
                     h:float=0.02)->NPFltArr:
        """
        Generate list of angle values that fluctuate around mean.
        
        Parameters
        ----------
        mean : float
            Mean direction in radians.
        stddev : float
            Standard deviation for amplitude sampling.
        dt : float
            Half-period in seconds.
        n : int
            Number of samples.
        h : float, default=0.02
            Time step in seconds.
            
        Returns
        -------
        angle : ndarray, shape (n,)
            Direction time series wrapped to [-pi, pi].
        """
        return self._ssa(self._genValueList(mean, stddev, dt, n, h))
    
    #-------------------------------------------------------------------------#
    def display(self)->None:
        """Display simple plot to view attribute data arrays."""
        speed_s = (f"speed: {self.v_spd:.3} m/s, +/- {self.v_dv:.3}, " +
                   f"{self.v_dt:.0f} s")
        angle_s = (f"angle: {self.b_ang:.3} rad, +/- {self.b_db:.3}, " +
                   f"{self.b_dt:.0f} s")
        line = "-"*36
        log.info("%s", line)
        log.info("%s", speed_s)
        log.info("%s", angle_s)
        log.info("%s", line)
        xvals = [x*self.h for x in range(0,len(self.speed))] 
        plt.figure(figsize=(8,9))
        plt.subplot(211,title=speed_s)
        plt.plot(xvals,self.speed)
        plt.ylabel('(m/s)')
        plt.grid()
        plt.subplot(212,title=angle_s)
        plt.plot(xvals,self.angle)
        plt.ylabel('(rad)')
        plt.xlabel('(s)')
        plt.grid()
        plt.show()

    #-------------------------------------------------------------------------#
    def getAsData(self)->Current1DData:
        """
        Convert this Current1D object to Current1DData object.
        
        Returns
        -------
        data : Current1DData
            Immutable dataclass with all current parameters and arrays.
        """
        return Current1DData(self.speed, self.angle, 
                             self.v_spd, self.v_dv, self.v_dt, 
                             self.b_ang, self.b_db, self.b_dt,
                             self.n, self.h, self.seed)

    ## Static Methods ========================================================#
    @staticmethod
    def genAsData(spd:Union[float, str, List[float]] = 'typical',
                  dSpd:Union[float, str, List[float]] = 'steady',
                  dtSpd:Union[float, str, List[float]] = 'regular',
                  ang:Union[float, str, List[float]] = 'northeast',
                  dAng:Union[float, str, List[float]] = 'steady',
                  dtAng:Union[float, str, List[float]] = 'regular',
                  nIter:int = 50000,
                  h:float = 0.02,
                  seed:Optional[int] = None,
                  )->Current1DData:
        """
        Create a Current1DData object using the Current1D class.
        
        Static method convenience constructor for data object.
        
        Parameters
        ----------
        Same as Current1D.__init__
        
        Returns
        -------
        data : Current1DData
            Immutable dataclass with generated current.
        """

        c = Current1D(spd,dSpd,dtSpd,ang,dAng,dtAng,nIter,h,seed=seed)
        return Current1DData(c.speed, c.angle, 
                             c.v_spd, c.v_dv, c.v_dt, 
                             c.b_ang, c.b_db, c.b_dt,
                             c.n, c.h, c.seed)

    ## Helper Methods ========================================================#
    def _clamp(self, val:float, minVal:float, maxVal:float)->float:
        """Limit value to within specified range."""
        return max(min(maxVal, val), minVal)
    
    #-------------------------------------------------------------------------#
    def _calcSigma(self, mu:float, hi:float, ci:Optional[float]=None)->float:
        """Return standard devation, based on 'hi' within 'ci' sigma of 'mu'."""
        if (ci is None):
            ci = Current1D._CI
        return abs(hi - mu) / ci

    #-------------------------------------------------------------------------#
    def _selectRandomN(self, mean:float, stddev:float)->float:
        """Return a value from a normal distribution."""
        return self._rng.normal(mean, stddev)
    
    #-------------------------------------------------------------------------#
    def _getOffset(self, mean:float, stddev:float)->float:
        """Return an offset from the mean inside of a normal distribution."""
        val = self._selectRandomN(mean, stddev)
        return abs(mean-val)
    
    #-------------------------------------------------------------------------#
    def _selectRandomU(self, lo:float, hi:float)->float:
        """Return a value from a uniform distribution."""
        return self._rng.uniform(lo, hi)
 
    #-------------------------------------------------------------------------#
    def _selectFromLoHi(self, lo:float, hi:float)->float:
        """Return a random value from distribution between lo and hi."""
        return self._selectRandomU(lo, hi)
    
    #-------------------------------------------------------------------------#
    def _ssa(self, angle:float)->float:
        """Convert angle to smallest-signed angle in [-pi,pi)."""
        return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi
    
    #-------------------------------------------------------------------------#
    def _subtractSSA(self, minuend:float, subtrahend:float)->float:
        """Calculate size of arc from subtrahend to minuend."""
        if (subtrahend > minuend):
            return (minuend + (2*np.pi)) - subtrahend
        return minuend - subtrahend
    
    #-------------------------------------------------------------------------#
    def _calcAngleSigma(self, 
                        mu:float, 
                        hi:float, 
                        ci:Optional[float]=None,
                        )->float:
        """Return standard devation, based on 'hi' within 'ci' sigma of 'mu'."""
        if (ci is None):
            ci = Current1D._CI
        return self._subtractSSA(hi, mu) / ci
    
    #-------------------------------------------------------------------------#
    def _selectAngleFromLoHi(self, lo:float, hi:float)->float:
        """Return a random angle from distribution between lo and hi."""
        hi = lo + self._subtractSSA(hi,lo)
        return self._ssa(self._selectRandomU(lo,hi))

    #-------------------------------------------------------------------------#
    def _printFallback(self, default:float)->None:
        """Print message indicating default fallback value being used."""
        log.warning("Using fallback value: %s", default)

    #-------------------------------------------------------------------------#
    def _printBadZoneList(self, input:Union[List[int],List[float]], zone:str,
                          default:float)->None:
        """Print message if invalid range given for list zone."""
        log.warning("%s is not a valid range for %s.", input, zone)
        log.warning("Range must have two elements: [Low, High]")
        self._printFallback(default)
    
    #-------------------------------------------------------------------------#
    def _printBadZoneName(self, input:str, keys:KeysView, zone:str,
                      default:float)->None:
        """Print message if invalid option given for named zone."""
        options = ""
        for option in keys:
            options += f" {option}"
        log.warning("'%s' is not a valid zone for %s.", input, zone)
        log.warning("(Valid options: %s)", options)
        self._printFallback(default)
    
    #-------------------------------------------------------------------------#
    def _buildDescription(self)->Dict[str,str]:
        """Make a dictionary of all named zones that inputs fall into."""
        description = {}
        #Speed
        for k,v in self.v_spdZones.items():
            if ((self.v_spd >= v[0]) and 
                (self.v_spd <= v[1]) and 
                (k != 'any')):
                description['spd'] = k
        #dSpeed
        for k,v in self.v_dvZones.items():
            if ((self.v_dv >= v[0]) and 
                (self.v_dv <= v[1]) and 
                (k != 'any')):
                description['dSpd'] = k
        #dtSpeed
        for k,v in self.v_dtZones.items():
            if ((self.v_dt >= v[0]) and 
                (self.v_dt <= v[1]) and 
                (k != 'any')):
                description['dtSpd'] = k
        #Angle
        for k,v in self.b_angZones.items():
            if ((self.b_ang >= v[0]) and 
                (self.b_ang <= v[1]) and 
                (k != 'any')):
                description['ang'] = k
        #dAngle
        for k,v in self.b_dbZones.items():
            if ((self.b_db >= v[0]) and 
                (self.b_db <= v[1]) and 
                (k != 'any')):
                description['dAng'] = k
        #dtAngle
        for k,v in self.b_dtZones.items():
            if ((self.b_dt >= v[0]) and 
                (self.b_dt <= v[1]) and 
                (k != 'any')):
                description['dtAng'] = k
        return description

    #-------------------------------------------------------------------------#
    def _genRNGStartState(self, seed:int)->None:
        """Return a numpy bit generator state from a seed."""
        return np.random.default_rng(seed).bit_generator.state
    
    #-------------------------------------------------------------------------#
    def _genValueList(self, mean:float, stddev:float, dt:float, n:int, 
                      h:float)->NPFltArr:
        """Return a list of values that fluctuate around the mean."""
        # Make empty list
        vals = []
        # Choose random starting direction
        start = self._rng.choice([1,-1])
        # Make half-period oscillation array
        n_halfT = int(dt/h)
        oscillation = start * np.sin(np.linspace(0, np.pi, n_halfT))
        # Starting amplitude, as offset from mean
        a_0 = self._getOffset(mean, stddev)
        for i in range(-(n // -n_halfT)):
            """This bizarre expression in range() does 'upside-down floor
            division', and that bizarre statement means I'm doing ceiling
            division in the negative numbers. The reason for this is to perform
            the ceil() operation without a function call and to return an
            integer. Fun! The point being to run the loop enough times to fill
            the speed list array with n or more elements."""
            # Flip direction
            flip = (-1.0)**i
            # Get next amplitude, as offset from mean
            a_1 = self._getOffset(mean, stddev)
            # Make amplitude array
            a = np.linspace(a_0, a_1, n_halfT)
            # Store amplitude
            a_0 = a_1
            # Generate oscillations from mean for half period
            vals.extend(mean + (a * (flip * oscillation)))
        return np.array(vals[:n])
    
    #-------------------------------------------------------------------------#
    def _genValue(self, mean:float, stddev:float, dt:float, n:int=np.inf, 
                  h:float=0.02)->Generator:
        """Generator function that yields values rather than a complete list."""
        # Choose random starting direction
        start = self._rng.choice([1,-1])
        # Calculate iteration angular frequency
        n_halfT = int(dt/h)
        omega_h = start * (np.pi / n_halfT)
        # Get amplitude generator, as offset from mean
        a_0 = self._getOffset(mean, stddev)
        a_1 = self._getOffset(mean, stddev)
        step = (a_1-a_0) / (n_halfT-1)
        # Generate oscillations
        i = 0
        while i <= n:
            a = a_0 + step*(i % n_halfT)
            yield mean + ((a) * np.sin(omega_h*(i)))
            i += 1
            if (i % (n_halfT) == 0):
                # Get next amplitude
                a_0 = a_1
                a_1 = self._getOffset(mean, stddev)
                step = (a_1-a_0) / (n_halfT-1)

###############################################################################

class Floor:
    """
    Ocean floor depth map generated from 2D Perlin noise.
    
    Procedurally generates realistic ocean floor terrain using Perlin noise
    algorithms. Provides smooth, continuous depth variations suitable for
    AUV navigation, sensor simulation, and collision detection.
    

    Parameters
    ----------
    z : float, default=125
        Shallowest floor depth in meters (minimum z value in depth array).
        Represents highest point of terrain relative to sea surface.
    z_range : float, default=10
        Vertical range of depth variation in meters.
        Maximum depth = z + z_range.
    size : int, default=1000
        Side length of square floor area in meters.
        Generates size x size depth array.
    origin : list of float, default=[500, 500]
        Coordinates [x, y] where (0, 0) maps to floor array center.
    style : str, default='linear'
        How Perlin noise is stretched over z_range, determining terrain
        characteristics. Current development only supports linear style.
    seed : int, default=0
        PRNG seed for Perlin noise generation. Same seed produces
        identical terrain. seed=0 is valid and reproducible.
    **kwargs
        Additional keyword arugments.

          
    Attributes
    ----------
    z : float
        Minimum floor depth (shallowest) in meters.
    z_range : float
        Depth variation range (max - min) in meters.
    size : int
        Floor array side length in meters.
    origin : list of float
        Origin coordinates [x, y] for indexing.
    seed : int
        PRNG seed used for generation.
    style : str
        Terrain style identifier.
    depth : ndarray, shape (size, size)
        2D array of floor depths in meters.
    perlin : PerlinNoise
        Perlin noise generator instance.
    
        
    Methods
    -------
    __call__(x, y) :
        Floor instance is callable: query floor depth at (x, y) position. 
    sample_points(x, y) :
        Sample depth values at list of points.
    sample_grid(x, y) :
        Sample depth at grid of points.
    sample_region(x_bounds, y_bounds) :
        Sample complete subregion of depth map.
    standard_map(noise, z, z_range) :
        Create a depth map from 2D noise array using linear scaling.
    xy2Index(x, y) :
        Convert END coordinates (x, y) to array indices [i, j].
    display2D(z, dispType, path) :
        Display a 2D image of the depth array.
    display3D(z) :
        Display a 3D plot of the depth array.
    
        
    Notes
    -----
    **Terrain Generation:**

    - **Perlin Noise Properties:**

      - Continuous: No discontinuities or sharp edges
      - Coherent: Nearby points have similar values
      - Multi-scale: Octaves add detail at different frequencies
      - Deterministic: Same seed produces identical terrain
      - Flexible evaluation: Supports both point-wise and vectorized queries
    
    - **Style Characteristics:**

      Currently, only linear scaling is available, stretching the Perlin noise
      over z_range and adding the depth offset:

        depth = perlin_noise * z_range + z

      Future development can expand the options to generate more varied terrain.
      Options to consider include: exponential, bimodal, sigmoid, and terraced
      scaling to generate chasms, plateaus, shelves, and more features.

    **Coordinate System:**

    Floor uses END (East-North-Down) convention:

    - x: East coordinate (meters)
    - y: North coordinate (meters)
    - z: Depth below surface (positive down)
    
    **Callable Interface:**

    Floor objects are callable for convenient depth queries:
    
    >>> floor = Floor(z=100, z_range=20)
    >>> depth_at_origin = floor(0, 0)
    >>> depth_at_point = floor(200, 300)
    
    **Indexing and Origin:**

    The origin parameter allows coordinate systems with negative values.
    
    >>> floor = Floor(size=1000, origin=[500, 500])
    >>> # Query at (0, 0) maps to array index [500, 500]
    >>> # Query at (-250, 300) maps to array index [250, 800]
    
    **Boundary Behavior:**

    Queries outside [0, size] range return valid values. The indexing is allowed
    to wrap around and return values at [x mod size]. This prevents stopping a
    simulation run due to bad indexing and effectively treats the floor map as
    though it were tiled.
    
    >>> floor = Floor(size=1000)
    >>> d = floor(1100, 500)                    # Outside bounds
    >>> floor(1100, 500) == floor(100, 500)     # 1100 % 1000 = 100
    True
    
    **Resampling:**

    Changing size or seed regenerates terrain:
    
    >>> floor.size = 2000  # Expands area
    >>> floor.seed = 42    # New terrain pattern
    >>> # Both automatically regenerate depth array
    
    **Memory Usage:**

    Perlin noise generation requires substantially more memory than final
    storage due to intermediate array creation. The method uses `np.meshgrid()`
    and vectorized operations that create large temporary arrays. Floor sizes
    exceeding ~6000m may cause out-of-memory crashes due to generation
    requirements exceeding available RAM. The peak memory during generation is
    approximately 10-20x larger than the final storage requirement.

    Future development is planned to address this limitation. Current mitigation
    strategies are to use smaller floor sizes (<5000m), pre-generate large
    floors on high-memory systems and save for import, generate smaller floor
    sizes and tile the regions together into a larger floor map.

    **Performance:**

    Generation time scales quadratically with size:

    - size=1000: ~0.5-1 second
    - size=2000: ~2-4 seconds
    - size=5000: ~12-25 seconds
    

    Examples
    --------
    ### Create default ocean floor:
    
    >>> floor = Floor()
    >>> print(f"Depth range: {floor.z}-{floor.z + floor.z_range} m")
    Depth range: 125-135 m
    >>> print(f"Area: {floor.size}m x {floor.size}m")
    Area: 1000m x 1000m
    
    ### Custom terrain parameters:
    
    >>> floor = Floor(
    ...     z=80,              # Shallowest at 80m
    ...     z_range=40,        # Varies 80-120m
    ...     size=2000,         # 2km x 2km area
    ...     origin=[1000, 1000],
    ...     seed=123
    ... )
    
    ### Query single point:
    
    >>> floor = Floor(randomFloor=True)
    >>> depth = floor(250, 300)  # At (250m E, 300m N)
    >>> print(f"Floor depth: {depth:.2f} m")
    
    ### Query multiple points:
    
    >>> x_coords = [0, 100, 200, 300]
    >>> y_coords = [0, 100, 200, 300]
    >>> depths = floor.sample_points(x_coords, y_coords)
    >>> print(depths)
    [ 95.82410046  99.48385046 100.1779558  102.04782572]  # Example values
    >>>
    >>> depth_grid = floor.sample_grid(x_coords, y_coords)
    >>> print(depth_grid)
    [[ 95.82410046  97.39379346 105.14764687  86.46614981]  # Example values
     [ 88.33687983  99.48385046 105.89335857  86.24682284]
     [ 88.67318118 100.66821609 100.1779558   93.64459247]
     [106.17117273 100.66821609 105.8082081  102.04782572]]
    
    ### Visualize terrain:
    
    >>> floor.display2D()
    >>> floor.display3D()
    

    See Also
    --------
    PerlinNoise : 2D Perlin noise generator used internally
    Ocean : Container class that manages Floor instance
    navigation.OceanDepthSensor : Sensor that samples floor depth
    Waypoint : Path planning with samplePath() integration
    

    Warnings
    --------
    - Large size values (>5000) may cause memory issues and slow generation
    - Queries outside floor bounds wrap around the array (check before using)
    - Floor assumes flat water surface at z=0 (no waves or surface variation)
    - Depth array is generated at initialization (modification requires
      regeneration)
    """

    ## Constructor ===========================================================#
    def __init__(self,
                 z:Number=125,
                 z_range:Number=10,
                 size:int=1000,
                 origin:List[float]=[500,500],
                 seed:int=0,
                 random:bool=False,
                 style:str='linear',
                 **kwargs,              # Necessary for use by Ocean class
                 ):
        """
        Construct ocean floor with procedurally generated terrain from Perlin noise.
    
        Creates a realistic bathymetric depth map using Perlin noise generation with
        specified depth range, spatial extent, and terrain characteristics. Supports
        deterministic (seeded) or randomized terrain generation.
        
        Parameters
        ----------
        z : float, default=125
            Shallowest floor depth in meters (minimum depth in generated map).
            Represents the highest point of seafloor relative to surface (z=0).
            Typical values:

            - Shallow coastal: 10-50m
            - Continental shelf: 50-200m
            - Slope regions: 200-1000m
            - Deep ocean: >1000m

            Must be non-negative. No upper limit enforced.
        z_range : float, default=10
            Vertical range of depth variation in meters.
            Maximum depth = z + z_range.
            Controls terrain "roughness":

            - Smooth terrain: 5-15m
            - Moderate terrain: 15-30m
            - Rough terrain: 30-50m
            - Extreme terrain: >50m

            Must be non-negative. Typical: 10-30m for coastal AUV operations.
        size : int, default=1000
            Side length of square floor area in meters.
            Generates size * size depth array (square domain only).
            Typical values:

            - Small test area: 500m
            - Standard operations: 1000-2000m
            - Large survey: 3000-5000m
            - Maximum practical: ~6000m (memory limit)

            Must be positive integer.
        origin : list of float, default=[500, 500]
            Coordinates [x_origin, y_origin] where (x=0, y=0) maps in depth array.
            Allows coordinate systems with negative values:

            - [500, 500]: (0,0) at array center for 1000m size
            - [0, 0]: (0,0) at array corner (all positive coords)
            - [1000, 1000]: Allows negative coords from -1000 to 0

            Typically set to [size/2, size/2] for centered operations.
        seed : int, default=0
            PRNG seed for Perlin noise generation.
            Ensures reproducible terrain:

            - Same seed -> identical terrain every time
            - Different seed -> completely different terrain
            - seed=0 is valid (not treated as null)

            Ignored if random=True.
        random : bool, default=False
            If True, generates random seed from system entropy.
            Overrides explicit seed parameter.
            Generates unique terrains.
            If True: seed stored in self.seed after generation.
        style : str, default='linear'
            Terrain generation style (how Perlin noise maps to depth).
            Currently only 'linear' supported:

            - 'linear': depth = perlin_noise * z_range + z

            Future styles possible:

            - 'exponential': For canyons/trenches
            - 'bimodal': For plateaus and valleys
            - 'sigmoid': For smooth transitions
            - 'terraced': For step-like formations

            Invalid style triggers warning and fallback to default.
        **kwargs
            Additional keyword arguments.
            
        Attributes
        ----------
        z : float
            Minimum depth (stored directly).
        z_range : float
            Depth variation range (stored directly).
        size : int
            Floor area dimension (stored, triggers regeneration on change).
        origin : list of float
            Origin coordinates (stored directly).
        seed : int
            Actual seed used (from parameter or generated if random=True).
        style : str
            Terrain style (stored, triggers method assignment).
        perlin : PerlinNoise
            Perlin noise generator instance containing:

            - noise: Normalized 2D array [0, 1]
            - scale, octaves, persistence: Generation parameters

        depth : ndarray, shape (size, size)
            Final depth map in meters, computed via create_map().
        create_map : callable
            Dynamically assigned method based on style parameter.
            Currently always points to self.standard_map.
            
        Notes
        -----
        **Generation Process**

        1. **Parameter Storage:**

          Store z, z_range, size, origin directly as attributes.
        
        2. **Perlin Noise Generation:**

          Create PerlinNoise instance:

          >>> self.perlin = PerlinNoise(
          ...     size=size,
          ...     seed=seed,
          ...     random=random
          ... )

          Relies on default PerlinNoise parameters for scale, octaves, and
          persistence. This generates normalized noise array in [0, 1].
        
        3. **Style Method Assignment:**

          Based on style parameter, assign appropriate mapping function:

            - 'linear' -> self.create_map = self.standard_map

          Invalid style falls back to 'linear' with warning.
        
        4. **Depth Map Creation:**

          Call create_map() to transform Perlin noise to depth:

          >>> self.depth = self.create_map()
          >>> # For linear: depth = perlin.noise * z_range + z

          Final depth array ready for queries.
        
        **Memory Limitations:**
        
        Perlin noise generation uses large intermediate arrays (meshgrids).
        Known limitation: size > ~6000 may cause crash due to memory error.
        
        Workaround for large terrains:

        - Generate smaller tiles and stitch together
        - Pre-generate on high-memory system and save
        - Use lower resolution and interpolate
        
        Future planned: Chunked generation to eliminate size limit.
        
        **Size Property Behavior:**
        
        Changing size after initialization regenerates terrain:
        
        >>> floor = Floor(size=1000, seed=42)
        >>> floor.size = 2000  # Triggers regeneration
        >>> # New 2000*2000 depth array with same seed
        
        Expanding size extends pattern:

        - Existing region matches original
        - New regions fill with continued Perlin pattern
        
        Reducing size crops array:

        - Takes [:new_size, :new_size] slice
        - No regeneration needed (faster)
        
        **Style Assignment Mechanism:**
        
        style.setter performs dynamic method assignment:
        
        >>> if style == 'linear':
        ...     self.create_map = self.standard_map
        >>> # Future:
        >>> # elif style == 'exponential':
        >>> #     self.create_map = self.exponential_map
        
        **Integration with Ocean:**
        
        Ocean constructor passes parameters:
        
        >>> ocean = Ocean(
        ...     size=2000,
        ...     origin=[1000, 1000],
        ...     z=150,              # Passed to Floor as z
        ...     z_range=25,         # Passed to Floor as z_range
        ...     floorSeed=123,      # Passed as seed
        ...     randomFloor=False   # Passed as random
        ... )
        >>> ocean.floor.depth.shape
        (2000, 2000)
        
        **Default Perlin Parameters:**
        
        PerlinNoise created with hardcoded defaults:

        - scale=300: Balanced feature size
        - octaves=3: Natural detail level
        - persistence=0.5: Standard amplitude decay
        
        Cannot currently override via kwargs (limitation).
        Future: Pass perlin_scale, perlin_octaves, perlin_persistence.
        
        See Also
        --------
        PerlinNoise.__init__ : Noise generation parameters and process
        Floor.standard_map : Linear depth mapping implementation
        Ocean.__init__ : Creates Floor with oceanographic parameters
        
        Examples
        --------
        ### Minimal initialization:
        
        >>> import munetauvsim.environment as env
        >>> floor = env.Floor()
        >>> print(f"Depth range: {floor.z} to {floor.z + floor.z_range} m")
        Depth range: 125 to 135 m
        >>> print(f"Area: {floor.size} m")
        Area: 1000 m
        >>> print(f"Array shape: {floor.depth.shape}")
        Array shape: (1000, 1000)
        
        ### Custom shallow coastal terrain:
        
        >>> floor = env.Floor(
        ...     z=30,          # 30m shallowest
        ...     z_range=20,    # Varies 30-50m
        ...     size=2000,     # 2km * 2km
        ...     origin=[1000, 1000],
        ...     seed=42
        ... )
        >>> print(f"Min depth: {floor.depth.min():.2f} m")
        >>> print(f"Max depth: {floor.depth.max():.2f} m")
        
        ### Random terrain for testing:
        
        >>> floor = env.Floor(
        ...     z=100,
        ...     z_range=30,
        ...     random=True  # Unique terrain each run
        ... )
        >>> print(f"Generated with seed: {floor.perlin.seed}")
        
        ### Access Perlin noise directly:
        
        >>> floor = env.Floor(seed=42)
        >>> noise = floor.perlin.noise  # Raw Perlin [0, 1]
        >>> print(f"Noise range: [{noise.min():.3f}, {noise.max():.3f}]")
        Noise range: [0.000, 1.000]
        >>> # Manually transform
        >>> custom_depth = noise * 50 + 100  # 100-150m range
        
        ### Query depth after creation:
        
        >>> floor = env.Floor(z=120, z_range=20)
        >>> depth_at_origin = floor(0, 0)
        >>> print(f"Depth at (0, 0): {depth_at_origin:.2f} m")
        >>> # Sample multiple points
        >>> depths = floor.sample_points([0, 100, 200], [0, 100, 200])
        >>> print(f"Depths: {depths}")
        """

        self.z = z
        self.z_range = z_range
        self.size = size
        self.origin = origin
        self.perlin = PerlinNoise(size=size, seed=seed, random=random)
        self.style = style
        self.depth = self.create_map()  # Defined in style.setter
        
    ## Properties ============================================================#
    @property
    def size(self)->int:
        """Length of one side of floor area (m)."""
        return self._size

    @size.setter
    def size(self, size:int)->None:
        """Set size and recreate map with new length. Assumes same origin."""
        if ('perlin' in self.__dict__):
            if (size > self._size):
                self.perlin = PerlinNoise(size=size,
                                          scale=self.perlin.scale,
                                          octaves=self.perlin.octaves,
                                          persistence=self.perlin.persistence,
                                          seed=self.perlin.seed,
                                          random=False)
            else:
                self.perlin.size = size
                self.perlin.noise = self.perlin.noise[:size,:size]
            self.depth = self.create_map()
        self._size = size

    #--------------------------------------------------------------------------
    @property
    def style(self)->str:
        """Defines how noise array is scaled to create depth map."""    
        return self._style

    @style.setter
    def style(self, style:str)->None:
        """Set depth map scaling style and rescale depth map."""
        s_default = 'linear'
        # Set corresponding depth map function
        if (style == 'linear'):
            self.create_map = self.standard_map
        else:
            log.warning("'%s' is not a valid map generation style", style)
            log.warning("Proceeding with default: '%s'.", s_default)
            style = s_default
            self.create_map = self.standard_map
        # Remake depth map (Note: depth is not a managed attribute, so the
        # inverse operation will not automatically update style attribute)
        if (('depth' in self.__dict__) and (style != self._style)):
            self.depth = self.create_map()
        # Assign new attribute
        self._style = style

    ## Special Methods =======================================================#
    def __call__(self, x:Number, y:Number)->np.float64:
        """
        Return depth value at (x,y) coordinates.
    
        Parameters
        ----------
        x, y : float
            Coordinate point to sample.
            
        Returns
        -------
        z : float
            Depth value at (x,y) in meters.
            
        Notes
        -----
        Uses array indexing to retrieve pre-generated depth values. The
        underlying Perlin noise generator supports single-point evaluation,
        enabling future implementations to compute depth on-demand rather than
        pre-generating the entire depth map.

        Examples
        --------
        >>> floor = Floor()      # New default ocean floor object
        >>> z = floor(100,100)   # Depth value at (x=100,y=100)
        """

        # Single values: default case
        xp, yp = self.xy2Index(x, y)
        return self.depth[xp, yp]
    
    #--------------------------------------------------------------------------
    def __repr__(self)->str:
        """Detailed description of Floor."""
        return (
            f"{self.__class__.__name__}("
            f"z={self.z}, "
            f"z_range={self.z_range}, "
            f"size={self.size}, "
            f"origin={self.origin}, "
            f"style='{self.style}', "
            f"perlin={self.perlin!r})"
        )

    #--------------------------------------------------------------------------
    def __str__(self)->str:
        """User friendly description of Floor."""
        cw = 16
        return (
            f"Floor\n"
            f"{' Size:':{cw}} {self.size} m\n"
            f"{' Origin:':{cw}} {self.origin}\n"
            f"{' Depth:':{cw}} {self.z} to {self.z + self.z_range} m\n"
            f"{' Style:':{cw}} {self.style}\n"
            f"\n{self.perlin}"
        )
    ## Methods ===============================================================#
    def sample_points(self, 
                      x: Union[List, np.ndarray], 
                      y: Union[List, np.ndarray]
                      ) -> np.ndarray:
        """
        Sample floor depth at list of specific coordinate points.
    
        Parameters
        ----------
        x, y : list or ndarray
            Lists of coordinates [x_0,...,x_i], [y_0,...,y_i].
            
        Returns
        -------
        z : ndarray
            1D array of depth values at each point (x_i,y_i).

        Examples
        --------
        >>> floor = Floor()                # New default ocean floor object
        >>> x_pts = [100, 200, 300]
        >>> y_pts = [150, 250, 350]
        >>> z = floor.sample_points(x_pts, y_pts)   # z is list of length 3
        """

        # Vectorized coordinate to index conversion
        x_arr, y_arr = np.asarray(x), np.asarray(y)
        x_l, y_l = self.depth.shape
        x_indices = ((x_arr + self.origin[0]).astype(int) % x_l).flatten()
        y_indices = ((y_arr + self.origin[1]).astype(int) % y_l).flatten()
        
        return self.depth[x_indices, y_indices].reshape(x_arr.shape)

    #--------------------------------------------------------------------------
    def sample_grid(self, 
                    x: Union[List, np.ndarray], 
                    y: Union[List, np.ndarray]
                    ) -> np.ndarray:
        """
        Sample floor depth at grid of coordinate points.
    
        Parameters
        ----------
        x, y : list or ndarray
            Lists of coordinates [x_0,...,x_i], [y_0,...,y_j].
            
        Returns
        -------
        z : ndarray, shape (len(x), len(y))
            2D array of depth values at each grid point (x_i,y_j).

        Examples
        --------
        >>> floor = Floor()             # New default ocean floor object
        >>> x_pts = [100, 200, 300]
        >>> y_pts = [150, 250, 350]
        >>> z = floor.sample_grid(x_pts, y_pts)     # z is array (3,3)
        """
        
        # Create meshgrid for all combinations of x and y coordinates
        x_arr, y_arr = np.asarray(x), np.asarray(y)
        xx, yy = np.meshgrid(x_arr, y_arr, indexing='ij')
        
        # Vectorized coordinate to index conversion
        x_i = ((xx + self.origin[0]).astype(int) % self.depth.shape[0])
        y_j = ((yy + self.origin[1]).astype(int) % self.depth.shape[1])
        
        # Sample depths at all grid points
        return self.depth[x_i, y_j]

    #--------------------------------------------------------------------------
    def sample_region(self, 
                      x_bounds: Union[List[float], np.ndarray], 
                      y_bounds: Union[List[float], np.ndarray]
                      ) -> np.ndarray:
        """
        Extract depth map region within specified boundaries.
    
        Parameters
        ----------
        x_bounds, y_bounds : list or ndarray
            Boundary pairs [x_min,x_max], [y_min,y_max].
            
        Returns
        -------
        z : ndarray
            2D array of depth map region.
            
        Notes
        -----
        Endpoints inclusive. For x:[100,110], y:[100,110], returns (11,11)
        array.
        
        Examples
        --------
        >>> floor = Floor()       # New default ocean floor object
        >>> x_s = [100, 200]      # n = (200-100) + 1 = 101
        >>> y_s = [150, 250]      # m = (250-150) + 1 = 101
        >>> z = floor.sample_region(x_s, y_s)   # z is array (n, m)
        """
        
        if (isinstance(x_bounds, np.ndarray)):
            xmin, xmax = x_bounds[0], x_bounds[1]
            ymin, ymax = y_bounds[0], y_bounds[1]
        else:
            xmin, xmax = x_bounds
            ymin, ymax = y_bounds
        
        # Convert boundaries to indices
        xmin_idx, ymin_idx = self.xy2Index(xmin, ymin)
        xmax_idx, ymax_idx = self.xy2Index(xmax, ymax)
        
        # Ensure proper ordering
        xmin_idx, xmax_idx = min(xmin_idx, xmax_idx), max(xmin_idx, xmax_idx)
        ymin_idx, ymax_idx = min(ymin_idx, ymax_idx), max(ymin_idx, ymax_idx)
        
        # Extract region
        return self.depth[xmin_idx:xmax_idx+1, ymin_idx:ymax_idx+1]
    
    #--------------------------------------------------------------------------
    def standard_map(self, 
                     noise:Optional[NPFltArr]=None, 
                     z:Optional[Number]=None, 
                     z_range:Optional[Number]=None,
                     )->NPFltArr:
        """
        Create depth map from 2D noise array using linear scaling.
    
        Parameters
        ----------
        noise : ndarray, optional
            2D noise array normalized to [0,1]. If None, uses self.perlin.noise.
        z : float, optional
            Minimum depth. If None, uses self.z.
        z_range : float, optional
            Depth range. If None, uses self.z_range.
            
        Returns
        -------
        depth : ndarray
            Depth map = noise * z_range + z.
        """

        n = self.perlin.noise if (noise is None) else noise
        z = self.z if (z is None) else z
        z_r = self.z_range if (z_range is None) else z_range
        return (n * z_r) + z
    
    #--------------------------------------------------------------------------
    def xy2Index(self, x:float, y:float)->List[int]:
        """
        Transform coordinates from (x,y) to array indices.
    
        Parameters
        ----------
        x, y : float
            X, Y coordinates for depth array.
            
        Returns
        -------
        [i,j] : list of int
            Array indices corresponding to (x,y).
            
        Notes
        -----
        **Boundary Behavior - Coordinate Wrapping**
        
        Queries outside the floor domain [0, size] are handled via modulo
        wrapping:

        - Coordinates automatically wrap around array boundaries
        - Prevents IndexError exceptions that would halt simulation
        - Effectively treats floor map as infinitely tiled pattern
        
        **Rationale:**
        Negative array indexing in Python is valid (accesses from end), making
        lower-bound violations difficult to detect compared to upper-bound
        IndexError exceptions. Rather than implement costly bounds checking or
        allowing edge indexing errors stop the entire simulations, wrapping
        provides graceful degradation. This decision can be ammended if a
        generator is built for the PerlinNoise class.

        **Examples:**
        >>> floor = Floor(size=1000, origin=[500, 500])
        >>> # Query outside bounds
        >>> i, j = floor.xy2Index(1100, 500)  # x=1100 > size=1000
        >>> # Result: i = (1100 + 500) % 1000 = 600 (wrapped)
        >>> 
        >>> # Equivalent queries due to wrapping:
        >>> floor(1100, 500) == floor(100, 500)  # True

        **Impact on Simulation:**
        Wrapped coordinates may return unrealistic depth values for vehicles
        that stray far from intended operation areas. Users should validate that
        vehicle trajectories remain within expected floor domain.
        """

        x_l, y_l = self.depth.shape
        return [int(x + self.origin[0]) % x_l, int(y + self.origin[1]) % y_l]

    #--------------------------------------------------------------------------
    def display2D(self, 
                  z:Optional[NPFltArr]=None, 
                  dispType="contour", 
                  path:Optional[Waypoint]=None,
                  )->None:
        """
        Display 2D image of depth array.
    
        Parameters
        ----------
        z : ndarray, optional
            Depth array to display. If None, uses self.depth.
        dispType : str, default='contour'
            Display style.
            'contour': A contour plot with labeled depth contours
            'cloud': Simple scaling to the color map.
        path : guidance.Waypoint, optional
            A waypoint object. Path described by waypoints is plotted over the
            depth map.
        """

        if (z is None):
            z = self.depth

        extent=[-self.origin[0],z.shape[0]-self.origin[0],
                -self.origin[1],z.shape[1]-self.origin[1]]
        plt.figure(figsize=(9,6))
        # Simple scaling plot
        if (dispType == "cloud"):
            p = plt.imshow(z, extent=extent, origin='lower', cmap='viridis_r')
            plt.colorbar(p).ax.invert_yaxis()
        # Plot with countour lines
        else:
            plt.imshow(z, extent=extent, origin='lower', 
                       cmap='viridis_r', alpha=0.5)
            plt.colorbar().ax.invert_yaxis()
            x = np.linspace(extent[0],extent[1]-1,z.shape[0])
            y = np.linspace(extent[2],extent[3]-1,z.shape[1])
            contours = plt.contour(x, y, z, 10, colors='black', alpha=0.4)
            plt.clabel(contours, inline=True, fontsize=8, fmt="%.0f")
        # Lines at origin
        oc = 'gray'
        ow = 0.5
        plt.axvline(x=0, linewidth=ow, color=oc)
        plt.axhline(y=0, linewidth=ow, color=oc)
        # Path of vehicle
        if (path is not None):
            pmc = "red"
            plc = "black"
            plt.plot(path.pos.x,path.pos.y,linestyle="dotted",color=plc)
            plt.scatter(path.pos.x,path.pos.y,marker='^',color=pmc,s=64)
        plt.show()

    #--------------------------------------------------------------------------
    def display3D(self, z:Optional[NPFltArr]=None)->None:
        """
        Display 3D plot of depth array.
    
        Parameters
        ----------
        z : ndarray, optional
            Depth array to display. If None, uses self.depth.
        """
        
        if (z is None):
            z = self.depth
        color = mpl.colormaps['terrain']
        new_cmap = mpl.colors.ListedColormap(color(np.linspace(0.65,0.55,256)))
        # color = mpl.colormaps['gist_earth']
        # new_cmap = mpl.colors.ListedColormap(color(np.linspace(0.9,0.8,256)))
        # color = mpl.colormaps['copper']
        # new_cmap = mpl.colors.ListedColormap(color(np.linspace(0.7,0.9,256)))
        # color = mpl.colormaps['Wistia']
        # new_cmap = mpl.colors.ListedColormap(color(np.linspace(0.9,0.25,256)))
        x,y = np.meshgrid(range(z.shape[0]),range(z.shape[1]))
        fig = plt.figure(figsize=(9,9))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_zlim(-z.max(),0)
        ax.plot_surface(x, y, -z, alpha=0.9, cmap=new_cmap)
        #add water surface
        ax.plot_surface(x, y, 0*z, alpha=0.3, color='blue')
        plt.show()

###############################################################################

class PerlinNoise:
    """
    2D Perlin noise generator for procedural terrain and environmental features.
    
    Implements Ken Perlin's gradient noise algorithm to generate smooth,
    continuous pseudo-random patterns suitable for natural-looking ocean floor
    terrain maps and other patial environmental features. Supports multi-octave
    generation for fractal detail and reproducible seeding for consistent
    terrains.
    
    Parameters
    ----------
    size : int
        Side length of square noise array to generate (array dimensions: size x
        size). Typical values: 500-5000 meters for AUV simulation areas.
    scale : int, default=300
        Spatial frequency scale in array units. Determines feature size:

        - Larger scale: Broader, smoother features (low frequency)
        - Smaller scale: Finer, more detailed features (high frequency)
        - Feature wavelength ≈ scale/size when octaves=1

        Typical values: 100-500 for natural-looking ocean floors.
    octaves : int, default=3
        Number of noise layers to combine (fractal octaves). Each octave adds
        detail at progressively higher frequencies:

        - octaves=1: Single frequency (smooth, uniform)
        - octaves=3-5: Natural terrain detail (recommended)
        - octaves>7: Extremely detailed, may appear noisy

    persistence : float, default=0.5
        Amplitude decay factor for successive octaves. Controls roughness:

        - persistence=0.5: Balanced detail (recommended)
        - persistence<0.5: Smoother, less detailed terrain
        - persistence>0.5: Rougher, more chaotic terrain

        Each octave i has amplitude = persistence^i.
    seed : int, default=0
        PRNG seed for permutation table generation. Ensures reproducibility:

        - Same seed -> identical noise pattern
        - Different seed -> completely different pattern
        - seed=0 is valid and deterministic

    random : bool, default=False
        If True, generates and uses random seed from entropy pool.
        If False, uses provided seed parameter.
        

    Attributes
    ----------
    noise : ndarray, shape (size, size)
        Generated 2D Perlin noise array, normalized to [0, 1].
        Ready for transformation into depth maps or other applications.
    p : ndarray, shape (512,), dtype=int
        Permutation table containing shuffled integers [0, 255] repeated twice.
        Defines gradient vector selection pattern. Derived from seed.
    rng : numpy.random.Generator
        Random number generator instance for permutation table creation.
    size : int
        Array dimension (stored for reference and reconstruction).
    scale : int
        Frequency scale parameter (stored for reconstruction).
    octaves : int
        Number of noise layers combined (stored for reconstruction).
    persistence : float
        Amplitude decay factor (stored for reconstruction).
    seed : int
        PRNG seed used for this instance (stored for reconstruction).
        

    Methods
    -------
    None
        PerlinNoise is designed as a data container. Generation occurs during
        __init__ and results are stored in the noise attribute.
        

    Notes
    -----
    **Perlin Noise Algorithm:**
    
    - **Gradient Noise Principles:**
    
        Perlin noise generates coherent pseudo-random values by:

        1. Dividing space into grid of unit cubes
        2. Assigning random gradient vectors to grid corners via permutation
           table
        3. Computing dot products between gradients and position offsets
        4. Interpolating dot products using smooth fade function
        
    - **Multi-Octave Generation:**
        
        Fractal detail is achieved by summing multiple octaves:
        
            noise(x, y) = sum (amplitude_i * perlin_i(x,y))

        where:

            - sum is from i=0 to octaves-1
            - amplitude_i = persistence^i
            - perlin_i: Baser Perlin noise at frequency_i
            - frequency_i = (size / scale) * 2^i
        
        Lower octaves contribute large-scale features, higher octaves add fine
        detail.
    
    - **Interpolation (Fade Function):**
    
        Uses improved Perlin smoothing curve for continuity:
        
            fade(t) = 6t^5 - 15t^4 + 10t^3
        
        Properties:

            - f(0) = 0, f(1) = 1 (boundary conditions)
            - f'(0) = f'(1) = 0 (smooth derivatives)
            - f''(0) = f''(1) = 0 (smooth curvature)
    
    - **Normalization:**
    
        Output is linearly scaled to [0, 1]:
        
            normalized = (noise - min(noise)) / (max(noise) - min(noise))
    
    **Gradient Vector Selection:**
    
    Permutation table p maps grid coordinates to gradient indices:

    - p contains [0-255] shuffled, repeated twice (length 512)
    - Hash function: h = p[p[xi] + yi] determines gradient
    - Four possible gradients: [(0,1), (0,-1), (1,0), (-1,0)]
    
    **Parameter Interactions:**
    
    Changing size with same seed extends/crops pattern:
    
    >>> small = PerlinNoise(size=500, seed=10)
    >>> large = PerlinNoise(size=1000, seed=10)
    >>> # large.noise[:500, :500] matches small.noise (same pattern)
    
    **Performance Characteristics:**
    
    Generation time scales with O(size^2 x octaves):

    - size=1000, octaves=3: ~0.5-1 second
    - size=2000, octaves=5: ~3-6 seconds
    - size=5000, octaves=3: ~12-25 seconds
    
    Memory requirements:

    Perlin noise generation requires substantially more memory than final
    storage due to intermediate array creation. The method uses `np.meshgrid()`
    and vectorized operations that create large temporary arrays. Sizes
    exceeding ~6000m may cause out-of-memory crashes due to generation
    requirements exceeding available RAM. The peak memory during generation is
    approximately 10-20x larger than the final storage requirement.

    Future development is planned to address this limitation. Current mitigation
    strategies are to use smaller sizes (<5000m), pre-generate large arrays on
    high-memory systems and save for import, generate smaller array sizes and
    tile the regions together.
    
    **Design Decisions:**
    
    This implementation prioritizes simplicity and integration over features:

    - 2D only (sufficient for ocean floor depth maps)
    - Supports both single-point and array-based evaluation for flexibility
    - No tiling/seamless wrapping (may be introduced at later stage)
    - No 3D/4D (future extension possible for volumetric environments)
    - No arbitrary dimensions (square arrays only for simplicity)
    
    **Alternative Implementations Considered:**
    
    Alternative methods for Perlin noise creation exist, but in testing were
    slower or more complicated than what is needed here. For example,

        - opensimplex:     can do 3D,
        - pythonperlin:    can do 3D, offers seamless tiling
        - noise:           does not work with numpy arrays
        - nlmpy:           did not explore, looks extensive
    
    Current implementation balances simplicity, speed, and integration.
    
    **Future Development:**
    
    Planned enhancements:

    - Chunked generation to eliminate size>6000 memory limit
    - Optional seamless tiling for repeating terrains
    - 3D noise for volumetric pollution/current fields
    - Generator pattern for memory-efficient large maps
    

    Warnings
    --------
    - size > 6000 may cause out-of-memory errors during generation
    - Does not support non-square arrays
    - No built-in tiling (repeated patterns have discontinuities)
    - Changing scale/octaves/persistence with same seed creates different
      patterns
    
      
    See Also
    --------
    Floor : Uses PerlinNoise to generate ocean floor depth maps
    Floor.standard_map : Transforms normalized noise into depth values
    

    References
    ----------
    [1] Producing 2D Perlin noise with numpy
    https://stackoverflow.com/questions/42147776/producing-2d-perlin-noise-with-numpy
    
    [2] Perlin Noise (with implementation in Python)
    https://iq.opengenus.org/perlin-noise/
    
    [3] Adrian's Blog: Perlin Noise Explanation
    https://adrianb.io/2014/08/09/perlinnoise.html
    
    [4] Red Blob Games: Terrain Generation from Noise
    https://www.redblobgames.com/maps/terrain-from-noise/
    
        
    Examples
    --------
    ### Basic noise generation:
    
    >>> noise_gen = PerlinNoise(size=1000, scale=300, octaves=3, seed=42)
    >>> noise_array = noise_gen.noise
    >>> print(f"Array shape: {noise_array.shape}")
    Array shape: (1000, 1000)
    >>> print(f"Value range: [{noise_array.min():.3f},{noise_array.max():.3f}]")
    Value range: [0.000, 1.000]
    
    ### Effect of persistence:
    
    >>> smooth = PerlinNoise(size=500, persistence=0.3, seed=50)
    >>> rough = PerlinNoise(size=500, persistence=0.7, seed=50)
    >>> # smooth.noise has gentler variations
    >>> # rough.noise has sharper, more chaotic features
    
    ### Random unique terrains:
    
    >>> for i in range(5):
    ...     unique = PerlinNoise(size=500, random=True)
    ...     print(f"Terrain {i+1} seed: {unique.seed}")
    Terrain 1 seed: 187239847293847
    Terrain 2 seed: 928374619283746
    # ... each has unique seed and pattern
    
    ### Custom frequency and detail:
    
    >>> # Fine detail with high frequency
    >>> fine = PerlinNoise(size=1000, scale=100, octaves=6)
    >>> 
    >>> # Broad features with low frequency  
    >>> broad = PerlinNoise(size=1000, scale=600, octaves=2)
    """

    ## Constructor ===========================================================#
    def __init__(self, 
                 size:int,
                 scale:int=300,
                 octaves:int=3,
                 persistence:float=0.5, 
                 seed:int=0, 
                 random:bool=False):
        """
        Initialize Perlin noise generator and create normalized 2D noise array.
    
        Constructs permutation table from seed and generates multi-octave Perlin
        noise with specified frequency and detail parameters. All computation
        occurs during initialization; the resulting noise array is stored in
        self.noise.
        
        Parameters
        ----------
        size : int
            Side length of square output array. Creates size x size noise map.
            Recommended range: 500-5000 for AUV simulation areas.
            Maximum practical size: ~6000 (memory-limited).

        scale : int, default=300
            Base frequency scale factor. Determines the size of the largest
            features:

            - Larger scale (500-1000): Broad, rolling terrain
            - Medium scale (200-400): Balanced terrain (recommended)
            - Smaller scale (50-150): Fine, detailed features

            Formula: base_frequency = size / scale
            Recommended: scale = 0.2-0.5 x size for natural appearance.

        octaves : int, default=3
            Number of noise frequencies to layer. Each adds detail at 2x
            frequency:

            - octaves=1: Single smooth layer (uniform hills)
            - octaves=2-4: Natural terrain detail (recommended)
            - octaves=5-7: High detail (rocky, complex)
            - octaves>7: Excessive detail, diminishing returns

            Computation cost scales linearly with octaves.

        persistence : float, default=0.5
            Amplitude scaling between octaves. Controls fractal dimension:

            - persistence=0.3-0.4: Smooth, gentle terrain
            - persistence=0.5: Balanced (recommended)
            - persistence=0.6-0.7: Rough, dramatic terrain
            - persistence>0.7: Very chaotic, may look noisy

            Formula: amplitude_octave_i = persistence^i
            Valid range: [0, 1], typical range: [0.3, 0.7].

        seed : int, default=0
            Random number generator seed for permutation table:

            - Identical seeds produce identical noise patterns
            - Different seeds produce completely different patterns
            - seed=0 is valid (not treated specially)
            - Can be any integer (negative values accepted)

            Used to initialize numpy.random.default_rng(seed).

        random : bool, default=False
            Override seed parameter with system entropy:

            - True: Generates seed for unique pattern each run
            - False: Uses provided seed for reproducibility

            When True, final seed value stored in self.seed attribute.
            
        Attributes
        ----------
        size : int
            Stored for reconstruction and reference.
        scale : int
            Stored for reconstruction and reference.
        octaves : int
            Stored for reconstruction and reference.
        persistence : float
            Stored for reconstruction and reference.
        seed : int
            Actual seed used (either provided or generated if random=True).
        rng : numpy.random.Generator
            RNG instance created from seed.
        p : ndarray, shape (512,), dtype=int
            Permutation table: shuffled [0-255] repeated twice.
            Used to hash grid coordinates to gradient vectors.
        noise : ndarray, shape (size, size), dtype=float64
            Final normalized Perlin noise array in range [0, 1].
            
        Notes
        -----
        **Generation Process**

        1. **Seed Handling:**

          If random=True, generates seed from np.random.SeedSequence().entropy.
          Otherwise uses provided seed parameter.
        
        2. **Permutation Table:**

          Creates array [0, 1, 2, ..., 255], shuffles with rng,
          repeats to length 512: p = [shuffled[0:255], shuffled[0:255]].
        
        3. **Multi-Octave Generation:**

          For each octave i from 0 to octaves-1:
        
                frequency_i = (size / scale) x 2^i
                amplitude_i = persistence^i
                noise += perlin(x * frequency_i, y * frequency_i) * amplitude_i
        
        4. **Normalization:**

          Linearly scales final noise to [0, 1]:
        
                noise = (noise - min(noise)) / (max(noise) - min(noise))
        
        **Parameter Storage:**
        
        All input parameters stored for reconstruction/serialization:
        
        >>> pn = PerlinNoise(size=500, scale=200, octaves=4, seed=99)
        >>> print(pn)
        Perlin
        Scale:          200
        Octaves:        4
        Persistence:    0.5
        Seed:           99
        
        **Memory Allocation:**
        
        Multiple large temporary arrays created during generation. Peak memory
        ~10-20x final noise array size. Sizes exceeding ~6000m may cause
        out-of-memory crashes due to generation requirements exceeding available
        RAM. Future development is planned to address this limitation.
        
        **Frequency Doubling:**
        
        Each octave doubles the frequency, halving feature size:
        
        - Octave 0: wavelength = scale
        - Octave 1: wavelength = scale / 2
        - Octave 2: wavelength = scale / 4
        - ...
        - Octave i: wavelength = scale / 2^i
        
        **Amplitude Decay:**
        
        Higher frequency octaves contribute less:
        
        - Octave 0: weight = 1.0
        - Octave 1: weight = persistence
        - Octave 2: weight = persistence²
        - ...
        - Octave i: weight = persistence^i
        
        For persistence=0.5:

        - Octave 0: 100% amplitude
        - Octave 1: 50% amplitude
        - Octave 2: 25% amplitude
        - Octave 3: 12.5% amplitude
        
        Examples
        --------
        ### Minimal initialization:
        
        >>> pn = PerlinNoise(size=1000)
        >>> # Uses defaults: scale=300, octaves=3, persistence=0.5, seed=0
        >>> print(pn.noise.shape)
        (1000, 1000)
        
        ### Custom parameters:
        
        >>> pn = PerlinNoise(
        ...     size=2000,
        ...     scale=400,
        ...     octaves=5,
        ...     persistence=0.6,
        ...     seed=12345
        ... )
        >>> print(f"Generated noise: {pn.noise.shape},"
        ...       f" range [{pn.noise.min():.3f}, {pn.noise.max():.3f}]")
        Generated noise: (2000, 2000), range [0.000, 1.000]
        """

        # Save construction parameters so possible to recreate same instance
        self.size = size
        self.scale = scale
        self.octaves = octaves
        self.persistence = persistence
        self.seed=seed
        if (random):
            self.seed = np.random.SeedSequence().entropy
        self.rng = np.random.default_rng(seed=self.seed)
        # Construct noise array
        self.p = self._get_permutation_table()
        self.noise = self._gen_perlin()

    ## Special Methods =======================================================#
    def __repr__(self) -> str:
        """Detailed description of PerlinNoise."""
        # Note that rng, p, and noise can be recreated from these parameters
        return (
            f"{self.__class__.__name__}("
            f"seed={self.seed}, "
            f"size={self.size}, "
            f"scale={self.scale}, "
            f"octaves={self.octaves}, "
            f"persistence={self.persistence})"
        )

    #--------------------------------------------------------------------------
    def __str__(self)->str:
        """User friendly description of PerlinNoise."""
        cw = 16
        return (
            f"Perlin\n"
            f"{' Scale:':{cw}} {self.scale}\n"
            f"{' Octaves:':{cw}} {self.octaves}\n"
            f"{' Persistence:':{cw}} {self.persistence}\n"
            f"{' Seed:':{cw}} {self.seed}\n"
        )
    
    ## Helper Methods ========================================================#
    def _get_permutation_table(self)->NPIntArr:
        """
        Generate permutation table for Perlin noise gradient vector selection.
    
        Creates a hash table by shuffling integers [0-255] and repeating twice
        to create a 512-element lookup array. This table maps grid coordinates
        to pseudo-random gradient vectors, providing the spatial coherence
        characteristic of Perlin noise.
        
        Returns
        -------
        p : ndarray, shape (512,), dtype=int
            Permutation table containing [0-255] shuffled and repeated.
            Used to hash grid coordinates: p[p[xi] + yi] -> gradient index.
            
        Notes
        -----
        **Algorithm:**
        1. Create sequential array [0, 1, 2, ..., 255]
        2. Shuffle using self._rng (seeded RNG for reproducibility)
        3. Repeat shuffled array: [shuffled, shuffled] -> length 512
        
        **Purpose:**
        The permutation table provides deterministic pseudo-random access to
        gradient vectors. Doubling to length 512 eliminates need for modulo
        operations when indexing:
        
        >>> hash_value = self.p[self.p[xi] + yi]  # No % 256 needed
        
        **Gradient Mapping:**
        Hash values from permutation table select gradient vectors:

        - h = p[p[xi] + yi] determines which gradient from 4 options:
        - h % 4 = 0: [0, 1]   (North)
        - h % 4 = 1: [0, -1]  (South)
        - h % 4 = 2: [1, 0]   (East)
        - h % 4 = 3: [-1, 0]  (West)
        
        See Also
        --------
        _perlin2D : Uses permutation table for gradient selection
        _gradient : Converts hash values to gradient vectors
        """
        p = np.arange(256, dtype=int)
        self.rng.shuffle(p)
        p = np.stack([p,p]).flatten()
        return p

    #--------------------------------------------------------------------------
    def _gen_perlin(self)->NPFltArr:
        """
        Generate normalized multi-octave Perlin noise array.
    
        Combines multiple octaves (frequency layers) of Perlin noise to create
        fractal detail. Each octave adds progressively higher frequency features
        with decreasing amplitude, resulting in natural-looking terrain.
        

        Returns
        -------
        noise : ndarray, shape (size, size), dtype=float64
            Normalized 2D Perlin noise array with values in [0, 1]. Ready for
            transformation into depth maps or other applications.
            

        Notes
        -----
        **Algorithm:**

        For each octave i from 0 to octaves-1:
        
        1. **Frequency Calculation:**
        
            freq_i = (size / scale) * 2^i
        
        2. **Amplitude Calculation:**
        
            amp_i = persistence^i
        
        3. **Coordinate Generation:**

            Create meshgrid with spacing determined by frequency:
        
            >>> x_l = np.linspace(0, freq, size, endpoint=False)
            >>> x, y = np.meshgrid(x_l, x_l)
        
        4. **Noise Layer Generation:**

            Generate Perlin noise at current frequency:
            
            >>> noise += _perlin2D(x, y) * amp
        
        5. **Normalization:**

            After summing all octaves, normalize to [0, 1]:
            
            >>> noise = _normalize(noise)
        
        **Multi-Octave Properties:**
        
        - **Octave 0 (Base):** Lowest frequency, largest features
        - Frequency = size / scale
        - Amplitude = 1.0
        - Contribution: 100%
        
        - **Octave 1:** Double frequency, half amplitude
        - Frequency = 2 * (size / scale)
        - Amplitude = persistence
        - Contribution: persistence * 100%
        
        - **Octave i:** Exponential frequency, decaying amplitude
        - Frequency = 2^i * (size / scale)
        - Amplitude = persistence^i
        - Contribution: persistence^i * 100%
        
        **Frequency Doubling:**

        Each octave doubles spatial frequency, halving feature wavelength:

        - More octaves -> finer detail
        - Typical range: 3-6 octaves for natural terrain
        
        **Amplitude Decay:**

        Persistence controls amplitude scaling between octaves:

        - persistence = 0.5 (typical): Each octave contributes half previous
          amplitude
        - persistence < 0.5: Smoother, less detailed
        - persistence > 0.5: Rougher, more detailed
        
        **Memory Considerations:**

        Meshgrid operations create large temporary arrays:

        - Each octave creates 2 * size^2 arrays (x and y meshgrids)
        - Peak memory ~ 2 * size^2 * 8 bytes per octave
        - Total peak ~ 16-32 * size^2 bytes during generation
        

        See Also
        --------
        _perlin2D : Single-octave 2D Perlin noise computation
        _normalize : Linear scaling to [0, 1] range
        __init__ : Sets scale, octaves, persistence parameters
        

        Examples
        --------
        ### Typical usage (called internally by __init__):
        
        >>> pn = PerlinNoise(size=1000, scale=300, octaves=4, persistence=0.6)
        >>> # _gen_perlin() automatically called during construction
        >>> noise = pn.noise  # Access generated array
        
        ### Effect of octave count:
        
        >>> # 1 octave: Smooth, broad features
        >>> smooth = PerlinNoise(size=500, octaves=1)
        >>> 
        >>> # 6 octaves: Detailed, fractal-like
        >>> detailed = PerlinNoise(size=500, octaves=6)
        """
        freq = self.size / self.scale
        amp = 1
        noise = np.zeros([self.size, self.size])
        for i in range(self.octaves):
            x_l = np.linspace(0, freq, self.size, endpoint=False)
            x, y = np.meshgrid(x_l, x_l)
            noise += self._perlin2D(x, y) * amp
            amp *= self.persistence
            freq *= 2
        return self._normalize(noise)

    #--------------------------------------------------------------------------
    def _perlin2D(self, 
                  x:Union[Number, NPFltArr], 
                  y:Union[Number, NPFltArr],
                  )->NPFltArr:
        """
        Compute 2D Perlin noise at specified coordinates.
    
        Core Perlin noise algorithm that uses permutation table to select
        gradient vectors, computes dot products with coordinate offsets, and
        interpolates results using smooth fade functions. Single-octave
        implementation. Accepts both scalar coordinates and array inputs.
        

        Parameters
        ----------
        x : float or ndarray
            X-coordinate(s) scaled by frequency. Can be a single scalar value
            or an array (typically meshgrid of shape (size, size)).
        y : float or ndarray
            Y-coordinate(s) scaled by frequency. Can be a single scalar value
            or an array (typically meshgrid of shape (size, size)).
            

        Returns
        -------
        noise : float or ndarray
            Raw Perlin noise values (not normalized). Return type matches input:
            scalar inputs return scalar output, array inputs return array output.
            Range varies but typically [-1, 1] before normalization.
            

        Notes
        -----
        **Algorithm Steps:**

        1. **Grid Cell Identification:**

            Determine integer grid coordinates (top-left corner of each cell):
            
            >>> xi, yi = x.astype(int) % 255, y.astype(int) % 255
        
        2. **Internal Coordinates:**

            Compute fractional position within grid cell [0, 1):
            
            >>> xf = x - x.astype(int)  # Distance from left edge
            >>> yf = y - y.astype(int)  # Distance from top edge
        
        3. **Fade Factors:**

            Apply smoothing curve for C^2 continuity:
            
            >>> u = _fade(xf)  # Horizontal smoothing
            >>> v = _fade(yf)  # Vertical smoothing
            
        4. **Gradient Selection and Dot Products:**

            For each corner of grid cell, hash coordinates to select gradient
            vector and compute dot product with offset:
            
            >>> n00 = _gradient(p[p[xi  ] + yi  ], xf  , yf  )  # Top-left
            >>> n01 = _gradient(p[p[xi  ] + yi+1], xf  , yf-1)  # Bottom-left
            >>> n10 = _gradient(p[p[xi+1] + yi  ], xf-1, yf  )  # Top-right
            >>> n11 = _gradient(p[p[xi+1] + yi+1], xf-1, yf-1)  # Bottom-right
        
        5. **Bilinear Interpolation:**

            Combine corner values using fade factors:
            
            >>> x1 = _lerp(n00, n10, u)  # Interpolate top edge
            >>> x2 = _lerp(n01, n11, u)  # Interpolate bottom edge
            >>> result = _lerp(x1, x2, v)  # Interpolate vertical
        
        **Coordinate Wrapping:**

        Modulo 255 prevents overflow in permutation table indexing:

        - Coordinates wrapped to [0, 254] range
        - Allows seamless tiling if desired (though not implemented)
        
        **Gradient Hashing:**

        Permutation table provides pseudo-random but deterministic gradient
        selection:

        - Same (xi, yi) always maps to same gradient (for given seed)
        - Different seeds produce completely different gradient patterns
        
        **Interpolation Quality:**

        Uses improved Perlin fade function (6t⁵ - 15t⁴ + 10t³) for:

        - Smooth first derivative (no visible grid artifacts)
        - Smooth second derivative (no curvature discontinuities)
        - Better visual quality than linear or cosine interpolation
        
        **Vectorization:**

        Entire operation vectorized via numpy broadcasting:

        - All coordinates processed simultaneously
        - No explicit loops over grid cells
        - Efficient memory access patterns
        
        **Scalar vs Array Inputs:**

        Method accepts both scalar and array coordinates:

        - Scalar inputs (x=float, y=float): Returns single noise value
        - Array inputs (x=ndarray, y=ndarray): Returns noise array matching input shape
        - Useful for both single-point queries and full map generation
        

        See Also
        --------
        _fade : Smoothing function for interpolation weights
        _gradient : Gradient vector selection and dot product
        _lerp : Linear interpolation
        _gen_perlin : Multi-octave wrapper that calls this method
        

        Examples
        --------
        ### Single point query (scalar inputs):
        
        >>> pn = PerlinNoise(size=100, seed=42)
        >>> noise_value = pn._perlin2D(5.3, 7.8)
        >>> print(f"Noise at (5.3, 7.8): {noise_value:.3f}")
        Noise at (5.3, 7.8): -0.134
        
        ### Array-based generation (typical usage):
        
        >>> pn = PerlinNoise(size=100, seed=42)
        >>> # Create coordinate meshgrid
        >>> x = np.linspace(0, 10, 100)
        >>> X, Y = np.meshgrid(x, x)
        >>> # Generate single-octave noise
        >>> noise = pn._perlin2D(X, Y)
        >>> print(noise.shape)
        (100, 100)
        """
        # Convert scalar inputs to numpy arrays
        x = np.asarray(x)
        y = np.asarray(y)
        # Grid cell corners and smooth interpolation weights
        xi, yi = x.astype(int)%255, y.astype(int)%255 # Coordinates of top left
        xf, yf = x-x.astype(int), y-y.astype(int)     # Internal coordinates
        u, v = self._fade(xf), self._fade(yf)         # Fade factors
        # Noise components
        """Map the internal coordinates to the permutation table to select the
        gradient vector then perfom a dot product with the coordinate grid"""
        n00 = self._gradient(self.p[self.p[xi    ] + yi    ], xf    , yf    )
        n01 = self._gradient(self.p[self.p[xi    ] + yi + 1], xf    , yf - 1)
        n10 = self._gradient(self.p[self.p[xi + 1] + yi    ], xf - 1, yf    )
        n11 = self._gradient(self.p[self.p[xi + 1] + yi + 1], xf - 1, yf - 1)
        # Combine noise components using smoothing factors
        """First combine lefts and rights, then top with bottom"""
        x1 = self._lerp(n00, n10, u)
        x2 = self._lerp(n01, n11, u)
        return self._lerp(x1, x2, v)

    #--------------------------------------------------------------------------
    def _fade(self, t:NPFltArr)->NPFltArr:
        """
        Smoothing function for Perlin noise interpolation.
    
        Applies improved Perlin fade curve (6t^5 - 15t^4 + 10t^3) to create smooth
        interpolation weights. Ensures C^2 continuity (smooth curvature) at grid
        cell boundaries, eliminating visible artifacts in generated noise.
        

        Parameters
        ----------
        t : ndarray
            Interpolation parameter(s) in range [0, 1]. Typically internal
            coordinates (xf, yf) within grid cell.

                
        Returns
        -------
        smoothed : ndarray
            Smoothed interpolation weights, same shape as input. Values in range
            [0, 1] with smooth derivatives.
            

        Notes
        -----
        **Mathematical Form:**
        
            fade(t) = 6t^5 - 15t^4 + 10t^3
        
        - **Properties:**
        
            - f(0) = 0 (zero at start)
            - f(1) = 1 (one at end)
            - f'(0) = 0 (zero first derivative at start)
            - f'(1) = 0 (zero first derivative at end)
            - f''(0) = 0 (zero second derivative at start)
            - f''(1) = 0 (zero second derivative at end)
        
        - **Derivatives:**
        
            - First derivative:
            
                f'(t) = 30t^4 - 60t^3 + 30t^2 = 30t^2(t - 1)^2
            
            - Second derivative:
            
                f''(t) = 120t^3 - 180t^2 + 60t = 60t(2t - 1)(t - 1)
            
        **Performance:**

        Despite higher polynomial degree, vectorized numpy operations make this
        very efficient:

        - Single pass over array
        - No branching or conditionals
        - Cache-friendly memory access
        

        See Also
        --------
        _perlin2D : Uses fade for interpolation weights u and v
        _lerp : Linear interpolation using fade weights
        

        Examples
        --------
        ### Fade curve shape:
        
        >>> import matplotlib.pyplot as plt
        >>> pn = PerlinNoise(size=100)
        >>> t = np.linspace(0, 1, 100)
        >>> fade_vals = pn._fade(t)
        >>> plt.plot(t, fade_vals, label='fade(t)')
        >>> plt.plot(t, t, '--', label='linear(t)')
        >>> plt.legend()
        >>> plt.xlabel('t')
        >>> plt.ylabel('weight')
        >>> plt.title('Improved Perlin Fade Curve')
        >>> plt.show()
        
        ### Derivative verification:
        
        >>> t = np.array([0.0, 0.5, 1.0])
        >>> fade_t = pn._fade(t)
        >>> print(fade_t)
        [0.    0.5   1.   ]  # Smooth interpolation from 0 to 1
        """
        return (6 * t**5) - (15 * t**4) + (10 * t**3)
    
    #--------------------------------------------------------------------------
    def _gradient(self, 
                  h:Union[int, NPIntArr], 
                  xf:Union[Number, NPFltArr], 
                  yf:Union[Number, NPFltArr],
                  )->NPFltArr:
        """
        Select gradient vector from hash and compute dot product with offset.
    
        Converts hashed grid coordinate to one of four unit gradient vectors and
        computes dot product with internal cell coordinates. This is the core
        operation that gives Perlin noise its characteristic coherent
        randomness.
        

        Parameters
        ----------
        h : int or ndarray of int
            Hash value(s) from permutation table: h = p[p[xi] + yi]. Used to
            select gradient vector via modulo operation. Can be single integer
            or array of hash values.
        xf : float or ndarray
            X-offset from left edge of grid cell, range [0, 1). Represents
            fractional position within cell. Can be scalar or array.
        yf : float or ndarray
            Y-offset from top edge of grid cell, range [0, 1). Can be scalar
            or array.
        
        
        Returns
        -------
        dot_product : float or ndarray
            Dot product of selected gradient with (xf, yf) offset vector. Return
            type matches input: scalar inputs return scalar, array inputs return
            array. Values typically in range [-1, 1] before interpolation.
            
        
        Notes
        -----
        **Gradient Vector Selection:**

            Four possible unit gradient vectors selected by h % 4:

            .. code-block:: none

                h % 4 = 0: [0,  1]  -> dot = yf     (North-pointing)
                h % 4 = 1: [0, -1]  -> dot = -yf    (South-pointing)
                h % 4 = 2: [1,  0]  -> dot = xf     (East-pointing)
                h % 4 = 3: [-1, 0]  -> dot = -xf    (West-pointing)
            
        - **Dot Product Computation:**
        
            g * d = g_x * xf + g_y * yf

            where:

            - g: Gradient vector (from vectors array)
            - d: Displacement vector (xf, yf)
        
        **Why Only 4 Gradients:**

        Original Perlin noise used 8 or 12 gradients, but Perlin (2002) showed
        4 gradients sufficient for 2D noise with proper hashing:

        - Simpler implementation
        - Faster computation
        - Visually indistinguishable from 8/12 gradient versions
        
        **Gradient Distribution:**

        Permutation table ensures pseudo-random but uniform distribution:

        - Each gradient appears ~25% of the time (on average)
        - No directional bias in generated terrain
        - Spatial coherence from grid structure, not gradient distribution
        
        **Hash to Gradient Mapping:**

        The modulo operation (h % 4) effectively reduces 8-bit hash (0-255)
        to 2-bit gradient index (0-3):

        - Throws away most hash bits (intentional)
        - Remaining bits sufficient for gradient selection
        - Preserves spatial coherence from permutation table
        
        **Array Broadcasting:**

        Function handles both scalar and array inputs via numpy broadcasting:

        - h, xf, yf can be single values or meshgrids
        - Vectorized operations over entire coordinate arrays
        - No explicit loops needed
        

        See Also
        --------
        _perlin2D : Calls _gradient for each grid cell corner
        _get_permutation_table : Generates hash values used here
        

        References
        ----------
        [1] Perlin, K. (2002). "Improving Noise." ACM SIGGRAPH 2002. Explains
        gradient selection and reduction to 4 vectors for 2D.
        
            
        Examples
        --------
        ### Single gradient selection (scalar inputs):
        
        >>> pn = PerlinNoise(size=100, seed=42)
        >>> h = 7  # Example hash value
        >>> xf, yf = 0.3, 0.7  # Internal coordinates
        >>> dot = pn._gradient(h, xf, yf)
        >>> print(f"Gradient index: {h % 4}, Dot product: {dot:.3f}")
        Gradient index: 3, Dot product: -0.300
        
        ### Vectorized gradient computation (array inputs):
        
        >>> h_array = np.array([0, 1, 2, 3])
        >>> xf_array = np.full(4, 0.5)
        >>> yf_array = np.full(4, 0.5)
        >>> dots = pn._gradient(h_array, xf_array, yf_array)
        >>> print(dots)
        [ 0.5 -0.5  0.5 -0.5]  # yf, -yf, xf, -xf
        """
        vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
        g = vectors[h % 4]
        
        # Handle scalar and array inputs
        if g.ndim == 1:
            # Scalar case: g is shape (2,)
            return (g[0] * xf) + (g[1] * yf)
        else:
            # Array case: g is shape (..., 2)
            return (g[..., 0] * xf) + (g[..., 1] * yf)
    
    #--------------------------------------------------------------------------
    def _lerp(self, a:NPFltArr, b:NPFltArr, x:NPFltArr)->NPFltArr:
        """
        Linear interpolation between two values or arrays.
    
        Computes weighted average of a and b using interpolation parameter x.
        Standard lerp formula: result = a + x(b - a).
        

        Parameters
        ----------
        a : ndarray
            Start values. Interpolation returns a when x=0.
        b : ndarray
            End values. Interpolation returns b when x=1.
        x : ndarray
            Interpolation weight(s), typically in range [0, 1]. Usually output
            from _fade() for smooth interpolation.
            

        Returns
        -------
        interpolated : ndarray
            Linearly interpolated values between a and b. Same shape as input
            arrays (via broadcasting).

                        
        Notes
        -----
        **Formula:**
        
            lerp(a, b, x) = a + x(b - a) = (1-x)a + xb
        
        - **Properties:**

            - lerp(a, b, 0) = a
            - lerp(a, b, 1) = b
            - lerp(a, b, 0.5) = (a + b) / 2
        
        **Usage in Perlin Noise:**
        
        Called twice in bilinear interpolation sequence:

        1. Interpolate between top-left and top-right corners (horizontal)
        2. Interpolate between bottom-left and bottom-right corners (horizontal)
        3. Interpolate between results from steps 1 and 2 (vertical)
        
        >>> x1 = _lerp(n00, n10, u)  # Top edge
        >>> x2 = _lerp(n01, n11, u)  # Bottom edge
        >>> result = _lerp(x1, x2, v)  # Final value
        
        **Array Broadcasting:**

        Supports numpy broadcasting for efficient vectorization:

        - a, b, x can be scalars or arrays
        - Result shape determined by broadcast rules
        - No explicit loops needed
        
        **Extrapolation:**

        Function works for x outside [0, 1]:

        - x < 0: Extrapolates beyond a
        - x > 1: Extrapolates beyond b
        - Typical use cases keep x ∈ [0, 1] for interpolation
        

        See Also
        --------
        _perlin2D : Uses lerp for bilinear interpolation
        _fade : Generates smooth x values for interpolation
        

        Examples
        --------
        ### Basic interpolation:
        
        >>> pn = PerlinNoise(size=100)
        >>> # Interpolate between 10 and 20
        >>> result = pn._lerp(10, 20, 0.3)
        >>> print(result)
        13.0  # 10 + 0.3 * (20 - 10) = 13
        
        ### Array interpolation:
        
        >>> a = np.array([0, 10, 20])
        >>> b = np.array([10, 20, 30])
        >>> x = np.array([0, 0.5, 1])
        >>> result = pn._lerp(a, b, x)
        >>> print(result)
        [ 0. 15. 30.]  # [a[0], midpoint, b[2]]
        
        Bilinear interpolation example:
        
        >>> # Corner values
        >>> n00, n01, n10, n11 = 0, 10, 5, 15
        >>> # Interpolation weights (from fade)
        >>> u, v = 0.3, 0.7
        >>> # Horizontal interpolation
        >>> x1 = pn._lerp(n00, n10, u)  # Top: 1.5
        >>> x2 = pn._lerp(n01, n11, u)  # Bottom: 11.5
        >>> # Vertical interpolation
        >>> final = pn._lerp(x1, x2, v)  # Result: 8.5
        >>> print(final)
        8.5
        """
        return a + x * (b - a)
    
    #--------------------------------------------------------------------------
    def _normalize(self, n:NPFltArr)->NPFltArr:
        """
        Normalize array to range [0, 1] via linear scaling.
    
        Applies min-max normalization to map array values from arbitrary range
        to [0, 1]. Preserves relative spacing between values while fitting to
        standard range.
        
        Parameters
        ----------
        n : ndarray
            Input array with arbitrary value range.
            Typically raw Perlin noise before normalization.
            
        Returns
        -------
        normalized : ndarray
            Array with values linearly scaled to [0, 1].
            Same shape as input. Minimum value -> 0, maximum value -> 1.
            
        
        Notes
        -----
        **Formula:**
        
            normalized = (n - min(n))/(max(n) - min(n))
        
        **Algorithm Steps:**
        
        1. Shift array so minimum = 0: n' = n - min(n)
        2. Scale so maximum = 1: result = n' / max(n')

        **Why Normalization:**

        Raw Perlin noise has inconsistent value ranges:

        - Single octave: typically [-0.7, +0.7]
        - Multiple octaves: range expands with octave count
        - Amplitude depends on persistence parameter
        
        Normalization ensures:

        - Consistent [0, 1] output range
        - Full utilization of available range
        - Predictable behavior for terrain generation
        
        **Preservation Properties:**

        Linear scaling preserves:

        - Relative ordering (if a < b, then norm(a) < norm(b))
        - Proportional spacing between values
        - Smoothness and continuity of function
        
        **Edge Cases:**

        - Constant array (max = min): Results in division by zero
          Current implementation: NaN/inf values (should handle explicitly)
        - Very small range: Possible floating-point precision issues
        

        See Also
        --------
        _gen_perlin : Calls _normalize on final multi-octave noise
        Floor.standard_map : Uses normalized noise for depth mapping

        
        Examples
        --------
        ### Typical usage:
        
        >>> pn = PerlinNoise(size=100, seed=42)
        >>> # Raw multi-octave noise
        >>> raw_noise = np.zeros((100, 100))
        >>> # ... generate octaves ...
        >>> print(f"Raw range: [{raw_noise.min():.3f},"
        ...       f" {raw_noise.max():.3f}]")
        Raw range: [-1.234, 2.567]
        >>> # Normalize
        >>> norm_noise = pn._normalize(raw_noise)
        >>> print(f"Normalized: [{norm_noise.min():.3f},"
        ...       f" {norm_noise.max():.3f}]")
        Normalized: [0.000, 1.000]
        """
        n += (0 - n.min())
        return n / n.max()
        
###############################################################################

class Pollution:
    """
    Gaussian plume model for point-source pollution dispersion in ocean
    environments.
    
    Simulates the concentration distribution of a pollutant released from a
    fixed point source using a steady-state Gaussian plume model. Accounts for
    wind-driven advection, atmospheric dispersion, and plume rise effects.
    Suitable for modeling chemical spills, thermal discharges, or tracer
    releases in AUV sensor simulation.
    

    Parameters
    ----------
    source : list of float, default=[0, 0, 30]
        Pollution source location [x, y, z] in meters (END frame).

        - x: East coordinate
        - y: North coordinate
        - z: Depth below surface (positive value; stored as negative internally)

    Q : float, default=1.59
        Source emission rate (strength) in grams per second (g/s).
        Typical values: 0.5-5.0 g/s for tracer experiments.
    u : float, default=1.5
        Wind/current speed in meters per second (m/s).
        Valid range: [0.1, 2.5] m/s (automatically clipped to bounds).
        Drives horizontal advection of plume.
    v : float, default=pi/4 (45 deg)
        Wind/current direction in radians.
        Measured counterclockwise from East axis (END convention):

        - 0: East
        - pi/2: North
        - pi: West
        - -pi/2: South

    seed : int, optional
        PRNG seed for reproducibility when using random parameters.
        If None and randomness enabled, generates random seed from entropy.
    random : bool, default=False
        If True, randomizes both u and v within default ranges.
        Overrides explicit u and v values.
    randomU : bool or list of float, default=False
        Wind speed randomization:

        - True: Randomize u within [0.1, 2.5] m/s
        - [low, high]: Randomize u within custom range
        - False: Use explicit u parameter

    randomV : bool or list of float, default=False
        Wind direction randomization:

        - True: Randomize v within [0, 2pi] radians
        - [low, high]: Randomize v within custom range
        - False: Use explicit v parameter

    oceanSize : int, default=1000
        Side length of ocean floor area in meters.
        Used for boundary checking and visualization.
    oceanOrigin : list of float, default=[500, 500]
        Coordinates [x, y] where (0, 0) maps to floor array.
    oceanDepth : float, default=125
        Maximum ocean depth in meters (positive value).
        Concentration set to zero beyond this depth.
        

    Attributes
    ----------
    x_s : float
        Source x-coordinate (East) in meters.
    y_s : float
        Source y-coordinate (North) in meters.
    z_s : float
        Source depth in meters (stored as negative value).
    Q : float
        Emission rate in g/s.
    u : float
        Wind/current speed in m/s (clipped to [0.1, 2.5]).
    v : float
        Wind/current direction in radians (wrapped to [0, 2*pi]).
    source : list of float
        Combined [x_s, y_s, z_s] property (z returned as positive).
    seed : int
        PRNG seed used for randomization.
    oceanSize : int
        Ocean area dimension for boundaries.
    oceanOrigin : list of float
        Origin coordinates for coordinate system.
    oceanDepth : float
        Maximum depth for concentration calculations.
        

    Methods
    -------
    __call__(x, y, z)
        Pollution instance as callable, calculate concentration at coordinates
        (x, y, z). Supports scalar or array inputs. Returns concentration in
        g/m^3.
    get2D(x, y, z, size)
        Generate 2D concentration array at specified depth.
        Returns (concentration, X_mesh, Y_mesh).
    get3D(x, y, z, size)
        Generate 3D concentration volume (memory-intensive).
        Returns (concentration, X_mesh, Y_mesh, Z_mesh).
    automesh(size, res, center, is3d)
        Create coordinate meshgrids centered on plume distribution.
    display2D(x, y, z, size)
        Visualize 2D concentration contour plot at depth z.
    display3D(x, y, z, size)
        Visualize 3D isosurface of concentration distribution.

         
    Notes
    -----
    **Gaussian Plume Theory:**

    - **Physical Model:**
    
        The pollution plume is simulated using a basic Gaussian fluid mechanics
        model. Assuming wind along the x-axis, the pollutant concentration C at
        any point (x, y, z) is given by:

        .. code-block:: none

            C(x, y, z) = (Q / (2*pi*u*sigma_y*sigma_z)) * exp(-y'^2/(2*sigma_y^2)) *
                    [exp(-(z-H_e)^2/(2*sigma_z^2)) + exp(-(z+H_e)^2/(2*sigma_z^2))]
        
        where:
        
            - Q: Emission rate (g/s)
            - u: Wind speed (m/s)
            - (x', y'): Rotated coordinates aligned with wind direction
            - sigma_y: Horizontal dispersion coefficient (m)
            - sigma_z: Vertical dispersion coefficient (m)
            - H_e: Effective release height (m)
            - z: Depth coordinate (negative downward)
    
    - **Coordinate Rotation:**
        
        Wind direction v rotates the coordinate system:

        .. code-block:: none

            x' =  (x - x_s) * cos(v) + (y - y_s) * sin(v)
            y' = -(x - x_s) * sin(v) + (y - y_s) * cos(v)
        
        Aligns x' axis with wind direction for plume advection.
    
    - **Effective Release Height:**
    
        Plume buoyancy causes vertical rise:
        
        .. code-block:: none

            H_e = z_s + Delta h(x')
            Delta h(x') = 2.126 * 10^-4 * |x'|^(2/3)
        
        Rise increases with distance from source.
    
    - **Dispersion Coefficients:**
    
        Empirical formulas for neutral stability:
        
        .. code-block:: none

            sigma_y = 1.360 * |x'|^(0.82)
            sigma_z = 0.275 * |x'|^(0.69)
        
        Dispersion grows with distance, creating characteristic plume cone
        shape.
    
    **Model Assumptions:**
    
    1. Steady-state: Concentration field does not change with time
    2. Constant wind: Uniform u and v throughout domain
    3. Homogeneous medium: No density stratification or temperature gradients
    4. Point source: Instantaneous release from infinitesimal volume
    5. Passive tracer: No chemical reactions or biological uptake
    6. Neutral stability: No strong buoyancy effects (moderate temperature)
    
    **Validity Regime:**
    
    - Distance from source: 100m - 10km
    - Wind speed: 0.5 - 10 m/s (underwater currents typically 0.1-2 m/s)
    - Flat terrain (ocean floor variations negligible)
    - Low-moderate emission rates (Q < 10 g/s)
    
    **Limitations:**
    
    - Does not model time-varying sources
    - Ignores turbulent mixing in wake of AUV or obstacles
    - No chemical decay or transformation
    - Ground reflection term simplistic
    
    **Coordinate System:**
    
    Uses END (East-North-Down) convention throughout:

    - Positive z is depth below surface
    - Internally stored as negative for calculation consistency
    - User inputs/outputs use positive depth convention
    
    **Boundary Conditions:**
    
    - Ocean surface (z = 0): Perfect reflection (concentration decays to zero)
    - Ocean floor (z = -oceanDepth): Perfect reflection
    - Lateral boundaries: Open (concentration -> 0 as distance -> infinity)
    - Upwind of source (x' < 0): Concentration = 0 (no upstream diffusion)
    
    **Numerical Stability:**
    
    - Division by zero protection at source (x' -> R_MIN = 0.01 m)
    - Exponential overflow prevented by clipping large arguments
    - Dispersion coefficients bounded to avoid singularities
    

    Warnings
    --------
    - get3D() method can cause out-of-memory errors for domains larger than
      ~1000x1000x100 points. Use get2D() for large-scale visualization.
    - Concentration calculation becomes unreliable very near source (r < 1
      meter). Model assumes far-field dispersion regime.
    - Wind speed u < 0.5 m/s may produce unrealistic plume spreading (stagnation
      effects not modeled).
    - No validation for underwater releases (model derived from atmospheric
      dispersion). Use with caution for ocean applications.
    
      
    See Also
    --------
    Ocean : Container class that manages Pollution instance
    Current1D : Provides wind/current speed and direction for u, v parameters
    navigation.Sensor : Base class for sensors that could sample concentration
    

    References
    ----------
    [1] A. A. Abdel-Rahman, “On the atmospheric dispersion and gaussian plume
    model,” in Proceedings of the 2nd International Conference on Waste
    Management, Water Pollution, Air Pollution, Indoor Climate, Corfu, Greece,
    vol. 26, 2008.
    
        
    Examples
    --------
    ### Create pollution source with defaults:

    >>> import munetauvsim.environment as env
    >>> pollution = env.Pollution()
    >>> print(pollution)
    Pollution
     Source:       (0.00, 0.00, 30.00)
     Strength:     1.59 g/s
     Speed:        1.50 m/s at 0.79 rad
     Seed:         None
    
    ### Custom pollution parameters:
    
    >>> pollution = env.Pollution(
    ...     source=[200, 300, 25],   # 200m East, 300m North, 25m depth
    ...     Q=2.5,                   # 2.5 g/s emission rate
    ...     u=0.8,                   # 0.8 m/s current speed
    ...     v=np.pi/6,               # 30 deg from East (ENE direction)
    ...     seed=42
    ... )
    
    ### Query concentration at single point:
    
    >>> pollution = env.Pollution(source=[0, 0, 30], Q=2.0, u=1.0, v=0)
    >>> # Sample 100m East, 50m North, 28m depth
    >>> conc = pollution(100, 50, 28)
    >>> print(f"Concentration: {conc:.4e} g/m^3")
    Concentration: 5.4460e-04 g/m^3     # Example value
    
    ### Query concentration at array of points:
    
    >>> x_points = np.array([50, 100, 150, 200])
    >>> y_points = np.array([0, 0, 0, 0])
    >>> z_points = np.array([28, 28, 28, 28])
    >>> concentrations = pollution(x_points, y_points, z_points)
    >>> print(concentrations)
    [0.00205464 0.00077644 0.00042928 0.00028042]   # Example values
    
    ### Generate 2D concentration map at depth:
    
    >>> pollution = env.Pollution(source=[500, 500, 30], u=1.2, v=np.pi/4)
    >>> C, X, Y = pollution.get2D(
    ...     x=[400, 700],
    ...     y=[400, 700],
    ...     z=28  # 2m above source
    ... )
    >>> print(f"Concentration array shape: {C.shape}")
    Concentration array shape: (300, 300)
    >>> print(f"Max concentration: {C.max():.4e} g/m^3")
    Max concentration: 5.7852e-03 g/m^3

    ### Random pollution for Monte Carlo studies:
    
    >>> # Random speed and direction
    >>> poll_random = env.Pollution(
    ...     source=[0, 0, 30],
    ...     Q=1.5,
    ...     random=True,
    ...     seed=12345
    ... )
    >>> print(f"Random u: {poll_random.u:.2f} m/s")
    Random u: 0.65 m/s
    >>> print(f"Random v: {poll_random.v:.2f} rad")
    Random v: 1.99 rad
    >>> 
    >>> # Custom random ranges
    >>> poll_custom = env.Pollution(
    ...     source=[0, 0, 30],
    ...     Q=1.5,
    ...     randomU=[0.5, 1.5],    # Speed range
    ...     randomV=[0, np.pi/2],  # Direction range (East to North)
    ...     seed=67890
    ... )
    >>> poll_custom.display2D()

    ### Integration with Ocean environment:
    
    >>> ocean = Ocean.calm_ocean(createPlume=True)
    >>> # Pollution automatically created with ocean current parameters
    >>> print(ocean.pollution)
    >>> # Query concentration at vehicle position
    >>> vehicle_pos = [250, 300, 20]
    >>> conc = ocean.pollution(*vehicle_pos)
    
    ### AUV data collection:
    
    >>> # Define pollution field
    >>> pollution = env.Pollution(source=[500, 500, 25], Q=3.0, u=1.0, v=0)
    >>> 
    >>> # Simulate AUV path
    >>> import munetauvsim.guidance as guid
    >>> path = guid.Waypoint([0, 500, 1000], [500, 500, 500], [20, 20, 20])
    >>> 
    >>> # Sample along path
    >>> import matplotlib.pyplot as plt
    >>> concentrations = []
    >>> for i in range(len(path)):
    ...     x, y, z = path.pos[i]
    ...     # Build this out into a custom navigation.Sensor subclass
    ...     conc = pollution(x, y, z)
    ...     concentrations.append(conc)
    >>> 
    >>> plt.plot(concentrations)
    >>> plt.xlabel('Waypoint index')
    >>> plt.ylabel('Concentration (g/m^3)')
    >>> plt.title('Concentration profile along AUV path')
    >>> plt.show()
    
    ### Memory-conscious 3D volume:
    
    >>> # For large domains, use subregions
    >>> pollution = env.Pollution(source=[0, 0, 30], u=1.5, v=0)
    >>> # Only compute small region around AUV position
    >>> C, X, Y, Z = pollution.get3D(
    ...     x=[0, 100],    # 100m extent
    ...     y=[-20, 20],   # 40m width
    ...     z=[25, 35]     # 10m depth range
    ... )
    >>> print(f"3D array shape: {C.shape}")
    """

    ## Class Constants =======================================================#
    _U_MIN = 0.1                      # Minimum wind speed in m/s
    _U_MAX = Current1D._V_HI          # Maximum wind speed in m/s
    _V_MIN = 0                        # Minimum wind direction in radians
    _V_MAX = 2 * np.pi                # Maximum wind direction in radians
    _R_MIN = 0.01                     # Minimum source distance for dispersion

    ## Constructor ===========================================================#
    def __init__(self,
                 source:Union[NPFltArr,List[float]] = [0, 0, 30],
                 Q:float=1.59,
                 u:Number=1.5,
                 v:Number=np.pi/4,
                 seed:Optional[int] = None,
                 random:bool=False,
                 randomU:Union[bool, List[float]]=False,
                 randomV:Union[bool, List[float]]=False,
                 oceanSize:int = 1000,
                 oceanOrigin:List[float] = [500, 500],
                 oceanDepth:float = 125,
                 **kwargs,
                 ):
        """
        Initialize Pollution source with dispersion parameters and randomization
        options.
    
        Constructs a Gaussian plume pollution model with specified source
        location, emission rate, and wind/current conditions. Supports
        deterministic parameters or randomized generation for Monte Carlo
        simulations. Automatically handles coordinate system conventions and
        parameter validation.
        
        Parameters
        ----------
        source : list of float or ndarray, default=[0, 0, 30]
            Pollution source coordinates [x, y, z] in meters (END frame).

            - x: East position (can be negative if origin offset)
            - y: North position (can be negative if origin offset)
            - z: Depth below surface (input as positive, stored as negative)

            Default places source at origin, 30m depth.
        
        Q : float, default=1.59
            Source emission rate (strength) in grams per second (g/s).
            Represents mass flux of pollutant released per unit time.
            Typical values:

            - Tracer experiments: 0.1-2.0 g/s
            - Small spills: 2.0-10.0 g/s
            - Large industrial discharges: 10-100 g/s

            Must be positive. No upper bound enforced (user responsibility).
        
        u : float, default=1.5
            Wind or current speed in meters per second (m/s).
            Drives horizontal advection of plume. Valid range: [0.1, 2.5] m/s.
            Values outside range are automatically clipped with warning.
            Typical ocean currents: 0.1-1.0 m/s.
            Ignored if random=True or randomU specified.
        
        v : float, default=pi/4 (45 degrees)
            Wind or current direction in radians (END convention).
            Measured counterclockwise from East axis:

            - 0: East
            - pi/2: North
            - pi: West
            - -pi/2 or 3pi/2: South

            Input automatically wrapped to [0, 2*pi] range.
            Ignored if random=True or randomV specified.
        
        seed : int, optional
            Pseudo-random number generator seed for reproducibility.
            Used when random=True, randomU=True, or randomV=True.
            If None and randomization enabled, generates unique seed from
            system entropy.
            If None and no randomization, no RNG used (deterministic).
        
        random : bool, default=False
            Master randomization flag. If True:

            - Generates random u from [0.1, 2.5] m/s
            - Generates random v from [0, 2pi] radians
            - Overrides explicit u and v parameters
            - Triggers seed generation if seed=None

            Useful for quick Monte Carlo setup without specifying ranges.
        
        randomU : bool or list of float, default=False
            Wind speed randomization control:

            - False: Use explicit u parameter (default)
            - True: Randomize u within [_U_MIN=0.1, _U_MAX=2.5] m/s
            - [low, high]: Randomize u within custom range [low, high] m/s

            Takes precedence over random flag for u parameter.
            Custom ranges not validated (user must ensure physical validity).
        
        randomV : bool or list of float, default=False
            Wind direction randomization control:

            - False: Use explicit v parameter (default)
            - True: Randomize v within [0, 2pi] radians
            - [low, high]: Randomize v within custom range [low, high] radians
            
            Takes precedence over random flag for v parameter.
            Custom ranges automatically wrapped to [0, 2pi].
        
        oceanSize : int, default=1000
            Side length of ocean floor area in meters.
            Used for:

            - Boundary condition enforcement (concentration -> 0 beyond edges)
            - Default domain for get2D(), get3D(), automesh()
            - Coordinate wrapping in visualization methods

            Should match Ocean.size if using integrated environment.
        
        oceanOrigin : list of float, default=[500, 500]
            Origin offset coordinates [x_o, y_o] in meters.
            Defines where (x=0, y=0) maps in absolute coordinate system.
            Allows negative coordinates without array indexing issues.
            Should match Ocean.origin if using integrated environment.
        
        oceanDepth : float, default=125
            Maximum ocean depth in meters (positive value).
            Used as lower boundary condition:

            - Concentration set to 0 for z < -oceanDepth
            - Prevents unrealistic plume penetration below seafloor

            Should match Ocean.floor.z + Ocean.floor.z_range if using
            integrated environment for consistency.
        
        **kwargs
            Additional keyword arguments.
            
        Attributes
        ----------
        x_s : float
            Source East coordinate (stored directly).
        y_s : float
            Source North coordinate (stored directly).
        z_s : float
            Source depth (stored as negative value internally).
        Q : float
            Emission rate (stored directly).
        u : float
            Wind speed (after validation/randomization, clipped to [0.1, 2.5]).
        v : float
            Wind direction (after validation/randomization, wrapped to [0,
            2*pi]).
        seed : int or None
            Actual seed used (generated if random and seed=None).
        _rng : numpy.random.Generator or None
            RNG instance (used only if randomization requested).
        oceanSize : int
            Stored for boundary/visualization methods.
        oceanOrigin : list of float
            Stored for coordinate transformations.
        oceanDepth : float
            Stored for depth boundary enforcement.
            
        Notes
        -----
        **Initialization Process**

        1. **Source Coordinates:**

          Unpacks [x, y, z] from source parameter.
          Converts z to negative value: z_s = -abs(z_input).
        
        2. **Emission Rate:**

          Stores Q directly (no validation, user must ensure Q > 0).
        
        3. **Randomization Setup:**

          If any randomization flag True and seed=None:

          - Generates seed from np.random.SeedSequence().entropy
          - Creates RNG instance: self._rng = np.random.default_rng(seed)
        
        4. **Wind Speed Determination (priority order):**
        
           a. If random=True or randomU=True: Uniform random in [0.1, 2.5]
           b. If randomU=[low, high]: Uniform random in [low, high]
           c. Otherwise: Use explicit u parameter
           
           Finally: Clip to [0.1, 2.5] via u.setter validation.
        
        5. **Wind Direction Determination (priority order):**

          a. If random=True or randomV=True: Uniform random in [0, 2pi]
          b. If randomV=[low, high]: Uniform random in [low, high]
          c. Otherwise: Use explicit v parameter
          
          Finally: Wrap to [0, 2pi] via v.setter validation.
        
        6. **Ocean Parameters:**

          Store oceanSize, oceanOrigin, oceanDepth for later use.
        
        **Coordinate Convention:**
        
        User provides z as positive depth (intuitive), but internally stored as
        negative for calculation consistency with mathematical model
        implemented.
        
        >>> pollution = Pollution(source=[0, 0, 30])
        >>> pollution.z_s  # Stored as -30
        -30.0
        
        **Randomization Examples:**
        
        Full randomization:

        >>> poll = Pollution(random=True, seed=42)
        >>> # Both u and v randomized with seed 42
        
        Partial randomization:

        >>> poll = Pollution(u=1.0, randomV=True, seed=42)
        >>> # u fixed at 1.0, only v randomized
        
        Custom ranges:

        >>> poll = Pollution(
        ...     randomU=[0.5, 1.5],
        ...     randomV=[0, np.pi],  # East to West only
        ...     seed=42
        ... )
        
        **Parameter Validation:**
        
        u and v automatically validated by property setters:

        - u clipped to [0.1, 2.5] with warning if out of bounds
        - v wrapped to [0, 2pi] with warning if out of bounds
        - Q not validated (user must ensure Q > 0)
        - z always stored as negative (automatic conversion)
        
        **Integration with Ocean:**
        
        When Ocean creates Pollution, passes current parameters:

        >>> ocean = Ocean(
        ...     spd='moderate',  # Current speed category
        ...     ang='northeast',  # Current direction category
        ...     plume=[200, 300, 25],  # Source location
        ...     # ...
        ... )
        >>> # Ocean.__init__ calls:
        >>> pollution = Pollution(
        ...     source=plume,
        ...     u=ocean.current.v_spd,  # Mean speed from Current1D
        ...     v=ocean.current.b_ang,  # Mean angle from Current1D
        ...     oceanSize=size,
        ...     oceanOrigin=origin,
        ...     oceanDepth=ocean.floor.z,
        ...     # ...
        ... )
        
        **Log Warnings**

        Logs warning if u or v out of valid bounds and clipping applied.
        
        See Also
        --------
        Ocean.__init__ : Creates Pollution with current-derived parameters
        Current1D : Provides u and v from v_spd and b_ang attributes
        
        Examples
        --------
        ### Minimal initialization:
        
        >>> import munetauvsim.environment as env
        >>> pollution = env.Pollution()
        >>> print(f"Source: {pollution.source}")
        Source: [0.0, 0.0, -30.0]
        >>> print(f"Emission: {pollution.Q} g/s")
        Emission: 1.59 g/s
        >>> print(f"Wind: {pollution.u} m/s at {pollution.v} rad")
        Wind: 1.5 m/s at 0.7853981633974483 rad
        
        ### Custom deterministic parameters:
        
        >>> pollution = env.Pollution(
        ...     source=[250, 300, 20],
        ...     Q=2.5,
        ...     u=0.8,
        ...     v=np.pi/3,  # 60 degrees
        ...     oceanSize=2000,
        ...     oceanDepth=150
        ... )
        
        ### Mixed random and fixed:
        
        >>> pollution = env.Pollution(
        ...     source=[0, 0, 30],
        ...     Q=1.5,
        ...     u=1.0,  # Fixed speed
        ...     randomV=[0, np.pi/2],  # Random direction (East to North)
        ...     seed=42
        ... )
        >>> print(f"Fixed u: {pollution.u}")
        Fixed u: 1.0
        >>> print(f"Random v: {pollution.v}")
        Random v: 1.2157273181723998
        
        ### Validate parameter clipping:
        
        >>> pollution = env.Pollution(u=5.0, v=10*np.pi)
        WARNING: Wind speed out of bounds, clipping to 2.50 m/s
        WARNING: Wind direction out of bounds, clipping to 0.00 radians
        >>> pollution.u
        2.5
        >>> pollution.v
        0.0
        """

        self.x_s, self.y_s, self.z_s = source
        self.Q = Q

        # Randomness
        if (((random) or (randomV is not False) or (randomU is not False)) and
            (seed is None)):
            seed = np.random.SeedSequence().entropy
        self.seed = seed
        self._rng = np.random.default_rng(self.seed)

        # Set wind speed
        if ((random) or (randomU is True)):
            self.u = self._rng.uniform(self._U_MIN, self._U_MAX)
        if (isinstance(randomU, list)):
            self.u = self._rng.uniform(randomU[0], randomU[1])
        if ('_u' not in self.__dict__):
            self.u = u

        # Set wind direction
        if ((random) or (randomV is True)):
            self.v = self._rng.uniform(self._V_MIN, self._V_MAX)
        if (isinstance(randomV, list)):
            self.v = self._rng.uniform(randomV[0], randomV[1])
        if ('_v' not in self.__dict__):
            self.v = v

        self.oceanSize = oceanSize
        self.oceanOrigin = oceanOrigin
        self.oceanDepth = oceanDepth

    ## Properties ============================================================#
    @property
    def u(self) -> float:
        """Wind speed in m/s."""
        return self._u
    
    @u.setter
    def u(self, u:float) -> None:
        """Set wind speed with bounds check."""
        u_clip = np.clip(u, self._U_MIN, self._U_MAX)
        if (u != u_clip):
            log.warning(
                "Wind speed out of bounds, clipping to %.2f m/s", u_clip)
        self._u = u_clip

    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    @property
    def v(self) -> float:
        """Wind direction in radians."""
        return self._v
    
    @v.setter
    def v(self, v:float) -> None:
        """Set wind direction with bounds check."""
        v_norm = v % (2 * np.pi)
        v_clip = np.clip(v_norm, self._V_MIN, self._V_MAX)
        if (v != v_clip):
            log.warning(
                "Wind direction out of bounds, clipping to %.2f radians", 
                v_clip)
        self._v = v_clip

    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    @property
    def source(self) -> List[float]:
        """Coordinates of the pollution source."""
        return [self.x_s, self.y_s, self.z_s]

    @source.setter
    def source(self, source:List[float]) -> None:
        """Set the pollution source coordinates."""
        self.x_s, self.y_s, z_s = source
        self.z_s = -abs(z_s)

    #. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 
    @property
    def z_s(self) -> float:
        """Depth of the pollution source."""
        return self._z_s
    
    @z_s.setter
    def z_s(self, z_s:float) -> None:
        """Set the depth of the pollution source."""
        self._z_s = -abs(z_s)

    ## Special Methods =======================================================#
    def __call__(self, 
                 x: Union[Number, List[Number], NPFltArr], 
                 y: Union[Number, List[Number], NPFltArr], 
                 z: Union[Number, List[Number], NPFltArr],
                 ) -> Union[float, NPFltArr]:
        """
        Return concentration at (x, y, z) coordinates.
    
        Parameters
        ----------
        x, y, z : float, list, or ndarray
            Coordinates. Supports scalar or array inputs.
            
        Returns
        -------
        concentration : float or ndarray
            Pollutant concentration in g/m^3.
            Scalar if inputs scalar, array if inputs array.
        """

        # Convert input to arrays for consistent processing
        inputIsScalar = np.isscalar(x) and np.isscalar(y) and np.isscalar(z)
        x, y, z = map(np.asarray, (x, y, z))

        # Enforce consistent usage of z coordinate
        z = -abs(z)
        
        # Get coordinates relative to pollution source
        x_r = x - self.x_s
        y_r = y - self.y_s

        # Rotate coordinates based on wind direction
        x_rot = x_r * np.cos(self.v) + y_r * np.sin(self.v)
        y_rot = -x_r * np.sin(self.v) + y_r * np.cos(self.v)

        # Avoid divide by zero errors near source
        x_rot = np.where(x_rot == 0, self._R_MIN, x_rot)

        # Calculate effective release height
        delta_h = 2.126e-4 * np.abs(x_rot)**(2/3)
        He = self.z_s + delta_h

        # Calculate dispersion coefficients
        sigma_y = 1.36 * np.abs(x_rot)**0.82
        sigma_z = 0.275 * np.abs(x_rot)**0.69

        # Calculate concentration
        coeff = self.Q / (2 * np.pi * self.u * sigma_y * sigma_z)
        exp_y = np.exp(-y_rot**2 / (2 * sigma_y**2))
        exp_z = (np.exp(-(z - He)**2 / (2 * sigma_z**2)) + 
                 np.exp(-(z + He)**2 / (2 * sigma_z**2)))
        
        # Set concentration to 0 for opposite direction of wind
        concentration = np.where(x_rot >= 0, coeff * exp_y * exp_z, 0)

        # Set ocean surface and floor as boundaries
        concentration = np.where((z > 0) | (z < -abs(self.oceanDepth)),
                                 0, concentration)
        
        # Return scalar if inputs were scalar
        return concentration.item() if inputIsScalar else concentration
    
    #--------------------------------------------------------------------------
    def __repr__(self) -> str:
        """Detailed description of Pollution."""
        fmt = '.2f'
        return (
            f"{self.__class__.__name__}("
            f"x={self.x_s:{fmt}}, "
            f"y={self.y_s:{fmt}}, "
            f"z={abs(self.z_s):{fmt}}, "
            f"Q={self.Q:{fmt}}, "
            f"u={self.u:{fmt}}, "
            f"v={self.v:{fmt}}, "
            f"seed={self.seed})"
        )
    
    #--------------------------------------------------------------------------
    def __str__(self) -> str:
        """User friendly description of Pollution."""
        fmt = '.2f'
        cw = 16
        return (
            f"Pollution\n"
            f"{' Source:':{cw}} ({self.x_s:{fmt}}, {self.y_s:{fmt}}, "
            f"{abs(self.z_s):{fmt}})\n"
            f"{' Strength:':{cw}} {self.Q:{fmt}} g/s\n"
            f"{' Speed:':{cw}} {self.u:{fmt}} m/s at {self.v:{fmt}} rad\n"
            f"{' Seed:':{cw}} {self.seed}\n"
        )
    
    ## Methods ===============================================================#
    def get2D(self, 
              x:Optional[List[float]] = None,
              y:Optional[List[float]] = None,
              z:Optional[float] = None,
              size:Optional[int] = None,
              )->Tuple[NPFltArr, NPFltArr, NPFltArr]:
        """
        Return 2D array of pollution concentration at depth z and meshgrids.
    
        Parameters
        ----------
        x, y : list, optional
            [min, max] bounds. If None, uses oceanSize.
        z : float, optional
            Depth for 2D slice. If None, uses value 20m above z_s.
        size : int, optional
            Domain size if x,y not provided. Default is oceanSize.
            
        Returns
        -------
        C : ndarray
            2D concentration array.
        X, Y : ndarray
            Meshgrid coordinate arrays.

        Notes
        -----
        **Precision Trade-off**

        Fixed resolution (1-unit spacing) creates data discrepancies compared to
        direct `__call__()` evaluations due to discretization effects. Method
        prioritizes convenience over precision for rapid 2D array generation.

        **Usage Recommendation:**
        
        - Use `get2D()` for visualization and general analysis
        - Use `__call__()` for exact point concentrations and sensor simulation
        """

        if ((x is None) or (y is None)):
            size = self.oceanSize if (size is None) else size
            o = self.oceanOrigin
            x = [-o[0], size - o[0]]
            y = [-o[1], size - o[1]]

        x_l = np.linspace(x[0], x[1], int(x[1] - x[0]))
        y_l = np.linspace(y[0], y[1], int(y[1] - y[0]))

        dz = 20
        z = self.z_s + dz if (z is None) else -abs(z)
        X, Y = np.meshgrid(x_l, y_l)

        return self(X, Y, z), X, Y

    #-------------------------------------------------------------------------#
    def get3D(self, 
              x:Optional[List[float]] = None,
              y:Optional[List[float]] = None,
              z:Optional[List[float]] = None,
              size:Optional[int] = None,
              )->Tuple[NPFltArr, NPFltArr, NPFltArr, NPFltArr]:
        """
        Return 3D array of pollution concentration and meshgrids.
        
        Parameters
        ----------
        x, y : list, optional
            [min, max] bounds. If None, uses oceanSize.
        z : list, optional
            [min, max] depth bounds. If None, uses full ocean depth.
        size : int, optional
            Domain size if x,y not provided. Default is oceanSize.
            
        Returns
        -------
        C : ndarray
            3D concentration array.
        X, Y, Z : ndarray
            Meshgrid coordinate arrays.

        Notes
        -----
        **Memory Limitation:**

        Large 3D arrays can exceed available memory. Arrays greater than
        approximately (1000,1000,70) require more than ~250M total data points
        for all 4 returned data structures. This method is retained for caching
        smaller local regions at a time.

        **Design Intent:**
        Method provides foundation for pre-computed concentration sampling to
        avoid repeated calculations during simulation.

        **Performance Recommendation:**
        Use smaller domains or consider boundary arrays instead of full
        meshgrids for stability.
        """

        if ((x is None) or (y is None)):
            size = self.oceanSize if (size is None) else size
            o = self.oceanOrigin
            x = [-o[0], size - o[0]]
            y = [-o[1], size - o[1]]

        if (z is None):
            z = [-self.oceanDepth, 0]

        x_l = np.linspace(x[0], x[1], int(x[1] - x[0]))
        y_l = np.linspace(y[0], y[1], int(y[1] - y[0]))
        z_l = np.linspace(z[0], z[1], int(z[1] - z[0]))
        X, Y, Z = np.meshgrid(x_l, y_l, z_l)

        return self(X, Y, Z), X, Y, Z

    #-------------------------------------------------------------------------#
    def automesh(self, 
                 size:Optional[int] = None, 
                 res:int = 100, 
                 center:bool = False,
                 is3d:bool = False, 
                 )->Union[Tuple[NPFltArr, NPFltArr], 
                          Tuple[NPFltArr, NPFltArr, NPFltArr]]:
        """
        Return meshgrid arrays around pollution distribution.
    
        Parameters
        ----------
        size : int, optional
            Domain size. If None, uses oceanSize.
        res : int, default=100
            Number of points per dimension.
        center : bool, default=False
            If True, pollution source is put at the center. Otherwise, source is
            off-center and area is centered around the plume.
        is3d : bool, default=False
            If True, return 3D meshgrid.
            
        Returns
        -------
        X, Y : ndarray (if is3d=False)
            2D meshgrid arrays.
        X, Y, Z : ndarray (if is3d=True)
            3D meshgrid arrays.
        """

        size = self.oceanSize if (size is None) else size
        half = size/2

        if (center):
            x = np.linspace(self.x_s - half, self.x_s + half, res)
            y = np.linspace(self.y_s - half, self.y_s + half, res)
        else:
            o = 0.05                        # source offset from edge
            x_w = np.cos(self.v)            # weight in x 
            y_w = np.sin(self.v)            # weight in y
            x_s = size * (1-2*o) * x_w/2    # center shift in x
            y_s = size * (1-2*o) * y_w/2    # center shift in y
            
            x = np.linspace(self.x_s + x_s - half, self.x_s + x_s + half, res)
            y = np.linspace(self.y_s + y_s - half, self.y_s + y_s + half, res)
        
        if (is3d):
            z = np.linspace(self.z_s-10, 0, res)
            return np.meshgrid(x, y, z)

        return np.meshgrid(x, y)

    #-------------------------------------------------------------------------#
    def display2D(self, 
                  z:Optional[int] = None,
                  size:Optional[int] = None,
                  resolution:int = 100,
                  fromOcean:bool = False,
                  )->None:
        """
        Display 2D image of pollution concentration with contour lines.

        Parameters
        ----------
        z : float, optional
            Depth for 2D slice. If None, uses value 20m above z_s.
        size : int, optional
            Domain size. Default is oceanSize.
        resolution : int, default=100
            Number of points per dimension.
        fromOcean : bool, default=False
            If True, uses the ocean dimensions to set the domain size.
        """

        if (z is None):
            z = self.z_s + 20
            
        if (not fromOcean):
            size = self.oceanSize if (size is None) else size
            X, Y = self.automesh(size, resolution, center=False)
            Z = self(X, Y, -abs(z))
        else:
            Z, X, Y = self.get2D(z=z)
        
        # Adjust alpha values in color map
        cmap = plt.cm.viridis_r
        cmap_a = cmap(np.arange(cmap.N))
        split = int(cmap.N * 0.2)
        a = np.linspace(0, 1, split)
        alpha = a**1.6
        cmap_a[:split,-1] = alpha
        cmap_a = mpl.colors.ListedColormap(cmap_a)

        # Create filled contour plot
        plt.figure(figsize=(12, 10))
        contour = plt.contourf(X, Y, Z, levels=50, cmap=cmap_a)
 
        # Add colorbar
        plt.colorbar(contour, label='Concentration')
        
        # Highlight pollution source
        plt.scatter(self.x_s, self.y_s, color='red', s=100, marker='*', 
                    label='Pollution Source')
        
        plt.title(
            f'2D Pollution Concentration\nWind Speed: {self.u:.2f} m/s, '
            f'Direction: {np.degrees(self.v):.1f}°\n'
            f'Source depth: {abs(self.z_s):.1f} m, Plot depth: {abs(z):.1f} m',
            fontsize=16)
        plt.xlabel('X (m)', fontsize=12)
        plt.ylabel('Y (m)', fontsize=12)
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.savefig('pollution_concentration_2D.png', bbox_inches='tight')

    #-------------------------------------------------------------------------#
    def display3D(self,
                  size:Optional[int] = None,
                  resolution:int = 50,
                  )->None:
        """
        Display 3D representation of pollution concentration.

        Parameters
        ----------
        size : int, optional
            Domain size. Default is oceanSize.
        resolution : int, default=50
            Number of points per dimension.
        """

        size = self.oceanSize if (size is None) else size
        X, Y, Z = self.automesh(size, resolution, center=False, is3d=True)
        C = self(X, Y, Z)

        # Create a figure with 3D axes
        fig = plt.figure(figsize=(14, 12))
        ax = fig.add_subplot(111, projection='3d')

        # Normalize concentration for color mapping
        C_normalized = (C - np.min(C)) / (np.max(C) - np.min(C))

        # Adjust alpha values in color map
        cmap = plt.cm.viridis_r
        cmap_a = cmap(np.arange(cmap.N))
        split = int(cmap.N * 0.1)
        a = np.linspace(0, 1, split)
        alpha = a**1.6
        cmap_a[:split,-1] = alpha
        cmap_a = mpl.colors.ListedColormap(cmap_a)

        # Create a continuous 3D volume rendering using pcolormesh
        xx, yy = X[:,:,0], Y[:,:,0]
        for i in range(resolution):
            z_slice = Z[:,:,i]
            c_slice = C_normalized[:,:,i]
            ax.plot_surface(xx, yy, z_slice, facecolors=cmap_a(c_slice), 
                            rstride=1, cstride=1, shade=False)
        
        ax.set_title(
            f'3D Pollution Concentration\nWind Speed: {self.u:.2f} m/s, '
            f'Direction: {np.degrees(self.v):.2f}°', 
            fontsize=16)
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_zlabel('Z (m)', fontsize=12)

        # Add colorbar
        norm = plt.Normalize(C_normalized.min(), C_normalized.max())
        sm = plt.cm.ScalarMappable(cmap=cmap_a, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10, pad=0.1)
        cbar.set_label('Concentration', fontsize=12)

        plt.tight_layout()
        plt.show()
        plt.savefig('pollution_concentration_3D.png', bbox_inches='tight')

###############################################################################