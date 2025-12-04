"""
Vehicle classes for AUV simulation with swarm coordination.

Implements abstract and concrete vehicle classes for autonomous underwater
vehicle simulation, including 6-DOF dynamics, modular GNC architecture, sensor
integration, and multi-agent communication capabilities.


Classes
-------
Vehicle
    Abstract base class for vehicle hierarchy.
AUV
    Abstract AUV class with sensor management and GNC interfaces.
Remus100s
    Concrete Remus 100 AUV implementation with full dynamics.
Model
    Lightweight vehicle state model for swarm coordination.

    
Functions
---------
buildGroup(num, gid, hasLeader, vehType, kwargs)
    Create a list of vehicle instances for multi-agent simulation.


Notes
-----
Based on Fossen's marine vehicle dynamics formulation and Python Vehicle
Simulator.


References
----------
[1] Fossen, T.I. Python Vehicle Simulator. GitHub repository.
https://github.com/cybergalactic/PythonVehicleSimulator

[2] Fossen, T.I. (2021). Handbook of Marine Craft Hydrodynamics and Motion 
Control. 2nd Edition, Wiley. https://www.fossen.biz/wiley
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from numpy.typing import NDArray
from abc import ABC, abstractmethod
import numpy as np
import os
import inspect
import math
from munetauvsim import communication as comm
from munetauvsim import navigation as nav
from munetauvsim import environment as env
from munetauvsim import guidance as guid
from munetauvsim import control as ctrl
from munetauvsim import aquanet_lib as aqnet
from munetauvsim import gnc
from munetauvsim import logger

###############################################################################

# Type Aliases
NPFltArr = NDArray[np.float64]

# Global Variables
log = logger.addLog('veh')

###############################################################################

class Vehicle(ABC):
    """Abstract base class for structural hierarchy of vehicle classes."""

    @abstractmethod
    def __init__(self)->None:
        """Define and set vehicle attributes."""

###############################################################################

class Model(Vehicle):
    """
    Lightweight data model representing another vehicle's state in multi-agent
    simulations.

    The Model class stores data about another vehicle for use in swarm
    coordination where vehicles maintain internal models of their neighbors'
    states based on received communication messages. Includes historical logging
    of position and velocity reports with automatic capacity expansion.


    Parameters
    ----------
    vehObj : Vehicle
        The source vehicle to create a model from. Copies the following
        attributes into the model: callSign, id, groupId, isLeader, eta,
        velocity, nextEta, nextVel. 

        
    Attributes
    ----------
    **Identity and Status:**

        callsign : str
            Vehicle's unique identifier
        id : int
            Vehicle ID number
        groupId : str
            Swarm group identifier
        isLeader : bool
            True if vehicle is a group leader
    
    **State Vectors:**

        eta : ndarray, shape (3,)
            Vehicle position [x,y,z] in END frame (m)
        velocity : ndarray, shape (3,)
            Vehicle velocity [vx, vy, vz] in END frame (m/s)
        nextEta : ndarray
            Projected next position (if available)
        nextVel : ndarray
            Projected next velocity (if available)

    **Communication:**

        nodeAddr : int
            Simulation comms address.
        timeLastMsg : float
            Timestamp the last received message was sent from this vehicle (s)
        delayLastMsg : float
            Time elapsed since last received message (s)
        rprtRecv : bool
            Flag indicating last report successfully received

    **Data Logs:**

        etaLog : ndarray, shape (logCap, 4)
            Log of received position reports, [x, y, z, time]
        velLog : ndarray, shape (logCap, 5)
            Log of received velocity reports, [vx, vy, vz, speed, time]
        logSize : int
            Current number of filled log entries, (0 to logCap)
        logCap : int
            Current capacity of preallocated log arrays, starts at 512 and doubles
            when full

            
    Notes
    -----
    - Uses __slots__ to minimize memory overhead. New attributes cannot
      by dynamically added, they must be defined in the class and added to the
      __slots__ declaration.
    - When logs reach capacity, both are automatically doubled in size by the
      communication.writeEtaVelLogs() function.
    """

    ## Define slots
    __slots__ = ('callSign','id','groupId','nodeAddr','isLeader',
                 'eta','velocity','nextEta','nextVel',
                 'timeLastMsg','delayLastMsg','etaLog','velLog','rprtRecv',
                 '_logSize','_logCap')

    ## Constructor
    def __init__(self,vehObj:Vehicle)->None:
        """
        Create vehicle model from existing vehicle object.
        

        Parameters
        ----------
        vehObj : Vehicle
            Source vehicle to copy attributes from (callSign, id, groupId,
            nodeAddr, isLeader, eta, velocity, nextEta, nextVel).

             
        Notes
        -----
        Initializes communication tracking (timeLastMsg, delayLastMsg) and
        preallocates log arrays with 512-row capacity.
        """

        # Copy Existing Attributes
        self.callSign = vehObj.callSign
        self.id = vehObj.id
        self.groupId = vehObj.groupId
        self.nodeAddr = vehObj.nodeAddr
        self.isLeader = vehObj.isLeader
        self.eta = np.copy(vehObj.eta[:3])
        self.velocity = np.copy(vehObj.velocity[:3])
        self.nextEta = np.copy(vehObj.nextEta)
        self.nextVel = np.copy(vehObj.nextVel)
        # New Attributes for Modelling
        self._logSize = 0
        self._logCap = 512
        self.timeLastMsg = 0.0
        self.delayLastMsg = 0.0
        self.etaLog = np.zeros((self._logCap, 4))
        self.velLog = np.zeros((self._logCap, 5))
        self.rprtRecv = False

    ## Special Methods
    def __repr__(self)->str:
        """Return concise string representation of Model object."""
        return f"<{self.__class__.__name__} {self.id} at {hex(id(self))}>"

###############################################################################

class AUV(Vehicle):
    """
    Abstract base class for autonomous underwater vehicles.
    
    Provides sensor management, GNC interfaces, and common vehicle attributes.
    Cannot be instantiated directly; must be subclassed by concrete vehicle
    types.
    

    Attributes
    ----------
    id : int
        Unique vehicle identifier.
    groupId : str
        Swarm group identifier.
    isLeader : bool
        Leadership status in swarm.
    callSign : str
        Custom unique vehicle identifier.
    eta : ndarray, shape (6,)
        Position and attitude [x, y, z, phi, theta, psi].
    nu : ndarray, shape (6,)
        Body-frame velocities [u, v, w, p, q, r].
    sensors : dict
        Installed sensor objects keyed by name.
    GuidSystem : callable
        Guidance system function (assigned via load methods).
    CommNetwork : object
        Communication network manager.
        

    Methods
    -------
    addSensor(name, sensor)
        Install a sensor on the vehicle.
    readSensor(name, kwargs)
        Read data from a specific sensor.
    collectSensorData(ocean, i)
        Update vehicle state from sensors (abstract method).
    """

    ## Constructor ===========================================================#
    @abstractmethod
    def __init__(self)->None:
        """
        Initialize base AUV attributes.
        
        Sets up identity attributes, state vectors, sensor dictionary, and
        GNC function handle placeholders. All assigned method attributes
        (GuidSystem, CommNetwork, etc.) are initialized to None.
        

        Notes
        -----
        This is an abstract class constructor called by subclass __init__.
        Concrete vehicles (e.g., Remus100s) must call super().__init__().
        """

        #---------------------------------------------------------------------#
        #   Identity                                                          #
        #---------------------------------------------------------------------#
        self.info = {}                              # instance data dictionary
        self.id = None                              # id number
        self.groupId = None                         # swarm group ID
        self.isLeader = False                       # leader status
        self.callSign = None                        # identification string

        #---------------------------------------------------------------------#
        #   Assigned Methods                                                  #
        #---------------------------------------------------------------------#
        ## Communication
        self.CommNetwork = None                     # Comms Network Manager
        self.CommSched = None                       # Communications Scheduler
        
        ## Guidance
        self.GuidSystem = None                      # Guidance System
        self.GuidLaw = None                         # Guidance Law

        ## Navigation
        self.DepthObs = None                        # Depth Observer
        self.HeadingObs = None                      # Heading Observer
        self.Attitude = nav.attitudeEuler           # Attitude Propagation

        ## Control
        self.DepthAP = None                         # Depth Auto Pilot
        self.HeadingAP = None                       # Heading Auto Pilot
        self.PropCmd = None                         # Propeller Command

        #---------------------------------------------------------------------#
        #   Communication                                                     #
        #---------------------------------------------------------------------#
        ## Network
        self.txAddr = None              # Transmission Address
        self.nodeAddr = None            # Local Node Address
        self.aqBaseDir = None           # Aquanet Files Base Directory
        self.monitorThread = None       # Incoming Message Monitor (thread)

        ## Message Schedules
        self.epsdDur = None             # Episode Cycle Duration (s)
        self.bcrqDur = None             # BCST/RQST Duration (s)
        self.rspnRqst = None            # RSPN REQUEST Received (Follower)(T/F)
        self.rspnSched = None           # RSPN Schedule
        self.rspnFrame = None           # RSPN Frame Start Time (s)
        self.rspnSlot = None            # RSPN Slot Start Time (s)
        self.rspnDur = None             # RSPN Slot Duration (s)

        ## Data Integrity
        self.dataRepair = {}            # Corrupt data handling strategies

        #---------------------------------------------------------------------#
        #   Guidance                                                          #
        #---------------------------------------------------------------------#
        ## Waypoints
        self.wpt = guid.Waypoint(0,0,0) # waypoint database
        self.wpt_k = 0                  # index to previous active waypoint
        self.R_switch = 10.0            # waypoint acceptance radius (m)

        ## Heading
        self.beta_c_hat = 0             # estimate of the crab angle
        self.Delta = 10.0               # positive look-ahead distance (m)
        self.gamma = 0.0006             # positive adaptive gain constant

        ## Swarm
        self.target = None              # target vehicle for tracking
        self.group = None               # list of swarm followers
        self.nextEta = None             # next position
        self.nextVel = None             # next velocity
        self.r_safe = 10.0              # minimum safe vehicle distance (m)
        self.r_avoid = 40.0             # avoidance radius (m)
        self.r_follow = 100.0           # following distance (m)

        #---------------------------------------------------------------------#
        #   Navigation                                                        #
        #---------------------------------------------------------------------#
        ## State
        """
        eta=[e,n,d,phi,theta,psi]:      x, y, z, roll, pitch, yaw
        nu=[u,v,w,p,q,r]:               linear & angular velocities: BODY frame
        velocity=[x_dot,y_dot,z_dot]    linear velocities: NED frame
        """
        self.eta = np.zeros(6)              # position & attitude vector
        self.nu = np.zeros(6)               # BODY velocity vector
        self.velocity = np.zeros(3)         # NED velocity vector
        self.clock = 0                      # simulation time
        self.immobilized = False            # mobility status flag
        self.sensors = {}                   # installed sensors

        ## State Estimators
        self.sampleTime = 0.02              # simulation iteration time step
        self.wn_d_z = 1/self.sampleTime #1/20# desired natural frequency, depth
        """wn_d_z also gets recalculated by the @sampleTime.setter"""
        self.z_d = 0.0                      # desired depth command
        self.theta_d = 0.0                  # desired pitch angle command
        self.psi_d = 0.0                    # desired heading angle command
        self.r_d = 0.0                      # desired yaw rate (rad/s)
        
        ## Environment
        self.V_c = 0                    # ocean current speed (m/s)
        self.beta_V_c = 0               # ocean current direction (rad)
        self.z_bed = 6000               # ocean floor depth (m)
        self.z_safe = 5                 # safety distance from ocean floor (m)

        #---------------------------------------------------------------------#
        #   Physics                                                           #
        #---------------------------------------------------------------------#
        ## Constants
        self.D2R = math.pi / 180        # deg2rad
        self.rho = 1026                 # density of water (kg/m^3)
        self.g = 9.81                   # acceleration of gravity (m/s^2)

    ## Properties ============================================================#
    @property
    def id(self)->int:
        """Unique vehicle ID number"""
        return self._id
    
    @id.setter
    def id(self, uid:int)->None:
        """Set unique vehicle ID number. Can only be set once."""
        if (('_id' not in self.__dict__) or
            ((self._id is None) and (uid is not None))):
            self._id = uid

    #--------------------------------------------------------------------------
    @property
    def groupId(self)->str:
        """The vehicle group ID"""
        return self._groupId

    @groupId.setter
    def groupId(self, gid:str)->None:
        """Set vehicle group ID and clear cached default callsign"""
        self._defaultCallSign = None
        self._groupId = gid

    #--------------------------------------------------------------------------
    @property
    def isLeader(self)->bool:
        """The vehicle Leader status"""
        return self._isLeader

    @isLeader.setter
    def isLeader(self, val:bool)->None:
        """Set vehicle Leader status and clear cached default callsign"""
        self._defaultCallSign = None
        self._isLeader = val

    #--------------------------------------------------------------------------
    @property
    def callSign(self)->str:
        """Unique vehicle identifier"""
        if (self._callSign is None):
            if (('_defaultCallSign' not in self.__dict__) or
                (self._defaultCallSign is None)):
                idGroup = str(self.groupId).upper() if (self.groupId) else ''
                idNum = f'{self.id:02}'
                isLeader = '-LEADER'if (self.isLeader) else ''
                self._defaultCallSign = idGroup+idNum+isLeader
            return self._defaultCallSign
        return self._callSign
    
    @callSign.setter
    def callSign(self, identifier:str)->None:
        """Set the vehicle identifier"""
        self._callSign = identifier
        if (identifier is not None):
            self.info.update([("Callsign", f"{identifier}")])

    #--------------------------------------------------------------------------
    @property
    def sampleTime(self)->float:
        """The iteration loop time step (seconds), default: 0.02"""
        return self._sampleTime
    
    @sampleTime.setter
    def sampleTime(self, n:float)->None:
        """Set the time step for simulation iteration loop (seconds)"""
        self._sampleTime = n
        self.wn_d_z = 1/n              #NOTE: This may be unique to Remus100s
    
    #--------------------------------------------------------------------------
    @property
    def Delta(self):
        """Positive look-ahead distance used by ALOS (m), typically 5-20 m"""
        return self._Delta
    
    @Delta.setter
    def Delta(self, delta:float)->None:
        """Set the positive look-ahead distance used by ALOS (m)"""
        if (delta <= 0):
            msg = "Delta must be greater than zero to avoid division error"
            log.critical(msg)
            raise ValueError(msg)
        self._Delta = delta

    #--------------------------------------------------------------------------
    @property
    def target(self)->Optional[Union[Model,Vehicle]]:
        """The vehicle target for tracking"""
        return self._target

    @target.setter
    def target(self, v:Optional[Union[Model,Vehicle]])->None:
        """Set the vehicle target for tracking"""
        if ((self.CommNetwork is not None) and 
            (not isinstance(v, Model))):
            self._target = Model(v)
        else:
            self._target = v
    
    #--------------------------------------------------------------------------
    @property
    def CommNetwork(self)->Optional[Union[aqnet.AquaNetManager, comm.MuNode]]:
        """The Communication Network Manager"""
        return self._CommNetwork

    @CommNetwork.setter
    def CommNetwork(self, comnet:Any)->None:
        """Set the Communication Network Manager"""
        self._CommNetwork = comnet
        # Re-set target to ensure properly using Model as required
        if (("_target" in self.__dict__) and
            (self._target is not None)):
            self.target = self._target
        
    #--------------------------------------------------------------------------
    @property
    def fullDetails(self)->str:
        """The full details of the all the AUV attributes"""
        out = []
        attrs = vars(self)
        for key,value in sorted(attrs.items()):
            out.append(f"{key} = {value!r}")
        return f"{self.__class__.__name__}(\n  " + ",\n  ".join(out) + "\n)"

    #--------------------------------------------------------------------------
    @property
    def immobilized(self)->bool:
        """Mobility status of AUV"""
        return self._immobilized
    
    @immobilized.setter
    def immobilized(self, status:bool)->None:
        """Set the mobility status of the AUV"""
        if (('_immobilized' in self.__dict__) and
            (not self._immobilized) and status):
            log.info('%s: **IMMOBILIZED! Dynamics inoperable...', self.callSign)
        elif (('_immobilized' in self.__dict__) and
            self._immobilized and (not status)):
            log.info('%s: **MOBILIZED! Dynamics operable...', self.callSign)
        self._immobilized = status

    ## Special Methods =======================================================#
    def __repr__(self)->str:
        """Return concise string representation of AUV object."""
        return f"<{self.__class__.__name__} {self.id} at {hex(id(self))}>"

    #--------------------------------------------------------------------------
    def __str__(self)->str:
        """Return user-friendly string representation with vehicle info."""
        out = []
        # Identity
        inGroup = f' - Group {self.groupId}' if (self.groupId) else ''
        isLeader = ' Leader' if (self.isLeader) else ''
        out.append(f'AUV {self.id:02}'+inGroup+isLeader)
        # Add info from dictionary
        if (not self.info):
            return out[0]
        # Field column width
        cw = max(len(k) for k in self.info) + 1
        # Field: value data
        for key,value in self.info.items():
            k = f"{key}:"
            out.append(f"{k:{cw}} {value}")
        # Total width
        tw = max([len(o) for o in out])
        # Add line decorators
        line = '-' * tw
        out = [out[0],line,*out[1:],line]
        return '\n'+'\n'.join(out)
    
    ## Methods ===============================================================#
    def addSensor(self, name:str, sensor:nav.Sensor)->None:
        """
        Install or replace a sensor in the vehicle's sensor suite.

        Adds a sensor object to the vehicles internal sensor dictionary, making
        it available for data collection during simulation. If a sensor with the
        specified name already exists, it is replaced. The vehilce's info
        dictionary is automatically updated to reflect the current sensor
        configuration.
        

        Parameters
        ----------
        name : str
            Unique identifier for the sensor. This name is used as the
            dictionary key and for subsequent sensor operations (read, remove).
            Names are case-sensitive and should be descriptive.

            Common conventions:

            - 'current': Ocean currentr sensor
            - 'depth': Ocean floor depth sensor

        sensor : nav.Sensor
            Sensor object to install on the vehicle. The sensor's collectData
            method will be called during simulation with relevant parameters
            (ocean, eta, etc.) passed as keyword arguments.
        
            
        Notes
        -----
        **Side Effects:**

        - Calls _updateSensorInfo() to update self.info['Installed Sensors']

        **Sensor Interface Requirements:**

        All sensors must inherit from nav.Sensor and implement collectData()
        method with the signature:

          ``def collectData(self, **kwargs)->Any``
        
        The collectData method should:

        - Extract only needed parameters from kwargs
        - Return sensor measurement(s) in appropriate format
        - Handle missing parameters gracefully (return None or raise ValueError)

        **Data Collection Timing:**

        Sensors are automatically read during simulation via the vehicle's
        collectSensorData() method, which is called once per simulation
        iteration. The simulator provides ocean state and other context via
        kwargs.
        
        **Sensor Data Flow:**
        
        1. Simulation calls vehicle.collectSensorData(ocean, i)
        2. Vehicle calls vehicle.readAllSensors() or readSensor(name)
        3. Sensor.collectData() extracts parameters and returns measurement
        4. Vehicle updates internal state (e.g., V_c, beta_Vc, z_bed, etc.)

        
        See Also
        --------
        addSensors : Add multiple sensors at once from dictionary
        removeSensor : Remove sensor by name
        removeSelectedSensors : Remove multiple sensors by name list
        removeAllSensors : Remove all sensors at once
        readSensor : Read data from specific sensor
        readSelectedSensors : Read multiple sensors by name list
        readAllSensors : Read data from all installed sensors
        collectSensorData : Main sensor data collection method called by
        simulator
        updateSensorInfo : Helper that updates info dictionary

        
        Examples
        --------
        ### Install default ocean sensors:

        >>> auv = Remus100s()
        >>> auv.addSensor('current', nav.OceanCurrentSensor())
        >>> auv.addSensor('depth', nav.OceanDepthSensor())
        >>> print(auv.info['Installed Sensors'])
        current, depth
        
        ### Install a custom depth sensor:

        >>> class CustomDepthSensor(nav.Sensor):
        ...     def collectData(self, **kwargs):
        ...         ocean = kwargs.get('ocean')
        ...         eta = kwargs.get('eta')
        ...         # custom sensor logic
        ...         return customData
        >>> auv.addSensor('depth', CustomDepthSensor())

        ### Read sensor during simulation:

        >>> speed, angle = auv.readSensor('current', ocean, i)
        >>> print(f"Current: {speed:.2f} m/s at {np.degrees(angle):.1f} deg")
        """

        self.sensors[name] = sensor
        self._updateSensorInfo()

    #--------------------------------------------------------------------------
    def addSensors(self, sensors:Dict[str,nav.Sensor])->None:
        """
        Install or replace multiple sensors on the vehicle simultaneously.

        Batch operation for adding several sensors at once from a dictionary.
        The vehicle's info dictionary is updated once after all sensors are
        processed.
        

        Parameters
        ----------
        sensors : dict {str : nav.Sensor}
            Dictionary mapping sensor names to sensor objects. Keys are unique
            identifier strings (same conventions as addSensor), values must be
            nav.Sensor instances.

            
        Notes
        -----
        **Side Effects:**

        - Calls _updateSensorInfo() to update self.info['Installed Sensors']
        - Invalid entries (not a nav.Sensor instance) are reported and skipped

        **Validation:**

        Each sensor value is checked for being an instance of nav.Sensor.
        Invalid objects generate an error log message and are not installed.
        This allows partial installation of valid sensors even if some entries
        are invalid.
        

        See Also
        --------
        addSensor : Add single sensor
        removeSensor : Remove sensor by name
        removeSelectedSensors : Remove multiple sensors by name list
        removeAllSensors : Remove all sensors at once
        readSensor : Read data from specific sensor
        readSelectedSensors : Read multiple sensors by name list
        readAllSensors : Read data from all installed sensors
        collectSensorData : Main sensor data collection method called by
        simulator
        updateSensorInfo : Helper that updates info dictionary

        
        Examples
        --------
        ### Install complete sensor kit all at once:

        >>> auv = Remus100s()
        >>> sensors = {
        ...     'current': nav.OceanCurrentSensor(),
        ...     'depth': nav.OceanCurrentSensor(),
        ...     'custom': someCustomSensor(),
        ... }
        >>> auv.addSensors(sensors)
        >>> print(auv.info['Installed Sensors'])
        current, depth, custom
        """

        for k,v in sensors.items():
            if (isinstance(v, nav.Sensor)):
                self.sensors[k] = v
            else:
                log.error("Invalid sensor for '%s'", k)
        self._updateSensorInfo()

    #--------------------------------------------------------------------------
    def removeSensor(self, name:str)->None:
        """
        Remove a sensor from the vehicle's sensor suite by name.

        Deletes the specified sensor from the vehicle's internal sensor
        dictionary and updates the info display.
        

        Parameters
        ----------
        name : str
            Unique identifier of the sensor to remove. Must match a key in the
            self.sensors dictionary exactly (case-sensitive).

            
        Notes
        -----
        **Side Effects:**

        - Calls _updateSensorInfo() to update self.info['Installed Sensors']

        **Safe Operation:**

        Attempting to remove a non-existent sensor does not raise an exception.
        Instead, a warning is reported and operation continues. This allows
        safely calling removeSensor without checking existence first.

        
        See Also
        --------
        addSensor : Add single sensor
        addSensors : Add multiple sensors at once from dictionary
        removeSelectedSensors : Remove multiple sensors by name list
        removeAllSensors : Remove all sensors at once
        readSensor : Read data from specific sensor
        readSelectedSensors : Read multiple sensors by name list
        readAllSensors : Read data from all installed sensors
        collectSensorData : Main sensor data collection method called by
        simulator
        updateSensorInfo : Helper that updates info dictionary
        

        Examples
        --------
        ### Remove a single sensor:
        
        >>> auv = Remus100s()
        >>> auv.addSensor('current', nav.OceanCurrentSensor())
        >>> auv.addSensor('depth', nav.OceanDepthSensor())
        >>> print(auv.info['Installed Sensors'])
        current, depth
        >>> auv.removeSensor('current')
        >>> print(auv.info['Installed Sensors'])
        depth
        """

        if (name in self.sensors):
            del self.sensors[name]
            self._updateSensorInfo()
        else:
            log.warning("Sensor '%s' not installed on vehicle", name)

    #--------------------------------------------------------------------------
    def removeSelectedSensors(self, names:List[str])->None:
        """
        Remove multiple sensors from the vehicle by providing a list of names.

        Batch removal operation that processes a list of sensor names. The
        vehicle's info dictionary is updated once after all sensors are
        processed.
        

        Parameters
        ----------
        names : list of str
            List of sensor identifier strings to remove. Each string should
            match a key in self.sensors dictionary (case-sensitive).
        
            
        Notes
        -----
        **Side Effects:**

        - Calls _updateSensorInfo() to update self.info['Installed Sensors']

        **Partial Removal:**

        If some sensor names in the list don't exist they are reported and
        skipped, while valid names are still removed.

        
        See Also
        --------
        addSensor : Add single sensor
        addSensors : Add multiple sensors at once from dictionary
        removeSensor : Remove sensor by name
        removeAllSensors : Remove all sensors at once
        readSensor : Read data from specific sensor
        readSelectedSensors : Read multiple sensors by name list
        readAllSensors : Read data from all installed sensors
        collectSensorData : Main sensor data collection method called by
        simulator
        updateSensorInfo : Helper that updates info dictionary
        

        Examples
        --------
        ### Remove multiple sensors at once:
        
        >>> auv = Remus100s()
        >>> auv.addSensors({
        ...     'current': nav.OceanCurrentSensor(),
        ...     'depth': nav.OceanDepthSensor(),
        ...     'custom': someCustomSensor(),
        ... })
        >>> auv.removeSelectedSensors(['current', 'custom'])
        >>> print(auv.info['Installed Sensors'])
        depth
        """
        
        removedAny = False
        for name in names:
            if (name in self.sensors):
                del self.sensors[name]
                removedAny = True
            else:
                log.warning("Sensor '%s' not installed on vehicle", name)
        if (removedAny):
            self._updateSensorInfo()

    #--------------------------------------------------------------------------
    def removeAllSensors(self)->None:
        """
        Remove all installed sensors from the vehicle's sensor suite.

        Clears the entire sensor dictionary and updates the info display.

        
        Notes
        -----
        Calls _updateSensorInfo() to remove sensor list from self.info

        
        See Also
        --------
        addSensor : Add single sensor
        addSensors : Add multiple sensors at once from dictionary
        removeSensor : Remove sensor by name
        removeSelectedSensors : Remove multiple sensors by name list
        readSensor : Read data from specific sensor
        readSelectedSensors : Read multiple sensors by name list
        readAllSensors : Read data from all installed sensors
        collectSensorData : Main sensor data collection method called by
        simulator
        updateSensorInfo : Helper that updates info dictionary

        
        Examples
        --------
        ### Reconfigure sensor suite:
        
        >>> auv.removeAllSensors()
        >>> auv.addSensors({
        ...     'custom_current': CustomCurrentSensor(),
        ...     'custom_depth': CustomDepthSensor(),
        ... })
        """
        
        if (self.sensors):
            self.sensors.clear()
            self._updateSensorInfo()
    
    #--------------------------------------------------------------------------
    def readSensor(self, name:str, *args, **kwargs)->Optional[Any]:
        """
        Read data from a single sensor by name with provided parameters.

        Calls the collectData method of the specified sensor, passing along any
        provided arguments. Returns the sensor's data output or None if the
        sensor is not found or if data collection fails.
        

        Parameters
        ----------
        name : str
            Unique identifier of the sensor to read from. Must match a key in
            self.sensors dictionary exactly (case-sensitive).
        *args : tuple, optional
            Positional arguments to pass to the sesnor's collectData method.
            Since this reads a single known sensor, positional arguments are
            supported for convenience.
        **kwargs : dict, optional
            Keyword arguments to pass to the sensor's collectData method. The
            sensor extracts only the parameters it needs. 
            
            Common kwargs include:

            - ocean : env.Ocean
                Ocean environment object for current and depth data
            - eta : ndarray
                Vehicle position for location-dependent sensing
            - i : int
                Simulation iteration counter for time-dependent sensing

                
        Returns
        -------
        data: Any or None
            Sensor data defined by the sensor's collectData method.
            
            Common return types:

            - tuple : (value1, value2, ...) for multi-valued sensors
            - float : Single scalar measurement
            - ndarray : Vector or array measurement
            - None : Sensor not found or data collection failed

            
        Notes
        -----
        **Side Effects:**

        Calls sensor.collectData() for named sensor

        **Argument Passing:**

        Supports positional arguments ``(*args)`` for convenience since the method
        only calls a single sensor.
        
        **Error Handling:**

        Returns None and reports issue if a sensor is not found or there is an
        error during data collection, allowing the simulation to continue.

        **Return Type Variability:**

        Return type depends on specific sensor implementation. Check sensor
        documentation for expected return format.

        
        See Also
        --------
        addSensor : Add single sensor
        addSensors : Add multiple sensors at once from dictionary
        removeSensor : Remove sensor by name
        removeSelectedSensors : Remove multiple sensors by name list
        removeAllSensors : Remove all sensors at
        readSelectedSensors : Read multiple sensors by name list
        readAllSensors : Read data from all installed sensors
        collectSensorData : Main sensor data collection method called by
        simulator
        updateSensorInfo : Helper that updates info dictionary

        
        Examples
        --------
        ### Read ocean current sensor with keyword arguments

        >>> auv = Remus100s()
        >>> ocean = env.Ocean(spd=0.5, ang=np.pi/4)
        >>> speed, angle = auv.readSensor('current', ocean=ocean, i=100)
        >>> print(f"Current: {speed:.2f} m/s at {np.degrees(angle):.1f} deg")
        Current: 0.50 m/s at 45.0 deg

        ### Read ocean depth sensor with positional arguments

        >>> depth = auv.readSensor('depth', ocean, auv.eta)
        >>> print(f"Ocean floor at {depth:.1f} m")
        Ocean floor at 132.9 m
        """
        
        if (name in self.sensors):
            try:
                return self.sensors[name].collectData(*args, **kwargs)
            except ValueError as e:
                log.error("Error reading sensor '%s': %s", name, e)
                return None
        log.error("Sensor '%s' not installed on vehicle", name) 
        return None
    
    #--------------------------------------------------------------------------
    def readSelectedSensors(self, names:List[str], **kwargs)->Dict[str,Any]:
        """
        Read data from multiple sensors by providing a list of sensor names.

        Batch operation that reads several sensors at once, collecting their
        data into a dictionary. Keyword arguments are passed to all sensors,
        with each sensor extracting only the parameters it needs. Missing
        sensors are logged as errors.
        

        Parameters
        ----------
        names : list of str
            List of sensor identifier strings to read from. Each string should
            match a key in self.sensors dictionary. Names are case-sensitive.
        **kwargs : dict
            Keyword arguments passed to each sensor's collectData method. All
            sensors receive the same kwargs dictionary but extract only needed
            parameters.

            Common kwargs:

            - ocean : env.Ocean
                Ocean environment object for current and depth data
            - eta : ndarray
                Vehicle position for location-dependent sensing
            - i : int
                Simulation iteration counter for time-dependent sensing

                
        Returns
        -------
        data : dict of {str : Any}
            Dictionary mapping sensor names to their collected data. Keys are
            sensor names from the input list (only for successfully read
            sensors). Values are dependent on what each sensor's collectData
            method returns. Sensors that are not found are omitted from the
            returned dictionary.

        
        Notes
        -----
        **Side Effects:**

        Calls collectData() on each sensor

        **Keyword Argument Sharing:**

        All sensors receive the same kwargs dictionary. Each sensor must be
        careful to extract only what it needs via kwargs.get() and not
        unintentionally modify shared kwargs.

        **Partial Success:**

        If some sensors in the list do not exist, data is still collected from
        valid sensors. The returned dictionary contains only successfully read
        sensor entries.

        **No Positional Arguments:**

        Unlike readSensor(), this method calls multiple sensors so does not
        support positional arguments because there are no requirements that each
        sensor takes the same set of arguments or takes them in the same order.

        
        See Also
        --------
        addSensor : Add single sensor
        addSensors : Add multiple sensors at once from dictionary
        removeSensor : Remove sensor by name
        removeSelectedSensors : Remove multiple sensors by name list
        removeAllSensors : Remove all sensors at
        readSensor : Read data from specific sensor
        readAllSensors : Read data from all installed sensors
        collectSensorData : Main sensor data collection method called by
        simulator
        updateSensorInfo : Helper that updates info dictionary

        
        Examples
        --------
        ### Read multiple sensors

        >>> auv.Remus100s()
        >>> ocean = env.Ocean(spd=0.5, ang=np.pi/4)
        >>> data = auv.readSelectedSensors(
        ...     ['current','depth'],
        ...     ocean=ocean, eta=auv.eta, i=100,
        ... )
        >>> print(f"Ocean Current Speed: {data['current'][0]:.2f} m/s")
        Ocean Current Speed: 0.50 m/s
        """
        
        data = {}
        for name in names:
            if (name in self.sensors):
                data[name] = self.sensors[name].collectData(**kwargs)
            else:
                log.error("Sensor '%s' not installed on vehicle", name)
        return data
    
    #--------------------------------------------------------------------------
    def readAllSensors(self, **kwargs)->Dict[str,Any]:
        """
        Read data from all installed sensors simultaneously.

        Batch operation that collects data from every installed sensor in the
        vehicle's sensor suite, returning a dictionary mapping sensor names to
        their data outputs. Keyword arguments are passed to all sensors, with
        each extracting only the parameters it needs.
        

        Parameters
        ----------
        **kwargs : dict
            Keyword arguments passed to each sensor's collectData method. All
            sensors receive the same kwargs dictionary.

            Common kwargs:

            - ocean : env.Ocean
                Ocean environment object for current and depth data
            - eta : ndarray
                Vehicle position for location-dependent sensing
            - i : int
                Simulation iteration counter for time-dependent sensing
            
                
        Returns
        -------
        data : dict of {str : Any}
            Dictionary mapping sensor names to their collected data. Keys are
            sensor names from the input list. Values are dependent on what each
            sensor's collectData method returns. If no sensors are installed,
            returns empty dictionary {}.
        
        
        Notes
        -----
        **Side Effects:**

        Calls collectData on each installed sensor

        **Use in Simulation:**

        This method is typically called by collectSensorData(), which is called
        once per simulation iteration by the Simulator.

        **Error Handling:**

        Individual sensor errors are not caught here. Method assumes each sensor
        is responsible for handling own exceptions, otherwise it propagates them
        up. If more robust error handling is needed, use readSelectedSensors or
        readSensor which catch errors.

        
        See Also
        --------
        addSensor : Add single sensor
        addSensors : Add multiple sensors at once from dictionary
        removeSensor : Remove sensor by name
        removeSelectedSensors : Remove multiple sensors by name list
        removeAllSensors : Remove all sensors at
        readSensor : Read data from specific sensor
        readSelectedSensors : Read multiple sensors by name list
        collectSensorData : Main sensor data collection method called by simulator
        updateSensorInfo : Helper that updates info dictionary

        
        Examples
        --------
        ### Read all sensors

        >>> auv.Remus100s()
        >>> ocean = env.Ocean(spd=0.5, ang=np.pi/4)
        >>> data = auv.readAllSensors(ocean=ocean, eta=auv.eta, i=100)
        >>> for sensor,value in data.items():
        ...     print(f"{sensor}: {value}")
        current: [0.50, 0.79]
        depth: 132.9
        """

        return {name: sensor.collectData(**kwargs) 
                for name,sensor in self.sensors.items()}
    
    #--------------------------------------------------------------------------
    @abstractmethod
    def collectSensorData(self)->None:
        """
        Collect data from installed sensors and update the vehicle state.
        

        Notes
        -----
        This method should be implemented by each vehicle class to collect data
        from and update the vehicle state accordingly.
        """
    
    ## Helper Methods ========================================================#
    def _updateSensorInfo(self)->None:
        """Update sensor information in the vehicle info dictionary."""
        
        if (self.sensors):
            names = sorted(self.sensors.keys())
            sensors = (f"{', '.join(names)}") if len(names) > 1 else names[0]
            self.info["Installed Sensors"] = sensors
        else:
            self.info.pop("Installed Sensors", None)

###############################################################################
        
class Remus100s(AUV):
    """
    Remus 100 cylinder-shaped AUV model for multi-agent swarm simulation
    research.

    This class provides a physically-based Remus 100 model with complete 6-DOF
    nonlinear dynamics, customized to support full swarm operation, modular GNC
    architecture, networking communication, swarm coordination capabilities, and
    sensor integration.

    
    Parameters
    ----------
    **kwargs : dict, optional
        Keyword arguments for vehicle customization. Common options include:
        
        groupId : str
            Swarm group identifier, single character (e.g., "A")
        isLeader : bool
            True if vehicle is a swarm group leader
        callSign : str
            Custom vehicle identification string
        
        Any vehicle attribute can be overridden via kwargs.

        
    Attributes
    ----------
    **State Vectors:**

        eta : ndarray, shape (6,)
            Position and attitude [x, y, z, roll, pitch, yaw]
        nu : ndarray, shape (6,)
            Body-frame velocities [u, v, w, p, q, r]
        velocity : ndarray, shape (3,)
            END-frame linear velocities [x_dot, y_dot, z_dot]
        u_actual : ndarray, shape (3,)
            Actual control inputs [rudder, stern, propeller]
            
    **Control Gains (PID Controllers):**

        Kp_z, Ki_z : float
            Depth control proportional and integral gains
        Kp_theta, Ki_theta, Kd_theta : float
            Pitch control PID gains
        Kp_psi, Ki_psi, Kd_psi : float
            Heading control PID gains
        K_f : float
            Observer gain for desired yaw angle
    
    **Swarm Coordination:**

        target : Vehicle or Model
            Target vehicle for tracking
        group : list of Vehicle or Model
            List of swarm group member vehicles
        r_safe : float
            Minimum safe vehicle separation distance, 10.0 m
        r_avoid : float
            Avoidance radius for APF repulsion, 40.0 m
        r_follow : float
            Preferred following distance, 100.0 m
            
    **Waypoint Navigation:**

        wpt : guidance.Waypoint
            Waypoint database for path following
        wpt_k : int
            Index to previous active waypoint
        R_switch : float
            Waypoint acceptance radius, 10.0 m
        Delta : float
            Look-ahead distance for ALOS, 10.0 m
        gamma : float
            Adaptive gain constant for ALOS, 0.0006

            
    Assigned Methods (Function Handles)
    ------------------------------------
    These attributes are designed to hold function references that are assigned
    via a custom load* method or manual assignment, enabling modular GNC
    architecture:
    
        CommNetwork : object
            Communication network manager (AquaNet or MuNet)
        CommSched : callable
            Communication scheduling function
        GuidSystem : callable
            Main guidance coordinator that integrates components of the guidance
            system at a high-level. Orchestrates observers and autopilots to
            generate control commands based on specified guidance law.
        GuidLaw : callable
            Core guidance algorithm implementing specific mathematical
            approaches for path generation. Returns either desired heading
            angles (e.g., ALOS) or velocity vectors (e.g. APF) based on current
            state and mission objectives.
        DepthObs : callable
            Depth command filter and state estimator. Processes desired depth
            through smoothing algorithm to generate feasible depth command while
            respecting vehicle and environmental constraints.
        HeadingObs : callable
            Heading state estimator and reference generator. Filters
            guidance-generated heading commands and estimates desired yaw
            rates.
        Attitude : callable
            Attitude propagation function (default: attitudeEuler)
        DepthAP : callable
            Depth autopilot implementing the control architecture, such as
            PI/PID, for pitch or stern plane deflection commands.
        HeadingAP : callable
            Heading autopilot implementing the control arichitecture, such as
            PI/PID, for rudder deflection commands.
        PropCmd : callable
            Propeller command generator for thrust management. Converts speed
            requirements to RPM commands, supporting both constant thrust
            profiles and dynamic speed control based on guidance demands

            
    Methods
    -------
    dynamics(u_control):
        Integrate 6-DOF equations of motion using Euler's method.
    xferN2U(rpm):
        Convert propeller RPM to vehicle speed (m/s).
    xferU2N(speed):
        Convert vehicle speed (m/s) to propeller RPM.
    collectSensorData(ocean, i)
        Read and update environmental sensor data.
    loadPathFollowing():
        Assign GNC for path following guidance system. Default is ALOS.
    loadTargetTracking(target, law, att, rep)
        Assign GNC for target tracking guidance system. Default is APF.
    loadConstantProp(n_setpt):
        Set constant propeller RPM command
    loadAquaNetTdmaLF(epDur, bcDur, rpDur):
        Configure TDMA communication via AquaNet for leader-follower systems.
    loadMuNetLF(network, kwargs)
        Configure communication via MuNet with FDMA or TDMA access modes.
    
        
    Notes
    -----
    **Remus 100 Physical Properties:**

    Controlled by a tail rudder, stern planes, and a propeller.
    
    - Length:         1.6 m
    - Diameter:       19 cm
    - Mass:           31.9 kg
    - Max Speed:      2.5 m/s @ 1525 rpm
    - Max Fin Angle:  30 deg
    - Max Depth:      100 m

    **Vehicle Model:**
    
    This implementation follows the physical and hydrodynamic parameters from
    the Remus 100 AUV specifications. The dynamics model includes:
    
    - Rigid body and added mass effects
    - Coriolis and centripetal forces
    - Linear and quadratic damping
    - Propeller thrust
    - Control surface lift and drag forces
    - Environmental coupling (ocean currents, depth limits)
    - Actuator dynamics with first-order time constants
    - Saturation limits on all control surfaces
    
    The modular GNC architecture allows arbitrary function assignment to the
    Assigned Methods attributes, enabling flexible experimentation with
    different guidance, navigation, and control algorithms.

    
    References
    ----------
    [1] Fossen, T.I. Python Vehicle Simulator. GitHub repository.
    https://github.com/cybergalactic/PythonVehicleSimulator
    
    [2] Fossen, T.I. (2021). Handbook of Marine Craft Hydrodynamics and Motion
    Control. 2nd Edition, Wiley. ISBN: 978-1-119-57505-4
    
    
    Examples
    --------
    ### Create a basic AUV:
    
    >>> auv = Remus100s()
    >>> print(f"Max speed: {auv.u_max} m/s, Max depth: {auv.z_max} m")
    Max speed: 2.5 m/s, Max depth: 100 m
    
    ### Assign custom function to guidance system:
    
    >>> def custom_guidance(vehicle):
    ...     # ... Custom guidance logic ...
    ...     return np.array([rudder, stern, prop])
    >>> auv.GuidSystem = custom_guidance
    >>> delta_r, delta_s, n = auv.GuidSystem(auv)
    
    ### Create swarm leader with waypoint path following:
    
    >>> leader = Remus100s(groupId="X", isLeader=True)
    >>> leader.wpt = guid.Waypoint([0, 100, 200], [0, 100, 0], [10, 20, 30])
    >>> leader.loadPathFollowing()
    >>> leader.loadConstantProp(n_setpt=1200)
    
    ### Create swarm follower with APF target tracking:
    
    >>> follower = Remus100s(groupId="X")
    >>> follower.loadTargetTracking(leader, law="APF", att="linearCBZ", 
    ...     rep="varExp")
    >>> print(follower)
    AUV 03 - Group X
    ----------------
    Guidance System: Target Tracking, APF
    Target: X02-LEADER
    """

    ## Class Attributes ======================================================# 
    __num = 0                           # number of Remus100s's created

    ## Constructor ===========================================================#
    def __init__(self, **kwargs: Any)->None:
        """
        Initialize Remus100s AUV with complete state, physics, and swarm
        parameters.

        Sets up all vehicle attributes including physical properties,
        hydrodynamic coefficients, control system gains, sensor suite,
        navigation parameters, swarm coordination settings, and communication
        interfaces. All systems are initialized to safe default values based on
        the Remus 100 specifications.

        Parameters
        ----------
        **kwargs : dict, optional
            Arbitrary keyword arguments to override default attributes. Any
            attribute defined in the class can be set via kwargs. Common
            customizations:
            
            - groupId : str
                Swarm group identifier, single character (e.g., "A")
            - isLeader : bool
                Set True to designate as group leader (default: False)
            - callSign : str
                Custom vehicle identifier for display and logging (default:
                id+groupId)
            
            Any other vehicle attribute can be overridden, but use caution: 
            modifying physical parameters affects dynamics accuracy.

            
        Notes
        -----
        **Attributes Initialized:**

        The __init__ method initializes the following attribute categories:
        
        - **Identity & Classification:**

            - Increments class instance count variable `num`
            - Inherits from parent AUV class
            - Sets modelName, modelType, controls description
            - Assigns unique id based on class instance count
        
        - **Physical & Hydrodynamic Properties:**

            - Dimensions: L (length), diam (diameter)
            - Mass properties: rigid body mass, moments of inertia
            - Hydrodynamic coefficients: added mass, damping, drag
            - Propeller: thrust/torque coefficients (Wageningen B-series)
            - Control surfaces: rudder and stern plane parameters
            - See Fossen (2021) Section 8.4.2 for derivations
        
        - **Control System Gains:**

            - Depth control: Kp_z, Ki_z (outer PI loop)
            - Pitch control: Kp_theta, Ki_theta, Kd_theta (inner PID loop)
            - Heading control: Kp_psi, Ki_psi, Kd_psi (PID)
            - Observer gains: K_f for heading estimation
            - Propeller: n_setpt, n_rate for thrust management
        
        - **State Vectors:**

            - eta: position/attitude [x, y, z, phi, theta, psi]
            - nu: body velocities [u, v, w, p, q, r]  
            - velocity: END velocities [x_dot, y_dot, z_dot]
            - u_actual: control inputs [rudder, stern, propeller]
        
        - **Sensor Suite:**

            - OceanCurrentSensor: measures current speed and direction
            - OceanDepthSensor: measures ocean floor depth
            - Additional sensors can be added via addSensor() method
        
        - **Navigation Parameters:**

            - z_max: maximum operating depth (100 m)
            - z_safe: safety distance from floor (5 m)
            - wn_d_z: desired natural frequency for depth (1/s)
        
        - **Swarm Coordination:**

            - r_safe: minimum safe separation (10 m)
            - r_avoid: avoidance radius for APF (40 m)  
            - r_follow: preferred following distance (100 m)
            - target: reference to leader vehicle (initially None)
            - group: list of swarm neighbors, excludes self (initially None)
        
        - **Waypoint Navigation:**

            - wpt: waypoint database (initialized empty at origin)
            - wpt_k: active waypoint index (0)
            - R_switch: waypoint acceptance radius (10 m)
            - Delta: look-ahead distance for ALOS (10 m)
            - gamma: adaptive gain for ALOS (0.0006)
            - betachat: crab angle estimate (0)
        
        - **Communication:**

            - All communication attributes initially None
            - Configured via custom loading methods (e.g. loadMuNetLF())
            
        - **GNC Function Handles:**

            - All assigned method attributes initially None
            - Configured via custom loading methods (e.g. loadPathFollowing())

        **Side Effects:**

        - Increments Remus100s.num class variable
        - Calls parent AUV.__init__() to set up inherited attributes
        - Installs default sensors (current and depth)
        - Updates vehicle info dictionary with model details

        **Initialization Order:**

        1. Update class instance counter
        2. Call parent class __init__
        3. Set identity and model information
        4. Initialize navigation parameters and sensors  
        5. Initialize control state vector (u_actual)
        6. Set control gains and propeller parameters
        7. Initialize physics and hydrodynamics (comprehensive)
        8. Apply user-specified kwargs overrides
        
        **Default Configuration:**

        The vehicle is initialized in a neutral, safe state:

        - Position at origin (0, 0, 0)
        - Zero velocity and angular rates
        - No active guidance, navigation, or control systems
        - Default sensor suite installed
        - Ready for configuration via manual assignment or custom loading
          methods
        
        **Kwargs Override Behavior:**

        Any attribute can be overridden via kwargs, but this occurs AFTER all
        default initialization, so computed attributes (like mass matrix) are
        not automatically recalculated if you override component parameters. For
        major physics changes, consider subclassing rather than using kwargs.
        """

        #---------------------------------------------------------------------#
        #   General                                                           #
        #---------------------------------------------------------------------#
        
        self.__class__.__num += 1       # Update Vehicle Class Counts
        super().__init__()              # Inherit Attributes

        #---------------------------------------------------------------------#
        #   Identity                                                          #
        #---------------------------------------------------------------------#
        self.modelName = "Remus 100s - Experimental Swarm Series"
        self.modelType = "Cylinder-Shaped AUV"
        self.controls = ["Tail rudder (deg)",
                         "Stern plane (deg)",
                         "Propeller revolution (rpm)"]
        self.id = self.__class__.__num  # ID number
        
        #---------------------------------------------------------------------#
        #   Communication                                                     #
        #---------------------------------------------------------------------#
        ## Data Integrity
        self.dataRepair = {
            'type':             ['predict'],
            'group_id':         ['predict', 'previous'],
            'reporter_id':      ['predict'], 
            'reporter_pos':     ['predict', 'previous'],
            'reporter_vel':     ['predict', 'previous', 'neutral'],
            'time_stamp':       ['predict', 'previous', 'neutral'],
            'leader_pos':       ['predict', 'previous'],
            'leader_vel':       ['predict', 'previous', 'neutral'],
            'leader_next_pos':  ['previous'],
            'leader_next_vel':  ['previous', 'neutral'],
            'num_followers':    ['skip'],
            'followers_pos':    ['predict', 'previous'],
            'followers_vel':    ['predict', 'previous', 'neutral'],
            'followers_rspn':   ['predict', 'previous', 'neutral'],
            'rspn_sched':       ['predict', 'previous', 'neutral'],
            'rspn_start':       ['predict', 'previous', 'neutral'],
        }   # Corrupted message data recovery strategies
        
        #---------------------------------------------------------------------#
        #   Navigation                                                        #
        #---------------------------------------------------------------------#
        ## State Estimators
        self.z_max = 100                # max vehicle operating depth (m)

        ## Sensors
        self.addSensor('current', nav.OceanCurrentSensor())
        self.addSensor('depth', nav.OceanDepthSensor())
        
        #---------------------------------------------------------------------#
        #   Control                                                           #
        #---------------------------------------------------------------------#
        ## State
        """
        u_actual=[rudder,stern,propeller]:  radians, radians, rpm
        """
        self.u_actual = np.array([0, 0, 0], float)     # control input vector

        ## Propeller / Thrust
        self.n_setpt = 0                # propeller rpm set-point
        self.n_rate = 0                 # propeller rpm rate (rpm/s)
        self.u_max = 2.5                # maximum vehicle speed (m/s)
        
        ## Stern / Depth
        self.z_int = 0                  # depth position integral state
        self.Kp_z = 0.054               # depth proportional gain (outer loop)
        self.Ki_z = 0.00216             # depth integral gain (outer loop)
        self.theta_int = 0              # pitch angle integral state
        self.Kp_theta = -0.36           # pitch proportional gain
        self.Ki_theta = -0.21           # pitch integral gain
        self.Kd_theta = -0.13           # pitch derivative gain

        ## Rudder / Heading
        self.psi_int = 0                # yaw angle integral state
        self.K_f = 0.5                  # observer gain for desired yaw angle
        self.Kp_psi = -7.5              # yaw proportional gain
        self.Ki_psi = -0.75             # yaw integral gain
        self.Kd_psi = -15.0             # yaw derivative gain
     
        #---------------------------------------------------------------------#
        #   Physics                                                           #
        #---------------------------------------------------------------------#
        ## Dimensions
        self.dimU = len(self.controls)  
        self.L = 1.6                    # length (m)
        self.diam = 0.19                # cylinder diameter (m)

        ## Actuator Dynamics
        self.deltaMax_r = 30 * self.D2R # max rudder angle (rad)
        self.deltaMax_s = 30 * self.D2R # max stern plane angle (rad)
        self.nMax = 1525                # max propeller revolution (rpm)    
        self.T_delta = 1.0              # rudder/stern plane time constant (s)
        self.T_n = 1.0                  # propeller time constant (s)

        ## Hydrodynamics
        """(Fossen 2021, Section 8.4.2)"""
        self.S = 0.7 * self.L * self.diam    # S = 70% of rectangle L * diam
        a = self.L/2                         # semi-axes
        b = self.diam/2                  
        self.r_bg = np.array([0, 0, 0.02], float)    # CG w.r.t. to the CO
        self.r_bb = np.array([0, 0, 0], float)       # CB w.r.t. to the CO

        ## Parasitic Drag Coefficient
        """CD_0, i.e. zero lift and alpha = 0
           F_drag = 0.5 * rho * Cd * (pi * b^2)   
           F_drag = 0.5 * rho * CD_0 * S"""
        Cd = 0.42                                   # Allen et al. (2000)
        self.CD_0 = Cd * math.pi * b**2 / self.S

        ## Rigid-Body Mass Matrix: Expressed in CO
        m = 4/3 * math.pi * self.rho * a * b**2     # mass of spheriod 
        Ix = (2/5) * m * b**2                       # moment of inertia
        Iy = (1/5) * m * (a**2 + b**2)
        Iz = Iy
        MRB_CG = np.diag([m, m, m, Ix, Iy, Iz])     # MRB expressed in the CG     
        H_rg = gnc.Hmtrx(self.r_bg)
        self.MRB = H_rg.T @ MRB_CG @ H_rg           # MRB expressed in the CO

        ## Weight and Buoyancy
        self.W = m * self.g
        self.B = self.W

        ## Added Moment of Inertia in Roll
        """A44 = r44 * Ix"""
        r44 = 0.3           
        MA_44 = r44 * Ix

        ## Lamb's K-Factors
        e = math.sqrt(1-(b/a)**2)
        alpha_0 = (2 * (1-e**2)/pow(e,3)) * (0.5 * math.log((1+e)/(1-e)) - e)  
        beta_0  = 1/(e**2) - (1-e**2) / (2*pow(e,3)) * math.log((1+e)/(1-e))
        k1 = alpha_0 / (2 - alpha_0)
        k2 = beta_0  / (2 - beta_0)
        k_prime = pow(e,4) * (beta_0-alpha_0) / ( 
            (2-e**2) * (2*e**2 - (2-e**2) * (beta_0-alpha_0)))
        
        ## Added Mass System Matrix: Expressed in CO
        self.MA = np.diag([m*k1, m*k2, m*k2, MA_44, k_prime*Iy, k_prime*Iy])

        ## Mass Matrix, Including Added Mass
        self.M = self.MRB + self.MA
        self.Minv = np.linalg.inv(self.M)

        ## Natural Frequencies in Roll and Pitch
        self.w_roll = math.sqrt(self.W * (self.r_bg[2]-self.r_bb[2]) / 
            self.M[3][3])
        self.w_pitch = math.sqrt(self.W * (self.r_bg[2]-self.r_bb[2]) / 
            self.M[4][4])
        
        ## Tail Rudder Parameters (single)
        self.CL_delta_r = 0.5       # rudder lift coefficient
        self.A_r = 2 * 0.10 * 0.05  # rudder area (m2)
        self.x_r = -a               # rudder x-position (m)

        ## Stern-Plane Paramaters (double)
        self.CL_delta_s = 0.7       # stern-plane lift coefficient
        self.A_s = 2 * 0.10 * 0.05  # stern-plane area (m2)
        self.x_s = -a               # stern-plane z-position (m)

        ## Propeller Parameters
        self.D_prop = 0.14     # propeller diameter corresponding to 5.5 inches
        self.t_prop = 0.1      # thrust deduction number
        self.Ja_max = 0.6632   # from: 0.944 * 2.5 / (0.14 * 1525/60) = 0.6632
        """Single-screw propeller with 3 blades and blade-area ratio = 0.718.
           Coefficients are computed using the Matlab MSS toolbox:     
           >> [KT_0, KQ_0] = wageningen(0,1,0.718,3)
           >> [KT_max, KQ_max] = wageningen(0.6632,1,0.718,3)"""
        self.KT_0 = 0.4566
        self.KQ_0 = 0.0700
        self.KT_max = 0.1798
        self.KQ_max = 0.0312

        ## Low-Speed Linear Damping Matrix Parameters
        self.T_surge = 20           # time constant in surge (s)
        self.T_sway = 20            # time constant in sway (s)
        self.T_heave = self.T_sway  # equal for for a cylinder-shaped AUV
        self.zeta_roll = 0.3        # relative damping ratio in roll
        self.zeta_pitch = 0.8       # relative damping ratio in pitch
        self.T_yaw = 5              # time constant in yaw (s)

        #---------------------------------------------------------------------#
        #   User Specified                                                    #
        #---------------------------------------------------------------------#
        ## User Keyword Attributes
        for key,value in kwargs.items():
            setattr(self, key, value)

    ## Methods ===============================================================#
    def dynamics(self, u_control:NPFltArr)->Tuple[NPFltArr,NPFltArr]:
        """
        Integrate 6-DOF nonlinear AUV equations of motion forward one time step.

        Computes vehicle acceleration and control actuator responses under
        current state, control commands, and environmental conditions.
        Implements complete Fossen-based marine vehicle dynamics with rigid
        body, added mass, hydrodynamic, and environmental coupling effects. Uses
        Euler forward integration for computational efficiency.

        
        Parameters
        ----------
        u_control : ndarray, shape (3,)
            Commanded control inputs [delta_r, delta_s, n]:

            u_control[0] : float, delta_r
                Commanded rudder angle in radians. Positive deflection creates
                starboard turn (right when viewed from above). Limited to
                deltaMax_r.
            u_control[1] : float, delta_s
                Commanded stern plane angle in radians. Positive deflection
                creates bow-down pitch moment. Limited to deltaMax_s.
            u_control[2] : float, n
                Commanded propeller revolution rate in rpm. Positive values
                produce forward thrust. Limited to nMax.
        
        Returns
        -------
        nu : ndarray, shape (6,)
            Updated vehicle velocity vector in body-fixed frame after the
            integration step [u, v, w, p, q, r]:

            nu[0] : float, u
                Surge velocity (forward/aft) in m/s
            nu[1] : float, v
                Sway velocity (port/starboard) in m/s
            nu[2] : float, w
                Heave velocity (up/down) in m/s
            nu[3] : float, p
                Roll rate about x-axis in rad/s
            nu[4] : float, q
                Pitch rate about y-axis in rad/s
            nu[5] : float, r
                Yaw rate about z-axis in rad/s

        u_actual :  ndarray, shape (3,)
            Actual control surface positions and propeller rate after actuator
            dynamics, time constants, and saturation limits [delta_r, delta_s, n]:

            u_actual[0] : float, delta_r
                Actual rudder angle in radians 
            u_actual[1] : float, delta_s
                Actual stern plane angle in radians
            u_actual[2] : float, n
                Actual propeller rpm

        Notes
        -----
        **Equations of Motion:**

        The dynamics follow Fossen's 6-DOF formulation:

            M * d/dt nu_r + C(nu_r) * nu_r + D(nu_r) * nu_r + g(eta) = tau

        where:

            - M: Mass matrix, M_RB + M_A (rigid body plus added mass)
            - d/dt nu_r: Relative acceleration
            - C(nu_r): Coriolis and centripetal matrix (function of relative
              velocity)
            - D(nu_r): Damping matrix (linear + quadratic drag)
            - g(eta): Restoring forces and moments (weight, buoyancy)
            - tau: Generalized force vector (propulsion + control surfaces)
            - nu_r: Relative velocity, nu - nu_c (accounting for ocean current)

        **Force and Moment Components:**

        1. **Propleller Thrust and Torque:**

            - Wageningen B-series approximation
            - Thrust depends on advance number J_a = V_a/(nD_prop)
            - Coefficients: K_T0, K_Q0, K_Tmax, K_Qmax
            - Thrust deduction factor t_prop accounts for hull interaction

        2. **Control Surface Forces:**

            - Rudder: Generates sway force Y_r and yaw moment N_r
            - Stern planes: Generate heave force Z_s and pitch moment M_s
            - Lift coefficient times dynamic pressure times surface area
            - Includes drag penalty (parasitic drag)

        3. **Lift and Drag:**

            - Angle of attack alpha = atan2(w_r, u_r)
            - Lift and drag coefficients from Beard & McLain (2012)
            - Cross-flow drag for lateral motion

        4. **Ocean Current Effects:**

            - Ocean current velocity nu_c added in END frame
            - Relative velocity nu_r = nu - R^(T(psi) * nu_c) used for
              hydrodynamic forces
            - Derivative of current (Coriolis-like term) included

        **Actuator Dynamics:**

        First-order lag models with time constants:

            d/dt delta_r = (delta_r,cmd - delta_r) / T_delta
            d/dt delta_s = (delta_s,cmd - delta_s) / T_delta
            d/dt n = (n_cmd - n) / T_n

        where:

            - T_delta: rudder / stern plane time constant
            - T_n: propeller time constant

        **Safety and Collision Detection:**

        - Surface breach (z < 0): Buoyancy set to zero, continues dynamics
        - Floor collision (z > z_bed): Vehicle immobilized, returns zero
          velocities
        - Immobilization flag set to True on floor collision, persists for
          simulation duration
                
        **Integration Method:**

        Forward Euler integration with step size h = sampleTime:

            nu(k+1) = nu(k) + h * d/dt nu(k)
            u_actual(k+1) = u_actual(k) + h * d/dt u_actual(k)

        This is computationally efficient but requires sufficiently small time
        steps, typically 0.02 s or smaller, for numerical stability.

        **Coordinate Frames:**

        - BODY frame: Fixed to vehicle, origin at center of origin (CO)
        - END frame: East-North-Down inertial reference frame
        - All forces / moments computed in BODY frame
        - Velocity transformation via rotation matrix R(psi)

        
        See Also
        --------
        gnc.m2c : Mass matrix to Coriolis matrix conversion
        gnc.forceLiftDrag : Lift and drag force calculation  
        gnc.crossFlowDrag : Cross-flow drag calculation
        gnc.gvect : Restoring force vector calculation
        navigation.attitudeEuler : Attitude integration for position update

        
        References
        ----------
        [1] Fossen, T.I. (2021). Handbook of Marine Craft Hydrodynamics and Motion
        Control. 2nd Edition, Wiley.
        
        [2] Beard, R.W. and McLain, T.W. (2012). Small Unmanned Aircraft: Theory
        and Practice. Princeton University Press.

            
        Examples
        --------
        ### Simulate one dynamics step with control commands:
        
        >>> auv = Remus100s()
        >>> auv.eta = np.array([0, 0, 10, 0, 0, 0])   # 10m depth
        >>> auv.nu = np.array([2.0, 0, 0, 0, 0, 0])   # 2 m/s forward
        >>> u_cmd = np.array([0.1, -0.05, 1200])      # rudder, stern, prop
        >>> nu_new, u_act = auv.dynamics(u_cmd)
        >>> print(f"New surge velocity: {nu_new[0]:.3f} m/s")
        >>> print(f"Actual rudder angle: {u_act[0]:.3f} rad")
        """

        # Check Status
        if (self.immobilized):
            return self.nu*0, self.u_actual*0
        
        # Input Parameters Loaded on Vehicle
        h = self.sampleTime
        eta = self.eta
        nu = self.nu
        u_actual = self.u_actual
        B = self.B

        # Check Environment
        if (eta[2] < 0):                # surface breach
            B = 0
        if (eta[2] > self.z_bed):       # floor collision
            log.warning('**COLLISION with ocean floor! Vehicle: ' +
                        '%s at [%.1f,%.1f,%.1f] %.2f s',
                        self.callSign, eta[0], eta[1], eta[2], self.clock)
            self.immobilized = True
            return nu*0, u_actual*0

        # Ocean Current Velocity
        u_c = self.V_c * math.cos(self.beta_V_c - eta[5])  # surge 
        v_c = self.V_c * math.sin(self.beta_V_c - eta[5])  # sway
        nu_c = np.array([u_c, v_c, 0, 0, 0, 0], float)     # velocity 
        Dnu_c = np.array([nu[5]*v_c, -nu[5]*u_c, 0, 0, 0, 0],float) # d/dt

        # Vehicle Velocity
        nu_r = nu - nu_c                               # relative vehicle vel.        
        alpha = math.atan2(nu_r[2], nu_r[0])           # angle of attack 
        U = math.sqrt(nu[0]**2 + nu[1]**2 + nu[2]**2)  # vehicle speed
        U_r = math.sqrt(nu_r[0]**2 + nu_r[1]**2 + nu_r[2]**2)  # relative speed

        # Commands and Actual Control Signals
        delta_r_c = u_control[0]    # commanded tail rudder (rad)
        delta_s_c = u_control[1]    # commanded stern plane (rad)
        n_c = u_control[2]          # commanded propeller revolution (rpm)
        delta_r = u_actual[0]       # actual tail rudder (rad)
        delta_s = u_actual[1]       # actual stern plane (rad)
        n = u_actual[2]             # actual propeller revolution (rpm)

        # Propeller Coefficients
        """KT and KQ are computed as a function of advance no.
           Ja = Va/(n*D_prop)        where 
           Va = (1-w)*U = 0.944 * U
           Allen et al. (2000)"""
        n_rps = n / 60      # propeller revolution (rps) 
        Va = U * 0.944      # advance speed (m/s)

        # Propeller Thrust and Propeller-Induced Roll Moment
        """Linear approximations for positive Ja values
           KT ~= KT_0 + (KT_max-KT_0)/Ja_max * Ja   
           KQ ~= KQ_0 + (KQ_max-KQ_0)/Ja_max * Ja """
        ## Forward Thrust
        if (n_rps > 0):
            X_prop = (self.rho * pow(self.D_prop,4) * 
                      (self.KT_0 * abs(n_rps) * n_rps + 
                       (self.KT_max-self.KT_0)/self.Ja_max * (Va/self.D_prop) *
                         abs(n_rps)))
            K_prop = (self.rho * pow(self.D_prop,5) * 
                      (self.KQ_0 * abs(n_rps) * n_rps + 
                       (self.KQ_max-self.KQ_0)/self.Ja_max * (Va/self.D_prop) *
                         abs(n_rps)))
        ## Reverse Thrust       
        else:
            X_prop = (self.rho * pow(self.D_prop,4) * self.KT_0 * abs(n_rps) * 
                      n_rps)
            K_prop = (self.rho * pow(self.D_prop,5) * self.KQ_0 * abs(n_rps) * 
                      n_rps)
        
        # Rigid-Body/Added Mass Coriolis/Centripetal Matrices: Expressed in CO
        CRB = gnc.m2c(self.MRB, nu_r)
        CA  = gnc.m2c(self.MA, nu_r)
        """Nonlinear quadratic velocity terms in pitch and yaw (Munk moments) 
           are set to zero since only linear damping is used"""
        CA[4][0] = 0  
        CA[4][3] = 0
        CA[5][0] = 0
        CA[5][1] = 0
        C = CRB + CA

        # Dissipative Forces and Moments
        D = np.diag([
            self.M[0][0] / self.T_surge,
            self.M[1][1] / self.T_sway,
            self.M[2][2] / self.T_heave,
            self.M[3][3] * 2 * self.zeta_roll  * self.w_roll,
            self.M[4][4] * 2 * self.zeta_pitch * self.w_pitch,
            self.M[5][5] / self.T_yaw])
        """For DOF 1,2,6 the D elements go to zero at higher speeds, i.e.
           drag and lift/drag dominate"""
        D[0][0] = D[0][0] * math.exp(-3*U_r)
        D[1][1] = D[1][1] * math.exp(-3*U_r)
        D[5][5] = D[5][5] * math.exp(-3*U_r)
        tau_liftdrag = gnc.forceLiftDrag(self.diam,self.S,self.CD_0,alpha,U_r)
        tau_crossflow = gnc.crossFlowDrag(self.L,self.diam,self.diam,nu_r)

        # Restoring Forces and Moments
        g = gnc.gvect(self.W,B,eta[4],eta[3],self.r_bg,self.r_bb)

        # Horizontal and Vertical-Plane Relative Speed
        U_rh = math.sqrt(nu_r[0]**2 + nu_r[1]**2)
        U_rv = math.sqrt(nu_r[0]**2 + nu_r[2]**2)

        # Rudder and Stern-Plane Drag
        X_r = -0.5*self.rho * U_rh**2*self.A_r * self.CL_delta_r*delta_r**2
        X_s = -0.5*self.rho * U_rv**2*self.A_s * self.CL_delta_s*delta_s**2

        # Rudder sway force 
        Y_r = -0.5*self.rho * U_rh**2*self.A_r * self.CL_delta_r*delta_r

        # Stern-plane heave force
        Z_s = -0.5*self.rho * U_rv**2*self.A_s * self.CL_delta_s*delta_s

        # Generalized Force Vector
        tau = np.array([
            (1-self.t_prop) * X_prop + X_r + X_s, 
            Y_r, 
            Z_s,
            K_prop,
            self.x_s * Z_s,
            self.x_r * Y_r], float)

        # AUV Dynamics
        tau_sum = tau + tau_liftdrag + tau_crossflow - np.matmul(C+D,nu_r) - g
        nu_dot = Dnu_c + np.matmul(self.Minv, tau_sum)

        # Actuator dynamics
        delta_r_dot = (delta_r_c - delta_r) / self.T_delta
        delta_s_dot = (delta_s_c - delta_s) / self.T_delta
        n_dot = (n_c - n) / self.T_n

        # Forward Euler Integration: [k+1]
        nu += h * nu_dot
        delta_r += h * delta_r_dot
        delta_s += h * delta_s_dot
        n += h * n_dot

        # Actual Control Response
        """Bounds of control signals are enforced by amplitude saturation"""
        if (abs(delta_r) >= self.deltaMax_r):
            delta_r = np.sign(delta_r) * self.deltaMax_r
        if (abs(delta_s) >= self.deltaMax_s):
            delta_s = np.sign(delta_s) * self.deltaMax_s          
        if (abs(n) >= self.nMax):
            n = np.sign(n) * self.nMax
        u_actual = np.array([delta_r, delta_s, n], float)
        
        return nu, u_actual

    #--------------------------------------------------------------------------
    def xferN2U(self, rpm:float)->float:
        """
        Transfer function from propller RPM to vehicle Speed.
        

        Parameters
        ----------
        rpm:
            Number of propeller Rotations Per Minute.
        
            
        Returns
        -------
        speed:
            Vehicle speen (m/s).
        
            
        Notes
        -----
        Based on a simple linear speed test in a zero ocean current
        environment. The function domain is split into three regions with a
        linear approximation for each.
        """

        # Domain Divisions (Upper Boundary)
        low = 100       # low:    0-100
        mid = 400       # mid:  100-400
                        # high: 400-1525

        # Y-intercept and Slope
        if (rpm <= low):
            b = 0.0
            m = 0.000611108
        elif (rpm <= mid):
            b = -0.135849
            m = 0.00196552
        else:
            b = -0.0273616
            m = 0.0016922
        
        return (m*rpm + b)

    #--------------------------------------------------------------------------
    def xferU2N(self, speed:float)->float:
        """
        Transfer function from vehicle Speed to propeller RPM.

        
        Parameters
        ----------
        speed:
            Vehicle speed (m/s).

            
        Returns
        -------
        rpm:
            Number of propller Rotations Per Minute.
        
            
        Notes
        -----
        Based on a simple linear speed test in a zero ocean current
        environment. The function domain is split into three regions with a
        linear approximation for each.
        """

        # Domain Divisions (Upper Boundary)
        low = 0.0605    # low:     0-0.06
        mid = 0.6383    # mid:  0.06-0.64
                        # high: 0.64-2.55
        
        # Y-intercept and Slope
        if (speed <= low):
            b = 0.0
            m = 1636.37
        elif (speed <= mid):
            b = 69.2435
            m = 508.411
        else:
            b = 16.1836
            m = 590.939
        
        return (m*speed + b)

    #--------------------------------------------------------------------------
    def collectSensorData(self, ocean:env.Ocean, i:int)->None:
        """
        Update vehicle environmental state from installed sensors.

        Main sensor data collection method called once per simulation iteration.
        Reads installed sensors and updates vehicle attributes with
        environmental measurements. This is the primary interface between the
        simulation environment and the vehicle's internal state representation.

        
        Parameters
        ----------
        ocean : env.Ocean
            Ocean environment object containing:

            - current : env.OceanCurrent or None
                Ocean current model with speed and direction
            - floor : env.OceanFloor or None
                Ocean floor depth model (bathymetry)
            - Additional environment features as available
            
        i : int
            Current simulation iteration counter. Used for time-dependent
            sensing, logging, and sensor dynamics. Typically ranges from 0 to
            N-1 where N is total simulation iterations.

            
        Notes
        -----
        **Side Effects:**

        Updates the following vehicle attributes based on sensor readings:

        - V_c : float
            Ocean current speed in m/s (from 'current' sensor if installed)
        - beta_Vc : float
            Ocean current direction in radians (from 'current' sensor if
            installed)
        - z_bed : float
            Ocean floor depth in meters (from 'depth' sensor if installed)
            
        These attributes are used by the dynamics() method to compute
        environmental effects.

        **Present Implementation (Remus100s):**

        The present implementation specifically reads:
        
        1. **Ocean Current Sensor:**

        - Checks if ocean.current is not None
        - Calls readSensor('current', ocean, i)
        - Updates self.V_c and self.beta_Vc
        
        2. **Ocean Depth Sensor:**

        - Checks if ocean.floor is not None
        - Calls readSensor('depth', ocean, self.eta)
        - Updates self.z_bed
        
        **Extensibility:**

        To add additional sensors and state updates:
        
        1. Define a sensor that inherits nav.Sensor
        2. Install sensor via addSensor()
        3. Modify this method to read new sensor
        4. Update relevant vehicle attribute with sensor data

        **Simulator Integration:**

        This method is called by the Simulator during each iteration of the
        simulation loop, specifically in simulate(), simulateMuNet(), and
        simulateAquaNet() methods. The call occurs after communication updates
        but before GNC computations.

        
        See Also
        --------
        addSensor : Add single sensor
        addSensors : Add multiple sensors at once from dictionary
        removeSensor : Remove sensor by name
        removeSelectedSensors : Remove multiple sensors by name list
        removeAllSensors : Remove all sensors at
        readSensor : Read data from specific sensor
        readSelectedSensors : Read multiple sensors by name list
        readAllSensors : Read data from all installed sensors
        updateSensorInfo : Helper that updates info dictionary
        """
        
        # Read Ocean Environment
        if (ocean is not None):
            if (ocean.current is not None):
                self.V_c, self.beta_V_c = self.readSensor('current',ocean,i)
            if (ocean.floor is not None):
                self.z_bed = self.readSensor('depth',ocean,self.eta)
    
    #--------------------------------------------------------------------------
    def loadPathFollowing(self)->None:
        """
        Configure vehicle for waypoint-based path following guidance system.

        Sets up a complete autonomous navigation system for following a
        predefined sequence of waypoints. Configures all necessary guidance,
        navigation, and control components for integrated path-following
        behavior.

        Notes
        -----
        **Side Effects:**

        - Assigns function handles to the following vehicle attributes:

          - **Guidance System:**

            - self.GuidSystem : callable
                Main guidance system to coordinate components at a high-level.
            - self.GuidLaw : callable
                Core guidance algorithm implementing specific approach for path
                generation.

          - **Navigation Observers:**

            - self.DepthObs : callable
                Depth command filter and state estimator.
            - self.HeadingObs:
                Heading state estimator and reference generator.

          - **Control Autopilots:**

            - self.DepthAP : callable
               Depth autopilot implementing control commands for stern plane
               deflection.
            - self.HeadingAP : callable
               Heading autopilot implementing control commands for rudder
               defleciton.

        - Updates the vehicle's info dictionary with configuration details for
          display and logging.

        **Configuration Options:**

        Currently there is only one set of configuration options available for
        the path following guidance system, the Adaptive Line-of-Sight (ALOS)
        guidance law. See guidance.ALOSlaw for specific details.

        Attributes used by the assigned components can be customized by setting
        them manually. For example:

        ### Modify Guidance System Parameters:

        >>> auv.loadPathFollowing()
        >>> auv.Delta = 15.0    # Larger look-ahead distance
        >>> auv.R_switch = 5.0  # Tighter waypoint acceptance distance

        See each assigned function for more details and additional configuration
        options.

        
        See Also
        --------
        guidance.pathFollow : Main path-following guidance function
        guidance.ALOSlaw : Adaptive line-of-sight guidance law
        navigation.depthFilter : Depth measurement filter
        navigation.headingFilterLOS : LOS angle observer
        control.depthPID : Depth autopilot with cascade control
        control.headingPID : Heading autopilot
        guidance.Waypoint : Waypoint database class
        loadTargetTracking : Alternative guidance for swarm following
        loadConstantProp : Propeller command configuration

        
        References
        ----------
        [1] Fossen, T. I., "An Adaptive Line-of-Sight (ALOS) Guidance Law for
        Path Following of Aircraft and Marine Craft," in IEEE Transactions on
        Control Systems Technology, 31(6), 2887-2894, Nov. 2023, doi:
        10.1109/TCST.2023.3259819.
        
        [2] Fossen, T.I. (2021). Handbook of Marine Craft Hydrodynamics and Motion
        Control. 2nd Edition, Wiley.

        
        Examples
        --------
        ### Basic path following setup:
        
        >>> auv = Remus100s()
        >>> auv.wpt = guid.Waypoint(     # Define waypoint path
        ...     [0, 100, 200, 300],      # x coordinates (East)
        ...     [0, 0, 100, 100],        # y coordinates (North)
        ...     [10, 10, 10, 10]         # z coordinates (Down / Depth)
        ... )
        >>> auv.loadPathFollowing()      # Configure path following
        >>> print(auv.info['Guidance System'])
        'Path Following, ALOS'
        """

        # Load Vehicle Modules
        self.GuidSystem = guid.pathFollow
        self.GuidLaw = guid.ALOSlaw             # Adaptive Line of Sight
        self.DepthObs = nav.depthFilter
        self.HeadingObs = nav.headingFilterLOS  # LOS Observer
        self.DepthAP = ctrl.depthPID
        self.HeadingAP = ctrl.headingPID

        # Generate Information Parameters
        self.info.update([("Guidance System", "Path Following, ALOS"),
                          ("Waypoints", f"{len(self.wpt)}")])

    #--------------------------------------------------------------------------
    def loadTargetTracking(self,
                           target:Vehicle,
                           law:str='APF',
                           att:str='linearCBZ',
                           rep:str='varExp',
                           )->None:
        """
        Configure vehicle for swarm target tracking using artificial potential
        field methods.

        Sets up autonomous target-following behavior for swarm coordination
        using potential field-based guidance. Configures attraction to target
        vehicle and repulsion from obstacles/neighbors, with modular selection
        of attraction and repulsion functions.

        
        Parameters
        ----------
        target : Vehicle
            Target vehicle to track and follow. The target's position and
            veloctiy are tracked for coordinated motion.
        law : str
            Guidance law selection:

            - 'APF': Artificial Potential Field guidance
            - 'CB': Constant Bearing guidance

        att : str
            Attraction function type for APF guidance law:

            - 'linear': Variable linear attraction
            - 'linearCBZ': Variable linear zone attraction
            - 'cubic': Variable cubic attraction

        rep : str
            Repulsion function type for APF guidance law:

            - 'exp': Exponential repulsion
            - 'varExp': Variable exponential repulsion

            
        Notes
        -----
        **Side Effects:**

        - Assigns function handles to the following vehicle attributes:

          - **Guidance System:**

            - self.GuidSystem : callable
                Main guidance system to coordinate components at a high-level.
            - self.GuidLaw : callable
                Core guidance algorithm implementing specific approach for path
                generation.

          - **APF Component Functions (if law='APF'):**

            - self.GuidLaw.attraction : callable
                Artificial potential that contributes to attraction towards
                target.
            - self.GuidLaw.repulsion : callable
                Artificial potential that contributes to repulsion away from
                neighbors.

          - **Control Autopilots:**

            - self.DepthAP : callable
               Depth autopilot implementing control commands for stern plane
               deflection.
            - self.HeadingAP : callable
               Heading autopilot implementing control commands for rudder
               defleciton.

          See assigned functions for details and configuration options.

        - Updates the vehicles target attribute. Uses the passed Vehicle
          instance if no CommNetwork, otherwise creates a Model instance (a
          lightweight representation for tracking data).
        - Updates the vehicles groupId attribute, inheriting from the target
          vehicle.
        - Updates the vehicle's info dictionary with configuration details for
          display and logging.
        
        **Swarm Coordination Parameters:**

        The following vehicle attributes control swarm behavior:

        - r_safe : float
            Minimum safe separation distance, triggers strong repulsion
        - r_avoid : float
            Avoidance radius, repulsion active within this range
        - r_follow : float
            Preferred following distance from target
            
        Tune these parameters based on:

        - Vehicle size and maneuverability
        - Communication latency and update rate
        - Environmental conditions (currents, obstacles)
        - Desired formation tightness

        **Depth Control Difference:**

        Unlike path following which uses cascade depth control, target
        tracking uses direct pitch control. This provides:

        - Faster depth response for dynamic target following
        - Direct 3D tracking without waypoint constraints
        - More aggressive maneuvering capability
        
        **Wave Filter Note:**

        Current implementation does not include wave filtering for the potential
        field gradients. In high disturbance environments, the computed velocity
        commands may exhibit high frequency oscillations. Future versions should
        implement filtering to reduce influence of destabilizing forces.

        
        See Also
        --------
        guidance.targetTrack : Main target tracking guidance function
        guidance.velAPF : APF velocity guidance law
        guidance.velCB : Constant bearing velocity guidance law
        guidance.variableLinearZoneAttractionAPF : Recommended attraction function
        guidance.variableExpRepulsionAPF : Recommended repulsion function
        control.pitchPID : Direct pitch controller
        control.headingPID : Heading autopilot
        loadPathFollowing : Alternative guidance for waypoint following
        loadMuNetFdmaLF : Communication setup for swarm coordination

        
        Examples
        --------
        ### Basic leader-follower setup with default APF:

        >>> leader = Remus100s(groupId='A', isLeader=True)
        >>> leader.wpt = guid.Waypoint([0, 100], [0, 0], [0, 20])
        >>> leader.loadPathFollowing()
        >>> follwer = Remus100s()
        >>> follwer.loadTargetTracking(leader)
        >>> print(follower.info['Target'])
        A01-LEADER
        """

        # Load Vehicle Modules
        self.GuidSystem = guid.targetTrack
        self.DepthAP = ctrl.pitchPID
        self.HeadingAP = ctrl.headingPID

        # Load Guidance Law
        ## Artificial Potential Field
        if (law == 'APF'):
            self.GuidLaw = guid.velAPF
            ### Attraction Function
            if (att == 'linear'):
                self.GuidLaw.attraction = guid.variableLinearAttractionAPF
            elif (att == 'linearCBZ'):
                self.GuidLaw.attraction = guid.variableLinearZoneAttractionAPF
            elif (att == 'cubic'):
                self.GuidLaw.attraction = guid.variableCubicAttractionAPF
            ### Repulsion Function
            if (rep == 'exp'):
                self.GuidLaw.repulsion = guid.exponentialRepulsionAPF
            elif (rep == 'varExp'):
                self.GuidLaw.repulsion = guid.variableExpRepulsionAPF
        ## Constant Bearing
        elif (law == 'CB'):
            self.GuidLaw = guid.velCB

        # Swarm Group Data Acquisition Method
        ## Using Direct-Access
        if (self.CommNetwork is None):
            self.target = target
        ## Using Network Communication
        else:
            self.target = Model(target)
        
        # Set Parameters
        self.groupId = target.groupId

        # Generate Information Parameters
        self.info.update([("Guidance System", f"Target Tracking, {law}")])
        if (law == 'APF'):
            self.info.update([("Attraction Function", f"{att}"),
                              ("Repulsion Function", f"{rep}")])
        self.info.update([("Target", f"{self.target.callSign}")])

    #--------------------------------------------------------------------------
    def loadConstantProp(self,n_setpt:float=1200)->None:
        """
        Configure constant propeller RPM command.
        
        Sets up a simple propeller control that maintains constant RPM
        throughout simulation. Useful for basic testing and scenarios
        with fixed thrust.
        
        
        Parameters
        ----------
        n_setpt : float
            Propeller RPM setpoint (default: 1200 rpm, ~2.0 m/s forward speed).
            
        
        Notes
        -----
        Assigns ctrl.constProp to self.PropCmd and stores n_setpt.
        Does not implement thrust control or speed regulation.
        """

        # Assign Function as Vehicle Method
        self.PropCmd = ctrl.constProp

        # Load Parameters
        self.n_setpt = n_setpt

        # Generate Information Parameter
        self.info.update([('Propeller Command', f"Constant, {n_setpt} rpm")])
    
    #--------------------------------------------------------------------------
    def loadAquaNetTdmaLF(self, 
                          epDur:float=10,
                          bcDur:float=1,
                          rpDur:float=1,
                          )->None:
        """
        Configure TDMA communication via AquaNet for leader-follower swarm.
    
        
        Parameters
        ----------
        epDur : float
            Episode cycle duration in seconds (full comm cycle period).
        bcDur : float
            Broadcast/request frame duration in seconds (leader transmit time).
        rpDur : float
            Response frame duration in seconds (follower slot time).
            
        
        Notes
        -----
        - Sets up the Episode structure, which is the full set of transmission
          frames. 
        - The Episode starts with the Leader Broadcast / Request (BCRQ) frame,
          followed by a Response (RSPN) frame for each follower.
        - Configures network addresses, message schedules, and communication
          manager.
        """

        # AquaNet Manager Parameters
        self.txAddr = 255 if self.isLeader else self.target.nodeAddr
        self.nodeAddr = self.id
        self.aqBaseDir = os.path.dirname(inspect.getfile(aqnet))

        # Load Vehicle Modules
        commNetwork = aqnet.AquaNetManager(self.nodeAddr,self.aqBaseDir)
        commNetwork.start = commNetwork.initAquaNet
        commNetwork.monitor = comm.recvMsg  # Incoming Msg Monitor
        self.CommNetwork = commNetwork
        self.CommSched = comm.schedulerTdmaLF

        # Message Handling Parameters
        self.epsdDur = epDur                # Episode Cycle Duration (s)
        self.bcrqDur = bcDur                # BCST/RQST Duration (s)
        if (self.group is not None):
            self.rspnSched = [v.id for v in self.group] # using same as id list
        self.rspnFrame = 0                  # RSPN Frame Start Time (s)
        self.rspnSlot = 0                   # RSPN Slot Start Time (s)
        self.rspnDur = rpDur                # RSPN Time Slot Duration (s)
        if (not self.isLeader):
            self.rspnRqst = False           # RSPN REQUEST Received (Follower)
        else:
            self.nextEta, self.nextVel = guid.predictNextEtaVel(self)
        
        # Generate Information Parameters
        self.info.update([("Communication Network", "AquaNet"),
                          ("Access Mode", f"TDMA "+
                           f"({self.epsdDur}|{self.bcrqDur}-{self.rspnDur})s"),
                          ("Node Address", f"{self.nodeAddr}")])

    #--------------------------------------------------------------------------
    def loadMuNetLF(self, network:comm.MuNet, **kwargs)->None:
        """
        Configure communication via MuNet for leader-follower swarm.

        Parameters
        ----------
        network : comm.MuNet
            MuNet network object to register vehicle with
        **kwargs : dict
            Configuration options:

            - accessMode : str
                Network access mode ('fdma', 'tdma', 'tdma_lf')
            - epDur : float
                Episode Duration (s). Vehicle message transmission interval
            - bcDur : float
                TDMA - Broadcast-Request frame duration (s). (Leader)
            - rpDur : float
                TDMA - Response frame duration (s). (Followers)
            - txOset : float
                FDMA - Transmit offset (s). Time between each vehicle reporting.

        Notes
        -----
        - Supports FDMA and TDMA access modes for flexible swarm communication.
        - Registers vehicle with network and configures message timing parameters.
        """

        # Default Parameters
        defaults = {
            'accessMode': network.accessMode,
            'epDur': 5.0,
            'bcDur': 1.0,
            'rpDur': 1.0,
            'txOset': 0.5,
        }
        config = {**defaults, **kwargs}         # kwargs overwrite defaults

        # Communication Schedule
        accessMode = config['accessMode'].lower()
        accessStrategies = {
            'fdma': comm.schedulerFdmaLF,
            'tdma': comm.schedulerTdmaLF,
            'tdma_lf': comm.schedulerTdmaLF,
        }
        self.CommSched = accessStrategies[accessMode]

        # Message Timing Parameters
        self.epsdDur = config['epDur']          # TX interval (s)
        if (accessMode == 'fdma'):
            txOffset = config['txOset']         # TX offset between members (s)
            '''Dependency on id is a problem here, need to build *much* better
            parameter handling'''
            self.rspnSlot = txOffset * self.id  # Next TX time (s)
            details = f"({self.epsdDur}|{txOffset})s"
        elif (accessMode == 'tdma' or accessMode == 'tdma_lf'):
            self.bcrqDur = config['bcDur']      # BCST/RQST Duration (s)
            self.rspnDur = config['rpDur']      # RSPN Slot Start Time (s)
            self.rspnFrame = 0                  # RSPN Frame Start Time (s)
            self.rspnSlot = 0                   # RSPN Slot Start Time (s)
            if (not self.isLeader):
                self.rspnRqst = False           # REQUEST Received (Follower)
            if (self.group is not None):
                self.rspnSched = [m.id for m in self.group] # RSPN order
            details = f"({self.epsdDur}|{self.bcrqDur}-{self.rspnDur})s"

        # Generate Information Parameter
        # note: TX/RX channels info updated from munet after assignments
        self.nodeAddr = self.id
        self.info.update([('Communication Network', 'muNet'),
                          ('Access Mode', f"{accessMode.upper()} {details}"),
                          ('Node Address', f"{self.nodeAddr}"),
                          ('TX Channel IDs', '-'),
                          ('RX Channel IDs', '-'),])
        
        # Network Access Point
        self.CommNetwork = network.register(self)
        '''Use single TX for now. Node stores as set for future extension.'''
        self.txAddr = next(iter(self.CommNetwork.tx_channels), None)

        # Prime initial state parameters -> move this somwhere else, guid?
        if (self.isLeader):
            self.nextEta, self.nextVel = guid.predictNextEtaVel(self)

###############################################################################

def buildGroup(num:int,
               gid:str,
               hasLeader:bool=True,
               vehType:Vehicle=Remus100s,
               **kwargs)->List[Vehicle]:
    """
    Create a list of vehicle instances for swarm simulation.
    
    
    Parameters
    ----------
    num : int
        Number of vehicles in the group.
    gid : str
        Group identifier, single character (e.g., "A").
    hasLeader : bool
        If True, first vehicle in list is generated with isLeader=True.
    vehType : type
        Vehicle class to instantiate (default: Remus100s).
    **kwargs : dict
        Optional keyword arguments to pass to vehicle constructor.
        
    
    Returns
    -------
    group : list of Vehicle
        List of initialized vehicle objects with consistent group ID.
        If hasLeader=True, group[0].isLeader is True.
        
    
    Notes
    -----
    **Parameter Inheritance:**

    All vehicles in the group receive the same groupId and any keyword
    attributes passed in kwargs. The hasLeader parameter will only set
    isLeader=True for the first vehicle in the returned list.

    **Vehicle Numbering:**

    Vehicle IDs are assigned sequentially based on the class counter to avoid
    any duplicate vehicle IDs in a single session.

    
    Examples
    --------
    ### Basic group with default parameters:

    >>> swarm = buildGroup(5, "A", True)
    >>> for auv in swarm:
    ...     print(auv.callSign)
    ...
    A01-LEADER
    A02
    A03
    A04
    A05

    ### Group with custom vehicle parameters:

    >>> swarm = buildGroup(
    ...     num=3,
    ...     gid="B",
    ...     hasLeader=True,
    ...     vehType=Remus100s,
    ...     r_safe=15.0,
    ...     r_avoid=50.0
    ... )
    >>> print(swarm[0].sampleTime)
    0.01
    >>> print(swarm[1].r_safe)
    15.0
    
    ### Group without leader:
    
    >>> swarm = buildGroup(5, "C", hasLeader=False)
    >>> print(swarm[0].isLeader)
    False
    """

    group = []
    if (hasLeader):
        leader = vehType(groupId=gid, isLeader=True, **kwargs)
        group.append(leader)
        num -= 1
    for _ in range(num):
        group.append(vehType(groupId=gid, **kwargs))
    return group

###############################################################################
