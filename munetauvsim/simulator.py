"""
Core simulation driver for AUV multi-agent scenarios.

Provides the Simulator class for running autonomous underwater vehicle
simulations with support for multiple vehicles, swarm coordination,
communication networks, environmental modeling, data collection, and
visualization.


Classes
-------
Simulator
    Main simulation orchestrator for AUV scenarios.

    
Functions
---------
save(simulation, filename, format)
    Save Simulator object to file (pickle format).
load(filename, format)
    Load Simulator object from file.

    
Notes
-----
- Supports direct-access (no network), muNet, and AquaNet communication modes.
"""

from typing import Dict, List, Optional, Tuple, Union
from numpy.typing import NDArray
import os
import importlib
import inspect
import time
import datetime
import pickle
import matplotlib.pyplot as plt
import numpy as np
from munetauvsim import vehicles as veh
from munetauvsim import communication as comm
from munetauvsim import environment as env
from munetauvsim import plotTimeSeries as pltTS
from munetauvsim import logger

#-----------------------------------------------------------------------------#

# Type Aliases
NPFltArr = NDArray[np.float64]

###############################################################################

class Simulator:
    """
    Main simulation coordinator for multi-agent AUV scenarios.
    
    Manages the complete simulation workflow including time evolution, vehicle
    dynamics integration, environmental modeling, inter-vehicle communication,
    data collection, collision detection, and visualization. Supports multiple
    communication modes (direct-access, MuNet, AquaNet) and flexible deployment
    strategies for swarm coordination research.

    
    Parameters
    ----------
    name : str, default='Simulation'
        Simulation title. Used for output directory and file naming.
    sampleTime : float, default=0.02
        Iteration time step in seconds (50 Hz default). Used for discrete-time
        integration of vehicle dynamics and kinematics.
    N : int, default=60000
        Total number of simulation iterations. With default sampleTime, gives 20
        minutes of simulated time.
    ocean : env.Ocean, optional
        Ocean environment object containing Ocean current and floor depth map.
        If None, simulation runs without environmental effects.
    vehicles : list of Vehicle, optional
        List of vehicle objects to simulate. Can be added after initialization
        via the vehicles property.
    comnet : str, {'muNet', 'AquaNet', None}, optional
        Communication network type. If None, uses direct-access mode where
        vehicles can directly read each other's states (no network delays).
    logging : str, default='all'
        Main logger configuration. Options: 'all', 'none', 'noout', 'nofile',
        'quiet', 'onlyfile', 'onlyconsole'.
    commLogging : str, default='all'
        Communication logger configuration. Same options as logging parameter.
    **kwargs : dict
        Additional attributes to set on simulator instance.

        
    Attributes
    ----------
    **Time Management:**

        sampleTime : float
            Simulation time step (seconds per iteration).
        N : int
            Number of simulation iterations.
        runTime : float
            Total simulation time in seconds. Equal to N * sampleTime.
            Setting any of these three automatically updates the others.
        simTime : ndarray, shape (N+1,)
            Time vector for plotting. Includes t=0 start point.
        initTime : str
            Timestamp when simulator was created (YYMMDD-HHMMSS format).
            
    **Environment and Vehicles:**

        ocean : env.Ocean
            Ocean environment with currents, floor depth, etc.
        vehicles : list of Vehicle
            List of all vehicles in simulation.
        nVeh : int (read-only)
            Number of vehicles in simulation.
            
    **Communication Networks:**

        comnet : str
            Communication network type ('Direct-Access', 'muNet', 'AquaNet').
        muNet : comm.MuNet
            muNet network instance (if using muNet).
            
    **Output and Logging:**

        name : str
            Simulation name. Can only be set at initialization.
        outDir : str (read-only)
            Output directory path: outputs/<script_name>/<name>_<timestamp>/
        saveFile : str (read-only)
            Full path for saving simulator object (pickle).
        logFile : str
            Path to main log file.
        commFile : str
            Path to communication log file.
        gifFile : str
            Path to animated GIF output.
        log : logging.Logger
            Main simulation logger instance.
        logging : str
            Main logger configuration setting.
        commLogging : str
            Communication logger configuration setting.
            
    **Data Collection:**

        simData : ndarray, shape (nVeh, N+1, 18)
            Complete simulation data array containing vehicle states and
            controls. For each vehicle and timestep: [eta(6), nu(6),
            u_control(3), u_actual(3)]. Size N+1 to include initial conditions
            at t=0.
            
    **Visualization:**

        numDataPoints : int, default=200
            Number of trajectory points displayed in 3D animation.
        FPS : int, default=8
            Frames per second for animated GIF output.
            
    **Vehicle Proximity Monitoring:**

        vehProxMult : float, default=1.0
            Safety multiplier for vehicle contact radius. Contact radius =
            vehicle_length * vehProxMult.
        vehProxImmobilize : bool, default=True
            If True, vehicles stop moving on contact detection for the remainder
            of the simulation.
            
    **Internal Group Management (Private):**

        _groupsDict : dict
            Dictionary of non-leader vehicles by groupId.
        _leadersDict : dict
            Dictionary of leader vehicles by groupId.
        _noneGroup : list
            Vehicles without groupId or target assignment.
        _contactCount : int
            Total number of vehicle collisions detected.
        _contactRadius : ndarray
            Cached contact threshold distances for all vehicle pairs.

            
    Methods
    -------
    **Simulation Execution:**

        run()
            Execute complete simulation: loop, collect data, log stats, generate
            plots.
        simulate()
            Run iteration loop and return collected simulation data.
            
    **Visualization:**

        plot3D(numDataPoints, FPS, gifFile, vehicles, figNo, show*)
            Create 3D animated visualization and save as GIF.
            
    **Vehicle Deployment:**

        deployAtWpt(vehicle, posOnly)
            Deploy vehicle at next waypoint position in its database.
        deploySwarmGrid()
            Deploy vehicles in grid formation by group, West of leader.
            
    **Swarm Coordination:**

        linkSwarmGroup()
            Populate vehicle.group lists with references or Models for
            coordination.
            
    **Communication Setup:**

        loadMuNet(network, episode, txOffset, vehicles, kwargs)
            Configure MuNet communication network.
        loadAquaNet(episode, frames, vehicles)
            Configure AquaNet TDMA communication.
            
    **Monitoring and Logging:**

        initVehicleContactMonitor()
            Initialize collision detection system.
        logCommStats()
            Log network performance statistics.
            
    **Internal Simulation Loops (Private):**

        _simulateNoComm(nVeh, simData)
            Direct-access iteration loop (no network delays).
        _simulateMuNet(nVeh, simData)
            muNet communication iteration loop.
        _simulateAquaNet(nVeh, simData)
            AquaNet TDMA iteration loop with time synchronization.
        _simSync()
            Synchronize simulation time with AquaNet communication timing.

            
    Notes
    -----
    **Time Parameter Interdependence:**

    The attributes sampleTime, N, and runTime are coupled:
    
    - sampleTime can be set directly
    - N and runTime update automatically when one is set
    - Setting sampleTime recalculates runTime if N is defined
    
    Example:

    >>> sim.sampleTime = 0.01   # 100 Hz
    >>> sim.N = 10000           # 100 seconds at 100 Hz
    >>> print(sim.runTime)
    100.0
    
    **Communication Modes:**
    
    1. Direct-Access (comnet=None):

       - Vehicles directly read each other's states
       - No network delays or packet loss
       - Ideal for baseline comparisons
       
    2. muNet (comnet='muNet'):

       - Simulated acoustic modem network
       - Configurable packet loss, jitter, collision detection
       - Supports FDMA and TDMA access modes
       
    3. AquaNet (comnet='AquaNet'):

       - Real AquaNet stack via Unix sockets
       - TDMA protocol with leader-follower timing
       - Requires AquaNet installation and configuration
    
    **Output Directory Structure:**

    .. code-block:: none

        outputs/
            <script_name>/
                <name>_<timestamp>/
                    <name>_<timestamp>.log          # Main log
                    <name>_<timestamp>_comm.log     # Communication log
                    <name>_<timestamp>.gif          # Animation
                    <name>_<timestamp>.pickle       # Saved simulator

    **Vehicle Contact Detection:**
    
    The simulator continuously monitors vehicle-vehicle distances during
    simulation. When distance < contact_radius:
    
    - Warning logged with vehicle IDs and positions
    - Contact counter incremented
    - Vehicles immobilized (if vehProxImmobilize=True)
    
    Contact radius based on vehicle length and safety multiplier.

    
    Examples
    --------
    ### Basic simulation with path-following leader:
    
    >>> import munetauvsim.simulator as muSim
    >>> import munetauvsim.environment as env
    >>> import munetauvsim.vehicles as veh
    >>> import munetauvsim.guidance as guid
    >>> import munetauvsim.communication as com
    >>> 
    >>> # Create vehicles
    >>> leader = veh.Remus100s(groupId="A", isLeader=True)
    >>> leader.wpt = guid.Waypoint([0, 500], [0, 500], [0, 15])
    >>> leader.loadPathFollowing()
    >>> leader.loadConstantProp()
    >>> 
    >>> # Create simulator
    >>> sim = muSim.Simulator(
    ...     name="LeaderDemo",
    ...     sampleTime=0.02,
    ...     N=30000,  # 10 minutes
    ...     ocean=env.Ocean(),
    ...     vehicles=[leader]
    ... )
    >>> 
    >>> # Deploy and run
    >>> sim.deployAtWpt(leader)
    >>> sim.run()
    
    ### Multi-agent swarm with muNet communication:
    
    >>> # Create swarm
    >>> leader = veh.Remus100s(groupId="A", isLeader=True)
    >>> leader.wpt = guid.Waypoint([0, 1000], [0, 0], [0, 40])
    >>> leader.loadPathFollowing()
    >>> leader.loadConstantProp()
    >>> 
    >>> followers = veh.buildGroup(3, "A", hasLeader=False)
    >>> for f in followers:
    ...     f.loadTargetTracking(leader, law="APF")
    >>> 
    >>> # Setup communication
    >>> network = com.MuNet(PLR=0.05, MAX_JITTER=0.3)
    >>> 
    >>> # Create simulator
    >>> sim = muSim.Simulator(
    ...     name="SwarmDemo",
    ...     N=60000,  # 20 minutes
    ...     ocean=env.Ocean(spd=0.5, ang=0),
    ...     vehicles=[leader] + followers
    ... )
    >>> 
    >>> # Configure and run
    >>> sim.loadMuNet(network, episode=5.0, txOffset=0.5)
    >>> sim.deployAtWpt(leader)
    >>> sim.deploySwarmGrid()
    >>> sim.run()
    >>> muSim.save(sim)
    
    ### Load and analyze previous simulation:
    
    >>> sim = muSim.load(
    ...     "/path/to/outputs/SwarmDemo/SwarmDemo_241103-143000.pickle"
    ... )
    >>> print(f"Simulation ran for {sim.runTime} seconds")
    >>> print(f"Vehicle contacts: {sim._contactCount}")
    >>> 
    >>> # Re-plot with different settings
    >>> sim.plot3D(numDataPoints=500, FPS=12, showFloor=False)

    
    See Also
    --------
    vehicles.Remus100s : Main AUV vehicle class
    communication.MuNet : Simulated underwater acoustic network
    environment.Ocean : Ocean environment model
    save : Save simulator to file
    load : Load simulator from file
    """

    ## Constructor ===========================================================#
    def __init__(self,
                 name:str = 'Simulation',
                 sampleTime:float = 0.02,
                 N:int = 60000,
                 ocean:Optional[env.Ocean] = None,
                 vehicles:Optional[List[veh.Vehicle]] = None,
                 comnet:Optional[str] = None,
                 logging:str = 'all',
                 commLogging:str = 'all',
                 **kwargs,
                 )->None:
        """
        Initialize Simulator with time parameters, vehicles, and environment.
        

        Parameters
        ----------
        name : str
            Simulation title.
        sampleTime : float
            Time step per iteration in seconds. 
        N : int
            Number of simulation iterations.
        ocean : env.Ocean, optional
            Ocean environment object.
        vehicles : list of Vehicle, optional
            Vehicles to simulate.
        comnet : str, optional
            Communication network type.
        logging : str
            Main logger configuration.
        commLogging : str
            Communication logger configuration.
        **kwargs
            Additional attributes to set on simulator.
            

        Notes
        -----
        - Creates output directory structure:
          outputs/<script_name>/<simulation_name>_<timestamp>/
        - Initializes loggers.
        - Sets up simulation time array.
        """
        
        ## Time Stamp
        init_time = datetime.datetime.now()
        self.initTime = init_time.strftime("%y%m%d-%H%M%S")

        ## Data
        self.simData = None                         # data generated by the sim
        self.simTime = None                         # simulation times array

        ## Simulation
        self.name = name                            # simulation title
        self.sampleTime = sampleTime                # iteration time step (sec)
        self.N = N                                  # number of iterations
        
        ## Objects
        self.ocean = ocean                          # ocean object
        self.vehicles = vehicles                    # vehicles list
        self.comnet = comnet                        # communication method
        
        # Plotting
        self.numDataPoints = 200                    # number of 3D data points
        self.FPS = 8                                # GIF frames per second   

        ## Vehicle Proximity
        self.vehProxMult = 1.0                      # increase veh contact rad
        self.vehProxImmobilize = True               # stop vehicles on contact

        ## User Keyword Attributes
        for key,value in kwargs.items():
            if key not in {                         # computed attributes
                'simTime',
                'simData',
            }:
                setattr(self, key, value)

        ## Logging
        self.log = None                             # main logger
        self.logging = logging                      # logging setting
        self.commLogging = commLogging              # comms logging setting

    ## Properties ============================================================#
    @property
    def name(self)->str:
        """Get simulation name."""
        return self._name
    
    @name.setter
    def name(self, name:str)->None:
        """Set the simulation name. Can only be set at initialization."""
        if ('_name' in self.__dict__):
            self.log.warning("Cannot rename simulation. Attribute must be" +
                             "set at initialization.")
            return
        baseName = f"{name}_{self.initTime}"
        self._outDir = self._makeSaveDir(baseName)
        self._saveFile = os.path.join(self.outDir, baseName)
        self._name = name
    
    #--------------------------------------------------------------------------
    @property
    def sampleTime(self)->float:
        """Get simulation iteration time step in seconds."""
        return self._sampleTime
    
    @sampleTime.setter
    def sampleTime(self, h:float)->None:
        """
        Set simulation time step and update vehicles and ocean.
        
        Parameters
        ----------
        h : float
            Time step in seconds.
            
        Notes
        -----
        - Automatically updates runTime if N is already set.
        - Propagates sampleTime to all vehicles and ocean environment.
        """
        self._sampleTime = h
        if ('_N' in self.__dict__):
            self.N = self.N
        if (('_vehicles' in self.__dict__) and 
            (self._vehicles is not None)):
            # Push simulation sampleTime to vehicles
            for v in self._vehicles:
                v.sampleTime = h
        if (('_ocean' in self.__dict__) and
            (self._ocean is not None)):
            # Push simulation sampleTime to ocean
            self._ocean.sampleTime = h
    
    #--------------------------------------------------------------------------
    @property
    def N(self)->int:
        """Get number of simulation iterations."""
        return self._N
    
    @N.setter
    def N(self, n:int)->None:
        """
        Set number of iterations and compute time arrays.
        
        Parameters
        ----------
        n : int
            Number of simulation iterations.
            
        Notes
        -----
        - Automatically computes simTime array and runTime.
        - Propagates N+1 to ocean environment for array sizing.
        """
        self._N = n
        self.simTime = np.arange(start=0,
                                 stop=self.sampleTime*(n+1),
                                 step=self.sampleTime,
                                 )[:, None]
        self._runTime = self.simTime[-1][0]
        if (('_ocean' in self.__dict__) and
            (self._ocean is not None)):
            # Push number of simulation iterations to ocean
            self._ocean.N = n+1
            
    #--------------------------------------------------------------------------
    @property
    def runTime(self)->float:
        """Get total simulation time in seconds."""
        return self._runTime
    
    @runTime.setter
    def runTime(self, n:float)->None:
        """
        Set total simulation time and compute number of iterations.
        
        Parameters
        ----------
        n : float
            Total simulation time in seconds.
            
        Notes
        -----
        Indirectly calls N.setter which updates all dependent attributes.
        """
        self.N = int(n/self.sampleTime)

    #--------------------------------------------------------------------------
    @property
    def ocean(self)->env.Ocean:
        """Get ocean environment object."""
        return self._ocean
    
    @ocean.setter
    def ocean(self, ocean:env.Ocean)->None:
        """
        Set ocean environment and enforce time parameters.
        
        Parameters
        ----------
        ocean : env.Ocean
            Ocean environment object.
            
        Notes
        -----
        Sets ocean.N and ocean.sampleTime to match simulator values.
        """
        if (ocean is not None):
            ocean.N = self.N+1
            ocean.sampleTime = self.sampleTime
        self._ocean = ocean

    #--------------------------------------------------------------------------
    @property
    def vehicles(self)->List[veh.Vehicle]:
        """Get list of simulation vehicles."""
        return self._vehicles
    
    @vehicles.setter
    def vehicles(self, vehicles:List[veh.Vehicle])->None:
        """
        Set vehicle list and update group dictionaries.
        
        Parameters
        ----------
        vehicles : list of Vehicle
            Vehicles to simulate.
            
        Notes
        -----
        - Enforces sampleTime consistency across all vehicles.
        - Builds internal group dictionaries for swarm coordination.
        """
        self._vehicles = vehicles
        self._nVeh = 0
        self._vCallSigns = None
        if (vehicles is not None):
            # Ensure vehicles sampleTime value agrees with sim
            for v in self._vehicles:
                v.sampleTime = self.sampleTime
            self._groupsDict, self._leadersDict, self._noneGroup = (
                self._buildGroupDicts())
            self._nVeh = len(self._vehicles)
            self._vCallSigns = [v.callSign for v in self._vehicles]

    #--------------------------------------------------------------------------
    @property
    def nVeh(self)->int:
        """Get number of vehicles in simulation."""
        return self._nVeh
    
    #--------------------------------------------------------------------------
    @property
    def comnet(self)->str:
        """Get communication network type."""
        return self._comm
    
    @comnet.setter
    def comnet(self, comnet:Optional[str])->None:
        """
        Set communication network and select simulation loop method.
        
        Parameters
        ----------
        comnet : str or None
            'AquaNet', 'MuNet', or None for direct-access.
            
        Notes
        -----
        Assigns appropriate simulation loop: _simulateAquaNet, _simulateMuNet,
        or _simulateNoComm based on network type.
        """
        if (comnet is not None):
            if ("AQUANET" in comnet.upper()):
                self._simLoop = self._simulateAquaNet
                self._comm = "AquaNet"
                self._syncTime = 0.0
            elif ("MUNET" in comnet.upper()):
                self._simLoop = self._simulateMuNet
                self._comm = "muNet"
            self.commLogging = self._commLogging
            return
        self._simLoop = self._simulateNoComm
        self._comm = "Direct-Access"

    #--------------------------------------------------------------------------
    @property
    def outDir(self)->str:
        """Get output directory path."""
        return self._outDir
    
    @outDir.setter
    def outDir(self, outDir:str)->None:
        """Attempt to set the output directory for the simulation."""
        self.log.warning("Cannot set output directory directly. Attribute is "+
                         "set at initialization by the 'name' attribute.")

    #--------------------------------------------------------------------------
    @property
    def saveFile(self)->str:
        """Get save file path."""
        return self._saveFile
    
    @saveFile.setter
    def saveFile(self, saveFile:str)->None:
        """Attempt to set the name of the save file."""
        self.log.warning("Cannot set save file name directly. Attribute is " +
                         "set at initialization by the 'name' attribute.")

    #--------------------------------------------------------------------------
    @property
    def logFile(self)->str:
        """Get main log file path."""
        if ('_logFile' not in self.__dict__):
            self._logFile = f"{self.saveFile}.log"
        return self._logFile
    
    @logFile.setter
    def logFile(self, logFile:str)->None:
        """Set main log file path. Can only bet set at initialization."""
        if ('_logFile' in self.__dict__):
            self.log.warning("Cannot rename log file. Attribute must be set" +
                             "at initialization.")
            return
        self._logFile = self._validFileName(logFile, '.log')
    
    #--------------------------------------------------------------------------
    @property
    def commFile(self)->str:
        """Get communication log file path."""
        if ('_commFile' not in self.__dict__):
            self._commFile = f"{self.saveFile}_comm.log"
        return self._commFile
    
    @commFile.setter
    def commFile(self, commFile:str)->None:
        """Set communication log file path. Can only set at initialization."""
        if ('_commFile' in self.__dict__):
            self.log.warning("Cannot rename comm log file. Attribute must be" +
                             "set at initialization.")
            return
        self._commFile = self._validFileName(commFile, '.log')

    #--------------------------------------------------------------------------
    @property
    def gifFile(self)->str:
        """Get animated GIF file path."""
        if ('_gifFile' not in self.__dict__):
            self._gifFile = f"{self.saveFile}.gif"
        return self._gifFile
    
    @gifFile.setter
    def gifFile(self, gifFile:str)->None:
        """Set animated GIF file path. Can only be set at initialization."""
        if ('_gifFile' in self.__dict__):
            self.log.warning("Cannot rename gif file. Attribute must be set" +
                             "at initialization.")
            return
        self._gifFile = self._validFileName(gifFile, '.gif')
    
    #--------------------------------------------------------------------------
    @property
    def logging(self)->str:
        """Get main logger configuration."""
        return self._logging
    
    @logging.setter
    def logging(self, logging:str)->None:
        """
        Set main logger configuration.
        
        Parameters
        ----------
        logging : str
            'all', 'none', 'noout', 'nofile', 'quiet', 'onlyfile',
            'onlyconsole'.
        """

        def setNoneLog()->None:
            """Set the main logger to no logging"""
            self.log = logger.noneLog(logger.MAIN_LOG)

        def setNoConsoleLog()->None:
            """Set the main logger to no console logging"""
            if (self.log is not None):
                logger.deepRemoveHandler(logger.consoleHandler)
                logger.removeLog(logger.MAIN_LOG)
            self.log = logger.setupMain(fileName=self.logFile,outFormat=None)

        def setNoFileLog()->None:
            """Set the main logger to no file logging"""
            if (self.log is not None):
                logger.deepRemoveHandler(logger.fileHandler)
                logger.removeLog(logger.MAIN_LOG)
            self.log = logger.setupMain(fileFormat=None)

        def setDefaultLog()->None:
            """Set the main logger to default logging to console and file"""
            if (self.log is not None):
                logger.removeLog(logger.MAIN_LOG)
            self.log = logger.setupMain(fileName=self.logFile)

        # Map the logging settings to logging setter functions
        logSettings = {
            # No logging
            'NONE': setNoneLog,
            'OFF': setNoneLog,
            # No console logging
            'NOOUT': setNoConsoleLog,
            'QUIET': setNoConsoleLog,
            'NOCONSOLE': setNoConsoleLog,
            'ONLYFILE': setNoConsoleLog,
            # No file logging
            'NOFILE': setNoFileLog,
            'ONLYOUT': setNoFileLog,
            'ONLYCONSOLE': setNoFileLog,
        }

        # Set the logging settings
        configLog = logSettings.get(logging.upper(), setDefaultLog)
        configLog()
        self._logging = logging
    
    #--------------------------------------------------------------------------
    @property
    def commLogging(self)->str:
        """Get communication logger configuration."""
        return self._commLogging
    
    @commLogging.setter
    def commLogging(self, commLogging:str)->None:
        """
        Set communication logger configuration.
        
        Parameters
        ----------
        commLogging : str
            'all', 'none', 'noout', 'nofile', 'quiet', 'onlyfile',
            'onlyconsole'.
        """

        if ('_commLogging' not in self.__dict__):
            self._commLogging = commLogging

        settings = commLogging.upper()
        changed = (settings != self._commLogging.upper())
        self._commLogging = commLogging

        # Update only if comnet has been set and changes are needed
        if (self.comnet != 'Direct-Access'):
            # Comm logger already exists
            if (comm.log is not None):
                # Same name
                if (comm.log.name == self.comnet):
                    # Same settings
                    if not (changed):
                        return
                    # Different settings
                    logger.removeHandlers(comm.log.name)
                # Different name
                else:
                    logger.removeLog(comm.log.name)
            
            # Define comm logger setting functions
            def setNoneComm()->None:
                """Set the comm logger to no logging"""
                comm.log = logger.setupComm(name=self.comnet,file=False,
                                            out=False)
            
            def setNoConsoleComm()->None:
                """Set the comm logger to no console logging"""
                comm.log = logger.setupComm(name=self.comnet,
                                            fileName=self.commFile,
                                            out=False)

            def setNoFileComm()->None:
                """Set the comm logger to no unique file logging"""
                comm.log = logger.setupComm(name=self.comnet,
                                            file=False)
                
            def setDefaultComm()->None:
                """Set the comm logger to default logging"""
                comm.log = logger.setupComm(name=self.comnet,
                                            fileName=self.commFile)

            # Map the comm logging settings to comm log setting functions
            commSettings = {
                # No console or unique file logging
                'NONE': setNoneComm,
                'OFF': setNoneComm,
                # No console logging
                'NOOUT': setNoConsoleComm,
                'QUIET': setNoConsoleComm,
                'NOCONSOLE': setNoConsoleComm,
                # No unique file logging
                'NOFILE': setNoFileComm,
            }

            # Set the comm logging settings
            configCommLog = commSettings.get(settings, setDefaultComm)
            configCommLog()

    ## Special Methods =======================================================#
    def __str__(self)->str:
        """
        Return user-friendly string representation of simulator configuration.
        """
        line = '*' * 64
        if (self.comnet.upper() == "MUNET"):
            commOut = f"\n{self.muNet}"
        else:
            commOut = f"Communication: {self.comnet}"
        
        vehicleOut = ["Vehicles: "]
        if (self.vehicles):
            modelNames = [v.modelName for v in self.vehicles]
            modelCounts = {m:modelNames.count(m) for m in set(modelNames)}
            vehicleOut.extend(f"({num}) {m}" for m,num in modelCounts.items())
            vehicleOut.extend(f"{v}" for v in self.vehicles)
            cr_min = self._contactRadius.min()
            cr_max = self._contactRadius.max()
            if (cr_max - cr_min < 1E-6):
                crState = f"Uniform vehicle sizes"
                crOut = f"Contact Radius: {cr_min:.1f} m"
            else:
                crState = f"Varied vehicle sizes"
                crOut = f"Contact Radius: ({cr_min:.1f} - {cr_max:.1f}) m"
            vehicleOut.extend([
                f"",
                f"Contact Monitoring",
                f" {crState}",
                f" Safety Multiplier: {self.vehProxMult:.1f}",
                f" {crOut}",
                f"",
            ])

        else:
            vehicleOut.append("None")

        return "\n".join([
            line,
            f"{self.__class__.__name__}: {self.name}",
            line,
            f"Sampling frequency: {round(1 / self.sampleTime)} Hz",
            f"Simulation time: {round(self.runTime)} seconds",
            commOut,
            f"{self.ocean if self.ocean else 'Ocean: None'}",
            *vehicleOut,
            line,
        ])

    ## Methods ===============================================================#
    def run(self)->None:
        """
        Execute complete simulation workflow: run, collect data, generate plots.
        
        Orchestrates the full simulation process including iteration loop
        execution, data collection, vehicle contact monitoring, performance
        logging, and 3D visualization generation.
        

        Notes
        -----
        Workflow:

        1. Initialize vehicle contact monitoring
        2. Execute simulation loop, calls simulate()
        3. Log performance metrics
        4. Generate 3D plots and animated GIF, calls plot3D()
        5. Display total execution time
        """

        self.initVehicleContactMonitor()
        self.log.info(f"{self}")
        start = time.time()
        self.simData = self.simulate()
        runTime = round(self.runTime)
        endData = round(time.time()-start)
        line = '*' * 64
        self.log.info(line)
        self.log.info(f'Vehicle contacts: %s', self._contactCount)
        self.log.info(f'Run Time:'+
                      f' (Real) {datetime.timedelta(seconds=endData)},'+
                      f' (Simulated) {datetime.timedelta(seconds=runTime)}')
        self.logCommStats()

        # # Commented out to reduce graphical output. For now just want to see 
        # # the multi-agent plots in 3d. -JPC 06/2023
        # for i in range(self.nVeh):
        #     pltTS.plotVehicleStates(self.simTime,
        #                             self.simData[i],
        #                             self.vehicles[i].id,
        #                             i*2+1)
        #     pltTS.plotControls(self.simTime,
        #                        self.simData[i],
        #                        self.vehicles[i],
        #                        i*2+2)
        
        self.plot3D()

        endTotal = round(time.time()-start)
        endPlot = round(endTotal - endData)
        self.log.info(f'Plotting Time: {datetime.timedelta(seconds=endPlot)}')
        self.log.info(f'Total Time: {datetime.timedelta(seconds=endTotal)}')
        self.log.info(line)

    #--------------------------------------------------------------------------
    def simulate(self)->NPFltArr:
        """
        Execute simulation iteration loop and collect vehicle data.
        

        Returns
        -------
        simData : ndarray, shape (n_vehicles, N+1, 18)
            Simulation data containing vehicle states and controls.
            Each row: [eta(6), nu(6), u_control(3), u_actual(3)].
            

        Notes
        -----
        - Initializes swarm groups via linkSwarmGroup() before starting
          iteration loop.
        - Delegates to appropriate simulation method based on communication
          network:

          - _simulateNoComm() for direct-access
          - _simulateMuNet() for MuNet
          - _simulateAquaNet() for AquaNet
        """

        ## Set Up Parameters
        DOF = 6                         # degrees of freedom
        dimU = self.vehicles[0].dimU    # number of vehicle controls
        
        # Initialize Simulation Data Storage Table
        simData = np.empty([self.nVeh, self.N+1, 2*DOF + 2*dimU], float)

        # Initialize Swarm Group
        self.linkSwarmGroup()

        # Simulation Loop
        self._simLoop(simData)
        
        return simData
    
    #--------------------------------------------------------------------------
    def plot3D(self,
               numDataPoints:Optional[int]=None,
               FPS:Optional[int]=None,
               gifFile:Optional[str]=None,
               vehicles:Optional[List[veh.Vehicle]]=None,
               figNo:Optional[int]=None,
               showClock:Optional[bool]=True,
               showData:Optional[bool]=True,
               showTraj:Optional[bool]=True,
               showPos:Optional[bool]=True,
               showCur:Optional[bool]=True,
               showFloor:Optional[bool]=True,
               )->None:
        """
        Create 3D visualization and animated GIF of simulation.
        

        Parameters
        ----------
        numDataPoints : int, optional
            Number of trajectory points to display (default:
            self.numDataPoints).
        FPS : int, optional
            Frames per second for GIF animation (default: self.FPS).
        gifFile : str, optional
            Output GIF filename (default: self.gifFile).
        vehicles : list of Vehicle, optional
            Vehicles to plot (default: all vehicles).
        figNo : int, optional
            Figure number used by Matplotlib for window reference.
        showClock : bool
            Display simulation time clock.
        showData : bool
            Display vehicle state data panel.
        showTraj : bool
            Show vehicle trajectory paths.
        showPos : bool
            Show vehicle position markers.
        showCur : bool
            Show ocean current vectors.
        showFloor : bool
            Show ocean floor surface.
            

        Notes
        -----
        - Wrapper for plotTimeSeries.plot3D().
        - Saves animated GIF to gifFile path.
        """
        
        if (numDataPoints is None):
            numDataPoints = self.numDataPoints
        if (FPS is None):
            FPS = self.FPS
        if (gifFile is None):
            gifFile = self.gifFile
        if (vehicles is None):
            vehicles = self.vehicles
        if (figNo is None):
            figNo = len(self.vehicles)*2 + 1
        
        pltTS.plot3D(self.simData,
                     self.sampleTime,
                     numDataPoints,
                     FPS,
                     gifFile,
                     vehicles,
                     self.ocean,
                     figNo,
                     showClock=showClock,
                     showData=showData,
                     showTraj=showTraj,
                     showPos=showPos,
                     showCur=showCur,
                     showFloor=showFloor)
        plt.show()
        plt.close()

    #--------------------------------------------------------------------------
    def deployAtWpt(self, vehicle:veh.Vehicle, posOnly:bool=False)->None:
        """
        Deploy vehicle at next waypoint position in its database.
        

        Parameters
        ----------
        vehicle : Vehicle
            Vehicle to deploy. Must have wpt, eta, wpt_k, z_d attributes.
        posOnly : bool
            If True, sets only position [x,y,z]. If False, also sets heading
            to point toward next waypoint.
            

        Notes
        -----
        - Sets vehicle.eta[0:3] from waypoint position.
        - Sets vehicle.z_d (desired depth) to current depth.
        - If posOnly=False, calculates and sets heading toward next waypoint.
        """

        # Set Vehicle State Parameters
        vehicle.eta[0:3] = np.copy(vehicle.wpt.pos[vehicle.wpt_k])
        vehicle.z_d = vehicle.eta[2]
        
        # Set Vehicle Attitude Parameters
        ## For now, only adjusting Yaw
        if (not posOnly):
            idx = vehicle.wpt_k
            if (idx > len(vehicle.wpt)-1):
                idx -= 1
            vehicle.eta[5] = vehicle.wpt.calcWptHeading(idx)
            vehicle.psi_d = vehicle.eta[5]

    #--------------------------------------------------------------------------
    def deploySwarmGrid(self)->None:
        """
        Deploy vehicles in grid formation by group, West of group leader.
        
        Arranges follower vehicles in a grid formation at specified spacing from
        the leader. Formation parameters (spacing, columns) are computed from
        vehicle swarm coordination attributes (r_follow, r_avoid).
        

        Notes
        -----
        - Grid layout: Rows North-South, Columns East-West, starting West of
          leader.
        - Spacing factor alpha=1.2 multiplies r_follow and r_avoid.
        - Default: 3 columns per row.
        - Assumes one leader per group (uses first found leader).
        """
        
        # Deploy by groupId
        for groupId in self._groupsDict:
            group = self._groupsDict[groupId]

            # Group has Leader-Follower system
            if (groupId in self._leadersDict):
                leader = self._leadersDict[groupId][0]

                # Set spatial parameters
                """
                Rows will be West from leader, aligned North-South, and columns
                will be aligned East-West.
                """
                alpha = 1.2                       # Spacing factor
                nCol = 3                          # Number of vehicales per row
                r_f = alpha * group[0].r_follow   # Distance of group to leader
                r_i = alpha * group[0].r_avoid    # Intragroup member distance

                # Set follower positions
                cnt = 0
                for v in group:
                    if (v is leader):
                        continue
                    # Copy leader position, velocity, and control commands
                    np.copyto(v.eta, leader.eta)
                    np.copyto(v.nu, leader.nu)
                    np.copyto(v.velocity, leader.velocity)
                    np.copyto(v.u_actual, leader.u_actual)
                    # Move follower to x,y grid position
                    v.eta[0] -= (r_f + r_i*(cnt//nCol))
                    v.eta[1] += (r_i * 
                                   (((cnt%nCol) * ((-1)**(cnt%nCol))) // 2))
                    # Set follower waypoint to own new position
                    v.wpt.insert(0, [*v.eta[0:3]])
                    v.wpt = v.wpt[0]
                    # Update follower count
                    cnt += 1

            # Group does not have Leader-Follower system
            else:
                pass

    #--------------------------------------------------------------------------
    def linkSwarmGroup(self)->None:
        """
        Link swarm group members on each vehicle for coordination.
        
        Populates vehicle.group lists with references (direct-access) or Models
        (networked communication) of other group members. Sets vehicle.target to
        leader if applicable.
        

        Notes
        -----
        - Direct-access mode: vehicle.group contains Vehicle references.
        - Network mode: vehicle.group contains Model objects for state tracking.
        - Assumes one leader per group.
        - Called automatically by simulate() before iteration loop.
        """

        # Link groups by groupId
        for groupId,group in self._groupsDict.items():

            # Leader in groupId
            if (groupId in self._leadersDict):
                leader = self._leadersDict[groupId][0]
                hasLeader = True
                useCommNet = (leader.CommNetwork is not None)
                if (leader in group):
                    group.remove(leader)
                if (useCommNet):
                    leader.group = [veh.Model(mem) for mem in group]
                else:
                    leader.group = group

            # No leader in groupId
            else:
                hasLeader = False
                useCommNet = (group[0].CommNetwork is not None)
            
            # Assign member group lists
            for member in group:
                others = [*group]
                others.remove(member)
                if (useCommNet):
                    member.group = [veh.Model(mem) for mem in others]
                    if ((hasLeader) and
                        ((member.target is None) or
                         (member.target.id != leader.id))): 
                        member.target = veh.Model(leader)
                else:
                    member.group = others
                    if ((hasLeader) and
                        ((member.target is None) or
                         (member.target.id != leader.id))):
                        member.target = leader

    #--------------------------------------------------------------------------
    def loadMuNet(self,
                  network:Optional[comm.MuNet]=None,
                  episode:float=5.0,
                  txOffset:float=0.5,
                  vehicles:Optional[List[veh.Vehicle]]=None,
                  **kwargs,
                  )->None:
        """
        Configure MuNet communication network for simulation vehicles.
        

        Parameters
        ----------
        network : comm.MuNet, optional
            MuNet network object. Creates new if None.
        episode : float
            Transmission episode duration in seconds (FDMA reporting interval).
        txOffset : float
            Time offset between vehicle transmissions.
        vehicles : list of Vehicle, optional
            Vehicles to configure (default: all simulation vehicles).
        **kwargs
            MuNet configuration parameters (PLR, MAX_JITTER, etc.).
            

        Notes
        -----
        Sets self.comnet = "muNet" and assigns network to self.muNet.
        Calls vehicle.loadMuNetLF() for each Remus100s vehicle.
        """

        self.comnet = "muNet"

        # Vehicles
        if (vehicles is None):
            try:
                vehicles = self.vehicles
            except (AttributeError):
                self.log.error("No vehicles provided and no vehicles in " +
                               "simulation. muNet not loaded.")
                return
                
        # Network
        if (network is None):
            try:
                network = self.muNet
            except (AttributeError):
                self.muNet = comm.MuNet(**kwargs)
                network = self.muNet
        else:
            self.muNet = network
        
        # Load muNet
        for v in vehicles:
            if (isinstance(v, veh.Remus100s)):
                v.loadMuNetLF(network=network,
                              epDur=episode,
                              txOset=txOffset,
                              **kwargs)

    #--------------------------------------------------------------------------
    def loadAquaNet(self, 
                    episode:Optional[float]=None,
                    frames:Union[float, List[float], None]=None, 
                    vehicles:Optional[List[veh.Vehicle]]=None,
                    )->None:
        """
        Configure AquaNet TDMA communication for simulation vehicles.
        

        Parameters
        ----------
        episode : float, optional
            Episode cycle duration (default: sum of all frame durations).
        frames : float or list of float, optional
            Frame durations in seconds. If list: [BCRQ_duration, RSPN_duration].
            Default: 1.0 second per frame.
        vehicles : list of Vehicle, optional
            Vehicles to configure (default: all simulation vehicles).

              
        Notes
        -----
        - TDMA structure: 
        
            - Leader broadcasts (BCRQ), then followers respond (RSPN).
            - Episode = BCRQ_frame + (RSPN_frame * num_followers).
            
        - Calls vehicle.loadAquaNetTdmaLF() for each Remus100s vehicle.
        """

        self.comnet = "AquaNet"

        # Vehicles
        if (vehicles is None):
            if (self.vehicles is None):
                self.log.error("No vehicles provided and no vehicles in " +
                               "simulation. AquaNet not loaded.")
                return
            vehicles = self.vehicles
        
        # Frame durations
        if (frames is None):
            bcrqDur, rspnDur = [1.0, 1.0]
        else:
            if (isinstance(frames, list)):
                bcrqDur, rspnDur = [*frames]
            else:
                bcrqDur, rspnDur = [frames, frames]
        
        # Episode duration
        if (episode is None):
            episode = bcrqDur + (rspnDur * (len(vehicles)-1))
    
        # Load AquaNet
        for v in vehicles:
            if (isinstance(v, veh.Remus100s)):
                v.loadAquaNetTdmaLF(epDur=episode,bcDur=bcrqDur,rpDur=rspnDur)

    #--------------------------------------------------------------------------
    def logCommStats(self)->None:
        """
        Log communication network performance statistics.
        
        Writes network performance metrics (message counts, delivery rates,
        latency statistics) to the simulation log. Only applies when using
        muNet networked communication (AquaNet not implemented).
        

        Notes
        -----
        - For muNet: Logs statistics for each network instance (supports
          multiple networks). Calls MuNet.getStatsReport() for formatted output.
        - For AquaNet: Statistics logging not yet implemented.        
        - For Direct-Access: No operation (no network to report).        
        - Called automatically by run() after simulation completes.
        """

        if (self.comnet is not None):
            # muNet
            if (self.comnet.upper() == "MUNET" and
                self.muNet is not None):
                network = self.muNet
                if (not isinstance(network, list)):
                    network = [network]
                for mu in network:
                    self.log.info(mu.getStatsReport())
                self.log.info("")

    #--------------------------------------------------------------------------
    def initVehicleContactMonitor(self)->None:
        """
        Initialize vehicle proximity collision detection system.
    
        Sets up data structures for monitoring vehicle-vehicle distances and
        detecting collisions during simulation. Contact radius for each vehicle
        pair is cached based on vehicle sizes and safety multiplier.
        
        
        Notes
        -----
        - **Attributes Initialized:**

            _contactCount : int
                Total number of vehicle contact events detected.
            _contactRadius : ndarray, shape (nVeh, nVeh)
                Pairwise contact threshold distances in meters.
            _activeContact : ndarray, shape (nVeh, nVeh), bool
                Boolean matrix tracking currently active contact pairs.
            _contactPosBuf : ndarray, shape (nVeh, 3)
                Position buffer for efficient distance calculation.
            _contactDistBuf : ndarray, shape (nVeh, nVeh)
                Distance matrix buffer.
            _contactMaskBuf : ndarray, shape (nVeh, nVeh), bool
                Mask buffer for contact detection logic.
            
        - Contact radius calculation:

          - Uniform vehicle sizes: r_contact = vehicle_length*vehProxMult
          - Mixed vehicle sizes: r_contact = max(length_i, length_j)*vehProxMult

        - Default vehProxMult = 1.0 (contact at vehicle length).
        - Called automatically by run() before simulation loop starts.
        """

        self._contactCount = 0           # Number of vehicle contacts
        self._contactRadius = None       # Matrix of veh-veh contact distances
        self._activeContact = None       # Matrix of active contact pairs
        self._contactPosBuf = None       # Buffer for writing positions
        self._contactDistBuf = None      # Buffer for writing distances
        self._contactMaskBuf = None      # Buffer for writing masks

        # No vehicles
        if (self.nVeh < 1):
            return
        
        # Initialize tracking structures
        self._activeContact = np.zeros((self.nVeh, self.nVeh), dtype=bool)
        self._contactPosBuf = np.empty((self.nVeh, 3), dtype=np.float32)
        self._contactDistBuf = np.empty((self.nVeh, self.nVeh), dtype=float)
        self._contactMaskBuf = np.empty((self.nVeh, self.nVeh), dtype=bool)
        
        # Cache contact radius for vehicle-vehicle pairs
        sizes = [v.L for v in self.vehicles]
        maxSizeDiff = 0.5          # max length difference to be "same size" (m)
        
        # All vehicles same size
        if (np.allclose(sizes, sizes[0], maxSizeDiff)):
            r_coll = sizes[0] * self.vehProxMult
            self._contactRadius = np.full((self.nVeh, self.nVeh), r_coll)
        
        # Vehicles have different sizes: use larger size in each pair
        else:
            sizes = np.array(sizes)
            r_coll = np.maximum(sizes[:, np.newaxis], sizes[np.newaxis, :])
            self._contactRadius = r_coll * self.vehProxMult

    #--------------------------------------------------------------------------
    def monitorContact(self)->None:
        """
        Detect vehicle-vehicle collisions using vectorized distance computation.
    
        Computes pairwise distances between all vehicles using NumPy
        broadcasting and matrix operations. Detects new contact events, logs
        warnings with position data, and optionally immobilizes colliding
        vehicles.
        

        Notes
        -----
        **Algorithm:**

        Uses vectorized pairwise Euclidean distance calculation to avoid
        creating large intermediate arrays:

            dist_ij = sqrt(||x_i||^2 + ||x_j||^2 - 2 * x_i * x_j)

        Computed via:

        1. Position buffer: Extract all vehicle positions (nVeh x 3)
        2. Squared norms: Compute ||x||^2 for each vehicle
        3. Gram matrix: Compute x·y^T via matrix multiplication
        4. Distance matrix: Combine terms and take square root
        5. Contact detection: Compare distances to cached contact radii

        **Contact Tracking:**

        - Maintains _activeContact matrix to track ongoing collisions
        - Reports only new contacts to prevent spam reporting for same collision
        - Clears finished contacts when vehicles separate
    
        **Side Effects:**

        - Increments self._contactCount for each new collision
        - Logs warning with vehicle names and positions
        - Sets vehicle.immobilized = True if vehProxImmobilize is True
        - Updates self._activeContact tracking matrix
        

        References
        ----------
        [1] Python Like You Mean It: Broadcasting.
        https://www.pythonlikeyoumeanit.com/Module3_IntroducingNumpy/Broadcasting.html#The-Final-Answer,-At-Last!
        
               
        See Also
        --------
        initVehicleContactMonitor : Initialize contact detection system
        """

        # Check if contacts possible
        if (self.nVeh < 2):
            return
        
        # Get vehicle positions
        positions = self._contactPosBuf
        for i,v in enumerate(self.vehicles):
            positions[i, :] = v.eta[0:3]                  # views, not copies
        
        # Calculate pairwise distances
        '''v_dist = sqrt(||x||^2 + ||y||^2 - 2 x·y)'''
        v_dist = self._contactDistBuf
        sq = np.einsum('ij,ij->i', positions, positions)  #x^2,y^2
        gram = positions @ positions.T                    #x*y
        np.add(sq[: ,None], sq[None, :], out=v_dist)      #d=x^2+y^2
        v_dist -= 2.0 * gram                              #d=x^2+y^2-2xy
        np.maximum(v_dist, 0.0, out=v_dist)               #clip float math errs
        np.sqrt(v_dist, out=v_dist)                       #d=sqrt(d)

        # Find current contact pairs
        all_contact = self._contactMaskBuf
        np.less(v_dist, self._contactRadius, out=all_contact)
        np.fill_diagonal(all_contact, False)
        np.bitwise_or(all_contact, all_contact.T, out=all_contact)

        # Construct matrix masks
        new_contact = np.bitwise_and(all_contact, 
                                     np.logical_not(self._activeContact))
        fin_contact = np.bitwise_and(self._activeContact,
                                     np.logical_not(all_contact))

        # Report new contacts
        '''Use only upper triangle to report once per pair'''
        i_idx, j_idx = np.where(np.triu(new_contact, k=1))
        for i, j in zip(i_idx, j_idx):
            dist = v_dist[i,j]
            r_coll = self._contactRadius[i,j]
            self.log.warning(
                '**CONTACT! Vehicles: *%s ** %s* @ (%.2f, %.2f, %.2f), '+
                'Vehicle Distance: %.2f, Contact Radius: %.2f ',
                self._vCallSigns[i], self._vCallSigns[j],
                positions[i,0], positions[i,1], positions[i,2],
                dist, r_coll,
            )
            self._contactCount += 1

            # Immobilize vehicles
            if (self.vehProxImmobilize):
                self.log.warning('Immobilizing vehicles: %s, %s',
                                 self._vCallSigns[i], self._vCallSigns[j])
                self.vehicles[i].immobilized = True
                self.vehicles[j].immobilized = True

        # Update active contacts matrix
        '''Assuming active, new, and finished contact matrices remain strictly
        symmetric with zero diagonal. Otherwise need to enforce before write.'''
        self._activeContact |= new_contact
        self._activeContact &= ~fin_contact

    ## Helper Methods ========================================================#
    def _simulateNoComm(self, simData:NPFltArr)->None:
        """
        Simulation loop for direct-access mode (no communication network).
        

        Parameters
        ----------
        nVeh : int
            Number of vehicles.
        simData : ndarray
            Preallocated array to store simulation data.

             
        Notes
        -----
        - Vehicles access each other's states directly (no network delays).
        - Iteration sequence: update clock, collect sensors, compute guidance,
          store data, integrate dynamics, propagate attitude.
        """

        for i in range(0, self.N+1):
            # Simulation time
            currentTime = self.simTime[i][0]
            logger.simTime = f'{currentTime:.2f}'

            for j in range(self.nVeh):
                v = self.vehicles[j]

                # Clock
                v.clock = currentTime

                # Collect Sensor Data
                v.collectSensorData(self.ocean,i)
                
                # Compute Control Commands
                u_control = v.GuidSystem(v)

                # Store Simulation Data
                signals = np.concatenate([v.eta, v.nu, u_control, v.u_actual])
                simData[j,i,:] = signals

                # Advance Position and Attitude Dynamics
                v.nu, v.u_actual = v.dynamics(u_control)
                v.eta, v.velocity = v.Attitude(v)
            
            # Monitor vehicle contact
            self.monitorContact()
    
    #--------------------------------------------------------------------------
    def _simulateMuNet(self, simData:NPFltArr)->None:
        """
        Simulation loop for muNet communication network.
    
        
        Parameters
        ----------
        nVeh : int
            Number of vehicles.
        simData : ndarray
            Preallocated array to store simulation data.
            

        Notes
        -----
        - Handles message transmission timing and network updates.
        - Iteration sequence: transmit messages per schedule, deliver messages,
          update vehicle states, compute guidance, integrate dynamics.
        - Supports multiple muNet networks if passed as list.
        """

        # Allow multiple muNets
        network = self.muNet
        if (not isinstance(network, list)):
            network = [network]

        # Start Simulation Loop
        for i in range(0, self.N+1):
            # Simulation time
            currentTime = self.simTime[i][0]
            logger.simTime = f'{currentTime:.2f}'

            # Advance Vehicles
            for j in range(self.nVeh):
                v = self.vehicles[j]

                # Clock
                v.clock = currentTime

                # Transmit According to Communication Schedule
                v.CommSched(v)

                # Collect Sensor Data
                v.collectSensorData(self.ocean,i)
                
                # Compute Control Commands
                u_control = v.GuidSystem(v)

                # Store Simulation Data
                signals = np.concatenate([v.eta, v.nu, u_control, v.u_actual])
                simData[j,i,:] = signals

                # Advance Position and Attitude Dynamics
                v.nu, v.u_actual = v.dynamics(u_control)
                v.eta, v.velocity = v.Attitude(v)
            
            # Update Communication Network Trasfers
            for mu in network:
                mu.deliver(self.simTime[i][0])
            
            # Monitor vehicle contact
            self.monitorContact()
            
    #--------------------------------------------------------------------------
    def _simulateAquaNet(self, simData:NPFltArr)->None:
        """
        Simulation loop for AquaNet TDMA communication network.
        

        Parameters
        ----------
        nVeh : int
            Number of vehicles.
        simData : ndarray
            Preallocated array to store simulation data.
            

        Notes
        -----
        - Starts AquaNet stack and message monitoring threads before loop.
        - Synchronizes simulation time with network communication timing.
        - Stops network and joins threads after loop completion.
        """

        # Turn on AquaNet and Start Listening
        for v in self.vehicles:
            try:
                v.CommNetwork.start()
            except (AttributeError):
                self.log.error(f'*{v.callSign}: {self.comnet} Not Found')
            else:
                v.monitorThread = v.CommNetwork.monitor(v)
        
        # Start Simulation Loop
        for i in range(0, self.N+1):
            # Simulation time
            currentTime = self.simTime[i][0]
            logger.simTime = f'{currentTime:.2f}'

            # Advance Vehicles
            for j in range(self.nVeh):
                v = self.vehicles[j]

                # Clock
                v.clock = currentTime

                # Transmit According to Communication Schedule
                v.CommSched(v)

                # Collect Sensor Data
                v.collectSensorData(self.ocean,i)
                
                # Compute Control Commands
                u_control = v.GuidSystem(v)

                # Store Simulation Data
                signals = np.concatenate([v.eta, v.nu, u_control, v.u_actual])
                simData[j,i,:] = signals

                # Advance Position and Attitude Dynamics
                v.nu, v.u_actual = v.dynamics(u_control)
                v.eta, v.velocity = v.Attitude(v)

            # Synchronize Simulation Time with Communication Network
            self._simSync()
            
            # Monitor vehicle contact
            self.monitorContact()

        # Stop Communication Network
        for v in self.vehicles:
            v.CommNetwork.stop()

        # Join threads and remove references
        for v in self.vehicles:
            v.monitorThread.join()
            v.monitorThread = None

    #--------------------------------------------------------------------------
    def _simSync(self)->None:
        """
        Synchronize simulation time with AquaNet communication network.
    
        AquaNet operates in separate threads outside the simulation's discrete
        time framework. While each simulation iteration represents exactly
        sampleTime seconds (typically 0.02s), AquaNet message transmission and
        processing occurs in real-time threads with variable delays.
        
        This creates a temporal mismatch: simulation time advances in fixed
        increments while the AquaNet network operations are processed in an
        independent time frame.
        
        This function pauses each simulation iteration for a duration equal to
        the sampleTime if AquaNet messages are in transit, preventing simulation
        time from advancing ahead of network communication time. This
        synchronizes the simulation duration with the actual time taken for
        AquaNet operations.
        

        Notes
        -----
        - Monitors TDMA Leader-Follower communication schedule states
        - In BCRQ Frame: waits for all followers to acknowledge broadcast  
        - In RSPN Frame: waits for scheduled follower to transmit report
        - Sleeps for sampleTime duration when messages are active
        - Logs total synchronization time when communication completes
        """
        
        simSync = False
        t = self.vehicles[0].clock
        h = self.vehicles[0].sampleTime

        # Check message conditions under TDMA Leader-Follower schedule:
        leaders = [v for v in self.vehicles if v.isLeader]

        for lead in leaders:
            # In BCRQ Frame
            if (t < lead.rspnFrame):
                # Check for 'ACK' on BCRQ
                idList = [fol.id for fol in lead.group]
                rspnRqst = [v.rspnRqst for v in self.vehicles 
                            if v.id in idList]
                simSync = (not all(rspnRqst))
                    
            # In RSPN Frame
            else:
                slotIdx = int((t - lead.rspnFrame) / lead.rspnDur)
                nFol = len(lead.rspnSched)
                simSync = ((slotIdx < nFol) and
                           (not lead.group[slotIdx].rprtRecv))

        if (simSync):
            self._syncTime += h
            time.sleep(h)
        elif (self._syncTime):
            self.log.info(f'Sim sync: {self._syncTime:.2f}s')
            self._syncTime = 0.0

    #--------------------------------------------------------------------------
    def _addToGroupDict(self, 
                        vehicle:veh.Vehicle, 
                        groupDict:Dict[str,veh.Vehicle],
                        )->None:
        """
        Add vehicle to group dictionary using groupId as key.
        

        Parameters
        ----------
        vehicle : Vehicle
            Vehicle to add. Must have groupId.
        groupDict : dict
            Dictionary to add vehicle to.
        """

        if (vehicle.groupId in groupDict):
            groupDict[vehicle.groupId].append(vehicle)
        else:
            groupDict[vehicle.groupId] = [vehicle]
    
    #--------------------------------------------------------------------------
    def _buildGroupDicts(self)->Tuple[Dict, Dict, List]:
        """
        Build dictionaries organizing vehicles by group ID.
        

        Returns
        -------
        groupsDict : dict
            Non-leader vehicles by groupId.
        leadersDict : dict
            Leader vehicles by groupId.
        noneGroup : list
            Vehicles with no groupId or target.
        """
        
        # Prepare dictionaries
        groupsDict = {}
        leadersDict = {}
        
        # Define vehicles to groups by groupId
        for v in self.vehicles:
            self._addToGroupDict(v, groupsDict)
            if (v.isLeader):
                self._addToGroupDict(v, leadersDict)

        # Determine if any vehicles have no groupId and no target specified
        if (None in groupsDict):
            noneGroup = groupsDict[None]
            del groupsDict[None]
            for v in noneGroup:
                if (v.target is not None):
                    v.groupId = v.target.groupId
                    self._addToGroupDict(v, groupsDict)
                    noneGroup.remove(v)
        else:
            noneGroup = []

        return groupsDict, leadersDict, noneGroup
    
    #--------------------------------------------------------------------------
    def _makeSaveDir(self, dirName:str)->str:
        """
        Create and return output directory path for simulation files.
        

        Parameters
        ----------
        dirName : str
            Directory name for this simulation.
            

        Returns
        -------
        outDir : str
            Full path to created output directory.
            

        Notes
        -----
        - Creates directory structure: outputs/<script_name>/<dirName>/
        - Automatically detects calling script name.
        """

        # Get the project directory
        modulePath = inspect.getfile(importlib.import_module('munetauvsim'))
        projDir = os.path.dirname(os.path.dirname(modulePath))
        
        # Get the user script name
        frame = inspect.currentframe()
        while frame.f_back:
            frame = frame.f_back
        if ('__file__' in frame.f_globals):
            scriptPath = os.path.abspath(frame.f_globals['__file__'])
            scriptName = os.path.splitext(os.path.basename(scriptPath))[0]
        else:
            scriptName = 'REPL'

        # Create the user script output directory if it doesn't exist
        scriptOutDir = os.path.join(projDir, 'outputs', scriptName)
        if not os.path.exists(scriptOutDir):
            os.makedirs(scriptOutDir)

        # Create a unique subdirectory within the user script output directory
        outDir = os.path.join(scriptOutDir, dirName)
        if not os.path.exists(outDir):
            os.makedirs(outDir)

        return outDir

    #--------------------------------------------------------------------------
    def _validFileName(self, fileName:str, extension:str)->str:
        """
        Validate filename and prepend output directory if needed.
        

        Parameters
        ----------
        fileName : str
            Filename to validate.
        extension : str
            Required file extension (e.g., '.log', '.gif').
            

        Returns
        -------
        validName : str
            Validated filename with correct extension and directory.
        """

        root, ext = os.path.splitext(fileName)
        if (ext != extension):
            fileName = f"{root}{extension}"
        if not (os.path.dirname(fileName)):
            fileName = os.path.join(self.outDir, fileName)
        return fileName

###############################################################################

def save(simulation:Simulator,
         filename:Optional[str] = None,
         format:str = 'pickle',
         )->None:
    """
    Save Simulator object to file.
    

    Parameters
    ----------
    simulation : Simulator
        Simulator object to save.
    filename : str, optional
        Output filename (default: simulation.saveFile).
    format : {'pickle', 'json'}
        Save format (default: 'pickle'). JSON not yet implemented.
        

    Notes
    -----
    Removes AquaNet socket references before pickling (sockets not
    serializable). Saves to simulation.outDir if filename has no directory.
    """

    # Map formats to file extensions
    formatExts = {
        'pickle': ['pickle', 'pkl'],
        'json': ['json']
    }

    # Determine filename and path
    if (filename is None):
        filename = simulation.saveFile
    elif not (os.path.dirname(filename)):
        filename = os.path.join(simulation.outDir, filename)

    # Check filename for extension and remove if one of the save formats
    root, ext = os.path.splitext(filename)
    if any(ext[1:].lower() in values for values in formatExts.values()):
        filename = root
    baseName = os.path.basename(filename)
    
    # Save to specified format
    fmt = format.lower()
    # Pickle
    if (fmt in formatExts['pickle']):
        # Clear AquaNet Manager references
        """The send_socket and recv_sockets are socket.socket types, which
        cannot be pickled. If preservation of the AquaNet Manager is needed,
        then custom __getstate__ and __setstate__ methods can be written for
        the sockets to allow pickling."""
        if (simulation.comnet == 'AquaNet'):
            simulation.log.info('Removing vehicle AquaNet Manager references.')
            for v in simulation.vehicles:
                # Use Duck Typing to avoid importing aquanet_lib
                if (hasattr(v.CommNetwork, "initAquaNet")):
                    v.CommNetwork = None
        with open(f"{filename}.pickle", "wb") as f:
            pickle.dump(simulation, f, pickle.HIGHEST_PROTOCOL)
        simulation.log.info(f"Saved Simulator object as: '{baseName}.pickle'.")
    # JSON
    elif (fmt in formatExts['json']):
        simulation.log.info("TODO: write JSON save code")
        return
    # Unknown Format
    else:
        simulation.log.info(f"simulator.save(): Unknown format: '{format}'.")
        simulation.log.info(f"Saving as 'pickle' format.")
        save(simulation, filename=filename, format='pickle')

###############################################################################

def load(filename:str,
         format:Optional[str] = None,
         )->Simulator:
    """
    Load Simulator object from file.
    

    Parameters
    ----------
    filename : str
        Path to saved simulator file.
    format : str, optional
        File format. Auto-detected from extension if None.
        

    Returns
    -------
    simulation : Simulator
        Loaded Simulator object, or None if load fails.
        
        
    Notes
    -----
    Supports pickle format. JSON loading not yet implemented.
    """
    
    # Map formats to file extensions
    formatExts = {
        'pickle': ['pickle', 'pkl'],
        'json': ['json']
    }

    # Determine data file type from file name extension
    if (format is None):
        root, ext = os.path.splitext(filename)
        baseName = os.path.basename(filename)
        # No format and no file extension
        if not (ext):
            print(f"simulator.load(): No format specified for '{baseName}'.")
            return None
        # Specify format from file extension
        ext = ext[1:].lower()
        for formatType, extension in formatExts.items():
            if (ext in extension):
                format = formatType
                break
        # No format matches given extension
        else:
            print(f"simulator.load(): Unknown format: '{ext}'.")
            return None

    # Pickle Format
    fmt = format.lower()
    if (fmt in formatExts['pickle']):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    # JSON Format
    if (fmt in formatExts['json']):
        print("TODO: write JSON load code")
        return None
    
    # Unknown Format
    print(f"simulator.load(): Unknown format: '{format}'.")
    return None

###############################################################################