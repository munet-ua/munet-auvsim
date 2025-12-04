"""
demo.py - Interactive Demo and Tutorial for muNet-AUVsim

This script is a demonstration and tutorial of the muNet-AUVsim capabilities. It
guides new users through scenarios with informational content that accompanies
the project documentation.
"""

import numpy as np
import munetauvsim as mn
from textwrap import dedent

# ============================================================================
# MAIN CONTENT & DEMO SCENARIOS
# ============================================================================

def mainMenu() -> str:
    """Display main menu and get user selection"""
    print("\n" + "="*74)
    print("MAIN MENU - Select a demonstration scenario:")
    print("="*74)

    options = {
        '1': 'Basic Path Following (ALOS Guidance)',
        '2': 'Multi-Vehicle Swarm with APF Target Tracking',
        'q': 'Exit'
    }

    for key, desc in options.items():
        print(f"  ({key}) {desc}")
    
    print("\n" + "─"*74)
    print("  COMING SOON:")
    print("    * Communication Network Customization")
    print("    * Ocean Environment Customization")
    print("    * Sensor Configuration")
    print("    * Custom Scenario Builder")
    print("="*74)

    while True:
        choice = input("\nEnter your choice (1-2 or q): ").strip().lower()
        if choice in options:
            return choice
        print("Invalid choice. Please select 1-2 or q.")

##############################################################################

def demoPathFollowing() -> None:
    """Scenario 1: Basic path following demonstration"""
    printd("""
    
    ╔════════════════════════════════════════════════════════════════════════╗
    ║ SCENARIO 1: BASIC PATH FOLLOWING                                       ║
    ╚════════════════════════════════════════════════════════════════════════╝

    This scenario demonstrates a single AUV following a predefined path using
    the Adaptive Line of Sight (ALOS) guidance law.
    """)

    print("="*74)
    print("SIMULATION CONFIGURATION")
    print("="*74)
    print("Input new values or press Enter to accept the default value.")

    simParams = {}
    name = "Basic_Path_Follow"

    # -------------------------------------------------------------------------
    # WAYPOINT PATH SELECTION
    # -------------------------------------------------------------------------
    printd("""
    WAYPOINT PATH SELECTION
    -----------------------
    Path following is a fundamental autonomous guidance problem where a
    vehicle must navigate a path defined by a sequence of waypoints. The only
    constraints are the positions where the vehicle needs to be, allowing the
    guidance system to determine appropriate trajectories.
           
    Our ALOS guidance accepts waypoint success within 10 meters (R_switch) in
    the xy-plane. Vertical distance to the waypoint is not enforced.
           
    Path Options:
      1) Square pattern    - Simple square path for basic testing
      2) Figure-8 pattern  - S-curve path for smooth transitions
      3) Spiral pattern    - Descending spiral for survey missions
      4) Grid survey       - Lawn-mower pattern for area coverage
      5) Random waypoints  - Randomly generated path
      6) Custom waypoints  - Enter your own coordinates
    """)

    pathChoice = input("Select path type (1-6) [1]: ").strip() or "1"

    # Create path
    if pathChoice == "1":
        # Square pattern
        path = createSquarePath(sideLength=100)
        pathStr = "Square"
        print("> Created square path pattern")

    elif pathChoice == "2":
        # Figure-8 pattern
        path = createFigure8Path(nVehicles=1)
        pathStr = "Figure8"
        print("> Created figure-8 path pattern")

    elif pathChoice == "3":
        # Spiral pattern
        path = createSpiralPath(nVehicles=1)
        pathStr = "Spiral"
        print("> Created spiral path pattern")
    
    elif pathChoice == "4":
        # Grid survey pattern
        path = createGridSurveyPath(nVehicles=1)
        pathStr = "Grid"
        print("> Created grid survey path pattern")

    elif pathChoice == "5":
        # Random path
        nWaypoints = int(input("Number of waypoints [5]: ").strip() or 5)
        path = mn.guidance.generateRandomPath(nWaypoints, rLims=[50, 100], zLims=[5, 20])
        pathStr = f"Random_{nWaypoints}"
        print(f"> Generated random path with {nWaypoints} waypoints")

    else:
        # Custom waypoints
        path = createCustomPath()
        pathStr = f"Custom_{len(path)}"
        print(f"> Created custom path with {len(path)} waypoints")

    # -------------------------------------------------------------------------
    # OCEAN ENVIRONMENT SELECTION
    # -------------------------------------------------------------------------
    printd("""
    OCEAN ENVIRONMENT
    -----------------
    The simulation includes environmental effects. The Ocean environment 
    affects vehicle performance through currents and floor topology. Vehicle 
    dynamics are affected by currents, and vehicles are inoperable outside of 
    the upper and lower boundaries.
    
      * Ocean Currents: 
        Realistic 1D current models are uniform across the ocean space, not
        varying with position or depth, but do vary in time. The magnitude and
        direction of the current vary stochastically around given mean values
        and mean deviations, at given oscillation periods.
    
      * Ocean Floor:
        Depth maps are procedurally generated with Perlin noise to produce
        seabed terrain that can be stretched to given min and max depths.
          
    The Ocean is fully customizable, but also offers 3 simple pre-made 
    environments:
          
      1. Calm Ocean:   Minimal currents, steady conditions
      2. Stormy Ocean: Strong varying currents, challenging conditions
      3. Dead Ocean:   Zero current, deep floor
    """)

    oceanType = input("Select ocean type - (c)alm, (s)tormy, or (d)ead [c]: ").strip().lower()
    size, origin = path.estimateAreaRoot(padding=100)

    if oceanType == 's':
        ocean = mn.environment.Ocean.stormy_ocean(size=size, origin=origin, randomFloor=True)
        oceanStr = "Stormy"
        print("> Created stormy ocean environment")
    elif oceanType == 'd':
        ocean = mn.environment.Ocean.dead_ocean(size=size, origin=origin, randomFloor=True) 
        oceanStr = "Dead"
        print("> Created dead calm ocean environment")
    else:
        ocean = mn.environment.Ocean.calm_ocean(size=size, origin=origin, randomFloor=True)
        oceanStr = "Calm"
        print("> Created calm ocean environment")

    simParams['ocean'] = ocean

    # -------------------------------------------------------------------------
    # VEHICLE CONFIGURATION
    # -------------------------------------------------------------------------
    printd("""
    VEHICLE CONFIGURATION
    ---------------------
           
    VEHICLE DYNAMICS: Remus 100 AUV
    
    This vehicle model simulates realistic six degrees-of-freedom dynamics:
    
    Degrees of Freedom:
      1. Surge (u)    : forward/backward motion
      2. Sway  (v)    : port/starboard motion  
      3. Heave (w)    : up/down motion
      4. Roll  (p)    : rotation about longitudinal axis
      5. Pitch (q)    : rotation about lateral axis
      6. Yaw   (r)    : rotation about vertical axis
      
    Physical Effects Modeled:
      * Rigid body inertia (mass and moments of inertia)
      * Added mass (hydrodynamic mass of surrounding water)
      * Coriolis and centripetal forces
      * Linear and quadratic damping (drag)
      * Control surface lift and drag (rudder, stern planes)
      * Propeller thrust and torque
      * Environmental coupling (ocean currents, depth limits)
           
    
    CONTROL ARCHITECTURE
    
    The Remus 100 is controlled by a tail rudder, stern planes, and a 
    propeller. The rudder and stern planes are controlled with PID 
    controllers, while the propeller controlled by direct RPM commands.
    
      * Heading Control: 
        PID controller generates yaw moment by commanding rudder deflection
        until desired heading is achieved.
    
      * Pitch Control: 
        PID controller generates pitch moment by commanding stern plane
        deflection until desired pitch angle is achieved.
          
      * Depth Control: 
        PI controller in cascade configuration with pitch PID as the inner 
        loop. Generates desired pitch angle until desired depth is achieved.
    
      * Propeller Control:
        The Remus 100 AUV has maximum speed of 2.5 m/s at 1525 RPM. A 
        simplified linear conversion from RPM to speed (U) is used to 
        transform between the two based on empirical study of the vehicle
        dynamics.
    """)
    
    auvSpeed = float(input("Select vehicle speed (max 2.5 m/s) [2.0]: ").strip() or 2.0)
    auvSpeed = max(0.1, min(2.5, auvSpeed))
    print(f"> {auvSpeed:.1f}")
    
    # Create vehicle
    printd("""
    VEHICLE CALL SIGN
    
    Vehicles are assigned immutable ID numbers by a class counter, ensuring a
    unique identification handle for each vehicle generated during a session.
    Each vehicle also receives an automatic call sign used in logs and
    communication (e.g., "01" or "B05-LEADER"), but these are informal and 
    can be customized with your own creative names if desired.
    """)

    auv = mn.vehicles.Remus100s()
    callSign = input(f"Vehicle call sign [{auv.callSign}]: ").strip().upper() or f"{auv.callSign}"
    callSign = callSign.replace(' ', '_')
    auv.callSign = callSign
    auvRPM = round(auv.xferU2N(auvSpeed), -1)
    rpmStr = f"{int(auvRPM)}"
    auv.wpt = path
    auv.loadPathFollowing()
    auv.loadConstantProp(n_setpt=auvRPM)
    print(f"> Created vehicle \"{auv.callSign}\" following path at {rpmStr} rpm")
    
    simParams['vehicles'] = [auv]

    # ------------------------------------------------------------------------
    # SIMULATOR CONFIGURATION
    # ------------------------------------------------------------------------
    printd("""
    SIMULATOR CONFIGURATION
    -----------------------
           
    SIMULATION RUN TIME
    
    Each simulation runs a specified number of iterations (N), with each 
    iteration representing a discrete time-step (sampleTime). The total 
    simulated run time is the sum of all time steps.
           
    The estimated run time is calculated from the speed of the vehicle along
    the distance of the vehicle's path. Variations in guidance and ocean
    currents make this imprecise.
    """)

    estimatedRunTime = round(int(path.estimateTime(speed=auvSpeed, coeff=1.0)), -1)
    runTime = int(input(f"Simulation run time in seconds [estimated {estimatedRunTime}]: ").strip() or estimatedRunTime)
    runTime = max(10, min(runTime, 36000))
    print(f"> {runTime}")

    printd("""
    SIMULATION NAME
           
    Simulations can be assigned unique names. These appear on the output files
    generated for each run (with time stamps for uniqueness). We recommend
    using a descriptive name that remains concise but informative, and 
    captures key parameters of the simulation scenario.
    """)

    generatedName = '_'.join([name, pathStr, oceanStr, rpmStr])
    name = input(f"Simulation name [{generatedName}]: ").strip()
    name = name.replace(' ', '_') if name else generatedName
    print(f"> {name}")

    simParams['runTime'] = runTime
    simParams['name'] = name

    # ------------------------------------------------------------------------
    # SUMMARY & CONFIRMATION
    # ------------------------------------------------------------------------
    print("\n" + "-"*74)
    print("SELECTED SIMULATION PARAMETERS")
    print("-"*74)
    print(f"  Name:              {name}")
    print(f"  Vehicles:          1 ({auv.callSign})")
    print(f"  Vehicle speed:     {auvSpeed:.1f} m/s ({rpmStr} rpm)")
    print(f"  Path type:         {pathStr} ({len(path)} waypoints)")
    print(f"  Path details:")
    print(path)
    print(f"  Ocean:             {oceanStr}")
    print(f"  Run time:          {runTime} s")
    print("-"*74)

    ocean.floor.display2D(path=path)
    
    printSimRunEndInfo()

    confirm = input("\nRun this scenario? (y/n) [y]: ").strip().lower()
    if confirm != 'n':
        print("> Running simulation scenario...\n")

        # Create simulation
        sim = mn.simulator.Simulator(**simParams)

        # Deploy vehicle
        sim.deployAtWpt(auv)

        # Run simulation
        sim.run()
        printSimSummary(sim)

        # Keep data?
        saveData = input("Save simulation data file (pickle)? (y/n) [n]: ").strip().lower()
        if saveData == 'y':
            print("> Saving simulation data.")
            mn.simulator.save(sim)
    else:
        print("> Scenario cancelled.")

##############################################################################

def demoSwarmTargetTracking() -> None:
    """Scenario 2: Multi-vehicle swarm with target tracking"""
    printd("""

    ╔════════════════════════════════════════════════════════════════════════╗
    ║ SCENARIO 2: MULTI-VEHICLE SWARM TARGET TRACKING                        ║
    ╚════════════════════════════════════════════════════════════════════════╝

    This scenario demonstrates multiple AUVs using Artificial Potential Field
    (APF) algorithms to follow a leader while avoiding collisions.
    """)

    print("="*74)
    print("SIMULATION CONFIGURATION")
    print("="*74)
    print("Input new values or press Enter to accept the default value.")

    name = "Swarm_Target_Track"

    # ------------------------------------------------------------------------
    # APF CONFIGURATION
    # ------------------------------------------------------------------------
    printd("""
    ARTIFICIAL POTENTIAL FIELD (APF) CONFIGURATION
    ----------------------------------------------
    As a guidance system, APF models vehicles as charged particles with
    attraction and repulsion forces. The combined result generates desired
    velocity commands and produces emergent formation behavior.
    """)

    printd("""\
    ATTRACTION FUNCTION
           
      * Followers attracted to leader position
      * Creates velocity field pointing toward target
      * Attraction increases with distance
  
      1) Linear               - Simple linear function
      2) Linear w/ CB zone    - Linear with central constant bearing zone
      3) Cubic                - Smooth cubic polynomial with flat inflection
    """)

    attChoice = input("Select attraction function (1-3) [2]: ").strip() or "2"
    attMap = {'1': 'linear', '2': 'linearCBZ', '3': 'cubic'}
    attStr = attMap.get(attChoice, 'linearCBZ')
    print(f"> {attStr}")

    printd("""
    REPULSION FUNCTION
           
      * Followers repelled away from all swarm members
      * Creates velociy fields pointing away from target and neighbors
      * Repulsion decreases with distance
      
      1) Exponential             - Simple exponential decay function
      2) Variable Exponential    - Scales with velocity along line-of-sight
    """)

    repChoice = input("Select repulsion function (1-2) [2]: ").strip() or "2"
    repMap = {'1': 'exp', '2': 'varExp'}
    repStr = repMap.get(repChoice, 'varExp')
    print(f"> {repStr}")

    # ------------------------------------------------------------------------
    # FORMATION PARAMETERS
    # ------------------------------------------------------------------------
    printd("""
    FORMATION CONTROL PARAMETERS
    ----------------------------
    Three distances influence APF swarm formation behavior:
    
    * Following Distance (r_follow): 
      - The preferred spacing away from the leader vehicle
      - APF should match target velocity at this distance
      - Larger values create looser formation with more stability, while
        smaller values create tighter formation with more responsiveness
      
    * Avoidance Distance (r_avoid): 
      - The distance at which repulsion becomes active
      - APF should be neutral at this distance
      - Larger values create stricter separation enforcement, while smaller 
        values allow closer approach before repulsion activates
         
    * Safety Distance (r_safe): 
      - The critical minimum separation limit
      - Inside this distance, maximum repulsion is experienced
           
    The goal is to balance these distance values within the selected APF
    functions to maintain stable group formation while minimizing oscillations
    and violent reactions.
    """)

    r_follow = float(input("Following distance [100m]: ").strip() or 100)
    r_avoid = float(input("Avoidance distance [40m]: ").strip() or 40)
    r_safe = float(input("Safety distance [10m]: ").strip() or 10)
    formStr = f"{r_follow:.0f}-{r_avoid:.0f}-{r_safe:.0f}"
    print(f"> {r_follow}, {r_avoid}, {r_safe}")

    # ------------------------------------------------------------------------
    # SWARM SIZE & GROUP
    # ------------------------------------------------------------------------
    printd("""
    SWARM CONFIGURATION
    -------------------
           
    LEADER-FOLLOWER
           
    This scenario uses a hierarchical leader-follower pattern:
           
      * 1 Leader Vehicle: Executes autonomous path-following guidance
      * N Follower Vehicles: Use target-tracking guidance to follow leader
    """)

    nFollowers = int(input("Number of follower vehicles (1-6) [3]: ").strip() or 3)
    nFollowers = max(1, min(nFollowers, 6))
    numStr = f"{nFollowers + 1}"
    print(f"> {nFollowers}")

    printd("""
    GROUP ID
    
    Each swarm group is tagged with a group ID. For simplicity and to minimize
    communication we use only a single capital letter.
    """)

    gid = input("Swarm group identifier (A-Z) [A]: ").strip().upper()
    if ((not gid) or (len(gid) != 1) or (gid < "A") or (gid > "Z")):
        gid = "A"
    print(f"> {gid}")

    # ------------------------------------------------------------------------
    # COMMUNICATION NETWORK
    # ------------------------------------------------------------------------
    printd("""
    COMMUNICATION NETWORK
    ---------------------
    Communication enables state sharing for swarm coordination. Various 
    message types exist but the critical information being shared is position
    and velocity. Three different modes are supported:
    
      * muNet:    Simulated acoustic network with realistic propagation,
                  packet loss, bit errors, and collision effects.
      
      * AquaNet:  Real AquaNet protocol stack (Linux only).
      
      * None:     Direct access (vehicles have perfect knowledge).
                  Use for baseline and experimentation.
    """)

    commChoice = input("Communication network - (m)uNet, (a)quaNet, or (n)one [n]: ").strip().lower()
    commMap = {'a': 'AquaNet', 'm': 'muNet', 'n': 'None'}
    commStr = commMap.get(commChoice, 'None')
    print(f"> {commStr}")

    # ------------------------------------------------------------------------
    # LEADER PATH
    # ------------------------------------------------------------------------
    printd("""
    LEADER PATH CONFIGURATION
    -------------------------
    The leader executes path following while followers use APF guidance.\
    """)

    path = mn.guidance.generateRandomPath(6, zLims=[10, 40])
    print("> Generated random leader path with 6 waypoints")

    # ------------------------------------------------------------------------
    # CREATE SWARM
    # ------------------------------------------------------------------------
    printd("""
    BUILDING SWARM VEHICLES
    -----------------------\
    """)

    # Create vehicles
    vehicles = mn.vehicles.buildGroup(
        num = nFollowers + 1,
        gid = gid,
        hasLeader = True,
        r_follow = r_follow,
        r_avoid = r_avoid,
        r_safe = r_safe,
    )
    leader = vehicles[0]
    leader.wpt = path
    leader.loadPathFollowing()
    leaderSpeed = 2.0
    leaderRPM = round(leader.xferU2N(leaderSpeed), -1)
    leader.loadConstantProp(n_setpt=leaderRPM)
    print(f"> Leader: \"{leader.callSign}\" path following at {leaderSpeed:.1f} m/s")

    for follower in vehicles[1:]:
        follower.loadTargetTracking(
            target = leader,
            att = attStr,
            rep = repStr,
        )
        print(f"> Follower: \"{follower.callSign}\" APF target tracking")

    # ------------------------------------------------------------------------
    # OCEAN ENVIRONMENT
    # ------------------------------------------------------------------------
    printd("""
    OCEAN ENVIRONMENT
    -----------------
    For this scenario a simple calm ocean environment is used. A deep ocean
    with slow current speeds, and steady variation of magnitude and direction
    at calm rates.\
    """)

    size, origin = path.estimateAreaRoot(padding=100)
    ocean = mn.environment.Ocean.calm_ocean(size=size, origin=origin, randomFloor=True)
    print("> Created calm ocean environment")

    # ------------------------------------------------------------------------
    # SIMULATOR CONFIGURATION
    # ------------------------------------------------------------------------
    printd("""
    SIMULATOR CONFIGURATION
    -----------------------
           
    SIMULATION RUN TIME
    
    Each simulation runs a specified number of iterations (N), with each 
    iteration representing a discrete time-step (sampleTime). The total 
    simulated run time is the sum of all time steps.
           
    The estimated run time is calculated from the speed of the leader vehicle
    along the distance of the leader's path. Variations in guidance and ocean
    currents make this imprecise.
    """)

    estimatedRunTime = round(int(path.estimateTime(speed=leaderSpeed, coeff=1.0)), -1)
    runTime = int(input(f"Simulation run time in seconds [estimated {estimatedRunTime}]: ").strip() or estimatedRunTime)
    runTime = max(10, min(runTime, 36000))
    print(f"> {runTime}")

    printd("""
    SIMULATION NAME
    
    Simulations can be assigned unique names. These appear on the output files
    generated for each run (with time stamps for uniqueness). We recommend
    using a descriptive name that remains concise but informative, and 
    captures key parameters of the simulation scenario.
    """)

    generatedName = '_'.join([name, numStr, attStr, repStr, formStr, commStr])
    name = input(f"Simulation name [{generatedName}]: ").strip()
    name = name.replace(' ', '_') if name else generatedName
    print(f"> {name}")

    # ------------------------------------------------------------------------
    # SUMMARY & CONFIRMATION
    # ------------------------------------------------------------------------
    print("\n" + "-"*74)
    print("SELECTED SIMULATION PARAMETERS")
    print("-"*74)
    print(f"  Name:                   {name}")
    print(f"  Number of vehicles:     {numStr}")
    print(f"  Leader:                 Path following, Random, {len(path)-1} waypoints")
    print(f"  Followers:              APF guidance")
    print(f"  APF attraction:         {attStr}")
    print(f"  APF repulsion:          {repStr}")
    print(f"  Following distance:     {r_follow} m")
    print(f"  Avoidance distance:     {r_avoid} m")
    print(f"  Safety distance:        {r_safe} m")
    print(f"  Communication Network:  {commStr}")
    print(f"  Ocean:                  Calm")
    print(f"  Run time:               {runTime} s")
    print("-"*74)

    printSimRunEndInfo()

    confirm = input("\nRun this scenario? (y/n) [y]: ").strip().lower()
    if confirm != 'n':
        print("> Running simulation scenario...\n")

        # Create simulator
        sim = mn.simulator.Simulator(
            name = name,
            runTime = runTime,
            vehicles = vehicles,
            ocean = ocean,
        )

        # Deploy vehicles
        sim.deployAtWpt(leader)
        sim.deploySwarmGrid()

        # Setup communication
        if commChoice == 'a':
            try:
                sim.loadAquaNet()
                print("> Loaded AquaNet communication.")
            except Exception as e:
                print(f"> AquaNet not available: {e}")
                print("> Using muNet instead...")
                commChoice = 'm'
        
        if commChoice == 'm':
            sim.loadMuNet()

        # Run simulation
        sim.run()
        printSimSummary(sim)

        # Keep data?
        saveData = input("Save simulation data file (pickle)? (y/n) [n]: ").strip().lower()
        if saveData == 'y':
            print("> Saving simulation data.")
            mn.simulator.save(sim)
    else:
        print("> Scenario cancelled.")

# ============================================================================
# HELPER FUNCTIONS - PRINTING
# ============================================================================

def printd(text: str) -> None:
    """Wrapper to print dedented text"""
    print(dedent(text))

##############################################################################
def printBanner() -> None:
    """Display the welcome banner"""
    printd("""
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                     muNet-AUVsim Interactive Demo                      ║
    ║                                                                        ║
    ║  Multi-Agent Autonomous Underwater Vehicle (AUV) Simulation Framework  ║
    ║         with GNC, Ocean, Communication, and Swarm Coordination         ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """)

##############################################################################

def printSimRunEndInfo() -> None:
    """Print information about what to expect during and after sim run"""
    printd("""
    During the simulation run, log information will be printed to the
    terminal. If you selected a communication network, the communication log
    is also shown. The logs include simulation parameters and relevant data 
    for later reproduction, reference, and analysis.
           
    When a simulation completes:
            
    * A 3D plot of vehicle paths will display. Click and drag on the plot to
        view the paths from different angles.
    
    * A folder is created in the outputs directory containing:
        - An animated GIF of the scenario
        - Information and communication logs
        - Simulation data file (optional)\
    """)

##############################################################################

def printSimSummary(sim: mn.simulator.Simulator) -> None:
    """Print summary of simulation results"""
    print("\n" + "="*74)
    print("SIMULATION RESULTS SUMMARY")
    print("="*74)

    print(f"  Simulation name:      {sim.name}")
    print(f"  Simulation run time:  {sim.runTime:.1f}s (simulated)")
    print(f"  Vehicles:             {len(sim.vehicles)}")
    print(f"  Sample rate:          {1/sim.sampleTime:.0f} Hz")
    print(f"  Data points:          {sim.N}")

    if hasattr(sim, 'contactCount'):
        print(f"  Vehicle contacts:     {sim.contactCount}")

    # Communication stats
    if (hasattr(sim, 'muNet') and (sim.muNet is not None)):
        networks = sim.muNet if isinstance(sim.muNet, list) else [sim.muNet]
        for net in networks:
            stats = net.stats
            print(f"\n  Communication Stats:")
            print(f"    Packets sent:      {stats.get('packetSent', 0)}")
            print(f"    Packets delivered: {stats.get('packetDelivered', 0)}")
            if stats.get('packetSent', 0) > 0:
                deliveryRate = (stats.get('packetDelivered', 0) / 
                              stats.get('packetSent', 1) * 100)
                print(f"    Delivery rate:     {deliveryRate:.1f}%")

    print("="*74)
    print("\n> 3D trajectory plot and animation generated.")
    print("> Animation and log saved in:")
    print(f"  {sim.outDir}")

    # Option to show additional analysis
    showDetail = input("\nShow additional details? (y/n) [n]: ").strip().lower()
    if showDetail == 'y':
        printSimDetailAnalysis(sim)

##############################################################################

def printSimDetailAnalysis(sim: mn.simulator.Simulator) -> None:
    """Show detailed simulation analysis"""
    print("\n" + "-"*74)
    print("DETAILED ANALYSIS")
    print("-"*74)

    for i, vehicle in enumerate(sim.vehicles):
        print(f"\n  Vehicle {i+1}: {vehicle.callSign}")

        # Extract position data
        positions = sim.simData[i, :, :3]  # x, y, z positions

        # Calculate distances traveled
        distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))
        totalDistance = np.sum(distances)

        print(f"    Total distance traveled: {totalDistance:.1f} m")
        print(f"    Average speed: {totalDistance/sim.runTime:.2f} m/s")

        # Final position
        finalPos = positions[-1]
        print(f"    Final position: ({finalPos[0]:.1f}, {finalPos[1]:.1f}, {finalPos[2]:.1f}) m")

        # Depth profile
        depths = positions[:, 2]  # Convert END to positive depth
        print(f"    Depth range: {depths.min():.1f} m to {depths.max():.1f} m")

    print("\n" + "-"*74 + "\n")

# ============================================================================
# HELPER FUNCTIONS - PATHS
# ============================================================================

def createSquarePath(sideLength: float = 100) -> mn.guidance.Waypoint:
    """Create a square path"""
    half = sideLength / 2
    depth = 20

    x = [-half, half, half, -half, -half]
    y = [-half, -half, half, half, -half] 
    z = [depth] * len(x)

    return mn.guidance.Waypoint(x, y, z)

##############################################################################

def createFigure8Path(nVehicles: int = 1) -> mn.guidance.Waypoint:
    """Create a smooth figure-8 shaped path"""
    scale = 1.0 if nVehicles == 1 else 3.0
    size = 300 * scale
    depth = 20
    numPts = 16

    t = np.linspace(0, 2*np.pi, numPts)
    x = (size/3) * np.sin(t)
    y = (size/6) * np.sin(2*t)
    z = [depth] * len(x)

    return mn.guidance.Waypoint(x.tolist(), y.tolist(), z)

##############################################################################

def createSpiralPath(nVehicles: int = 1) -> mn.guidance.Waypoint:
    """Create a spiral survey pattern"""
    x, y, z = [], [], []
    scale = 1.0 if nVehicles == 1 else 2.5

    # Scale pattern
    nLoops = 5                      # Number of complete spirals
    pointsPerLoop = 8               # Waypoints per 360 deg rotation
    radiusStart = 50 * scale        # Initial radius (m)
    radiusIncrement = 25 * scale    # Radius increase per loop (m)
    depthStart = 20                 # Starting depth (m)
    depthIncrement = 5 * scale      # Depth change per loop (m)
    
    nTotalPoints = nLoops * pointsPerLoop

    for i in range(nTotalPoints):
        # Angle increases by 2pi/pointsPerLoop per waypoint
        angle = (i / pointsPerLoop) * 2 * np.pi
        
        # Radius increases linearly with waypoint number
        radius = radiusStart + (i / pointsPerLoop) * radiusIncrement
        
        # Calculate position
        x.append(radius * np.cos(angle))
        y.append(radius * np.sin(angle))

        # Descend gradually
        z.append(depthStart + (i / pointsPerLoop) * depthIncrement)

    wpt = mn.guidance.Waypoint(x, y, z)
    
    return wpt

##############################################################################

def createGridSurveyPath(nVehicles: int = 1) -> mn.guidance.Waypoint:
    """Create a grid survey pattern"""
    x, y, z = [], [], []
    scale = 1.0 if nVehicles == 1 else 3.0
    size = 80
    depth = 20
    
    # Create lawn-mower pattern
    for row in range(5):
        yPos = (row * size/2 - size) * scale
        if row % 2 == 0:
            # Left to right
            xRow = [x * scale for x in range(-size, size+1, size//2)]
        else:
            # Right to left
            xRow = [x * scale for x in range(size, -(size+1), -size//2)]

        x.extend(xRow)
        y.extend([yPos] * len(xRow))
        z.extend([depth] * len(xRow))

    return mn.guidance.Waypoint(x, y, z)

##############################################################################

def createCustomPath() -> mn.guidance.Waypoint:
    """Create a custom path from user input waypoints"""
    printd("""
    Enter custom waypoints x, y, z in meters.
      x: East direction
      y: North direction
      z: Down direction (depth)
    Enter coordinates separated by commas (no parenthesis).
    Press Enter when done.
    """)

    x, y, z = [], [], []
    while True:
        try:
            point = input(f"Waypoint {len(x)+1} - x,y,z or Enter to finish: ").strip()
            if not point:
                break
            coords = [float(c) for c in point.split(',')]
            if len(coords) != 3:
                print("Please enter exactly 3 coordinates x,y,z")
                continue
            x.append(coords[0])
            y.append(coords[1]) 
            z.append(coords[2])
        except ValueError:
            print("Invalid input. Please enter numbers only, separated by commas.")
            print("For example: 100, 120, 30")

    if len(x) < 2:
        print("Need at least 2 waypoints. Using default square path.")
        return createSquarePath()

    return mn.guidance.Waypoint(x, y, z)

# ============================================================================
# MAIN LOOP
# ============================================================================

def main() -> None:
    """Main program loop"""
    printBanner()

    while True:
        choice = mainMenu()

        if choice == 'q':
            print("\nThank you for trying muNet-AUVsim Interactive Demo!")
            print("Visit the project documentation for more information:\n")
            print("Documentation:  https://munet-ua.github.io/munet-auvsim/")
            print("Repository:     https://github.com/munet-ua/munet-auvsim")
            print()
            break

        try:
            if choice == '1':
                demoPathFollowing()
            elif choice == '2':
                demoSwarmTargetTracking()

            input("\nPress Enter to return to main menu...")

        except KeyboardInterrupt:
            print("\n\nOperation cancelled by user.")
            input("Press Enter to return to main menu...")

        except Exception as e:
            print(f"\n\nError occurred: {e}")
            print("This may be due to missing dependencies or configuration issues.")
            input("Press Enter to return to main menu...")

if __name__ == "__main__":
    main()