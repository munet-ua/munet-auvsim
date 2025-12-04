"""
Visualization functions for AUV simulation data.

Provides plotting functions for time-series analysis and 3D animated
visualization of vehicle trajectories, states, controls, and ocean environment.
Based on Fossen's Python Vehicle Simulator plotting utilities.


Functions
---------
plotVehicleStates(simTime, simData, vehId, figNo)
    Plot 6-DOF vehicle states (position, attitude, velocities) vs time.
plotControls(simTime, simData, vehicle, figNo)
    Plot vehicle control inputs (commanded and actual) vs time.
plot3D(simData, sampleTime, numDataPoints, FPS, filename, vehicles, ocean, ...)
    Create 3D animated visualization and save as GIF.

    
Utility Functions
-----------------
R2D(value)
    Convert radians to degrees.
cm2inch(value)
    Convert centimeters to inches for figure sizing.

    
Notes
-----
Default plot parameters (figure size, DPI, legend size) are defined as
module-level globals and can be modified before calling plot functions.


References
----------
[1] Fossen, T.I. Python Vehicle Simulator. GitHub repository.
https://github.com/cybergalactic/PythonVehicleSimulator
"""

from typing import List, Optional, Tuple
from numpy.typing import NDArray
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.artist import Artist
from matplotlib.offsetbox import AnchoredText
import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import math
from munetauvsim.environment import Ocean
from munetauvsim.vehicles import Vehicle
from munetauvsim.gnc import ssa
from munetauvsim import logger

#-----------------------------------------------------------------------------#

# Type Aliases
NPFltArr = NDArray[np.float64]

# Global Variables
log = logger.addLog('pltTS')

# Plot Parameters
legendSize = 10         # legend size
figSize1 = [25, 13]     # figure1 size in cm
figSize2 = [25, 13]     # figure2 size in cm
dpiValue = 150          # figure dpi value

###############################################################################

def R2D(value:float)->float:
    """
    Convert radians to degrees.
    

    Parameters
    ----------
    value : float
        Angle in radians.
        

    Returns
    -------
    degrees : float
        Angle in degrees.
    """

    return value * 180 / math.pi

###############################################################################

def cm2inch(value:float)->float:
    """
    Convert centimeters to inches for matplotlib figure sizing.
    

    Parameters
    ----------
    value : float
        Length in centimeters.
        

    Returns
    -------
    inches : float
        Length in inches.
    """

    return value / 2.54

###############################################################################

def plotVehicleStates(simTime:NPFltArr,
                      simData:NPFltArr,
                      vehId:int,
                      figNo:int)->None:
    """
    Plot 6-DOF vehicle states (position, attitude, velocities) versus time.
    
    Creates a multi-subplot figure showing comprehensive vehicle state evolution
    including position, attitude angles, velocities, and derived quantities like
    course angle and crab angle.
    
    
    Parameters
    ----------
    simTime : ndarray, shape (N+1,)
        Time vector in seconds. Includes t=0 initial condition.
    simData : ndarray, shape (N+1, 18)
        Simulation data for single vehicle: 
        [eta(6), nu(6), u_control(3), u_actual(3)].
    vehId : int
        Vehicle identification number for plot title.
    figNo : int
        Figure number for plot window.

          
    Notes
    -----
    - Creates 9 subplots:

      1. Position (path) in North-East plane (x-y plane)
      2. Depth (z) vs time
      3. Roll (phi) and pitch (theta) angles vs time
      4. Speed vs time
      5. Course angle (chi) vs time
      6. Pitch angle (theta) and flight path angle (alpha_c) vs time
      7. Body-frame velocities (surge (u), sway (v), heave (w)) vs time
      8. Angular rates (roll (p), pitch (q), yaw (r)) vs time
      9. Yaw angle (psi) and crab angle (beta_c) vs time

    - All angles displayed in degrees. 
    - Uses smallest signed angle (ssa) wrapping.
    """

    # Time vector
    t = simTime

    # State vectors
    x = simData[:, 0]
    y = simData[:, 1]
    z = simData[:, 2]
    phi = R2D(ssa(simData[:, 3]))
    theta = R2D(ssa(simData[:, 4]))
    psi = R2D(ssa(simData[:, 5]))
    u = simData[:, 6]
    v = simData[:, 7]
    w = simData[:, 8]
    p = R2D(simData[:, 9])
    q = R2D(simData[:, 10])
    r = R2D(simData[:, 11])

    # Speed
    U = np.sqrt(np.multiply(u, u) + np.multiply(v, v) + np.multiply(w, w))

    # crab angle, beta_c
    beta_c  = R2D(ssa(np.arctan2(v, u)))
    # flight path angle
    alpha_c = R2D(ssa(np.arctan2(w, u)))
    # course angle, chi=psi+beta_c
    chi = R2D(ssa(simData[:, 5] + np.arctan2(v, u)))

    # Plots
    plt.figure(figNo, 
               figsize=(cm2inch(figSize1[0]), cm2inch(figSize1[1])),
               dpi=dpiValue)
    plt.title("Vehicle states", fontsize=12)
    plt.suptitle(f'Vehicle {vehId:02}')
    plt.grid()

    plt.subplot(3, 3, 1)
    plt.plot(x, y)
    plt.legend(["North-East positions (m)"], fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 2)
    plt.plot(t, z)
    plt.gca().invert_yaxis()
    plt.legend(["Depth (m)"], fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 3)
    plt.plot(t, phi, t, theta)
    plt.legend(["Roll angle (deg)", "Pitch angle (deg)"], fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 4)
    plt.plot(t, U)
    plt.legend(["Speed (m/s)"], fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 5)
    plt.plot(t, chi)
    plt.legend(["Course angle (deg)"], fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 6)
    plt.plot(t, theta, t, alpha_c)
    plt.legend(["Pitch angle (deg)",
                "Flight path angle (deg)"], 
                fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 7)
    plt.plot(t, u, t, v, t, w)
    plt.xlabel("Time (s)", fontsize=12)
    plt.legend(["Surge velocity (m/s)", 
                "Sway velocity (m/s)", 
                "Heave velocity (m/s)"],
                fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 8)
    plt.plot(t, p, t, q, t, r)
    plt.xlabel("Time (s)", fontsize=12)
    plt.legend(["Roll rate (deg/s)", "Pitch rate (deg/s)", "Yaw rate (deg/s)"],
               fontsize=legendSize)
    plt.grid()

    plt.subplot(3, 3, 9)
    plt.plot(t, psi, t, beta_c)
    plt.xlabel("Time (s)", fontsize=12)
    plt.legend(["Yaw angle (deg)", "Crab angle (deg)"], fontsize=legendSize)
    plt.grid()

###############################################################################

def plotControls(simTime:NPFltArr,
                 simData:NPFltArr,
                 vehicle:Vehicle,
                 figNo:int):
    """
    Plot vehicle control inputs (commanded and actual) versus time.
    
    Creates subplots comparing commanded control inputs (u_control) with actual
    control responses (u_actual) after actuator dynamics and saturation.
    

    Parameters
    ----------
    simTime : ndarray, shape (N+1,)
        Time vector in seconds.
    simData : ndarray, shape (N+1, 18)
        Simulation data: [eta(6), nu(6), u_control(3), u_actual(3)].
    vehicle : Vehicle
        Vehicle object. Must have attributes: id, dimU, controls.
    figNo : int
        Matplotlib figure number.
        

    Notes
    -----
    - Number of subplots = vehicle.dimU (number of control inputs).
      For Remus100s: 3 subplots (rudder, stern plane, propeller).
    - Angles automatically converted to degrees based on control description
      strings.
    - Each subplot shows commanded (blue) vs actual (orange) control trajectory.
    """

    DOF = 6

    # Time vector
    t = simTime

    plt.figure(figNo,
               figsize=(cm2inch(figSize2[0]),cm2inch(figSize2[1])),
               dpi=dpiValue)
    plt.suptitle(f'Vehicle {vehicle.id:02}')

    # Columns and rows needed to plot vehicle.dimU control inputs
    col = 2
    row = int(math.ceil(vehicle.dimU / col))

    # Plot the vehicle.dimU active control inputs
    for i in range(0, vehicle.dimU):

        u_control = simData[:, 2 * DOF + i]           # control input, commands
        u_actual = simData[:, 2 * DOF+vehicle.dimU+i] # actual control input

        # convert angles to deg
        if vehicle.controls[i].find("deg") != -1:
            u_control = R2D(u_control)
            u_actual = R2D(u_actual)

        plt.subplot(row, col, i + 1)
        plt.plot(t, u_control, t, u_actual)
        plt.legend([vehicle.controls[i]+", command",
                    vehicle.controls[i]+", actual"],
                    fontsize=legendSize)
        plt.xlabel("Time (s)", fontsize=12)
        plt.grid()

###############################################################################

def plot3D(simData:NPFltArr,
           sampleTime:float,
           numDataPoints:int,
           FPS:int,
           filename:str,
           vehicles:List[Vehicle],
           ocean:Ocean,
           figNo:int=1,
           showClock:bool=True,
           showData:bool=True,
           showTraj:bool=True,
           showPos:bool=True,
           showCur:bool=True,
           showFloor:bool=True,
           )->None:
    """
    Create 3D animated visualization of vehicle trajectories and save as GIF.
    
    Generates animated 3D plot showing vehicle motion in East-North-Down frame
    with optional ocean environment features (current vectors, floor bathymetry,
    surface). Includes time clock and simulation data display.
    

    Parameters
    ----------
    simData : ndarray, shape (n_vehicles, N+1, 18)
        Simulation data for all vehicles. 
        (n_vehicles, N, [eta, nu, u_control, u_actual])
    sampleTime : float
        Simulation time step in seconds.
    numDataPoints : int
        Number of data points to use in plot. Divides simData into numDataPoints
        samples (downsampling).
    FPS : int
        Frames per second for GIF animation.
    filename : str
        Animated GIF file name.
    vehicles : list of Vehicle
        Vehicle objects in order corresponding to simData.
    ocean : Ocean
        Ocean environment object with floor and current data.
    figNo : int, default=1
        Figure number for plot window.
    showClock : bool, default=True
        Display simulation time clock overlay.
    showData : bool, default=True
        Display simulation info panel (communication mode, depth range,
        current).
    showTraj : bool, default=True
        Plot vehicle trajectory paths as lines.
    showPos : bool, default=True
        Plot vehicle current positions as points.
    showCur : bool, default=True
        Display animated ocean current vector field.
    showFloor : bool, default=True
        Display ocean floor surface (if ocean.floor exists).

         
    Notes
    -----
    **Performance Considerations:**

    - Downsampling via numDataPoints reduces computation time
    - Ocean floor rendering is major performance bottleneck (~2500 grid points
      default)
    - Animation saving takes ~1-5 minutes depending on numDataPoints and
      features
    
    **Visual Elements:**

    - Ocean surface: Transparent blue plane at z=0
    - Current vectors: Blue arrows at fixed grid, updated each frame
    - Floor: Terrain-colored surface with depth-based shading
    - Trajectories: Colored lines, one per vehicle
    - Positions: Moving points along trajectories
    - Waypoints: Triangle markers at target positions
    
    **Coordinate System:**

    END (East-North-Down)
    
    **Data Display Box Contents:**

    - COM: Communication mode (Direct Access, FDMA, TDMA)
    - DEP: Depth range min-max in meters
    - CUR: Ocean current speed and direction
    
    **Animation Function:**

    - Uses matplotlib.animation.FuncAnimation with custom frame generation.
    - Each frame updates positions, trajectories, current vectors, and display
      boxes.
    """

    # Animation function
    def anim_function(num:int,
                      dataSet:NDArray,
                      dt:float,
                      clockBox:AnchoredText,
                      dataBox:AnchoredText,
                      staticData:dict,
                      lines:Optional[List[Artist]],
                      points:Optional[List[Artist]],
                      currentQuiver:Optional[List[Artist]],
                      currentSpd:Optional[List[float]],
                      currentAng:Optional[List[float]],
                      currentXyzl:Optional[List[float]],
                      currQuivConfig:Optional[dict],
                      floorMap:Optional[Poly3DCollection],
                      )->Tuple[Optional[List[Artist]], Optional[List[Artist]]]:
        """
        Update animation frame for 3D vehicle visualization.
    
        Internal callback function for matplotlib.animation.FuncAnimation that
        updates all visual elements for each animation frame including
        trajectories, positions, current vectors, and display boxes.

        
        Parameters
        ----------
        num : int
            Present frame number. Used by FuncAnimation to control frame
            sequencing. Effectively plotting input data from [start:num] for
            each frame.
        dataSet : ndarray, shape (nVeh, numDataPoints, 3)
            Downsampled vehicle position data [x, y, -z] for all vehicles.
        dt : float
            Time per frame in seconds (sampleTime * downsample_factor).
        clockBox : AnchoredText
            Text box displaying simulation time clock.
        dataBox : AnchoredText
            Text box displaying simulation data.
        staticData : dict
            Dictionary of non-timeseries simulation data.
        lines : list of Artist or None
            Line3D objects for vehicle trajectory paths (if showTraj=True).
        points : list of Artist or None
            Line3D objects for vehicle position markers (if showPos=True).
        currentQuiver : list or None
            Contains Quiver3D object for ocean current vector field.
        currentSpd : list of float or None
            Ocean current magnitude values for each frame.
        currentAng : list of float or None
            Ocean current direction in radians for each frame.
        currentXyzl : list or None
            Ocean current vector grid positions and scaling: 
            [xx, yy, zz, length, max].
        currQuivConfig : dict or None
            Keyword arguments for quiver plot formatting.
        floorMap : Poly3DCollection or None
            Static ocean floor surface object (passed for persistence).
            

        Returns
        -------
        lines : list of Artist or None
            Updated trajectory line objects.
        points : list of Artist or None
            Updated position marker objects.
            

        Notes
        -----
        - Updates performed each frame:

            - Clock display: Converts frame time to HH:MM:SS format
            - Data box: Updates current speed/direction from frame data
            - Current vectors: Regenerates quiver with new direction/magnitude
            - Trajectories: Extends lines from start to current frame
            - Positions: Moves markers to current frame location
            
        - The floorMap parameter is not modified but must be included in
          signature to persist as static background element across frames.        
        """
        # Clock
        t = int((num - 1) * dt) # ignore fractional seconds
        h, m, s = t // 3600, (t % 3600) // 60, t % 60
        if (showClock):
            clockBox.txt.set_text(f"{h:02d}:{m:02d}:{s:02d}\nT+{t:>6}")
        # Data
        if (showData):
            if (currentSpd is not None) and (currentAng is not None):
                cspd, cang = currentSpd[num-1], currentAng[num-1]
            else:
                cspd, cang = 0.0, 0.0
            comstr = staticData['com']
            depstr = (f"{staticData['dep'][0]:.0f} - "
                      f"{staticData['dep'][1]:.0f} m")
            curstr = f"{cspd:>5.2f} m/s {np.degrees(cang):>6.1f}°E"
            cw = max([len(s) for s in [comstr,depstr,curstr]]) + 1
            dataBox.txt.set_text(
                f"COM {comstr:>{cw}}\n"
                f"DEP {depstr:>{cw}}\n"
                f"CUR {curstr:>{cw}}"
            )
        # Current
        if (showCur) and (ocean is not None) and (ocean.current is not None):
            if (currentQuiver is not None):
                cvMag = currentSpd[num-1] / currentXyzl[4] * currentXyzl[3]
                cin = (np.cos(currentAng[num-1]) * cvMag 
                       * np.ones_like(currentXyzl[0]))
                cjn = (np.sin(currentAng[num-1]) * cvMag
                       * np.ones_like(currentXyzl[0]))
                ckn = np.zeros_like(cxx)
                currentQuiver[0].remove()
                currentQuiver[0] = ax.quiver(
                    *currentXyzl[:3],
                    cin, cjn, ckn,
                    **currQuivConfig,
                )
        # Points and no lines
        if lines is None:
            for data, point in zip(dataSet, points):
                point.set_data_3d(data[num-1:num,0], 
                                  data[num-1:num,1], 
                                  data[num-1:num,2])
        # Lines and no points
        elif points is None:
            for data, line in zip(dataSet, lines):
                line.set_data_3d(data[:num,0], data[:num,1], data[:num,2])
        # Lines and Points
        else:
            for data, line, point in zip(dataSet, lines, points):
                line.set_data_3d(data[:num,0], data[:num,1], data[:num,2])
                point.set_data_3d(data[num-1:num,0], 
                                  data[num-1:num,1], 
                                  data[num-1:num,2])
        return lines, points,

    log.info('Generating plots...')
    
    # Attaching 3D axis to the figure
    fig = plt.figure(figNo,
                     figsize=(cm2inch(figSize1[0]),cm2inch(figSize1[1])),
                     dpi=dpiValue)
    ax = p3.Axes3D(fig, auto_add_to_figure=False, computed_zorder=False)
    fig.add_axes(ax)

    # Invert Z-axis for END coordinates (positive = down)
    ax.invert_zaxis()

    # Title of plot
    ax.set_title('East-North-Down Coordinates')

    # Axes labels
    ax.set_xlabel('X / East (m)')
    ax.set_ylabel('Y / North (m)')
    ax.set_zlabel('Z / Down (m)')

    # Downsample data set: take sample at every nDP-th time step
    nDP = len(simData[0,:,0]) // numDataPoints
    '''Explicit np.array call makes deep copy. This is important for protecting
    simData from being altered.'''
    dataSet = np.array((simData[:,:,0:3])[:,::nDP])
    timeStep = sampleTime * nDP # Time per sample unit

    # Define plot boundaries, set axes to same scale
    X = simData[:,:,0]
    Y = simData[:,:,1]
    Z = simData[:,:,2]
    max_range = np.array([X.max()-X.min(), 
                          Y.max()-Y.min(), 
                          Z.max()-Z.min()]).max() / 2.0
    mid_x = (X.max()+X.min()) * 0.5
    mid_y = (Y.max()+Y.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    
    [x_min, x_max] = ax.get_xlim()
    [y_min, y_max] = ax.get_ylim()

    # Configure data display boxes
    ## Clock
    clockBox = None
    if (showClock):
        clockBox = AnchoredText(
            "00:00:00\nT+     0",
            loc='upper left',
            prop=dict(family='monospace',
                    size=7,
                    ha='right',),
            frameon=True,
            pad=0.4,
            borderpad=0.5,
            zorder=10,
        )
        clockBox.patch.set_alpha(0.85)
        clockBox.patch.set_linewidth(0.8)
        ax.add_artist(clockBox)

    ## Data
    dataBox = None
    staticData = {}
    if (showData):
        dataBox = AnchoredText(
            "COM             None\n" +
            "DEP  ----- - ----- m\n" +
            "CUR  0.0 m/s   0.0°E",
            loc='upper right',
            prop=dict(family='monospace',
                    size=7,),
                    # ha='right',),
            frameon=True,
            pad=0.4,
            borderpad=0.5,
            zorder=10,
        )
        dataBox.patch.set_alpha(0.85)
        dataBox.patch.set_linewidth(0.8)
        ax.add_artist(dataBox)
        # depth
        if (ocean is not None) and (ocean.floor is not None):
            staticData['dep'] = [ocean.floor.z,
                                 ocean.floor.z+ocean.floor.z_range]
        # communication
        if (vehicles[0].CommNetwork is None):
            staticData['com'] = 'Direct Access'
        else:
            staticData['com'] = vehicles[0].info['Access Mode']

    # Capture color cycle to maintain plot colors
    colorWheel = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    def vehicleColor(i):
        return colorWheel[i % len(colorWheel)]

    # Plot Ocean Environment
    ## Ocean Surface: Plot static 2D surface for z=0
    x_grid = np.arange(x_min-20, x_max+20, 10)
    y_grid = np.arange(y_min-20, y_max+20, 10)
    [xx, yy] = np.meshgrid(x_grid, y_grid)
    zz = 0 * xx
    ax.plot_surface(xx, yy, zz, alpha=0.3, color='blue', zorder=10)

    ## Ocean Current: Plot 2D quiver for ocean current vector
    currentSpd, currentAng, = None, None
    if ((ocean is not None) and (ocean.current is not None)):
        currentSpd = ocean.current.speed[::nDP]
        currentAng = ocean.current.angle[::nDP]
    cv_xyzl, currQuivConfig, currentQuiver = None, None, None
    if (showCur and (ocean is not None) and (ocean.current is not None)):

        # Create arrow plot positions
        nArrows = 4
        cx = (x_max-x_min) / nArrows
        cy = (y_max-y_min) / nArrows
        cx_grid = np.arange(x_min, x_max, cx) + (cx / 2)
        cy_grid = np.arange(y_min, y_max, cy) + (cy / 2)
        cxx, cyy = np.meshgrid(cx_grid, cy_grid)
        ff = max_range
        if (ocean.floor is not None) and (showFloor):
            ff = ocean.floor.z * 0.95
        czz = np.ones_like(cxx) * ff

        # Package length and position for plot function
        cv_len = 0.8 * min(cx, cy)
        cv_max = ocean.current._V_HI
        cv_xyzl = [cxx,cyy,czz,cv_len,cv_max]

        # Configure vector arrows appearance
        currQuivConfig = {
            'length': 1.0,              # use length from cv_len
            'normalize': False,         # use the computed lengths
            'arrow_length_ratio': 0.15, # arrow heads
            'linewidth': 1.2,           # arrow shafts
            'color': 'steelblue',
            'alpha': 0.3,
            'zorder': 9,
        }

        # Fill quiver with initial vectors
        ci0 = ((np.cos(currentAng[0]) * currentSpd[0]/cv_max * cv_len) 
               * np.ones_like(cxx))
        cj0 = ((np.sin(currentAng[0]) * currentSpd[0]/cv_max * cv_len) 
               * np.ones_like(cxx))
        ck0 = np.zeros_like(cxx)
        currentQuiver = [ax.quiver(
            *cv_xyzl[:3],           # x,y,z grid of arrow positions
            ci0, cj0, ck0,          # i,j,k vector components
            **currQuivConfig,
        )]

    ## Ocean Floor: Plot static 3D surface at z=floor depth
    floorMap = None
    if (showFloor and (ocean is not None) and (ocean.floor is not None)):
        # Define fixed resolution grid
        '''Grid size is main driver on speed of animation saving when displaying
        the ocean floor.'''
        approxGridSize = 2500 
        aspectRatio = (x_max - x_min) / (y_max - y_min)
        x_res = int(np.sqrt(approxGridSize * aspectRatio))
        y_res = int(np.sqrt(approxGridSize / aspectRatio))

        # Sample floor depth map at grid points
        x_pts = np.linspace(x_min, x_max, x_res)
        y_pts = np.linspace(y_min, y_max, y_res)
        floorGrid = ocean.floor.sample_grid(x_pts, y_pts)

        # Create meshgrid for plotting floor surface
        x_f, y_f = np.meshgrid(x_pts, y_pts)
        z_f = floorGrid.T

        # Set floor color map
        terrain_cmap = mpl.colormaps['terrain']
        colors = terrain_cmap(np.linspace(0.65, 0.55, 256))
        colors[:, -1] = 0.65
        custom_cmap = mpl.colors.ListedColormap(colors)

        # Render floor surface
        floorMap = ax.plot_surface(
            x_f, y_f, z_f,
            cmap=custom_cmap,
            shade=True,
            edgecolor='none',
            linewidth=0,
            rasterized=True,   # True for some performance increase
            rstride=1,         # row stide - additional downsampling options
            cstride=1,         # column stride - can be done here
            antialiased=False, # False for some performance increase
            zorder=0,
        )

        # Use grid sample for data box
        if (showData):
            staticData['dep'] = [floorGrid.min(), floorGrid.max()]

    # Plot Vehicle Paths and Waypoints
    ## Line for Vehicle Trajectory
    lines = None
    if showTraj:
        lines = []
        for i, data in enumerate(dataSet):
            ln, = ax.plot(data[:,0],
                          data[:,1],
                          data[:,2],
                          lw=1,
                          zorder=1,
                          color=vehicleColor(i))
            lines.append(ln)
    ## Point for Vehicle Position
    points = None
    if showPos:
        points = []
        for i, data in enumerate(dataSet):
            pt, = ax.plot(data[:,0], 
                          data[:,1], 
                          data[:,2], 
                          marker='.', 
                          markersize=3,
                          zorder=2,
                          color=vehicleColor(i))
            points.append(pt)
    ## Waypoints
    for i, vehicle in enumerate(vehicles):
        if (vehicle.wpt is not None):
            wp = ax.plot(vehicle.wpt.pos.x, 
                         vehicle.wpt.pos.y, 
                         vehicle.wpt.pos.z, 
                         '^',
                         alpha=0.5,
                         zorder=3,
                         color=vehicleColor(i))[0]
    
    # Update Z limits before displaying
    if (floorMap is not None):
        max_depth = ocean.floor.z + ocean.floor.z_range
        ax.set_zlim(max_depth * 1.05, -5)
    else:
        ax.set_zlim(max_range * 1.05, -max_range * 0.05)
    """
    Uncomment the following line to scale z axis 1:1 with x and y
    """ 
    # ax.set_aspect('equal')

    # Create the animation object
    ani = animation.FuncAnimation(fig, 
                                  anim_function, 
                                  frames=numDataPoints, 
                                  fargs=(dataSet,
                                         timeStep,
                                         clockBox,
                                         dataBox,
                                         staticData,
                                         lines,
                                         points,
                                         currentQuiver,
                                         currentSpd,
                                         currentAng,
                                         cv_xyzl,
                                         currQuivConfig,
                                         floorMap),
                                  interval=1000//FPS, 
                                  blit=False,
                                  repeat=True)
    
    # Save the 3D animation as a gif file
    log.info('Saving animation...')
    ani.save(filename, writer=animation.PillowWriter(fps=FPS))
    log.info('Done saving animation.')

###############################################################################
