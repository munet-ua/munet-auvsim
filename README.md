# muNet AUV Sim

**Version:** 0.1.0-beta  
**Status:** Beta Release

## Description

This project is aimed at developing a comprehensive Python-based simulation
framework for multi-agent Autonomous Underwater Vehicle (AUV) swarm research.
The framework provides a complete environment for modeling, testing, and
evaluating underwater vehicle coordination, communication networks, and control
algorithms in realistic ocean conditions.

### Features

- **AUV dynamics modeling**: 6-DOF nonlinear vehicle dynamics based on Fossen's
  marine vehicle modeling
- **Virtual ocean environment**: Ocean currents, bathymetry, and environmental
  effects
- **Underwater Communication**: Simulated acoustic networking and real AquaNet
  stack integration
- **Modular GNC Architecture**: Flexible guidance, navigation, and control
  systems with plug-and-play algorithms
- **Swarm Coordination**: Multi-agent groups, path following, and target
  tracking
- **Comprehensive Visualization**: Animated 3D trajectory plots with real-time
  vehicle states
- **Data Collection & Analysis**: Complete state logging, performance metrics,
  and reproducible simulations

### Research Applications

- Swarm coordination algorithms
- Underwater acoustic communication protocols
- Formation control strategies
- Path planning and navigation
- Cooperative sensing and surveillance
- Multi-agent underwater robotics

## Table of Contents

- [Installation](#installation)
- [Documentation](#documentation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Core Concepts](#core-concepts)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Authors and Acknowledgments](#authors-and-acknowledgments)
- [Contact and Support](#contact-and-support)
- [Project Status](#project-status)

## Installation

### Prerequisites

- Python 3.8 or higher
- Git (for cloning repository)

### Platform Compatiblity

muNet-AUVsim runs best on Linux. The included AquaNet communication stack is not
compatible with Windows, but use of AquaNet is optional. The core simulation
framework supports Windows using the muNet network simulator or direct-access.

*Use of AquaNet is not required to run muNet-AUVsim.* It is only one of the
available options for simulating a communication network.

**Core Simulation Framework:**
- **Linux** - Full support (tested)
- **macOS** - Expected to work (not extensively tested)
- **Windows** - Core features supported with muNet communication

**AquaNet Communication Stack:**
- **Linux** - AquaNet requires Unix domain sockets and Linux binaries
- **macOS** - May work but untested
- **Windows** - Not supported: use muNet instead

#### Windows Users

Windows users can run the complete simulation framework with the following limitation:

**Available:**
- All vehicle dynamics and guidance, navigation, and control features
- muNet acoustic network simulator (full-featured simulation)
- Ocean environment modeling
- Visualization and data collection
- Simulation saving and loading

**Not Available:**
- AquaNet protocol stack integration
- Use `loadMuNet()` instead of `loadAquaNet()` in your simulations

### Step-by-Step Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/munet-ua/munet-auvsim.git
    cd munet-auvsim
    ```

2. **Create a Python virtual environment:**
    ```sh
    # Choose your preferred virtual environment name (e.g. 'munet')
    python -m venv munet
    ```
    This creates a directory for the virtual environment named 'munet' in your
    project folder.

3. **Add the virtual environment folder to your git exclude list:**
    ```sh
    # Open in your preferred text editor
    gedit .git/info/exclude
    ```
    Add the virtual environment directory name on a new line and save:
    ```text
    munet/
    ```

4. **Add the project to the PYTHONPATH for your virtual environment:**

    **Linux/Mac:**
    ```sh
    # Open virtual environment activation script
    gedit munet/bin/activate
    ```

    Add at the end of the file (replace with your actual path):
    ```bash
    export PYTHONPATH="/path/to/your/project/munet-auvsim"
    ```
    
    **Windows:**
    ```cmd
    # Open virtual environment activation script
    notepad munet\Scripts\activate.bat
    ```
    Add before the final line:
    ```batch
    set PYTHONPATH=C:\path\to\munet-auvsim;%PYTHONPATH%
    ```

5. **Activate the python virtual environment:**

    **Linux/Mac:**
    ```sh
    source munet/bin/activate
    ```
    
    **Windows:**
    ```cmd
    munet\Scripts\activate
    ```

    You should see that your prompt is now updated with `(munet)` to indiciate
    you are working inside a python virtual environment (venv). To exit the
    venv, type `deactivate`.

6. **Install the required dependencies:**

    Make sure you are in the virtual environment (e.g. '(munet)' shows at
    prompt). Enter:
    ```sh
    pip install -r requirements.txt
    ```

### Verifying Installation

Test your installation:
```python
python -c "import munetauvsim as mn; print(f'muNet-AUVsim v{mn.__version__} loaded successfully')"
```

Verify everything works by running the quick-start example:

```sh
cd munet-auvsim
python scripts/example.py
```

Expected output:

- Simulation runs for ~60 seconds
- Simulation progress printed to console
- Output files created in `outputs/example/Example_YYMMDD-HHMMSS` directory:
  - Simulation log file
  - 3D trajectory animation (GIF)
  - Simulation data file (pickle)

If the script runs without errors, your installation is verified and ready to use.

For a more detailed interactive experience, try the tutorial script:

```sh
python scripts/demo.py
```

**Windows users:** If you see an error related to `aquanet_lib`, this is expected. AquaNet is Linux-only. The core simulation framework will work correctly with muNet communication.

## Documentation

### Online Documentation

The full project documentation is available at:
[GitHub Pages](https://munet-ua.github.io/munet-auvsim/)

This documentation is up to date with the latest release and is the best place to start if you're new to the project or need reference information.

### Using the Documentation

- **Browse:**
  Use the left navigation bar to browse API Reference, containing a complete
  list of modules, classes, methods, and functions available in the project. The
  documentation provides quick reference, class/module hierarchies, and example
  usage.

- **Search:**
  Use the built-in search bar at the top-left of the page to quickly find
  elements by name or keyword across the entire API.

- **Index:**
  Browse the complete alphabetical listing of all API elements (classes,
  functions, methods, attributes) with links to their definitions. Useful for
  quick lookups and discovering what's available.

### Build the Docs Locally

You can also build the documentation from source if you wish to experiment, preview unpublished changes, or work offline:

1. **Activate your virtual environment first:**

    **Linux/Mac:**
    ```sh
    source munet/bin/activate
    ```

    **Windows:**
    ```sh
    munet\Scripts\activate
    ```

2. **Install Sphinx and theme:**

    ```sh
    pip install sphinx sphinx-copybutton sphinx-rtd-theme
    ```

3. **Build Documentation:**

    ```sh
    # Navigate to the docs directory
    cd docs
    ```

    **Linux/Max:**
    ```sh
    # Build html files
    make html
    ```

    **Windows:**
    ```sh
    make.bat html
    ```

4. **View Documentation:**

    Open munet-auvsim/docs/build/html/index.html in your browser.

The Sphinx configuration and source files are provided in the `docs/` directory, excluding the built HTML.

### Alternative Documentation Access

Building the docs locally is completely optional. The project is fully functional without installing Sphinx or building the docs. All Python modules, classes, methods, and functions in muNet-AUVsim include rich docstrings. You can inspect usage and function signatures in your IDE or with built-in help:

```python
import munetauvsim as mn
help(mn.communication.MuNet)
help(mn.vehicles.Remus100s)
help(mn.Simulator)
```

## Quick Start

Here's a minimal example to get started quickly:

```python
import munetauvsim as mn

# Create a leader vehicle with waypoint path
leader = mn.vehicles.Remus100s(groupId="A", isLeader=True)
leader.wpt = mn.guidance.generateRandomPath(num=5)
leader.loadPathFollowing()
leader.loadConstantProp(n_setpt=1200)

# Create follower vehicles
followers = mn.vehicles.buildGroup(num=3, gid="A", hasLeader=False)
for f in followers:
    f.loadTargetTracking(leader)

# Create ocean environment
ocean = mn.environment.Ocean.calm_ocean(size=2000, randomFloor=True)

# Create and configure simulation
sim = mn.Simulator(
    name="QuickDemo",
    N=5000,
    ocean=ocean,
    vehicles=[leader] + followers
)

# Deploy and run
sim.deployAtWpt(leader)
sim.deploySwarmGrid()
sim.run()
```

### Example Scripts

Two example scripts are provided in the `scripts/` directory:

- **`example.py`** -- Simple starter script with minimal code. Useful for a
  quick test to confirm the project is functional, or as a reference for the
  basic workflow to build a simulation scenario.
  Run: `python scripts/example.py`

- **`demo.py`** -- Interactive tutorial with detailed information and user
  prompts. Useful for learning the framework and exploring features with guided
  scenarios.
  Run: `python scripts/demo.py`

For a comprehensive feature demonstration and guided tour, see
`scripts/demo.py`.

## Usage

### Basic Workflow

The typical simulation workflow consists of:

1. **Vehicle Creation**: Instantiate AUV models and configure parameters
2. **Path Planning**: Define waypoints or generate paths
3. **Guidance Configuration**: Assign control strategies
4. **Environment Setup**: Create ocean with currents and bathymetry
5. **Communication Setup**: Configure acoustic network
6. **Simulation Creation**: Instantiate simulator with time parameters
7. **Vehicle Deployment**: Position vehicles at starting locations
8. **Execution**: Run simulation and collect data
9. **Analysis**: Visualize trajectories and examine results

### Running Simulations

User scripts should be placed in the `scripts/` directory:

```sh
# Navigate to project root
cd munet-auvsim

# Run your simulation script
python scripts/your_simulation.py
```

### Output Files

Simulation outputs are organized in `outputs/{scriptname}/{simname-timestamp}/`:

- `{simname-timestamp}.log`: Main simulation log
- `{simname-timestamp}_comm.log`: Communication network log  
- `{simname-timestamp}.gif`: Animated 3D trajectory visualization
- `{simname-timestamp}.pickle`: Saved simulation state (for later analysis)

### Loading Previous Simulations

```python
import munetauvsim as mn

# Load saved simulation
sim = mn.load("outputs/demo/Demo-YYMMDD-HHMMSS/Demo-YYMMDD-HHMMSS.pickle")
print(sim)

# Re-plot with different settings
sim.plot3D(numDataPoints=500, FPS=12, showFloor=True)

# Access collected data
auv1_data = sim.simData[0, :, :]
```

## Project Structure

The project follows a standard directory structure:
```
munet-auvsim/
├── munetauvsim/                # Main package directory
│   ├── __init__.py             # Package initialization
│   ├── aquanet_lib/            # AquaNet communication stack (external)
│   │   ├── __init__.py         # AquaNet Python interface (customized)
│   │   ├── scripts/            # AquaNet utility scripts
│   │   │   └── stack-stop.sh   # Script to stop AquaNet stack
│   │   └── tmp/                # AquaNet runtime log files from each Node
│   ├── communication.py        # Acoustic network simulation (muNet)
│   ├── control.py              # Vehicle control algorithms
│   ├── environment.py          # Ocean environment modeling
│   ├── gnc.py                  # Low-level GNC and dynamics mathematics
│   ├── guidance.py             # Path planning and guidance laws
│   ├── navigation.py           # Sensors and state estimation
│   ├── plotTimeSeries.py       # Visualization and plotting
│   ├── simulator.py            # Main simulation coordinator
│   └── vehicles.py             # AUV vehicle models and dynamics
├── docs/                       # Documentation (Sphinx)
│   ├── source/                 # Sphinx source files
│   │   ├── conf.py             # Sphinx configuration
│   │   ├── index.rst           # Documentation homepage
│   │   ├── installation.rst    # Installation guide
│   │   ├── quickstart.rst      # Quick start guide
│   │   └── api/                # API reference files
│   │       └── modules.rst     # Module documentation structure
│   ├── build/                  # Generated HTML/PDF (excluded from repo)
│   ├── Makefile                # Build commands (Linux/Mac)
|   └── make.bat                # Build commands (Windows)
├── outputs/                    # Generated output files
├── scripts/                    # User simulation scripts
│   ├── demo.py                 # Usage and API demonstration
│   └── example.py              # Basic example script
├── CITATION.cff                # Metadata for academic referencing
├── CONTRIBUTING.md             # Contribution guidelines
├── LICENSE                     # MIT License
├── README.md                   # This file
└── requirements.txt            # Python dependencies
```

### Module Overview

| Module | Description |
|--------|-------------|
| **simulator** | Main simulation coordinator; orchestrates vehicles, environment, communication, data collection |
| **vehicles** | AUV models, dynamics, sensor interfaces |
| **guidance** | Waypoint management, path planning, guidance laws |
| **control** | Autopilots, actuator commands, propeller commands |
| **navigation** | Sensor models, state estimation, filters |
| **communication** | muNet acoustic network simulator, message protocols, AquaNet integration |
| **environment** | Ocean currents, floor bathymetry, environmental elements |
| **gnc** | Core mathematical functions for GNC and dynamics computations |
| **plotTimeSeries** | 3D trajectory animation, state time series plots |
| **logger** | Logging configuration and utilities |

## Core Concepts

### Vehicles

The framework provides the **Remus100s** AUV model with complete 6-DOF dynamics:

```python
import munetauvsim as mn

# Create individual vehicle
auv = mn.vehicles.Remus100s(groupId="A", callSign="SCOUT-01")

# Create vehicle group
swarm = mn.vehicles.buildGroup(num=5, gid="B", hasLeader=True)
```

**Key Vehicle Attributes:**
- `eta`: Position/attitude [x, y, z, phi, theta, psi]
- `nu`: Body-frame velocities [u, v, w, p, q, r]
- `velocity`: END-frame velocities [vx, vy, vz]
- `u_actual`: Control inputs [rudder, stern, propeller]

### Guidance Systems

Two primary guidance modes are currently implemented:

#### 1. Path Following (ALOS)

Adaptive Line-of-Sight guidance for waypoint-based paths:

```python
# Define waypoints
leader.wpt = mn.guidance.Waypoint(
    xPos=[0, 500, 1000],
    yPos=[0, 0, 500],
    zPos=[15, 15, 20]
)

# Configure path following
leader.loadPathFollowing()
```

#### 2. Target Tracking (APF)

Artificial Potential Field guidance for swarm coordination:

```python
follower.loadTargetTracking(
    target=leader,
    law="APF",          # Guidance law
    att="linearCBZ",    # Attraction function
    rep="varExp"        # Repulsion function
)

# Tune swarm parameters
follower.r_safe = 10.0    # Minimum safe distance (m)
follower.r_avoid = 40.0   # Avoidance radius (m)
follower.r_follow = 80.0  # Following distance (m)
```

### Ocean Environment

Create realistic ocean conditions:

```python
# Predefined calm ocean
ocean = mn.environment.Ocean.calm_ocean(
    size=2000,
    origin=[0, 0],
    randomFloor=True
)

# Custom ocean with currents
ocean = mn.environment.Ocean(
    spd=0.8,           # Current speed (m/s)
    ang=np.pi/6,       # Current direction (rad)
    z=150.0,           # Average depth (m)
    size=3000
)
```

### Communication Networks

The simulation can run in direct-access mode with no simulated communication network between vehicles, but two simulated communication networks are supported.

#### muNet: Simulated Acoustic Network

**Platform Support:** Linux, macOS, Windows

```python
network = mn.communication.MuNet(
    accessMode="fdma",      # MAC protocol
    PLR=0.02,               # Packet loss rate
    BER=1e-3,               # Bit error rate
    MAXJITTER=0.3,          # Max delay (s)
    collType="corrupt",     # Collision handling
    seed=42                 # Reproducible network errors
)

sim = mn.Simulator()
sim.vehicles = mn.vehicles.buildGroup(3, gid='A', hasLeader=True)
sim.loadMuNet(network=network, accessMode='fdma', episode=5.0, txOffset=0.5)
```

**Supported MAC Protocols:**
- `shared`: Single broadcast channel
- `tdma`: Time-division multiple access
- `tdma_lf`: TDMA leader-follower
- `fdma`: Frequency-division multiple access

#### AquaNet: Real Protocol Stack (Linux Only)

**Platform Support:** Linux only. Requires Unix domain sockets and compiled binaries.

```python
sim.linkSwarmGroup()
sim.loadAquaNet(
    episode=10.0,      # TDMA cycle duration
    frames=[1.0, 1.0]  # [BCRQ, RSPN] durations
)
```

**Note for Windows users:** AquaNet integration is not available on Windows due to platform-specific dependencies. Use muNet for all communication simulation needs.

### Simulation Configuration

```python
sim = mn.Simulator(
    name="MySimulation",
    sampleTime=0.02,        # 50 Hz iteration rate
    N=30000,                # 10 minutes
    ocean=ocean,
    vehicles=swarm,
    logging="all"
)

# Alternative: set by duration
sim.runTime = 600  # 10 minutes in seconds
```

### Vehicle Deployment

```python
# Deploy leader at first waypoint
sim.deployAtWpt(leader)

# Deploy followers in grid formation
sim.deploySwarmGrid()
```

## Examples

### Example 1: Leader-Follower Path Following

```python
import munetauvsim as mn

# Create leader with waypoint path
leader = mn.vehicles.Remus100s(groupId="A", isLeader=True)
leader.wpt = mn.guidance.Waypoint(
    xPos=[0, 500, 1000, 1000, 500, 0],
    yPos=[0, 0, 500, 1000, 1000, 1000],
    zPos=[0, 15, 20, 20, 15, 15]
)
leader.loadPathFollowing()
leader.loadConstantProp(n_setpt=1200)

# Create 3 followers tracking leader
followers = mn.vehicles.buildGroup(3, "A", hasLeader=False)
for f in followers:
    f.loadTargetTracking(leader, law="APF", att="linearCBZ", rep="varExp")

# Setup environment and simulation
ocean = mn.environment.Ocean.calm_ocean(size=1200, randomFloor=True)
sim = mn.Simulator(
    name="LeaderFollowerDemo",
    N=66500,
    ocean=ocean,
    vehicles=[leader] + followers
)

# Deploy and run
sim.deployAtWpt(leader)
sim.deploySwarmGrid()
sim.run()
```

### Example 2: Multi-Group Swarm with Communication

```python
import munetauvsim as mn

# Create two swarm groups
groupA = mn.vehicles.buildGroup(3, "A", hasLeader=True)
groupB = mn.vehicles.buildGroup(3, "B", hasLeader=True)

# Configure leaders
for group in [groupA, groupB]:
    leader = group[0]
    pos = leader.id*100
    leader.wpt = mn.guidance.generateRandomPath(num=6, start=[pos, pos, 15])
    leader.loadPathFollowing()
    leader.loadConstantProp(n_setpt=1200)
    
    # Configure followers
    for f in group[1:]:
        f.loadTargetTracking(leader)

# Setup communication
network = mn.communication.MuNet(accessMode="fdma", PLR=0.03, seed=42)
ocean = mn.environment.Ocean(size=1200, randomFloor=True)

# Create simulation
sim = mn.Simulator(
    name="MultiGroupSwarm",
    runTime=900,
    ocean=ocean,
    vehicles=groupA + groupB
)

# Link communication network nodes
sim.loadMuNet(network, episode=5.0, txOffset=0.5)

# De-register groups from RX on each others channels
for auv in groupA:
    auv.CommNetwork.rx_channels -= {4, 5, 6}
for auv in groupB:
    auv.CommNetwork.rx_channels -= {1, 2, 3}

# Deploy both groups
for group in [groupA, groupB]:
    sim.deployAtWpt(group[0])
sim.deploySwarmGrid()

sim.run()
```

## Contributing

We welcome contributions from the research community! Whether you're fixing bugs, adding features, improving documentation, or reporting issues, your help is appreciated.

Please see [CONTRIBUTING.md](CONTRIBUTING.md) for complete details, workflow examples, and code guidelines.

### Quick Workflow

1. **Fork** the repository to your GitHub account
2. **Clone** your fork and create a feature branch
3. **Make changes** and test thoroughly
4. **Commit** with clear, descriptive messages
5. **Push** to your fork and open a Pull Request
6. **Address feedback** from maintainers during review

**Important:** Never push directly to `main`. All changes must go through the pull request review process.

### Contribution to Documentation

If you spot errors, want to add examples, or expand the user guide:
- Fork the repository and edit `.rst` files in `docs/source/`
- Preview your changes locally with `make html` in the `docs/` directory
- Submit a Pull Request summarizing your improvements

Documentation is automatically built and published to [GitHub Pages](https://munet-ua.github.io/munet-auvsim/) when changes are merged.

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE)
file for details.

## Authors and Acknowledgments

### Authors

- **JP Crawford** -- *Primary developer and maintainer*  
  [@jpcrawford](https://github.com/jpcrawford)

  - Restructured and extended single-vehicle dynamics and design into a
    multi-agent AUV swarm simulation framework.
  - Integrated existing GNC design with custom communication, guidance, and
    environment modeling components.
  - Implemented swarm behavior algorithms, target tracking with artificial
    potential fields, and leader-follower coordination strategies.      
  - Designed and implemented ocean environment simulation, including currents
    and Perlin noise terrain generation.
  - Built the simulation logging, save/load functionality, and enhanced the
    3D trajectory visualization and animation tools.
  - Developed message serialization and application layer protocols and
    integrated the AquaNet protocol stack.
  - Created the muNet acoustic network simulator with physical layer models
    (BER, PLR, collision) and multiple MAC protocols (FDMA, TDMA).

### Contributors

- **Michael Qin** -- *Developed the 'Pollution' object for representing a static
  Gaussian plume dispersion*  
  [Michael](https://munet-gitlab.eng.ua.edu/Michael)

### Acknowledgments

This work was developed at the **muNet Lab** at the The University of Alabama
under the supervision and support of [Dr. Aijun
Song](https://eng.ua.edu/eng-directory/dr-aijun-song/).

**Key External Resources:**

- **Marine Vehicle Dynamics**: This project extensively utilizes the work of
  [Dr. Thor I. Fossen](https://www.fossen.biz/). The 6-DOF vehicle dynamics
  implementation is based on his [Python Vehicle
  Simulator](https://github.com/cybergalactic/PythonVehicleSimulator) and the
  textbook *Handbook of Marine Craft Hydrodynamics and Motion Control* (2nd
  Edition, Wiley, 2021).

- **AquaNet Communication Stack**: The AquaNet module (`aquanet_lib/`) is
  written by [Dmitrii Dugaev](https://github.com/dugdmitry/aquanet_lib). AquaNet
  provides a real network protocol stack implementation.

### Citing This Work

If you use muNet-AUVsim in your research, please cite:

```bibtex
@software{munet_auvsim,
  author = {Crawford, JP and Song, Aijun},
  title = {muNet-AUVsim: Multi-Agent AUV Simulation Framework},
  year = {2025},
  version = {0.1.0-beta},
  url = {https://github.com/munet-ua/munet-auvsim}
}
```

## Contact and Support

- **Issues**: Report bugs and request features via [GitHub
  Issues](https://github.com/munet-ua/munet-auvsim/issues)

## Project Status

**Current Version**: 0.1.0-beta  
**Status**: Beta Release - Core features stable, API subject to refinement

### Roadmap

Short list of bug fixes and features intended for future development:

**Near Term:**

- [ ] **Improved Ocean Modeling** - Remove size limitations in Perlin terrain
  generation for larger simulation areas
- [ ] **Better Network Tools** - Enhanced node registration and channel
  assignment utilities for muNet
- [ ] **Visualization Improvements** - Separate 3D plot generation from
  animation rendering for better performance and user control
- [ ] **APF Performance** - Investigate wave filtering in APF and errant vehicle
  behavior

---
---