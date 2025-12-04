"""
muNet-AUVsim: Multi-Agent Autonomous Underwater Vehicle Simulation Framework

A comprehensive simulation environment for autonomous underwater vehicle
swarm coordination, communication networks, and environmental modeling.

Modules
-------
vehicles : Vehicle models and dynamics
environment : Ocean environment and effects  
communication : Network protocols and models
guidance : Path planning and waypoint systems
control : Low-level vehicle control laws
navigation : Sensor models and state estimation
gnc : Guidance, navigation, and control math
simulator : Main simulation coordination
plotTimeSeries : Visualization and plotting utilities
logger : Logging configuration and utilities

Examples
--------
### Basic AUV simulation:

>>> import munetauvsim as mn
>>> 
>>> # Create vehicle and environment
>>> auv = mn.vehicles.Remus100s()
>>> ocean = mn.environment.Ocean()
>>> 
>>> # Setup simulation
>>> sim = mn.Simulator(
...     name="BasicDemo",
...     vehicles=[auv],
...     ocean=ocean,
...     N=30000
... )
>>> sim.run()

### Multi-agent swarm with communication:

>>> # Create swarm
>>> leader = mn.vehicles.Remus100s(groupId="A", isLeader=True)
>>> followers = mn.vehicles.buildGroup(3, "A", hasLeader=False)
>>> 
>>> # Setup network
>>> network = mn.communication.MuNet()
>>> 
>>> # Run simulation
>>> sim = mn.Simulator(
...     name="SwarmDemo", 
...     vehicles=[leader] + followers
... )
>>> sim.loadMuNet(network)
>>> sim.run()
"""

# Core modules - import for direct access
from . import communication
from . import control
from . import environment  
from . import gnc
from . import guidance
from . import logger
from . import navigation
from . import plotTimeSeries
from . import simulator
from . import vehicles

# Classes and functions for convenience
from .simulator import Simulator
from .simulator import save, load

# Version info
__version__ = "0.1.0-beta"
__author__ = "JP Crawford"

# Define what gets imported with "from munetauvsim import *"
__all__ = [
    # Modules
    'communication',
    'control',
    'environment', 
    'gnc',
    'guidance',
    'logger',
    'navigation',
    'plotTimeSeries',
    'simulator',
    'vehicles',
    # Main classes
    'Simulator',
    'save',
    'load',
]