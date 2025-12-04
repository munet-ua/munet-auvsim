"""
example.py - Simple Example for muNet-AUVsim

This is a basic example script demonstrating the workflow for setting up and
running an AUV swarm simulation. It uses default values and built-in convenience
functions for quick setup. For advanced configuration options and detailed
explanations, see demo.py and refer to the project documentation.
"""

import munetauvsim as mn

#------------------------------------------------------------------------------#
#    Set Up Simulation                                                         #
#------------------------------------------------------------------------------#

sim = mn.simulator.Simulator(name='Example')   # create a simulation object
path = mn.guidance.generateRandomPath(num=6)   # make a random waypoint path
sim.runTime = path.estimateTime()              # estimate simulation run time

#------------------------------------------------------------------------------#
#    Ocean                                                                     #
#------------------------------------------------------------------------------#

size, origin = path.estimateAreaRoot()         # fit ocean around path
sim.ocean = mn.environment.Ocean.calm_ocean(   # create basic calm ocean
    size=size,                                 # side length of xy ocean square
    origin=origin,                             # position of origin in square
)

#------------------------------------------------------------------------------#
#    Vehicles                                                                  #
#------------------------------------------------------------------------------#

swarm = mn.vehicles.buildGroup(                # build swarm vehicles
    num = 4,                                   # number of vehicles: 4
    gid = 'A',                                 # group ID: A
    hasLeader=True                             # first vehicle is leader
)

# Leader AUV
leader = swarm[0]                              # handle to leader vehicle
leader.wpt = path                              # load path waypoints
leader.loadPathFollowing()                     # assign guidance system
leader.loadConstantProp()                      # set constant propeller

# Follower AUVs
for auv in swarm[1:]:                          # rest of vehicles are followers
    auv.loadTargetTracking(leader)             # assign guidance system

## Deploy AUVs
sim.vehicles = swarm                           # load swarm in simulator
sim.deployAtWpt(leader)                        # deploy leader at first waypoint
sim.deploySwarmGrid()                          # deploy swarm behind leader

#------------------------------------------------------------------------------#
#    Communication                                                             #
#------------------------------------------------------------------------------#

sim.loadMuNet()                                # acoustic network communication

#------------------------------------------------------------------------------#
#    Run Simulation                                                            #
#------------------------------------------------------------------------------#

sim.run()                                      # start the simulation
mn.simulator.save(sim)                         # save simulation data file