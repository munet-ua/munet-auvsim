Quick Start
===========

Minimal Example
---------------

Here's a complete minimal simulation:

.. code-block:: python

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

For a comprehensive feature demonstration and guided tour, see :ref:`example-scripts`.

Basic Workflow
--------------

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

Running Simulations
-------------------

User scripts should be placed in the ``scripts/`` directory:

.. code-block:: bash

    # Navigate to project root
    cd munet-auvsim

    # Run your simulation script
    python scripts/your_simulation.py

Simulations can also be built and run from a Python REPL session.

Output Files
------------

Simulation outputs are saved to::

    outputs/{scriptname}/{simname-timestamp}/

Files generated:

- ``{simname-timestamp}.log``: Main simulation log
- ``{simname-timestamp}_comm.log``: Communication log
- ``{simname-timestamp}.gif``: Animated 3D trajectory
- ``{simname-timestamp}.pickle``: Saved simulation state

If run from a REPL, `REPL` is used as the `{scriptname}`.

Loading Previous Simulations
-----------------------------

You can reload saved simulations for additional analysis or re-plotting:

.. code-block:: python

    import munetauvsim as mn

    # Load saved simulation
    sim = mn.load("outputs/demo/Demo-YYMMDD-HHMMSS/Demo-YYMMDD-HHMMSS.pickle")
    print(sim)

    # Re-plot with different settings
    sim.plot3D(numDataPoints=500, FPS=12, showFloor=True)

    # Access collected data
    auv1_data = sim.simData[0, :, :]

Next Steps
----------

- See :doc:`api/modules` for complete documentation

.. _example-scripts:

Example Scripts
---------------

Two example scripts are provided to help you get started:

**example.py** — Quick Start Script
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A basic script demonstrating how to build a simple simulation scenario:

.. code-block:: bash

    cd munet-auvsim
    python scripts/example.py

This script:

- Creates a small swarm group with a leader
- Assigns a random path to the leader and followers track leader
- Sets up a simple ocean environment and communication network
- Runs the simulation and saves the data

This can be used for a quick test to confirm the project is functional or as a
reference for understanding the basic workflow.

**demo.py** — Interactive Tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An interactive demonstration script with detailed explanations:

.. code-block:: bash

    python scripts/demo.py

This script:

- Presents interactive menus for scenario selection
- Provides educational context for key concepts
- Allows customization of parameters (paths, ocean, vehicles, etc.)
- Includes 2 demonstration scenarios, with more coming soon

This is best for learning the framework and exploring features in a hands-on
format.

Choose ``example.py`` if you want quick results with minimal setup. Choose ``demo.py`` 
if you want an interactive guided tour with explanations.
