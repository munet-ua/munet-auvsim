"""
Communication network simulation for multi-vehicle acoustic messaging.

This module implements the Communication components for simulating underwater
acoustic vehicle-to-vehicle networks. It provides a discrete-event network
simulator (MuNet) that models physical layer characteristics of acoustic
communication including propagation delay, packet loss, bit errors, and
collision dynamics.

The module supports multiple Medium Access Control (MAC) protocols for channel
allocation and provides configurable error models for realistic network behavior
under varying environmental conditions and communication ranges.

Classes
-------
**Network Infrastructure**
    MuNet
        Underwater acoustic network simulator with configurable MAC protocols,
        error models, and channel management. Main simulation engine for
        message transmission, reception, and delivery.
    MuNode
        Network node representing vehicle connection to acoustic network with
        TX/RX channel assignments and collision detection state.

**Data Structures**
    MuNetQEntry
        Queue entry dataclass for tracking message transmission and reception
        with metadata (times, addresses, channel info, distance) and payload.
    MuChannel
        Network channel specification with frequency, bandwidth, ID, and
        broadcast flag.

Functions
---------
**Network Management**
    getMsgStruct(msgType) : Return construct library binary structure for
        message type ('RPRT', 'LRPT', 'BCRQ').
    recvMsg(vehicle) : Spawn daemon receiver thread for vehicle network
        listening.
    recvMsgCallback(bytesMsg, vehicle) : Message dispatcher routing received
        messages to appropriate handler by type flag.

**REPORT Message Operations (Vehicle State Broadcast)**
    writeReport(vehicle, msgType='REPORT') : Serialize vehicle position,
        velocity, timestamp into 37-byte RPRT message.
    sendReport(vehicle, msgType='REPORT', address=None) : Serialize and
        transmit REPORT message over network.
    recvReport(vehicle, bytesMsg) : Parse REPORT message, validate fields,
        update group member state.

**LEADER-REPORT Message Operations (Leader→Follower)**
    writeLeaderReport(vehicle) : Serialize leader current state and predicted
        next waypoint into 57-byte LRPT message.
    sendLeaderReport(vehicle) : Transmit leader state to followers on
        broadcast channel.
    recvLeaderReport(vehicle, bytesMsg) : Parse leader report, validate
        fields, update target tracking state.

**BROADCAST/REQUEST Message Operations (TDMA_LF Swarm Coordination)**
    writeBroadcastReq(vehicle) : Serialize complete swarm state (leader +
        all followers) with response schedule into variable-length BCRQ
        message (65 + 32*N bytes).
    sendBroadcastReq(vehicle) : Transmit swarm broadcast from leader,
        reset follower response flags.
    recvBroadcastReq(vehicle, bytesMsg) : Parse swarm broadcast, validate
        multi-field structure, update target and group member states,
        compute response timing.

**Message Transmission Scheduling**
    schedulerTdmaLF(vehicle) : Compute and apply TDMA leader-follower
        transmission and response slots for group-synchronized messaging in
        TDMA_LF protocol.
    schedulerFdmaLF(vehicle) : Compute response and coordination timing for
        frequency-division (FDMA) leader-follower messaging.

**Position/Velocity Logging Utilities**
    writeEtaVelLogs(vehicle) : Append current position/velocity/timestamp
        to vehicle log.
    resizeLog(logArray, logSize, newCap) : Truncate or expand a position /
        velocity log to a target capacity, maintaining chronological order.
    readLastEtaLogs(model, n) : Retrieve the last n records of position/attitude
        (eta) history from a vehicle or model's log.
    readLastVelLogs(model, n) : Retrieve the last n records of velocity history
        from a vehicle or model's log.

**Data Validation and Recovery**
    dataIsCorrupted(data, field, kwargs) : Validate message field using
        type checks, range validation, and structural verification specific
        to field type.
    restoreCorrupt(vehicle, field, data, kwargs) : Attempt recovery of
        corrupted field using fallback strategies (logs, predictions,
        vehicle state, defaults).
    recoverPredict(vehicle, field, data) : Recover corrupted field using known
        data to compute estimates.
    recoverPrevious(vehicle, field, data) : Recover corrupted field using the
        last known valid value for that attribute.
    recoverNeutral(vehicle, field, data) : Recover corrupted field by replacing
        with a neutral / benign default.
    recoverSentinel(vehicle, field, data) : Replace corrupted field with a clear
        outlier or sentinel value to flag for further handling.
    recoverSkipField(vehicle, field, data) : Skip data recovery attempt for
        corrupted field, proceeding with rest of message.
    recoverSkip(vehicle, field, data) : Skip data recovery, rejecting entire
        message.

Notes
-----
**Communications Role in Simulation Architecture:**

The Communication block operates alongside Guidance, Navigation, and Control
blocks in the vehicle autonomy stack:

1. **Guidance Block** -> Computes desired trajectories and generates state
   change commands for formation control, path following, or target tracking.

2. **Navigation Block** -> Estimates vehicle state (position, attitude,
   velocity) from sensors and integrates kinematics for dead reckoning.

3. **Control Block** -> Generates actuator commands (rudder, stern plane,
   propeller) to track guidance references while maintaining stability.

4. **Communication Block** -> Enables information sharing between vehicles for
   cooperative behaviors (formation flying, swarm intelligence, distributed
   sensing) and coordination with command stations.

**Network Simulation Scope:**

Application layer functionality (message serialization) is provided by functions
in the Communication module, while the MAC layer is simulated by a combination
of functions and MuNet methods, and the physical layer (propagation, errors,
collisions) is simulated by MuNet alone:

- **Application Layer**: (module functions) Message serialization and parsing.
  Three message types carry vehicle state and coordination data with binary
  serialization via Python construct library.
  
- **MAC Layer**: (split implementation)

    - **Scheduler functions**: Message transmissing timing control. 
    - **MuNet class**: Channel allocation, node assignment, and access protocol
      infrastructure.
      
    Four access protocols manage channel allocation and scheduling:
    shared (single broadcast), TDMA (time division), TDMA_LF (leader-follower
    two-channel), FDMA (frequency division with dedicated channels).
  
- **Physical Layer**: Acoustic propagation modeled with distance-dependent
  travel time (speed of sound in water), packet jitter (environmental
  variability), packet loss (range and frequency dependent), bit errors
  (SNR or distance based), and collision effects (packet drop or XOR
  corruption).

**Message Types and Structure:**

Three binary message formats defined via construct library:

**REPORT (RPRT) - 37 bytes:**

Standard vehicle state broadcast for group awareness.

.. code-block:: none

    Field              Type       Bytes   Description
    -----              ----       -----   -----------
    type               bytes(4)   4       Message flag: b'RPRT' or b'RSPN'
    group_id           bytes(1)   1       Group identifier (ASCII char)
    reporter_id        uint32     4       Sender vehicle ID
    reporter_pos       float32[3] 12      Position [x, y, z] meters
    reporter_vel       float32[3] 12      Velocity [vx, vy, vz] m/s
    time_stamp         float32    4       Transmission time (seconds)

**LEADER-REPORT (LRPT) - 57 bytes:**

Leader state with predictive waypoint for follower coordination.

.. code-block:: none

    Field              Type       Bytes   Description
    -----              ----       -----   -----------
    type               bytes(4)   4       Message flag: b'LRPT'
    group_id           bytes(1)   1       Group identifier
    leader_pos         float32[3] 12      Current position
    leader_vel         float32[3] 12      Current velocity
    leader_next_pos    float32[3] 12      Next waypoint position
    leader_next_vel    float32[3] 12      Next waypoint velocity
    time_stamp         float32    4       Transmission time

**BROADCAST/REQUEST (BCRQ) - Variable:**

Complete swarm state with response scheduling for TDMA_LF.

.. code-block:: none

    Field              Type         Bytes       Description
    -----              ----         -----       -----------
    type               bytes(4)     4           Message flag: b'BCRQ'
    group_id           bytes(1)     1           Group identifier
    leader_pos         float32[3]   12          Leader current position
    leader_vel         float32[3]   12          Leader current velocity
    leader_next_pos    float32[3]   12          Leader next position
    leader_next_vel    float32[3]   12          Leader next velocity
    num_followers      uint32       4           Array size N
    followers_pos      float32[3*N] 12*N        Follower positions
    followers_vel      float32[3*N] 12*N        Follower velocities
    followers_rspn     float32[N]   4*N         Last message times
    rspn_sched         uint32[N]    4*N         Response order (IDs)
    rspn_start         float32      4           Schedule start time
    time_stamp         float32      4           Transmission time

    Total Size: 65 + 32*N bytes
    Example (9 followers): 65 + 288 = 353 bytes

**MAC Protocols and Channel Architecture:**

**Shared ('shared'):**

- Single broadcast channel at BASE_FREQ
- All nodes TX and RX on same channel
- Simplest protocol, highest collision risk

**TDMA ('tdma'):**

- Single channel using full BANDWIDTH
- Time-domain multiplexing via protocol schedulers
- Nodes transmit in assigned time slots
- Eliminates collisions via time separation
- Requires precise clock synchronization

**TDMA Leader-Follower ('tdma_lf'):**

- Two-channel architecture optimized for hierarchical swarms
- Broadcast channel (BCAST_ADDR): Leader -> All Followers
- Response channel (next available): All Followers -> Leader
- Leader sends BCRQ with response schedule
- Followers respond in assigned time slots
- Efficient for centralized coordination (1 leader + N followers)

**FDMA ('fdma'):**

- Frequency-division with dedicated TX channel per node
- Each node: TX on unique channel, RX on all others
- Maximum capacity:
    CHAN_MAXN = (BANDWIDTH - 2*GUARD_BW)/(CHAN_MINBW + GUARD_BW)
- Eliminates collisions via frequency separation
- Scalability to larger swarms is limited by bandwidth
- Channel allocation strategies: center_out (symmetric from BASE_FREQ) or
  edge_in (sequential from lowest frequency)

**Physical Layer Error Models:**

**Jitter (Packet Delay Variability):**

- Uniform: Random delay [0, MAX_JITTER] seconds
- Models environmental propagation variability
- Affects message ordering at receiver

**Packet Loss Rate (PLR):**

- Uniform: Constant drop probability PLR
- Exponential: Distance-based PLR = PLR * exp((d - d_nom)/d_char) * f_factor
- Sonar Simple: PLR = PLR * (d/d_nom)^1.5 * exp((d - d_nom)/d_char) * f_factor
- Frequency correction: +1% loss per kHz above BASE_FREQ

**Bit Error Rate (BER):**

- Poisson: Constant BER, error count ~ Poisson(totalBits * BER)
- Exponential: Distance-based BER = BER * exp((d - d_nom)/d_char)
- SNR: SNR = SNR_0 - 15*log10(d/d_nom), BER = BER * 10^(slope*(SNR_0 - SNR))
- Random bit positions flipped via XOR

**Collision Handling:**

- Drop: All overlapping packets removed from queue
- Corrupt: XOR of overlapping byte regions (additive interference model)
- Collision detection via per-node active RX dictionary tracking ongoing
  receptions by channel

**Queue Architecture and Message Flow:**

**TX Queue (Transmission):**

- Min-heap priority queue sorted by delivery start/end time
- Entries created during transmit() for each receiver
- Metadata: start_time, end_time, src_addr, dest_addr, cid, freq, bw,
  distance, message
- Sorting:

    - Collision disabled: heap sorted by end_time (delivery completion)
    - Collision enabled: heap sorted by start_time (delivery initiation)

**RX Queue (Active Reception):**

- Min-heap sorted by end_time (delivery completion)
- Used only when collision detection enabled
- Messages move TX->RX when reaching start_time
- Active RX dictionary (rxd) tracks overlapping receptions per channel

**Delivery Flow:**

1. Application: vehicle.CommNetwork.send(message, address)
2. Network.transmit(): Compute times, distances -> enqueue to TXQ
3. Simulation loop: Network.deliver(time) called each timestep
4. deliver() dispatcher:

   a. Collision disabled: _rxNoColl

      - Pop from TXQ when time ≥ end_time
      - Apply PLR -> Apply BER -> Deliver to node.recv()

   b. Collision enabled: _rxWithColl

      - Pop from TXQ when time ≥ start_time -> push to RXQ
      - Track in rxd (active RX dictionary)
      - Pop from RXQ when time ≥ end_time
      - Check collisions via rxd -> Apply PLR -> Apply BER -> Deliver

5. node.recv(): Call recvMsgCallback() dispatcher
6. recvMsgCallback(): Route by type flag to handler
7. Handler (recvReport, recvLeaderReport, recvBroadcastReq):

   - Parse message via getMsgStruct()
   - Validate all fields via dataIsCorrupted()
   - Attempt recovery via restoreCorrupt() if corruption detected
   - Update vehicle state (group, target, logs)

**Data Validation and Corruption Recovery:**

Underwater acoustic network settings allow simulating high error rates (BER,
packet collision corruption) requiring validation and recovery:

**Validation Strategy:**

- Type-specific checks for each message field
- Range validation (positions within 10E6 m, velocities within 100 m/s)
- Structural verification (array lengths, element types, finite values)

**Recovery Strategies:**

- Configurable per vehicle and per message field
- Uses a sequential fallback to allow multiple recovery attempts using a
  preference heirarchy
- Strategies include:

    - Predict: use available known data to estimate corrupted data
    - Previous: use previous known data to replace corrupted data
    - Neutral: use harmless neutral data to replace corrupted data
    - Sentinel: use obvious bad value to replace corrupted data
    - Skip Field: skip only corrupted message field
    - Skip: skip entire message

**Design Philosophy:**

The module uses strategy pattern for dynamic assignment of:

- Protocol handlers (channel builders, node assigners)
- Channel allocation generators (center-out, edge-in)
- Error models (PLR, BER, collision, jitter)


See Also
--------
vehicles.Vehicle : Vehicle state and dynamics
vehicles.Model : Data model for vehicle state
guidance.py : Formation control and path following
navigation.py : State estimation and sensor fusion
control.py : Actuator command generation
"""

from __future__ import annotations
from threading import Thread
from dataclasses import dataclass
from collections.abc import Generator
from typing import Any, List, Optional, TYPE_CHECKING, Tuple, Union
from numpy.typing import NDArray
if (TYPE_CHECKING):
    from munetauvsim.vehicles import Vehicle, Model
import construct as cst
import numpy as np
import heapq
from functools import lru_cache
from munetauvsim import navigation as nav
from munetauvsim import guidance as guid
from munetauvsim import logger

#-----------------------------------------------------------------------------#

# Type Aliases
NPFltArr = NDArray[np.float64]

# Global Variables
log = logger.setupComm(file=False)

###############################################################################

@dataclass
class MuNetQEntry:
    """
    Message entry for transmission/reception queue in network simulation.

    Container dataclass holding network metadata and message payload for
    tracking acoustic transmissions through TX/RX queues. Metadata used for
    network simulation and error model calculations; only payload transmitted as
    acoustic data.

    Attributes
    ----------
    start_time : float
        Time when message transmission begins reaching receiver.
    end_time : float
        Time when message transmission ends at receiver.
    src_addr : int
        Source node address.
    dest_addr : int
        Destination node address.
    cid : int
        Channel ID carrying transmission.
    freq : float
        Channel center frequency in kHz.
    bw : float
        Channel bandwidth in kHz.
    distance : float
        Distance from source to destination in meters (computed at creation).
    message : bytearray
        Message payload (this is the only part carried as acoustic data).

    Notes
    -----
    Heap queue ordering maintained by tuple (sort_time, qid):

    - TX queue with collision disabled: sorts by end_time
    - TX queue with collision enabled: sorts by start_time
    - RX queue: sorts by end_time

    Dataclass uses __slots__ for memory efficiency in large message volumes.
    """

    # Define slots
    __slots__ = ('start_time', 'end_time', 
                 'src_addr', 'dest_addr',
                 'cid', 'freq', 'bw', 
                 'distance', 'message',)
    
    # Queue Entry Fields
    start_time: float
    end_time: float
    src_addr: int
    dest_addr: int
    cid: int
    freq: float
    bw: float
    distance: float
    message: bytearray

###############################################################################

@dataclass
class MuChannel:
    """
    Network channel for acoustic transmission.

    Attributes
    ----------
    cid : int
        Unique channel identifier.
    freq : float
        Center frequency in kHz.
    bw : float
        Channel bandwidth in kHz.
    isBcst : bool
        True if broadcast channel (all nodes RX).
    """

    __slots__ = ('cid', 'freq', 'bw', 'isBcst')

    cid: int
    freq: float
    bw: float
    isBcst: bool

###############################################################################

class MuNode:
    """
    Network node representing vehicle connection to acoustic network.

    Abstracts vehicle communication interface with TX/RX channel assignments,
    active reception tracking for collision detection, and message send/receive
    methods.

    Attributes
    ----------
    address : int
        Unique node address on network.
    vehicle : Vehicle
        Vehicle object associated with this node.
    network : MuNet
        Reference to MuNet network instance.
    rxd : dict
        Active reception dictionary tracking overlapping transmissions by
        channel for collision detection. Maps channel_id -> [queue_entries].
    tx_channels : set
        Channel IDs available for transmission.
    rx_channels : set
        Channel IDs tuned for reception.

    Methods
    -------
    send(message, cid)
        Transmit message on specified channel.
    recv(message, cid)
        Receive message from specified channel.

    Notes
    -----
    The rxd (receive dictionary) tracks active message receptions by channel.
    When message completes reception, it is removed from rxd. If two messages
    overlap in time and channel, collision detection occurs. This design
    supports multi-channel collision detection without global search.
    """

    ## Constructor ===========================================================#
    def __init__(self, address:int, vehicle:Vehicle, network:MuNet)->None:
        """
        Initialize network node for vehicle communication interface.
        
        Parameters
        ----------
        address : int
            Unique node address on network.
        vehicle : Vehicle
            Vehicle object associated with this node.
        network : MuNet
            Reference to parent MuNet network instance.
        
        Notes
        -----
        Initializes empty rxd (active reception dictionary) and empty TX/RX
        channel sets. Channels assigned later via network._assignNodes().
        """
        
        self.address = address
        self.vehicle = vehicle
        self.network = network
        self.rxd = {}
        self.tx_channels = set()
        self.rx_channels = set()

    ## Methods ===============================================================#
    def send(self, message:bytes, cid:int)->None:
        """
        Transmit message from node on specified channel.
    
        Forwards message to network transmission queue via network.transmit().
        Logs transmission event with channel details.
        
        Parameters
        ----------
        message : bytes
            Outgoing message payload.
        cid : int
            Transmitting channel ID from node.tx_channels.
        
        Notes
        -----
        **Acoustic Network Limitation:**
        
        Acoustic network does not include physical layer for automatic sender
        identification. Senders must package source identity in message payload
        if receiver needs to identify sender.
        
        **Logging:**
        
        Logs transmission at INFO level with format:
        
        {sender_addr:channel_id} (cid|freq) message_bytes
        
        Example: `{003:__1} (1|24.50) b'RPRT...'`
        """
        
        freq = self.network.channels[cid].freq
        ch = str(cid).rjust(3,'_')
        log.info('{%03d:%s} (%d|%.2f) %s', self.address, ch, cid, freq, 
                 message)
        self.network.transmit(message, cid, self.address)
    
    #--------------------------------------------------------------------------
    def recv(self, message:bytes, cid:int)->None:
        """
        Receive message at node and route to vehicle message handler.
    
        Logs reception event and invokes recvMsgCallback() dispatcher to route
        message to appropriate handler based on message type flag.
        
        Parameters
        ----------
        message : bytes
            Incoming message payload.
        cid : int
            Receiving channel ID from node.rx_channels.
        
        Notes
        -----
        **Acoustic Network Limitation:**
        
        Receiver does not automatically know sender address unless packaged in
        message payload, or if sender is on dedicated channel (as in FDMA where
        channel ID implies sender).
        
        **Message Routing:**
        
        Calls recvMsgCallback(message, self.vehicle) which:
        1. Extracts 4-byte type flag from message header
        2. Validates type flag for corruption
        3. Routes to handler: recvReport, recvLeaderReport, or recvBroadcastReq
        
        **Logging:**
        
        Logs reception at INFO level with format:
        
        {channel_id:receiver_addr} (cid|freq) message_bytes
        
        Example: `{__1:003} (1|24.50) b'RPRT...'`
        
        **Error Handling:**
        
        Exceptions during message handling caught and logged by
        recvMsgCallback().
        """

        freq = self.network.channels[cid].freq
        ch = str(cid).rjust(3,'_')
        log.info('{%s:%03d} (%d|%.2f) %s', ch, self.address, cid, freq,
                 message)
        recvMsgCallback(message, self.vehicle)

###############################################################################

class MuNet:
    """
    Underwater acoustic network simulator with discrete-event message delivery.
    
    Simulates vehicle-to-vehicle acoustic communication including Medium Access
    Control (MAC) protocols, physical layer transmission characteristics, such
    as jitter, and error models, such as packet loss, bit errors, and
    collisions. Provides infrastructure for cooperative vehicle behaviors
    requiring information exchange in underwater swarm robotics.
    
    The simulator uses a priority queue-based message scheduling, dynamic
    strategy assignment for protocol and error model selection, and per-node
    channel management for frequency-division and time-division multiple access.

    Attributes
    ----------
    **Physical Layer Parameters:**
    
    C_WATER : float, default=1500.0
        Speed of sound in water (m/s). Determines acoustic propagation delay
        between vehicles. Constant approximation valid for shallow depths
        (<100m). For depth-dependent modeling, incorporate Mackenzie equation.
    
    BASE_FREQ : float, default=24.0
        Network base (center) acoustic frequency (kHz). Symmetric frequency
        allocation strategies (center_out) place first channel here. Typical
        underwater acoustic modems operate 10-30 kHz for medium-range
        communication.
    
    BANDWIDTH : float, default=10.0
        Total available bandwidth for all channels (kHz). Determines maximum
        number of frequency-division channels. Higher bandwidth supports more
        simultaneous channels but may increase interference.
    
    GUARD_BW : float, default=0.25
        Guard bandwidth between adjacent channels and at band edges (kHz).
        Prevents inter-channel interference from frequency drift and imperfect
        filtering. Larger guards reduce usable bandwidth but improve isolation.

    CHAN_MAXN : see Properties.
    
    CHAN_MINBW : float, default=1.0
        Minimum channel bandwidth for data transmission (kHz). With guard bands,
        determines channel spacing. Narrower channels support more frequency
        division but may limit data rate.
    
    BCAST_ADDR : int, default=255
        Reserved address for broadcast channel in shared and TDMA protocols.
        Messages sent to this address received by all nodes listening on
        broadcast channel.
    
    DATA_RATE : int, default=1000
        Channel data transmission rate (bits per second). Determines message
        transmission time: t_data = (message_bytes * 8) / DATA_RATE. Typical
        acoustic modems: 100-5000 bps depending on range and quality.
    
    MAX_JITTER : float, default=0.2
        Maximum random packet delay added to transmission (seconds). Simulates
        environmental propagation variability from temperature gradients,
        salinity variations, surface reflections. Uniform distribution [0,
        MAX_JITTER].
    
    NOM_DIST : float, default=1000.0
        Nominal reference distance for distance-based error models (meters).
        At this range, computed PLR/BER equals base value. Shorter distances
        see improved performance; longer distances see degradation.
    
    PLR : float, default=0.02
        Base Packet Loss Ratio in range [0.0, 1.0]. Fraction of packets randomly
        dropped before delivery. Default 2% represents moderate acoustic
        channel. Actual loss may be distance-dependent based on plrType
        strategy.
    
    PLR_CHARDIST : float, default=2000.0
        Characteristic distance for exponential PLR attenuation (meters).
        Controls rate of PLR increase beyond nominal distance in distance-based
        models. Smaller values -> faster degradation with range.
    
    BER : float, default=1e-3
        Base Bit Error Rate (errors per bit). Probability of random bit flip
        in message payload. Typical ranges: Poor 1e-1 to 1e-2, Moderate 1e-3
        to 1e-4, Good 1e-5 to 1e-6. Generated via Poisson distribution.
    
    BER_CHARDIST : float, default=3000.0
        Characteristic distance for exponential BER attenuation (meters).
        Controls rate of BER increase beyond nominal distance. Smaller values
        -> faster bit error degradation with range.
    
    SNR_NOMINAL : float, default=30.0
        Nominal Signal-to-Noise Ratio for SNR-based BER models (decibels).
        Reference SNR where BER equals base value. Path loss reduces SNR with
        distance using simplified spreading model.
    
    SNR_SLOPE : float, default=0.1
        Slope factor for SNR-based BER models (dimensionless). Controls BER
        sensitivity to SNR variations. Higher slope -> faster BER increase as
        SNR degrades.

    **MAC Protocol Configuration:**
    
    accessMode : {'shared', 'tdma', 'tdma_lf', 'fdma'}, default='fdma'
        Medium Access Control protocol for channel allocation and node
        assignment. Determines how vehicles share acoustic bandwidth:
        
        - 'shared': Single broadcast channel, all nodes TX/RX simultaneously.
          Simplest but highest collision risk.
        
        - 'tdma': Time-Division Multiple Access. Single channel using full
          bandwidth. Nodes transmit in assigned time slots. Eliminates
          collisions via temporal separation. Requires clock synchronization
          and scheduling coordination.
        
        - 'tdma_lf': TDMA Leader-Follower variant. Two-channel architecture:
          broadcast for leader->followers, dedicated response for followers->
          leader. Optimized for hierarchical swarms (1 leader + N followers).
          Efficient coordinated communication with response scheduling.
        
        - 'fdma': Frequency-Division Multiple Access. Dedicated TX channel per
          node, RX on all others. Eliminates collisions via frequency
          separation. Scalable to larger swarms limited by bandwidth capacity.
    
    orderMode : {'center_out', 'edge_in'}, default='center_out'
        Channel frequency allocation strategy in available bandwidth:
        
        - 'center_out': Symmetric allocation from BASE_FREQ. First channel at
          center, then alternates ±offset pairs expanding outward. Balances
          frequency distribution, minimizes worst-case interference.
        
        - 'edge_in': Sequential allocation from lowest frequency. Fills from
          band edge upward by lane increments. Simplifies allocation but may
          create uneven frequency usage.

    **Error Model Configuration:**

    jitterType : {'uniform', 'off'}, default='uniform'
        Packet delay variability strategy:
        
        - 'uniform': Random delay [0, MAX_JITTER] from uniform distribution.
          Models propagation variability from environmental effects.
        
        - 'off': Zero jitter. Packets arrive at deterministic propagation
          time. For idealized testing.
    
    collType : {'drop', 'corrupt', 'off'}, default='drop'
        Collision handling strategy when messages overlap in time/channel:
        
        - 'drop': All colliding packets removed from queue. Conservative
          approach simulating complete signal destruction from interference.
        
        - 'corrupt': XOR overlapping byte regions bidirectionally. Models
          additive interference destroying signals in overlap window. Packets
          delivered but corrupted.
        
        - 'off': No collision detection. Messages delivered independently
          even if overlapping. For capacity testing without interference.
    
    plrType : {'uniform', 'exp_simple', 'sonar_simple', 'off'}, default='sonar_simple'
        Packet Loss Rate strategy:
        
        - 'uniform': Constant PLR independent of distance. Simple baseline.
        
        - 'exp_simple': Distance-based exponential attenuation:
          PLR_d = PLR * exp((d - d_nom)/d_char) * (1 + (f - f_0)*0.01)
          Includes frequency correction (+1% per kHz above BASE_FREQ).
        
        - 'sonar_simple': Sonar equation approximation with geometric spreading
          and absorption:
          PLR_d = PLR * (d/d_nom)^1.5 * exp((d - d_nom)/d_char) * freq_factor
          Uses practical spreading exponent (1.5) between spherical (2) and
          cylindrical (1). More realistic for underwater acoustics.
        
        - 'off': No packet loss. All messages delivered (subject to collisions).
    
    berType : {'poisson', 'exp_simple', 'snr', 'off'}, default='poisson'
        Bit Error Rate strategy:
        
        - 'poisson': Constant BER. Error count sampled from Poisson(total_bits
          * BER). Random bit positions flipped via XOR. Distance-independent
          baseline.
        
        - 'exp_simple': Distance-based exponential BER:
          BER_d = BER * exp((d - d_nom)/d_char)
          Error count from Poisson(total_bits * BER_d).
        
        - 'snr': SNR-based BER using path loss model:
          SNR = SNR_0 - 15*log10(d/d_nom)
          BER_SNR = BER * 10^(slope*(SNR_0 - SNR))
          More realistic signal quality degradation with distance.
        
        - 'off': No bit errors. Messages delivered uncorrupted (subject to
          packet loss and collisions).
    
    seed : int, optional
        Random seed for NumPy RNG. Enables reproducible simulations with
        deterministic error patterns. If None, uses system entropy for
        non-reproducible runs.

    **Data Structures:**
    
    channels : dict
        Network channel registry. Key: channel ID (int), Value: MuChannel
        dataclass with (cid, freq, bw, isBcst). Populated during channel
        building phase based on accessMode.
    
    nodes : dict
        Registered network nodes. Key: node address (int), Value: MuNode
        object with vehicle reference, TX/RX channel sets, active reception
        dictionary. Populated during vehicle registration.
    
    txq : list
        Transmission queue (min-heap). Entries: (sort_time, qid, MuNetQEntry).
        Sorted by start_time (collision enabled) or end_time (collision
        disabled). Messages moved to txq during transmit(), popped during
        deliver().
    
    rxq : list
        Active reception queue (min-heap). Used only when collision detection
        enabled. Entries: (end_time, qid, MuNetQEntry). Messages moved from
        txq when reaching start_time, tracked in node.rxd for collision
        detection.
    
    rng : numpy.random.Generator
        NumPy random number generator instance. Initialized with seed parameter.
        Used for all stochastic operations (jitter, PLR, BER, bit positions).

    **Statistics Tracking:**

    stats : dict
        Network performance counters updated during message flow:
        
        - packetSent : int
            Total packets enqueued to txq during transmit() calls.
        
        - packetDelivered : int
            Packets successfully delivered to node.recv() after all error
            models.
        
        - packetDropPLR : int
            Packets dropped by Packet Loss Rate strategy.
        
        - packetDropColl : int
            Packets dropped by collision strategy (collType='drop' only).
        
        - packetFailedDel : int
            Packets that triggered exceptions during node.recv() callback.
        
        - inTransitTX : int
            Messages currently in txq (updated by calcStats()).
        
        - inTransitRX : int
            Messages currently in rxq (updated by calcStats()).
        
        - plrActual : float
            Actual PLR: packetDropPLR / packetSent (computed by calcStats()).
        
        - plrDeliveryRate : float
            Delivery rate: packetDelivered / packetSent (computed by
            calcStats()).
        
        - collDropRate : float
            Collision drop rate: packetDropColl / packetSent (computed by
            calcStats(), collType='drop' only).
        
        - inTransitRate : float
            In-transit ratio: (inTransitTX + inTransitRX) / packetSent (computed
            by calcStats()).
        
        - collisions : int
            Collision events detected (overlapping messages on same channel).
        
        - packetCorruptColl : int
            Packets corrupted by collision XOR (collType='corrupt' only).
        
        - packetBER : int
            Packets modified by bit error injection.
        
        - bitErrors : int
            Total individual bit flips applied across all messages.

    **Dynamically Assigned Methods:**

    The following methods are assigned during __init__ based on configuration,
    implementing strategy pattern for protocol and error model selection:
    
    _buildChannels : callable
        Channel builder for accessMode: _buildChannelsShared,
        _buildChannelsTDMA, _buildChannelsTDMALF, or _buildChannelsFDMA.
    
    _assignNodes : callable
        Node channel assigner for accessMode: _assignNodesShared,
        _assignNodesTDMA, _assignNodesTDMALF, or _assignNodesFDMA.
    
    _genChannel : callable
        Channel generator for orderMode: _genChannelCenterOut or
        _genChannelEdgeIn.
    
    _channelGen : Generator
        Active channel generator instance created from _genChannel.
    
    _generateJitter : callable
        Jitter generator for jitterType: _jitterUniform or _jitterDisabled.
    
    _applyColl : callable
        Collision handler for collType: _collDrop, _collCorrupt, or
        pass-through.
    
    _txEnqueue : callable
        TX queue enqueuer for collType: _txWithColl or _txNoColl.
    
    deliver : callable
        Message delivery dispatcher for collType: _rxWithColl or _rxNoColl.
    
    _applyPLR : callable
        PLR strategy for plrType: _plrUniform, _plrExpSimple, _plrSonarSimple,
        or _plrDisabled.
    
    _applyBER : callable
        BER strategy for berType: _berPoisson, _berExpSimple, _berSNR, or
        _berDisabled.

    Methods
    -------
    **Public Interface:**
    
    register(vehicle) : MuNode
        Create network node for vehicle with channel assignment. Returns MuNode
        with TX/RX channels configured per accessMode.
    
    transmit(message, cid, srcAddr) : None
        Queue message for transmission to all receivers on channel. Computes
        propagation times and enqueues to txq.
    
    deliver(time) : None
        Process message queues at simulation time. Dynamically assigned to
        _rxNoColl or _rxWithColl based on collision configuration.
    
    calcStats() : None
        Compute network performance statistics from raw counters. Updates
        derived metrics (delivery rate, PLR, collision rate).
    
    getStatsReport() : str
        Return formatted multi-line statistics report. Calls calcStats() first.
    
    **Queue Management:**
    
    _txEnqueue(rxTime, dataTime, srcAddr, rNode, channel, distances, message) : None
        Enqueue messages to txq. Dynamically assigned to _txNoColl or
        _txWithColl.
    
    _txNoColl(...) : None
        Enqueue without collision tracking. Sorts by end_time.
    
    _txWithColl(...) : None
        Enqueue with collision tracking. Sorts by start_time.
    
    _rxNoColl(time) : None
        Deliver messages without collision detection. Pop from txq, apply
        PLR/BER, deliver to node.
    
    _rxWithColl(time) : None
        Deliver with collision detection. Two-phase: txq->rxq (populate
        node.rxd), rxq->deliver (check collisions, apply PLR/BER).
    
    _nextQid() : int
        Return next sequential queue message ID for heap tie-breaking.
    
    _getChannelRXNodes(cid) : list[MuNode]
        Return nodes registered to receive on channel ID.
    
    **Error Models:**
    
    _generateJitter(size) : ndarray
        Generate jitter delays. Assigned to _jitterUniform or _jitterDisabled.
    
    _jitterUniform(size) : ndarray
        Uniform random delays [0, MAX_JITTER].
    
    _jitterDisabled(size) : ndarray
        Zero delays (array of zeros).
    
    _applyColl(entry, rNode) : bool
        Check/handle collisions. Assigned to _collDrop, _collCorrupt, or
        pass-through.
    
    _collDrop(entry, rNode) : bool
        Drop all colliding packets from rxq.
    
    _collCorrupt(entry, rNode) : bool
        XOR overlapping byte regions of colliding packets.
    
    _applyPLR(entry) : bool
        Test for packet loss. Assigned to _plrUniform, _plrExpSimple,
        _plrSonarSimple, or _plrDisabled.
    
    _plrUniform(entry) : bool
        Constant PLR test (random < PLR).
    
    _plrExpSimple(entry) : bool
        Distance-based exponential PLR.
    
    _plrSonarSimple(entry) : bool
        Sonar equation PLR with spreading and absorption.
    
    _plrDisabled(entry) : bool
        No packet loss (always False).
    
    _applyBER(entry) : None
        Inject bit errors. Assigned to _berPoisson, _berExpSimple, _berSNR,
        or _berDisabled.
    
    _berPoisson(entry) : None
        Constant BER with Poisson error count.
    
    _berExpSimple(entry) : None
        Distance-based exponential BER.
    
    _berSNR(entry) : None
        SNR-based BER with path loss model.
    
    _berDisabled(entry) : None
        No bit errors (pass-through).
    
    _flipBits(msg, bitPos) : None
        Flip specified bit positions in bytearray via XOR.
    
    **Channel Management:**
    
    _genChannel(cid, freq, bw, isBcst) : Generator[MuChannel]
        Generate channels per allocation strategy. Assigned to
        _genChannelCenterOut or _genChannelEdgeIn.
    
    _genChannelCenterOut(...) : Generator[MuChannel]
        Center-outward allocation from BASE_FREQ.
    
    _genChannelEdgeIn(...) : Generator[MuChannel]
        Edge-inward allocation from lowest frequency.
    
    _getChannelCenterLimits() : tuple
        Compute frequency bounds for channel centers. Returns (center, fmin,
        fmax, lane, chan_bw).
    
    _makeBroadcastChannel(cid, freq, bw) : MuChannel
        Create broadcast channel with optional parameter overrides.
    
    _buildChannels() : None
        Build channel registry. Assigned to _buildChannelsShared,
        _buildChannelsTDMA, _buildChannelsTDMALF, or _buildChannelsFDMA.
    
    _buildChannelsShared() : None
        Single broadcast channel for all nodes.
    
    _buildChannelsTDMA() : None
        Single TDMA channel using full bandwidth.
    
    _buildChannelsTDMALF() : None
        Two channels for leader-follower TDMA.
    
    _buildChannelsFDMA() : None
        Multiple channels for frequency division.
    
    _assignNodes(node) : str
        Assign TX/RX channels to node. Assigned to _assignNodesShared,
        _assignNodesTDMA, _assignNodesTDMALF, or _assignNodesFDMA.
    
    _assignNodesShared(node) : str
        Assign to broadcast channel.
    
    _assignNodesTDMA(node) : str
        Assign to TDMA channel.
    
    _assignNodesTDMALF(node) : str
        Assign based on leader/follower role.
    
    _assignNodesFDMA(node) : str
        Assign dedicated TX channel, RX on all others.
    
    _updateNodeRXFDMA() : None
        Update all nodes' RX channels when new TX channel added in FDMA.

    Properties
    ----------
    CHAN_MAXN : int
        Maximum channels in available bandwidth (read-only). Formula:
        (BANDWIDTH - 2*GUARD_BW) // (CHAN_MINBW + GUARD_BW). Determines FDMA
        capacity limit.
    
    Notes
    ------
    **Discrete-Event Simulation Architecture:**
    
    MuNet operates as a discrete-event simulator with two priority queues:
    
    1. **Transmitting Queue (txq)**: Messages pending transmission delivery

       - Min-heap sorted by start_time (collision enabled) or end_time 
         (collision disabled)
       - Entries: (sort_time, qid, MuNetQEntry)
       - Populated during transmit()
       - Drained during deliver()
    
    2. **Receiving Queue (rxq)**: Active message receptions (collision mode
       only)

       - Min-heap sorted by end_time
       - Entries: (end_time, qid, MuNetQEntry)
       - Populated from txq when time ≥ start_time
       - Drained when time ≥ end_time
       - Enables collision detection via node.rxd dictionary
    
    **Message Flow:**
    
    1. **Application** -> vehicle.CommNetwork.send(message, address)
    2. **Network.transmit()** -> Compute times/distances -> Enqueue to txq
    3. **Simulation loop** -> Network.deliver(time) each timestep
    4. **deliver() dispatcher**:

       - Collision disabled (_rxNoColl): 
       
         * txq -> Apply PLR -> Apply BER -> Deliver

       - Collision enabled (_rxWithColl):

         * Phase 1: txq -> rxq (populate node.rxd)
         * Phase 2: rxq -> Check collisions -> Apply PLR -> Apply BER -> Deliver

    5. **node.recv()** -> recvMsgCallback() -> Message handler
    6. **Message handler** -> Parse -> Validate -> Update vehicle state
    
    **Strategy Pattern Design:**
    
    Dynamic method assignment during __init__ enables protocol and error model
    selection without conditional branching during message flow:
    
    - MAC protocol strategies (4 variants)
    - Channel allocation strategies (2 variants)
    - Error model strategies (2 jitter * 3 collision * 4 PLR * 4 BER = 96
      combinations)
    
    Total configuration space: 768 unique network configurations.
    
    **Network Capacity Limits:**
    
    - Shared/TDMA: Unlimited nodes (single channel)
    - TDMA_LF: 1 leader + unlimited followers (two channels)
    - FDMA: Max nodes = CHAN_MAXN (typically 8-20 for 10 kHz bandwidth)
    

    See Also
    --------
    MuNode : Network node representing vehicle communication interface
    MuNetQEntry : Queue entry dataclass for message tracking
    MuChannel : Channel specification dataclass
    vehicles.Vehicle : Vehicle state and dynamics

    Examples
    --------
    >>> # Network creation
    >>> network = MuNet(
    ...     accessMode='fdma',
    ...     plrType='sonar_simple',
    ...     berType='snr',
    ...     collType='corrupt',
    ...     seed=42
    ... )
    
    >>> # Vehicle registration
    >>> vehicle.CommNetwork = network
    >>> vehicle.commNode = network.register(vehicle)
    
    >>> # Periodic transmission (in vehicle update loop)
    >>> if vehicle.clock % vehicle.reportInterval == 0:
    ...     sendReport(vehicle)
    
    >>> # Network delivery (in main simulation loop)
    >>> for timestep in simulation:
    >>>     # ... vehicle dynamics, guidance, control ...
    ...     network.deliver(time)  # Process queued messages
    >>>     # ... update visualizations ...
    
    >>> # Statistics
    >>> network.calcStats()
    >>> print(network.getStatsReport())
    """

    ## Class Attributes ======================================================# 
    __qid = 0                           # sequential queue message id counter

    ## Constructor ===========================================================#
    def __init__(self, **kwargs)->None:
        """
        Initialize acoustic network with physical parameters, protocols, and
        error models.
    
        Configures all network simulation parameters, assigns strategy methods
        for MAC protocols and error models, initializes data structures, and
        logs network configuration. All parameters customizable via kwargs for
        rapid prototyping.
        
        Parameters
        ----------
        **kwargs : dict
            Optional keyword arguments override default parameter values.
            Supported keys correspond to instance attributes documented in class
            docstring.
        
            
        Notes
        -----
        **Parameter Categories:**
    
        **Physical Constants (7 parameters):**

        - C_WATER, BASE_FREQ, BANDWIDTH, GUARD_BW, CHAN_MINBW, BCAST_ADDR,
          DATA_RATE
        - Define acoustic propagation, frequency allocation, addressing
        
        **Error Model Parameters (8 parameters):**

        - MAX_JITTER, NOM_DIST, PLR, PLR_CHARDIST, BER, BER_CHARDIST,
          SNR_NOMINAL, SNR_SLOPE
        - Control packet loss, bit errors, propagation variability
        
        **Protocol Configuration (6 parameters):**

        - accessMode, orderMode, jitterType, collType, plrType, berType
        - Select MAC protocol and error model strategies
        
        **Reproducibility (1 parameter):**

        - seed
        - Enables deterministic simulation runs
        
        **Initialization Sequence:**
        
        1. **Set Default Parameters**
        Physical layer constants, error model parameters, protocol configuration.
        
        2. **Apply Keyword Overrides**
        Custom parameters from kwargs update defaults via __dict__.update().
        
        3. **Initialize Data Structures**
        Empty channel/node dictionaries, TX/RX queues, statistics counters.
        
        4. **Create RNG Instance**
        NumPy random generator with seed for reproducibility.
        
        5. **Assign Protocol Strategies**
        Dynamic method binding for MAC protocols (channel builders, node
        assigners).
        
        6. **Create Channel Generator**
        Instantiate generator from _genChannel strategy.
        
        7. **Assign Error Model Strategies**
        Dynamic method binding for jitter, collisions, PLR, BER.
        
        8. **Log Configuration**
        Write network parameters and strategy selections to log.
        
        **Strategy Pattern Implementation:**
        
        Instead of conditional logic during message flow, __init__ assigns
        concrete implementations to abstract method names based on
        configuration keywords:
    
        - **accessMode**: Assigns channel builder and node assigner method pairs
        - **orderMode**: Assigns frequency allocation generator method  
        - **jitterType**: Assigns packet delay generator method
        - **collType**: Assigns collision handler, queue enqueuer, and message
                        delivery methods
        - **plrType**: Assigns packet loss rate strategy method
        - **berType**: Assigns bit error rate strategy method

        This enables runtime polymorphism without performance overhead of
        conditional branching during network operation.
    
        **Statistics Initialization:**
    
        All statistics counters reset to zero:

            - 'packetSent', 'packetDelivered', 'packetDropPLR',
              'packetDropColl', 'packetFailedDel', 'inTransitTX', 'inTransitRX',
              'plrActual', 'plrDeliveryRate', 'collDropRate', 'inTransitRate',
              'collisions', 'packetCorruptColl', 'packetBER', 'bitErrors'
        
        Statistics updated during message processing. Call calcStats() to
        compute derived metrics before reading rates.
        
        **Channel Generator Initialization:**
        
        After strategy assignment, instantiates a channel generator. Generator
        yields MuChannel objects on demand during channel building. Removes need
        to pre-allocate channel list, supports lazy allocation in FDMA as nodes
        register.
        
        **Error Model Interaction:**
        
        Multiple error models applied sequentially during message delivery:
        
        1. **Collision detection** (if enabled): Check rxd for overlaps

            - If collType='drop': Remove all colliding packets
            - If collType='corrupt': XOR overlapping byte regions
        
        2. **Packet Loss**: Test with _applyPLR()

            - If dropped: Increment stats, skip delivery
        
        3. **Bit Errors**: Apply with _applyBER()

            - Flip random bit positions in message payload
        
        4. **Delivery**: Call node.recv() with possibly corrupted message
        
        Order matters: PLR prevents BER computation on dropped packets, saving
        cycles.
        
        
        Examples
        --------
        ### Default network (FDMA, sonar PLR, poisson BER):
    
        >>> import munetauvsim.communication as comm
        >>> network = comm.MuNet()
        
        ### Custom configuration:
        
        >>> network = comm.MuNet(
        ...     accessMode='tdma_lf',
        ...     plrType='exp_simple',
        ...     PLR=0.05,
        ...     PLR_CHARDIST=1500,
        ...     berType='snr',
        ...     BER=1e-3,
        ...     SNR_NOMINAL=25,
        ...     collType='corrupt',
        ...     seed=42
        ... )
        
        ### Ideal network (no errors):
        
        >>> network = comm.MuNet(
        ...     jitterType='off',
        ...     collType='off',
        ...     plrType='off',
        ...     berType='off'
        ... )
        
        ### High-error scenario:
        
        >>> network = comm.MuNet(
        ...     PLR=0.20,
        ...     BER=1E-2,
        ...     MAX_JITTER=1.0,
        ...     collType='corrupt'
        ... )
        """

        # Parameters
        self.C_WATER = 1500.0           # Speed of Sound in Water (m/s)
        self.BASE_FREQ = 24.0           # Base Acoustic Frequency (kHz)
        self.BANDWIDTH = 10.0           # Total available bandwidth (kHz)
        self.GUARD_BW = 0.25            # Width of guard bands (kHz)
        self.CHAN_MINBW = 1.0           # Minimum channel bandwidth (kHz)
        self.BCAST_ADDR = 255           # Broadcast Address
        self.DATA_RATE = 1000           # Data rate (bps)
        self.MAX_JITTER = 0.2           # Maximum random packet delay (s)
        self.NOM_DIST = 1000.0          # Range where attenuation value=base (m)
        self.PLR = 0.02                 # Packet loss ratio (PLR*100 = %)
        self.PLR_CHARDIST = 2000.0      # Characteristic PLR attenuation (m)
        self.BER = 1E-3                 # Bit error rate (err/bits)
        self.BER_CHARDIST = 3000.0      # Characteristic BER attenuation (m)
        self.SNR_NOMINAL = 30.0         # Nominal SNR for BER models (dB)
        self.SNR_SLOPE = 0.1            # SNR slope factor for BER models

        # Configurations
        self.accessMode = 'FDMA'            # Channel access protocol
        self.orderMode = 'center_out'       # Channel placement sequence
        self.jitterType = 'uniform'         # Jitter strategy
        self.collType = 'drop'              # Collision strategy
        self.plrType = 'sonar_simple'       # PLR strategy
        self.berType = 'poisson'            # BER strategy
        self.seed = np.random.SeedSequence().entropy  # Random Seed

        # Customized Input Parameters
        self.__dict__.update(kwargs)        # Update any keyword inputs
        
        # Data Structures
        self.channels = {}                  # Channels dictionary {id:info}
        self.nodes = {}                     # Nodes dictionary {id:node}
        self.txq = []                       # Transmission queue
        self.rxq = []                       # Active reception queue
        self.rng = np.random.default_rng(self.seed) # Random Number Generator

        # Network Statistics
        """Some stats not accurate until calcStats() is called."""
        self.stats = {                      # Network Transmission Statistics
            'packetSent': 0,                # Number of packets sent
            'packetDelivered': 0,           # Number of packets delivered
            'packetDropPLR': 0,             # Packets dropped from PLR
            'packetDropColl': 0,            # Packets dropped from Collision
            'packetFailedDel': 0,           # Packets that failed delivery
            'inTransitTX': 0,               # Messages still in transit
            'inTransitRX': 0,               # Messages still in RX queue
            'plrActual': 0,                 # Calculated Actual PLR
            'plrDeliveryRate': 0,           # Calculated Delivery Rate
            'collDropRate': 0,              # Calculated Collision Drop Rate
            'inTransitRate': 0,             # Calculates In-Transit Rate
            'collisions': 0,                # Packet collision events
            'packetCorruptColl': 0,         # Packets corrupted by collision
            'packetBER': 0,                 # Packets that have BER
            'bitErrors': 0,                 # Number of bit errors introduced
        }

        # Dynamic Method Assignments
        log.info('*** muNet ONLINE ***')

        ## Channels / Nodes
        accessStrategies = {
            'shared': 
                (self._buildChannelsShared, self._assignNodesShared),
            'tdma': 
                (self._buildChannelsTDMA, self._assignNodesTDMA),
            'tdma_lf':
                (self._buildChannelsTDMALF, self._assignNodesTDMALF),
            'fdma': 
                (self._buildChannelsFDMA, self._assignNodesFDMA),
        }
        self._buildChannels, self._assignNodes = (
            accessStrategies[self.accessMode.lower()]
        )
        log.info('Access Mode: %s...', self.accessMode.upper())

        genChannelStrategies = {
            'center_out': self._genChannelCenterOut,
            'edge_in': self._genChannelEdgeIn,
        }
        self._genChannel = genChannelStrategies[self.orderMode]
        self._channelGen = self._genChannel()  # Iterable channel generator
        log.info('Channel Order Mode: %s...', self.orderMode.upper())

        ## Jitter
        jitterStrategies = {
            # add more jitter strategies here if there is value
            'uniform': self._jitterUniform,
            'off': self._jitterDisabled,
        }
        self._generateJitter = jitterStrategies[self.jitterType]
        if (self.jitterType == 'off'):
            log.info('Jitter DISABLED...')
        else:
            log.info('Jitter ENABLED: %s...', self.jitterType.upper())
        
        ## Collisions
        collStrategies = {
            'drop': 
                (self._collDrop, self._txWithColl, self._rxWithColl),
            'corrupt': 
                (self._collCorrupt, self._txWithColl, self._rxWithColl),
            'off': 
                (self._applyColl, self._txNoColl, self._rxNoColl),
        }
        self._applyColl, self._txEnqueue, self.deliver = (
            collStrategies[self.collType]
        )
        if (self.collType == 'off'):
            log.info('Packet Collisions DISABLED...')
        else:
            log.info('Packet Collisions ENABLED: %s...', self.collType.upper())
        
        ## PLR
        plrStrategies = {
            'uniform': self._plrUniform,
            'exp_simple': self._plrExpSimple,
            'sonar_simple': self._plrSonarSimple,
            'off': self._plrDisabled,
        }
        self._applyPLR = plrStrategies[self.plrType]
        if (self.plrType == 'off'):
            log.info('Packet Loss Rate DISABLED...')
        else:
            log.info('Packet Loss Rate ENABLED: %s...', self.plrType.upper())

        ## BER
        berStrategies = {
            'poisson': self._berPoisson,
            'exp_simple': self._berExpSimple,
            'snr': self._berSNR,
            'off': self._berDisabled,
        }
        self._applyBER = berStrategies[self.berType]
        if (self.berType == 'off'):
            log.info('Bit Error Rate DISABLED...')
        else:
            log.info('Bit Error Rate ENABLED: %s...', self.berType.upper())

    ## Properties ============================================================#
    @property
    def CHAN_MAXN(self)->int:
        """Maximum number of channels possible in bandwidth. Read-only."""
        bw_eff = self.BANDWIDTH - (2 * self.GUARD_BW)
        return int(bw_eff // (self.CHAN_MINBW + self.GUARD_BW))
    
    ## Special Methods =======================================================#
    def __repr__(self)->str:
        """Detailed description of the network."""
        return (
            f"{self.__class__.__name__}("
            f"C_WATER={self.C_WATER}, "
            f"BASE_FREQ={self.BASE_FREQ}, "
            f"BANDWIDTH={self.BANDWIDTH}, "
            f"GUARD_BW={self.GUARD_BW}, "
            f"CHAN_MINBW={self.CHAN_MINBW}, "
            f"CHAN_MAXN={self.CHAN_MAXN}, "
            f"BCAST_ADDR={self.BCAST_ADDR}, "
            f"DATA_RATE={self.DATA_RATE}, "
            f"MAX_JITTER={self.MAX_JITTER}, "
            f"NOM_DIST={self.NOM_DIST}, "
            f"PLR={self.PLR}, "
            f"PLR_CHARDIST={self.PLR_CHARDIST}, "
            f"BER={self.BER}, "
            f"BER_CHARDIST={self.BER_CHARDIST}, "
            f"SNR_NOMINAL={self.SNR_NOMINAL}, "
            f"SNR_SLOPE={self.SNR_SLOPE}, "
            f"accessMode={self.accessMode}, "
            f"orderMode={self.orderMode}, "
            f"jitterType={self.jitterType}, "
            f"collType={self.collType}, "
            f"plrType={self.plrType}, "
            f"berType={self.berType}, "
            f"channels={len(self.channels)}, "
            f"nodes={len(self.nodes)}, "
            f"stats={self.stats}, "
            f"seed={self.seed})"
        )
    
    #-------------------------------------------------------------------------#
    def __str__(self)->str:
        """User friendly description of the network."""
        # column spacing
        cw = 20
        cw2 = 12
        
        # jitter
        if (self.jitterType == 'off'):
            jitterStatus = 'Disabled'
        else:
            jitter = f"{self.MAX_JITTER} s"
            jitterStatus = f"{jitter:{cw2}}{self.jitterType}"

        # collisions
        if (self.collType == 'off'):
            collStatus = 'Disabled'
        else:
            collStatus = f"{'Enabled':{cw2}}{self.collType}"
        
        # plr
        plrParams = ''
        if (self.plrType == 'off'):
            plrStatus = 'Disabled'
        else:
            plr = f"{self.PLR:.2%}"
            plrStatus = f"{plr:{cw2}}{self.plrType}"
            if (self.plrType in {'exp_simple', 'sonar_simple'}):
                plrParams = (
                    f"{' PLR Nominal:':{cw}} {self.NOM_DIST:.0f} m\n"
                    f"{' PLR Character:':{cw}} {self.PLR_CHARDIST:.0f} m")

        # ber
        berParams = ''
        if (self.berType == 'off'):
            berStatus = 'Disabled'
        else:
            ber = f"{self.BER:.0E}"
            berStatus = f"{ber:{cw2}}{self.berType}"
            if (self.berType == 'exp_simple'):
                berParams = (
                    f"{' BER Nominal:':{cw}} {self.NOM_DIST:.0f} m\n"
                    f"{' BER Character:':{cw}} {self.BER_CHARDIST:.0f} m")
            elif (self.berType == 'snr'):
                berParams = (
                    f"{' SNR Nominal:':{cw}} {self.SNR_NOMINAL:.1f} dB\n"
                    f"{' SNR Slope:':{cw}} {self.SNR_SLOPE:.2f}")
                
        # channels
        nChannels = f"{len(self.channels)}"
        chanStatus = f"{nChannels:{cw2}}{self.orderMode}"

        # output
        out = [
            f"Communication: muNet",
            f"{'Frequency:':{cw}} {self.BASE_FREQ} kHz",
            f"{'Bandwidth:':{cw}} {self.BANDWIDTH} kHz",
            f"{'Min Channel BW:':{cw}} {self.CHAN_MINBW} kHz",
            f"{'Guard BW:':{cw}} {self.GUARD_BW} kHz",
            f"{'Max Channels:':{cw}} {self.CHAN_MAXN}",
            f"{'Data Rate:':{cw}} {self.DATA_RATE} bps",
            f"{'Max Jitter:':{cw}} {jitterStatus}",
            f"{'Collisions:':{cw}} {collStatus}",
            f"{'Packet Loss:':{cw}} {plrStatus}",
        ]
        if (plrParams):
            out.extend(plrParams.split('\n'))
        out.extend([
            f"{'Bit Error Rate:':{cw}} {berStatus}",
        ])
        if (berParams):
            out.extend(berParams.split('\n'))
        out.extend([
            f"{'Channels:':{cw}} {chanStatus}",
            f"{'Nodes:':{cw}} {len(self.nodes)}",
            f"{'Mode: ':{cw}} {self.accessMode}",
            f"{'RNG Seed:':{cw}} {self.seed}",
        ])
        line = '-' * max([len(line) for line in out])
        out.insert(1, line)
        out.append(line)
        return "\n".join(out)

    #-------------------------------------------------------------------------#
    def __getstate__(self)->dict:
        """
        Customize pickle serialization by removing non-serializable generator.

        Returns
        -------
        dict
            Network state with channel generator removed for safe pickling.

        Notes
        -----
        Generator objects cannot be pickled. This method removes _channelGen
        during serialization.
        """
    
        ch_gen = '_channelGen'
        state = self.__dict__.copy()
        if ch_gen in state:
            state[ch_gen] = None
        return state
    
    #-------------------------------------------------------------------------#
    def __setstate__(self, state)->None:
        """
         Restore network state and recreate channel generator after unpickling.

        Parameters
        ----------
        state : dict
            Network state dictionary from __getstate__.

        Notes
        -----
        Recreates channel generator according to current orderMode strategy.
        Generator does not restore to pre-pickle state (generates from start).
        """

        self.__dict__.update(state)
        self._channelGen = self._genChannel()

    ## Methods ===============================================================#
    def register(self, vehicle:Vehicle)->MuNode:
        """
        Create and register vehicle node on network with channel assignment.

        Parameters
        ----------
        vehicle : Vehicle
            Vehicle object with nodeAddr and callSign attributes.

            - nodeAddr : int, node network address
            - callSign : str, custom unique vehicle identifier

        Returns
        -------
        MuNode
            New network node for vehicle with assigned TX/RX channels.

        Notes
        -----
        - Automatically builds channels if not yet created. 
        - Assigns channels according to accessMode protocol. 
        - Logs status string 'CONFIRMED' if successful or 'DENIED' if channel
          capacity exceeded.
        """

        # Make new node
        node = MuNode(vehicle.nodeAddr, vehicle, self)

        # Add node to network dictionary
        self.nodes[node.address] = node

        # Assign channels according to access mode
        status = self._assignNodes(node)

        # Log client at new node address
        log.info('%s: ON MUNET AT %s... %s...', vehicle.callSign, node.address,
                 status)
        
        # Bad coupling here, but convenient for now
        if (status == 'CONFIRMED'):
            for n in self.nodes.values():
                n.vehicle.info.update([
                    ('TX Channel IDs', f"{sorted(n.tx_channels)}"),
                    ('RX Channel IDs', f"{sorted(n.rx_channels)}"),
                ])
        
        return node

    #--------------------------------------------------------------------------
    def transmit(self, message:bytes, cid:int, srcAddr:int)->None:
        """
        Queue message for transmission to all receivers on channel.

        Computes transmission times (data rate, travel time, jitter) and
        enqueues message copies to TX queue for each receiver on channel (except
        sender).

        Parameters
        ----------
        message : bytes
            Message payload to transmit.
        cid : int
            Channel ID carrying the transmission.
        srcAddr : int
            Source node address (sender).

        Notes
        -----
        Enqueues one message for each node registered as receiver on channel.
        For each message, computes:

        - Data transmission time: (message_bytes * 8) / DATA_RATE
        - Travel time: distance / C_WATER
        - Jitter: random delay via _generateJitter strategy

        Message delivery times computed as: 
            rxTime_start = srcTime+ travelTime + jitterTime
            rxTime_end = rxTime_start + dataTime
        """

        # Identify sender node
        sNode = self.nodes[srcAddr]

        # Identify receiver node(s)
        rNodes = self._getChannelRXNodes(cid)

        # Filter sender node out of receiver node(s)
        rNode = [n for n in rNodes if n.address != sNode.address]

        # Determine message data transmission time
        dataTime = ((len(message) * 8) / self.DATA_RATE)

        # Determine message travel time(s)
        positions = [n.vehicle.eta[0:3] for n in rNode]
        distances = np.linalg.norm(positions - sNode.vehicle.eta[0:3], axis=1)
        travelTime = distances / self.C_WATER

        # Generate message jitter time(s)
        jitterTime = self._generateJitter(len(rNode))

        # Determine end of message delivery time(s)
        rxTime = sNode.vehicle.clock + dataTime + (travelTime + jitterTime)

        # Enqueue message(s) to TXQ
        ch = self.channels[cid]
        for node, time in zip(rNode,rxTime):
            log.debug('[%03d:%03d] (%d|%.2f) (%.2fs@%.2fs) %s',
                      sNode.address, node.address, ch.cid, ch.freq, 
                      dataTime, time, message)
        self._txEnqueue(rxTime, dataTime, sNode.address, rNode, ch, distances,
                        message)

    #--------------------------------------------------------------------------
    def deliver(self, time:float)->None:
        """
        Dequeue messages from TX queue and deliver to RX nodes.

        Parameters
        ----------
        time : float
            Current simulation time.

        Notes
        -----
        Main entry point for message delivery. Dynamically assigned during
        initialization based on configuration specified by collType:

        - If collisions enabled, uses: _rxWithColl
        - If collisions disabled, uses: _rxNoColl
        """

        raise NotImplementedError(
            "deliver() method should be assigned during initialization")
    
    #--------------------------------------------------------------------------
    def calcStats(self)->None:
        """
        Compute network performance statistics from current state.

        Calculates derived metrics (delivery rate, collision rate, in-transit ratio)
        from raw statistics counters updated during message processing.

        Notes
        -----
        Statistics valid only after calling calcStats(). Some stats (in-transit
        counts) updated continuously; others (delivery rate) computed here.
        """

        totalSent = self.stats['packetSent']
        self.stats['inTransitTX'] = len(self.txq)
        self.stats['inTransitRX'] = len(self.rxq)
        if (totalSent > 0):
            self.stats['collDropRate'] = (
                self.stats['packetDropColl'] / totalSent)
            self.stats['plrActual'] = (
                self.stats['packetDropPLR'] / totalSent)
            self.stats['plrDeliveryRate'] = (
                self.stats['packetDelivered'] / totalSent)
            self.stats['inTransitRate'] = (
                self.stats['inTransitTX'] + self.stats['inTransitRX']/totalSent)
    
    #--------------------------------------------------------------------------
    def getStatsReport(self)->str:
        """
        Return formatted network performance statistics report.

        Returns
        -------
        str
            Multi-line formatted report of network traffic, quality, and
            performance metrics.
        """

        self.calcStats()
        cw = 22 if (self.stats['packetFailedDel'] <= 0) else 26
        cw2 = 10
        cw3 = 16

        # Configure outputs
        if (self.collType == 'off'):
            collStatus = f"{'Disabled':>{cw2}}"
        else:
            collStatus = f"{'Enabled':>{cw2}}{self.collType:>{cw3}}"
        if (self.plrType == 'off'):
            plrStatus = f"{'Disabled':>{cw2}}"
        else:
            plr = f"{self.PLR:>6.1%}"
            plrStatus = f"{plr:>{cw2+1}}{self.plrType:>{cw3-1}}"
        if (self.berType == 'off'):
            berStatus = f"{'Disabled':>{cw2}}"
        else:
            ber = f"{self.BER:.0E}"
            berStatus = f"{ber:>{cw2}}{self.berType:>{cw3}}"
        packetsAccounted = (self.stats['packetDropPLR'] +
                            self.stats['packetDropColl'] +
                            self.stats['packetDelivered'] +
                            self.stats['packetFailedDel'] +
                            self.stats['inTransitTX'] +
                            self.stats['inTransitRX'])
        inTransitTotal = self.stats['inTransitTX'] + self.stats['inTransitRX']
        
        # Construct output
        report = [
            f"\nmuNet: Network Performance Summary",
            f"Configuration",
            f"{' Collisions:':{cw}} {collStatus}",
            f"{' Packet Loss Rate:':{cw}} {plrStatus}",
            f"{' Bit Error Rate:':{cw}} {berStatus}",
            f"",
            f"Traffic",
            f"{' Packets Sent:':{cw}} {self.stats['packetSent']:>{cw2}}",
            f"{' Delivered:':{cw}} {self.stats['packetDelivered']:>{cw2}}",
            f"{' Dropped (PLR):':{cw}} {self.stats['packetDropPLR']:>{cw2}}",
        ]
        if (self.collType == 'drop'):
            report.append(
                f"{' Dropped (Collision):':{cw}} "
                f"{self.stats['packetDropColl']:>{cw2}}",
            )
        if (self.stats['packetFailedDel'] > 0):
            report.append(
                f"{' Failed Delivery (errors):':{cw}} "
                f"{self.stats['packetFailedDel']:>{cw2}}",
            )
        report.extend([
            f"{' In Transit:':{cw}} {inTransitTotal:>{cw2}}",
            f"{' Total Accounted:':{cw}} {packetsAccounted:>{cw2}}",
            f"",
            f"Quality",
        ])
        if (self.collType == 'corrupt'):
            nCorrupt = self.stats['packetBER'] + self.stats['packetCorruptColl']
            report.extend([
                f"{' Packets Corrupted:':{cw}} {nCorrupt:>{cw2}}",
                f"{' By BER:':{cw}} {self.stats['packetBER']:>{cw2}}",
                f"{' By Collision:':{cw}} "
                f"{self.stats['packetCorruptColl']:>{cw2}}",
            ])
        else:
            report.append(
                f"{' Packets with BER:':{cw}} {self.stats['packetBER']:>{cw2}}",
            )
        report.extend([
            f"",
            f"Performance",
            f"{' Bit Errors:':{cw}} {self.stats['bitErrors']:>{cw2}}",
            f"{' Collision Events:':{cw}} {self.stats['collisions']:>{cw2}}",
        ])
        if (self.collType == 'drop'):
            report.append(
                f"{' Collision Drop Rate:':{cw}} "
                f"{self.stats['collDropRate']:>{cw2+1}.1%}",
            )
        report.extend([
            f"{' Actual PLR:':{cw}} {self.stats['plrActual']:>{cw2+1}.1%}",
            f"{' Delivery Rate:':{cw}} {self.stats['plrDeliveryRate']:>{cw2+1}.1%}",
            f"{' In Transit:':{cw}} {self.stats['inTransitRate']:>{cw2+1}.1%}",
        ])
        line = '-' * max([len(line) for line in report])
        report.insert(1, line)
        report.append(line)
        return "\n".join(report)

    ## Helper Methods ========================================================#
    def _txEnqueue(self, 
                   rxTime:Union[List[float],NPFltArr],
                   dataTime:float,
                   srcAddr:int,
                   rNode:List[MuNode],
                   channel:MuChannel,
                   distances:NPFltArr,
                   message:bytes,
                   )->None:
        """
        Enqueue transmitted messages to network TX queue.

        Parameters
        ----------
        rxTime : array-like of float
            Delivery end times for each receiver.
        dataTime : float
            Message transmission duration.
        srcAddr : int
            Source node address.
        rNode : list of MuNode
            Receiver nodes.
        channel : MuChannel
            Transmitting channel.
        distances : ndarray
            Distances from source to each receiver.
        message : bytes
            Message payload.

        Notes
        -----
        Dynamically assigned during initialization based on configuration
        specified by collType:

        - If collisions enabled, uses: _txWithColl
        - If collisions disabled, uses: _txNoColl
        """

        raise NotImplementedError(
            "_txEnqueue() method should be assigned during initialization")

    #--------------------------------------------------------------------------
    def _txNoColl(self, 
                  rxTime:Union[List[float],NPFltArr],
                  dataTime:float,
                  srcAddr:int,
                  rNode:List[MuNode],
                  channel:MuChannel,
                  distances:NPFltArr,
                  message:bytes,
                  )->None:
        """
        Enqueue messages to network TX queue without collision handling.

        Parameters
        ----------
        rxTime : array-like of float
            Delivery end times for each receiver.
        dataTime : float
            Message transmission duration.
        srcAddr : int
            Source node address.
        rNode : list of MuNode
            Receiver nodes.
        channel : MuChannel
            Transmitting channel.
        distances : ndarray
            Distances from source to each receiver.
        message : bytes
            Message payload.

        Notes
        -----
        - Used when collType='off'.
        - Each receiver gets separate queue entry with computed arrival time.
        - Packages each message as MuNetQEntry dataclass with transmission metadata:

            - 'start_time'     : message delivery start time
            - 'end_time'       : message delivery end time
            - 'src_addr'       : source address
            - 'dest_addr'      : destination address
            - 'cid'            : channel id
            - 'freq'           : channel frequency
            - 'bw'             : channel bandwidth
            - 'distance'       : distance from src to dest
            - 'message'        : message

        - Heap sorting is done on: 'end_time'.
        """
        
        for time, node, dist in zip(rxTime, rNode, distances):
            log.debug('[%03d:%03d] (%d|%.2f) (@%.2fs) %s',
                      srcAddr, node.address, channel.cid, channel.freq,
                      time, message)
            entry = MuNetQEntry(
                start_time=(time-dataTime),
                end_time=time,
                src_addr=srcAddr,
                dest_addr=node.address,
                cid=channel.cid,
                freq=channel.freq,
                bw=channel.bw,
                distance=dist,
                message=bytearray(message),
            )
            heapq.heappush(self.txq, (entry.end_time, self._nextQid(), entry))
            self.stats['packetSent'] += 1

    #--------------------------------------------------------------------------
    def _rxNoColl(self, time:float)->None:
        """
        Dequeue TX messages and deliver without collision detection.

        Simple queue processing without active reception tracking or collision
        checking.

        Parameters
        ----------
        time : float
            Current simulation time.

        Notes
        -----
        - Processes TX queue sequentially: apply PLR, apply BER, then deliver to
          receiver node via recv() callback.
        - Used when collType='off'.
        - TX queue entries are MuNetQEntry dataclass with transmission metadata:

            - 'start_time'     : message delivery start time
            - 'end_time'       : message delivery end time
            - 'src_addr'       : source address
            - 'dest_addr'      : destination address
            - 'cid'            : channel id
            - 'freq'           : channel frequency
            - 'bw'             : channel bandwidth
            - 'distance'       : distance from src to dest
            - 'message'        : message

        - Heap sorting is done on: 'end_time'
        """

        # TX queue has entries
        try:
            # Reached message delivery end time
            while (self.txq and self.txq[0][0] <= time):
                # Pop message entry from TX queue
                _, _, entry = heapq.heappop(self.txq)

                # Identify receiver node
                rNode = self.nodes[entry.dest_addr]

                # Introduce packet loss
                if (self._applyPLR(entry)):
                    continue

                # Introduce bit error rate
                self._applyBER(entry)

                # Deliver message
                try:
                    delivery = bytes(entry.message)
                    log.debug('[%03d:%03d] (%d|%.2f) (>%.2fs) %s', 
                              entry.src_addr, entry.dest_addr, 
                              entry.cid, entry.freq, entry.end_time, delivery)
                    rNode.recv(delivery, entry.cid)
                    self.stats['packetDelivered'] += 1
                except Exception as e:
                    log.error('[%03d:%03d] (%d|%.2f) ' +
                              'MESSAGE DELIVERY FAILED: %s',
                              entry.src_addr, entry.dest_addr, 
                              entry.cid, entry.freq, str(e))
                    self.stats['packetFailedDel'] += 1
        
        # TX queue is empty
        except (IndexError):
            pass

    #--------------------------------------------------------------------------
    def _txWithColl(self, 
                    rxTime:Union[List[float],NPFltArr],
                    dataTime:float,
                    srcAddr:int,
                    rNode:List[MuNode],
                    channel:MuChannel,
                    distances:NPFltArr,
                    message:bytes,
                    )->None:
        """
        Enqueue messages to TX queue with collision detection preparation.

        Parameters
        ----------
        rxTime : array-like of float
            Delivery end times for each receiver.
        dataTime : float
            Message transmission duration.
        srcAddr : int
            Source node address.
        rNode : list of MuNode
            Receiver nodes.
        channel : MuChannel
            Transmitting channel.
        distances : ndarray
            Distances from source to each receiver.
        message : bytes
            Message payload.

        Notes
        -----
        - Used when collType in {'drop', 'corrupt'}.
        - Each receiver gets separate queue entry with computed arrival time.
        - Packages each message as MuNetQEntry dataclass with transmission
          metadata:

            - 'start_time'     : message delivery start time
            - 'end_time'       : message delivery end time
            - 'src_addr'       : source address
            - 'dest_addr'      : destination address
            - 'cid'            : channel id
            - 'freq'           : channel frequency
            - 'bw'             : channel bandwidth
            - 'distance'       : distance from src to dest
            - 'message'        : message

        - Heap sorting is done on 'start_time'. Enables populating receiver rxd
          (active RX dict) and monitoring for collisions until 'end_time'.
        """
        
        for time, node, dist in zip(rxTime, rNode, distances):
            log.debug('[%03d:%03d] (%d|%.2f) (@%.2f:%.2fs) %s',
                      srcAddr, node.address, channel.cid, channel.freq,
                      time - dataTime, time, message)
            entry = MuNetQEntry(
                start_time=(time-dataTime),
                end_time=time,
                src_addr=srcAddr,
                dest_addr=node.address,
                cid=channel.cid,
                freq=channel.freq,
                bw=channel.bw,
                distance=dist,
                message=bytearray(message),
            )
            heapq.heappush(self.txq, (entry.start_time, self._nextQid(), entry))
            self.stats['packetSent'] += 1            

    #--------------------------------------------------------------------------
    def _rxWithColl(self, time:float)->None:
        """
        Dequeue TX messages and deliver with collision detection.

        Parameters
        ----------
        time : float
            Current simulation time.
        
        Notes
        -----
        - Two-phase processing:

            1. TX -> RX: Move messages from TX queue to RX queue and active RX
               dict
            2. RX -> Deliver: Check for collisions, apply PLR/BER, deliver

        - Used when collType in {'drop', 'corrupt'}.
        - Queue entries are are MuNetQEntry dataclass with transmission metadata:

            - 'start_time'     : message delivery start time
            - 'end_time'       : message delivery end time
            - 'src_addr'       : source address
            - 'dest_addr'      : destination address
            - 'cid'            : channel id
            - 'freq'           : channel frequency
            - 'bw'             : channel bandwidth
            - 'distance'       : distance from src to dest
            - 'message'        : message

        - TXQ heap sorting is done on: 'start_time'
        - RXQ heap sorting is done on: 'end_time'
        - Maintains rxd (active RX dictionary) per receiver per channel: 
            {Key = source channel id, Value = [RX queue entry]}.
        - Colliding packets handled by _applyColl strategy (drop or corrupt).
        """

        # TX queue has entries: Move to RX queue
        try:
            # Reached message delivery start time
            while (self.txq and self.txq[0][0] <= time):
                # Pop message entry from TX queue
                _, _, entry = heapq.heappop(self.txq)

                # Identify receiver node
                rNode = self.nodes[entry.dest_addr]

                # Add RXQ entry to receiver node Active RX dictionary
                try:
                    rNode.rxd[entry.cid].append(entry)
                except (KeyError):
                    rNode.rxd[entry.cid] = [entry]
                
                # Enqueue message to RXQ
                log.debug('[%03d:%03d] (%d|%.2f) (@%.2f:%.2fs) TXQ>RXQ : %s',
                          entry.src_addr, entry.dest_addr, 
                          entry.cid, entry.freq,
                          entry.start_time, entry.end_time, bytes(entry.message))
                heapq.heappush(self.rxq, (entry.end_time, self._nextQid(), entry))

        # TX queue is empty
        except (IndexError):
            pass

        # RX queue has entries: Monitor collisions and deliver
        try:
            # Reached message delivery end time
            while(self.rxq and self.rxq[0][0] <= time):
                # Pop message entry from RX queue
                _, _, entry = heapq.heappop(self.rxq)

                # Identify receiver node
                rNode = self.nodes[entry.dest_addr]

                # Check receiver node active RX dictionary for collisions
                if (self._applyColl(entry, rNode)):
                    continue

                # Remove entry from active RX dictionary
                rNode.rxd[entry.cid].remove(entry)

                # Introduce packet loss
                if (self._applyPLR(entry)):
                    continue

                # Introduce bit error rate
                self._applyBER(entry)

                # Deliver message
                try:
                    delivery = bytes(entry.message)
                    log.debug('[%03d:%03d] (%d|%.2f) (>%.2fs) %s', 
                              entry.src_addr, entry.dest_addr, 
                              entry.cid, entry.freq,
                              entry.end_time, delivery)
                    rNode.recv(delivery, entry.cid)
                    self.stats['packetDelivered'] += 1
                except Exception as e:
                    log.error('[%03d:%03d] (%d|%.2f) ' + 
                              'MESSAGE DELIVERY FAILED: %s',
                              entry.src_addr, entry.dest_addr, 
                              entry.cid, entry.freq, str(e))
                    self.stats['packetFailedDel'] += 1
        
        # RX queue is empty
        except (IndexError):
            pass

    #--------------------------------------------------------------------------
    def _nextQid(self)->int:
        """
        Return next sequential TX/RX queue message ID number.
        
        Returns
        -------
        int
            Unique message identifier for heap sort tie-breaking.
        
        Notes
        -----
        Increments class-level counter __qid. Used to maintain sort order when
        messages have identical timestamps in priority queue.
        """
        self.__qid += 1
        return self.__qid

    #--------------------------------------------------------------------------
    def _getChannelRXNodes(self, cid:int)->List[MuNode]:
        """
        Return nodes registered to receive on channel.

        Parameters
        ----------
        cid : int
            Channel ID.

        Returns
        -------
        nodes : list of MuNode
            Nodes with cid in their rx_channels set.

        Notes
        -----
        Linear search over all nodes. For large networks (1000+ nodes), consider
        caching receiver lists by channel.
        """

        return [n for n in self.nodes.values() if cid in n.rx_channels]
    
    #--------------------------------------------------------------------------
    def _generateJitter(self, size:int)->NPFltArr:
        """
        Generate jitter values for network transmission delays.

        Parameters
        ----------
        size : int
            Number of jitter values to generate.

        Returns
        -------
        jitter : ndarray
            Array of jitter delays in seconds.

        Notes
        -----
        Dynamically assigned during initialization based on jitter
        configuration:

        - If jitterType='uniform', uses: _jitterUniform
        - If jitterType='off', uses: _jitterDisabled
        """

        raise NotImplementedError(
            "_generateJitter() method should be assigned during initialization")
    
    #--------------------------------------------------------------------------
    def _jitterUniform(self, size:int)->NPFltArr:
        """
        Generate random jitter values for network transmission delays.
        
        Applies random uniform delays to simulate network propagation
        variability on packet transmission timing. Values are drawn from a
        uniform distribution in the range [0, MAX_JITTER].

        Parameters
        ----------
        size : int
            Number of jitter values to generate.

        Returns
        -------
        jitter : ndarray
            Array of random jitter delays in seconds.
        """

        return self.rng.uniform(high=self.MAX_JITTER, size=size)
    
    #--------------------------------------------------------------------------
    def _jitterDisabled(self, size:int)->NPFltArr:
        """
        Return zero jitter values (jitter disabled).
        
        Parameters
        ----------
        size : int
            Number of values to generate.
        
        Returns
        -------
        ndarray
            Array of zeros with length=size.
        """

        return np.zeros(size)
    
    #--------------------------------------------------------------------------
    def _applyColl(self, entry:MuNetQEntry, rNode:MuNode)->bool:
        """
        Apply packet collision strategy to overlapping transmissions.

        Parameters
        ----------
        entry : MuNetQEntry
            Message entry from RX queue.
        rNode : MuNode
            Receiving node.

        Returns
        -------
        bool
            True if packet was dropped due to collision.

        Notes
        -----
        Dynamically assigned during initialization based on collision
        configuration:

        - If collType='drop', uses: _collDrop
        - If collType='corrupt', uses: _collCorrupt
        - If collType='off', returns: False (no collision)
        """
        
        return False

    #--------------------------------------------------------------------------
    def _collDrop(self, entry:MuNetQEntry, rNode:MuNode)->bool:
        """
        Apply network collision by dropping all overlapping packets.

        Checks receiver's active RX dictionary for collisions on entry's channel.
        If multiple messages active, removes all from RX queue and statistics.

        Parameters
        ----------
        entry : MuNetQEntry
            Message entry from RX queue.
        rNode : MuNode
            Receiving node with rxd (active RX dict).

        Returns
        -------
        bool
            True (packet dropped) if collision detected, False otherwise.

        Notes
        -----
        - Used if collType='drop'
        - Updates stats: collisions counter, packetDropColl counter.
        """
        
        # Check for collision
        if (len(rNode.rxd[entry.cid]) <= 1):
            return False
        
        # Remove all colliding messages from RX dictionary
        drop = list(rNode.rxd.get(entry.cid, [])) # get a copy 
        rNode.rxd[entry.cid].clear()

        # Remove messages from RX queue by excluding them
        self.rxq = [qentry for qentry in self.rxq if qentry[2] not in drop]
        heapq.heapify(self.rxq)

        # Track collisions
        nColl = len(drop)
        self.stats['collisions'] += 1
        self.stats['packetDropColl'] += nColl
        log.debug('[%03d:%03d] (%d|%.2f) COLLISION: %d PACKETS DROPPED', 
                              entry.src_addr, entry.dest_addr, 
                              entry.cid, entry.freq, nColl)
        
        return True

    #--------------------------------------------------------------------------
    def _collCorrupt(self, entry:MuNetQEntry, rNode:MuNode)->bool:
        """
        Apply collision by corrupting overlapping packet regions via XOR.

        Models additive interference. For each colliding message, XORs
        overlapping byte regions to simulate mutual signal destruction.

        Parameters
        ----------
        entry : MuNetQEntry
            Message entry from RX queue.
        rNode : MuNode
            Receiving node with rxd (active RX dict).

        Returns
        -------
        bool
            False (packet not dropped) - packets corrupted in-place but
            delivered.

        Notes
        -----
        - Used if collType='corrupt'
        - Updates stats: collisions counter, packetCorruptColl counter.
        - Overlap calculation: Determines byte indices for overlap window at
          data rate, then XORs corrupted regions bidirectionally.
        """

        # Check for collision
        if (len(rNode.rxd[entry.cid]) <= 1):
            return False
        
        # Convert data bit rate into bytes rate
        Rb = self.DATA_RATE / 8.0           # bytes/sec

        # Get current message details
        msg = entry.message                 # bytearray
        msgStart = entry.start_time
        msgEnd = entry.end_time
        msgLen = len(msg)

        # Track corrupt packet counts
        corruptPkts = []
        anyOlap = False

        # Apply corruption on overlapping message sections
        for other in rNode.rxd[entry.cid]:
            if (other is entry):
                continue
            
            # Get other message details
            oth = other.message
            othStart = other.start_time
            othEnd = other.end_time
            othLen = len(oth)

            # Find overlap interval
            olapStart = max(msgStart, othStart)
            olapEnd = min(msgEnd, othEnd)
            if (olapEnd <= olapStart):
                continue

            # Compute indices for byte overlaps
            msgWinStart = max(0, int((olapStart - msgStart) * Rb))
            othWinStart = max(0, int((olapStart - othStart) * Rb))
            msgWinSize = max(0, msgLen - msgWinStart)
            othWinSize = max(0, othLen - othWinStart)
            olapWinSize = max(0, int((olapEnd - olapStart) * Rb))
            winSize = min(msgWinSize, othWinSize, olapWinSize)
            if (winSize <= 0):
                continue
            msgWinEnd = msgWinStart + winSize
            othWinEnd = othWinStart + winSize

            # Update corrupt packet count trackers
            if (entry not in corruptPkts):
                corruptPkts.append(entry)
            if (other not in corruptPkts):
                corruptPkts.append(other)
            anyOlap = True
            
            # Perform full mutual XOR corruption across overlap interval
            msgWin = bytes(msg[msgWinStart:msgWinEnd])
            othWin = bytes(oth[othWinStart:othWinEnd])
            for i in range(winSize):
                msg[msgWinStart + i] = msgWin[i] ^ othWin[i]
                oth[othWinStart + i] = othWin[i] ^ msgWin[i]
            
            log.debug('[%03d:%03d] (%d|%.2f) COLLISION: %d BYTES CORRUPTED',
                        entry.src_addr, entry.dest_addr, 
                        entry.cid, entry.freq, winSize)
        
        # Network stats update
        if (anyOlap):
            self.stats['collisions'] += 1
            self.stats['packetCorruptColl'] += len(corruptPkts)

        return False

    #--------------------------------------------------------------------------
    def _applyPLR(self, entry:MuNetQEntry)->bool:
        """
        Apply packet loss strategy to determine if message should be dropped.

        Parameters
        ----------
        entry : MuNetQEntry
            Message entry from TX queue.

        Returns
        -------
        bool
            True if packet should be dropped.

        Notes
        -----
        Dynamically assigned during initialization based on PLR
        configuration:

        - If plrType='uniform', uses: _plrUniform
        - If plrType='exp_simple', uses: _plrExpSimple
        - If plrType='sonar_simple', uses: _plrSonarSimple
        - If plrType='off', uses: _plrDisabled
        """
            
        raise NotImplementedError(
            "_applyPLR() method should be assigned during initialization")

    #--------------------------------------------------------------------------
    def _plrUniform(self, entry:MuNetQEntry)->bool:
        """
        Apply uniform random packet loss.

        Compares value drawn from random uniform distribution [0,1] to PLR
        parameter. If random < PLR, drop packet.

        Parameters
        ----------
        entry : MuNetQEntry
            Message entry.

        Returns
        -------
        bool
            True if packet dropped.
        """
        
        # Check if packet loss application is valid
        if (self.PLR <= 0):
            return False
        
        # Apply packet loss according to random uniform distribution
        if (self.rng.random() < self.PLR):
            log.debug('[%03d:%03d] (%d|%.2f) PACKET DROPPED (PLR:UNIF)', 
                      entry.src_addr, entry.dest_addr, entry.cid, entry.freq)
            self.stats['packetDropPLR'] += 1
            return True
        return False

    #--------------------------------------------------------------------------
    def _plrExpSimple(self, entry:MuNetQEntry)->bool:
        """
        Apply distance-based exponential packet loss.

        PLR increases exponentially with distance beyond nominal range.
        Includes frequency correction (+1% per kHz above BASE_FREQ).

        Parameters
        ----------
        entry : MuNetQEntry
            Message entry with distance field.

        Returns
        -------
        bool
            True if packet dropped.

        Notes
        -----
        PLR reaches base rate at nominal distance and increases exponentially
        beyond the nominal range, with the rate modulated by the characteristic
        distance:

            distance_factor = exp((dist - NOM_DIST) / PLR_CHARDIST)

        A smaller characteristic distance produces a faster exponential
        increase. An additional frequency correction is included, adding 1%
        additional loss per kHz above base frequency:
        
            frequency_factor = 1 + (freq - BASE_FREQ) * 0.01
        
        Maximum PLR is capped at 100%.

            PLR_d = min(PLR * distance_factor * frequency_factor, 1.0)

        A value is drawn from a random uniform distribution [0,1]. If random <
        PLR_d, the packet is dropped.
        """

        # Check if packet loss application is valid
        if (self.PLR <= 0):
            return False
        
        # Calculate distance and frequency factors
        dist_f = np.exp((entry.distance - self.NOM_DIST) / self.PLR_CHARDIST)
        freq_f = 1 + (entry.freq - self.BASE_FREQ) * 0.01

        # Determine PLR at distance, limit by max PLR
        plr_d = min(self.PLR * dist_f * freq_f, 1.0)
        if (self.rng.random() < plr_d):
            log.debug('[%03d:%03d] (%d|%.2f) PACKET DROPPED ' +
                      '(PLR:EXP %.3f@%.1fm)', 
                      entry.src_addr, entry.dest_addr, entry.cid, entry.freq,
                      plr_d, entry.distance)
            self.stats['packetDropPLR'] += 1
            return True
        return False

    #--------------------------------------------------------------------------
    def _plrSonarSimple(self, entry:MuNetQEntry)->bool:
        """
        Apply distance-based PLR using simplified sonar equation approximation.

        Combines geometric spreading (distance^1.5, practical compromise between
        spherical and cylindrical) with exponential absorption.

        Parameters
        ----------
        entry : MuNetQEntry
            Message entry with distance field.

        Returns
        -------
        bool
            True if packet dropped.

        Notes
        -----
        This model stays in linear space and approximates a simplified sonar
        equation by considering sound intensity degradation as dominated by two
        components: geometric spreading and absorption. The geometric spreading
        describes how the signal weakens with distance as the waves expand,
        while the use of nominal and characteristic distance in an exponential
        can approximate the contributions of noise and attenuation. A small
        frequency-dependent loss is also included.
        
        The geometric power-law spreading uses the practical spreading
        compromise between spherical (^2) and cylindrical (^1):

            geo_spread = (dist / NOM_DIST)^1.5, 

        The spreading is combined with exponential absorption:
        
            exp_abs = exp((dist - NOM_DIST) / PLR_CHARDIST),
        
        which is equal to 1 at the nominal distance, and increases exponentially
        beyond that at a rate modulated by the characteristic distance. An
        additional frequency correction is included, adding 1% additional loss
        per kHz above base frequency:
        
            freq_factor = 1 + (freq - BASE_FREQ) * 0.01

        Maximum PLR is capped at 100%.
            
            PLR_d = min(PLR * geo_spread * exp_abs * freq_factor, 1.0)

        A value is drawn from a random uniform distribution in [0,1] and
        compared to PLR_d. If random < PLR_d, the packet is dropped.

        References
        ----------
        [1] bioSNR: The Sound Stops with the Passive Sonar Equation. CRAN bioSNR
        vignette. https://cran.r-project.org/web/packages/bioSNR/vignettes/sonar.html

        [2] Discovery of Sound in the Sea. Sonar Equation: Passive sonar 
        example. https://dosits.org/science/advanced-topics/sonar-equation/sonar-equation-example-passive-sonar/
        """

        # Check if packet loss application is valid
        if (self.PLR <= 0):
            return False
        
        # Calculate distance and frequency factors
        dist_f = ((entry.distance / self.NOM_DIST) ** 1.5 *
                  np.exp((entry.distance-self.NOM_DIST) / self.PLR_CHARDIST))
        freq_f = 1 + (entry.freq - self.BASE_FREQ) * 0.01

        # Determine PLR at distance, limit by max PLR
        plr_d = min(self.PLR * dist_f * freq_f, 1.0)
        if (self.rng.random() < plr_d):
            log.debug('[%03d:%03d] (%d|%.2f) PACKET DROPPED ' +
                      '(PLR:SONAR %.3f@%.1fm)', 
                      entry.src_addr, entry.dest_addr, entry.cid, entry.freq,
                      plr_d, entry.distance)
            self.stats['packetDropPLR'] += 1
            return True
        return False

    #--------------------------------------------------------------------------
    def _plrDisabled(self, entry:MuNetQEntry)->bool:
        """
        No packet loss applied (PLR disabled).
        
        Parameters
        ----------
        entry : MuNetQEntry
            Message entry from TX queue.
        
        Returns
        -------
        bool
            Always False (packet not dropped).
        """

        return False

    #--------------------------------------------------------------------------
    def _applyBER(self, entry:MuNetQEntry)->None:
        """
        Apply bit error strategy to corrupt message payload.

        Parameters
        ----------
        entry : MuNetQEntry
            Message entry with metadata and message payload to corrupt in-place.

        Notes
        -----
        Dynamically assigned during initialization based on BER
        configuration:

        - If berType='poisson', uses: _berPoisson
        - If berType='exp_simple', uses: _berExpSimple
        - If berType='snr', uses: _berSNR
        - If berType='off', uses: _berDisabled
        """

        raise NotImplementedError(
            "_applyBER() method should be assigned during initialization")

    #--------------------------------------------------------------------------
    def _berPoisson(self, entry:MuNetQEntry)->None:
        """
        Apply constant BER via Poisson-distributed bit flips.

        Samples number of bit errors from Poisson(totalBits * BER), then flips
        random bit positions.

        Parameters
        ----------
        entry : MuNetQEntry
            Message entry with metadata and message payload to corrupt.

        Notes
        -----
        Updates stats: packetBER counter, bitErrors counter.
        """

        # Check if bit errors application is valid
        if (self.BER <= 0):
            return
        
        # Grab message buffer
        msg = entry.message    # bytearray

        # Determine number of bits to toggle using Poisson distribution
        totalBits = len(msg) * 8
        numErrors = self.rng.poisson(totalBits * self.BER)

        # Limit number of errors to size of message
        if (numErrors == 0):
            return
        numErrors = min(numErrors, totalBits)

        # Select random bit positions to flip
        bitPos = self.rng.choice(totalBits, size=numErrors, replace=False)

        # Apply bit flips in-place on the bytearray using XOR
        self._flipBits(msg, bitPos)
        log.debug('[%03d:%03d] (%d|%.2f) BIT ERRORS: ' +
                  '%d BITS CORRUPTED (BER:POIS)',
                  entry.src_addr, entry.dest_addr, entry.cid, entry.freq,
                  numErrors)
        self.stats['packetBER'] += 1
        self.stats['bitErrors'] += numErrors

    #--------------------------------------------------------------------------
    def _berExpSimple(self, entry:MuNetQEntry)->None:
        """
        Apply distance-based exponential BER.

        Bit error rate increases exponentially with distance. Samples bit error
        count from Poisson(totalBits * ber_d).

        Parameters
        ----------
        entry : MuNetQEntry
            Message entry with metadata and message payload to corrupt.
        
        Notes
        -----
        BER reaches base rate at nominal distance and increases exponentially
        with distance beyond the nominal range, with the rate modulated by the
        characteristic distance:

            distance_factor = exp((dist - NOM_DIST) / BER_CHARDIST)

        A smaller characteristic distance produces a faster exponential
        increase. Maximum BER is capped at 100%.
        
            ber_d = min(BER * distance_factor, 1.0)

        The number of corrupted bits is selected with a Poisson distribution
        from the total length of the message payload, and bit positions are
        selected by random sample from the payload array. Finally, the bits are
        flipped in-place by XOR with a bit mask.
        """

        # Check if bit errors application is valid
        if (self.BER <= 0):
            return

        # Grab message buffer
        msg = entry.message    # bytearray

        # Calculate distance-based BER
        dist_f = np.exp((entry.distance - self.NOM_DIST) / self.BER_CHARDIST)
        ber_d = min(self.BER * dist_f, 1.0)  # Cap at 100%

        # Determine number of bits to toggle using Poisson distribution
        totalBits = len(msg) * 8
        numErrors = self.rng.poisson(totalBits * ber_d)

        # Limit number of errors to size of message
        if (numErrors == 0):
            return
        numErrors = min(numErrors, totalBits)

        # Select random bit positions to flip
        bitPos = self.rng.choice(totalBits, size=numErrors, replace=False)

        # Apply bit flips in-place on the bytearray using XOR
        self._flipBits(msg, bitPos)
        log.debug('[%03d:%03d] (%d|%.2f) BIT ERRORS: %d BITS CORRUPTED '+
                  '(BER:EXP %.2E@%.1fm)',
                  entry.src_addr, entry.dest_addr, entry.cid, entry.freq, 
                  numErrors, ber_d, entry.distance)
        self.stats['packetBER'] += 1
        self.stats['bitErrors'] += numErrors

    #--------------------------------------------------------------------------
    def _berSNR(self, entry:MuNetQEntry)->None:
        """
        Apply SNR-based BER model using path loss.

        Estimates SNR from distance via path loss model. BER scales with SNR
        deviation from nominal.

        Parameters
        ----------
        entry : MuNetQEntry
            Message entry with metadata and message payload to corrupt.
        
        Notes
        -----
        This model operates in decible (log) space, estimating SNR based on
        distance using simplified path loss model with practical spreading
        (15*log(dist), compromise between spherical (20) and cylindrical (10)):
        
            distance_factor = 15 * log10(distance / NOM_DIST).
        
        The signal is improved inside the nominal distance, has no affect at the
        nominal distance, and degrades beyond the nominal distance:
            
            SNR = SNR_NOMINAL - distance_factor.
            
        BER varies with SNR and the maximum is capped at 100%:
        
            BER_SNR = min(BER * 10^(SNR_SLOPE * distance_factor), 1.0).
        
        Where the SNR_SLOPE captures the sensitiviy of BER to signal quality
        changes: with a larger slope, the BER increases (signal degrades)
        faster.

        The number of corrupted bits is selected with a Poisson distribution
        from the total length of the message payload, and bit positions are
        selected by random sample from the payload array. Finally, the bits are
        flipped in-place by XOR with a bit mask.

        References
        ----------
        [1] Salim Wireless. "Relationship Between BER and SNR."
        https://www.salimwireless.com/2022/05/relationship-between-ber-and-snr.html

        [2] Northeastern University/Neu (Department of Electrical and Computer
        Engineering). "Handbook Chapter 5."
        https://millitsa.coe.neu.edu/publications/book/handbook-chap5.pdf 
        """
        
        # Check if bit errors application is valid
        if (self.BER <= 0):
            return

        # Grab message buffer
        msg = entry.message    # bytearray

        # Calculate SNR based on distance (simplified path loss model)
        # SNR decreases with distance due to spreading loss
        dist_r = max(entry.distance / self.NOM_DIST, 1e-6)  # Avoid log(0)
        dist_f = 15 * np.log10(dist_r)
        snr = self.SNR_NOMINAL - dist_f

        # Calculate BER based on SNR, limit at max BER
        ber_snr = min(self.BER * (10 ** (self.SNR_SLOPE * dist_f)), 1.0)

        # Determine number of bits to toggle using Poisson distribution
        totalBits = len(msg) * 8
        numErrors = self.rng.poisson(totalBits * ber_snr)

        # Limit number of errors to size of message
        if (numErrors == 0):
            return
        numErrors = min(numErrors, totalBits)

        # Select random bit positions to flip
        bitPos = self.rng.choice(totalBits, size=numErrors, replace=False)

        # Apply bit flips in-place on the bytearray using XOR
        self._flipBits(msg, bitPos)
        log.debug('[%03d:%03d] (%d|%.2f) BIT ERRORS: %d BITS CORRUPTED '+
                  '(SNR:%.1fdB, BER:%.2E@%.1fm)',
                  entry.src_addr, entry.dest_addr, entry.cid, entry.freq, 
                  numErrors, snr, ber_snr, entry.distance)
        self.stats['packetBER'] += 1
        self.stats['bitErrors'] += numErrors

    #--------------------------------------------------------------------------
    def _berDisabled(self, entry:MuNetQEntry)->None:
        """
        No bit error rate applied (BER disabled).
        
        Parameters
        ----------
        entry : MuNetQEntry
            Message entry from TX/RX queue.
        
        Notes
        -----
        Pass-through method that performs no operations on message.
        """

        pass
    
    #--------------------------------------------------------------------------
    def _flipBits(self, msg:bytearray, bitPos:NPFltArr)->None:
        """
        Flip specified bit positions in bytearray via XOR.

        In-place modification of message buffer.

        Parameters
        ----------
        msg : bytearray
            Message to modify.
        bitPos : ndarray of int
            Bit positions to flip (0-indexed from LSB of first byte).
        """

        for bp in bitPos:
            byteIdx = bp // 8
            bitIdx = bp % 8
            msg[byteIdx] ^= (1 << bitIdx)
        
    #--------------------------------------------------------------------------
    def _genChannel(self, 
                    cid:int = 0,
                    freq:Optional[float] = None, 
                    bw:Optional[float] = None,
                    isBcst:bool = False
                    )->Generator[MuChannel, None, None]:
        """
        Generator yielding network channels per allocation strategy.

        Parameters
        ----------
        cid : int, default=0
            Override starting channel id. Otherwise starts from the default
            value and increments by 1 on each yield.
        freq : float, optional
            Override starting frequency. Otherwise, starts from the value in the
            ordering strategy and increments according to channel order
            configuration on each yield.
        bw : float, optional
            Override channel bandwidth. Otherwise, uses the bandwidth determined
            by the channel order configuration on each yield.
        isBcst : bool, default=False
            Broadcast flag. All channels generated by the returned generator
            object will carry the same flag.

        Returns
        -------
        MuChannel Generator
            Yields channels generated according to channel order sequence set in
            configuration settings. Channel ids and frequencies are incremented
            on each iteration, bandwidth is static unless strategy determines
            otherwise, and broadcast flag is static.
        
        Notes
        -----
        Dynamically assigned during initialization based on channel order mode
        configuration:

        - If orderMode='center_out': uses _genChannelCenterOut
        - If orderMode='edge_in': uses _genChannelEdgeIn
        """

        raise NotImplementedError(
            "_genChannel() method should be assigned during initialization")
    
    #--------------------------------------------------------------------------
    def _genChannelCenterOut(self, 
                             cid:int = 0,
                             freq:Optional[float] = None, 
                             bw:Optional[float] = None,
                             isBcst:bool = False
                             )->Generator[MuChannel, None, None]:
        """
        Generates new network channels using a center-outward strategy.

        First channel is at BASE_FREQ, then alternates above and below until the
        bandwidth is full.

        Parameters
        ----------
        cid : int, default=0
            Override starting channel id.
        freq : float, optional
            Override starting frequency. Otherwise, starts from BASE_FREQ.
        bw : float, optional
            Override channel bandwidth. Otherwise, uses CHAN_MINBW
        isBcst : bool, default=False
            Broadcast flag. All channels generated will carry the same flag.

        Returns
        -------
        MuChannel Generator
            Yields channels with incrementing ID and frequencies alternating out
            from center. Bandwidth and broadcast flag are static.
        """

        # Get useable central frequency limits and bandwidth values
        center, fmin, fmax, lane, ch_bw = self._getChannelCenterLimits()

        # Use optional input
        center = float(center if freq is None else freq)
        ch_bw = float(ch_bw if bw is None else bw)

        # floating-point error prevention and bandwidth validation
        epsilon = 1e-12
        f_lo = fmin - epsilon
        f_hi = fmax + epsilon

        # Central channel first
        if not (f_lo <= center <= f_hi):
            return
        yield MuChannel(cid, center, ch_bw, isBcst)
        cid += 1

        # Generate channels alternating on either side of center going outward
        n = 1
        while True:
            allocated = False

            # Larger frequency
            freq = center + n * lane
            if (f_lo <= freq <= f_hi):
                yield MuChannel(cid, freq, ch_bw, isBcst)
                cid += 1
                allocated = True

            # Smaller frequency
            freq = center - n * lane
            if (f_lo <= freq <= f_hi):
                yield MuChannel(cid, freq, ch_bw, isBcst)
                cid += 1
                allocated = True
            
            if (not allocated):
                break
            n += 1

    #--------------------------------------------------------------------------
    def _genChannelEdgeIn(self, 
                          cid:int = 0,
                          freq:Optional[float] = None, 
                          bw:Optional[float] = None,
                          isBcst:bool = False
                          )->Generator[MuChannel, None, None]:
        """
        Generates new network channels using an edge-inward strategy.

        First channel borders the lowest frequency in the network bandwidth,
        then increments by channel bandwidth plus guard bandwidth until the
        bandwidth is full.

        Parameters
        ----------
        cid : int, default=0
            Override starting channel id.
        freq : float, optional
            Override starting frequency. Otherwise, starts from lowest available
            channel slot in network bandwith.
        bw : float, optional
            Override channel bandwidth. Otherwise, uses CHAN_MINBW
        isBcst : bool, default=False
            Broadcast flag. All channels generated will carry the same flag.

        Returns
        -------
        MuChannel Generator
            Yields channels with incrementing ID and frequencies. Bandwidth and
            broadcast flag are static.
        """

        # Get useable central frequency limits and bandwidth values
        _, fmin, fmax, lane, ch_bw = self._getChannelCenterLimits()

        # Use optional input
        epsilon = 1e-12     # floating-point error prevention
        if (freq is None):
            freq = fmin
        else:
            # Ensure the channel slots into the planner alignment
            n = max(0, int((freq - fmin) // lane))
            f = fmin + n * lane
            while (f < freq - epsilon):
                n += 1
                f = fmin + n * lane
            freq = f
        ch_bw = float(ch_bw if bw is None else bw)

        # Start channels at lowest edge, increment by lane size
        while (freq <= fmax + epsilon):
            yield MuChannel(cid, freq, ch_bw, isBcst)
            cid += 1
            freq += lane

    #--------------------------------------------------------------------------
    def _getChannelCenterLimits(self)->Tuple[float, float, float, float, float]:
        """
        Compute bounds for channel center frequencies from network bandwidth.

        Returns
        -------
        center : float
            Network base (center) frequency (self.BASE_FREQ).
        fmin_center : float
            Minimum frequency allowed for a channel center frequency (kHz).
        fmax_center : float
            Maximum frequency allowed for a channel center frequency (kHz).
        lane : float
            Bandwidth size of a channel with guard.
        chan_bw : float
            Network channel minimum bandwidth (self.CHAN_MIN).

        Notes
        -----
        Determines upper and lower bound symmetrically around central network
        band defined in BASE_FREQ:

            freq_min = center - half_bandwidth + guard_bw + channel_bw/2
            freq_max = center + half_bandwidth - guard_bw - channel_bw/2

        Uses guard bandwidth as inter-channel spacing.
        """

        center = float(self.BASE_FREQ)
        half_bw = float(self.BANDWIDTH) / 2.0
        guard_bw = float(self.GUARD_BW)
        chan_bw = float(self.CHAN_MINBW)
        freq_min = center - half_bw + guard_bw + chan_bw/2.0
        freq_max = center + half_bw - guard_bw - chan_bw/2.0
        lane = chan_bw + guard_bw
        return center, freq_min, freq_max, lane, chan_bw
        
    #--------------------------------------------------------------------------
    def _makeBroadcastChannel(self, 
                              cid:int=None, 
                              freq:float=None, 
                              bw:float=None)->MuChannel:
        """
        Create a broadcast channel with optional overrides.

        Parameters
        ----------
        cid : int, optional
            Channel ID. Default: BCAST_ADDR.
        freq : float, optional
            Frequency in kHz. Default: BASE_FREQ.
        bw : float, optional
            Bandwidth in kHz. Default: CHAN_MINBW.
        
        Returns
        -------
        MuChannel
            Broadcast channel with isBcst=True.
        """

        cid = self.BCAST_ADDR if cid is None else cid
        freq = float(self.BASE_FREQ) if freq is None else freq
        bw = float(self.CHAN_MINBW) if bw is None else bw

        return MuChannel(cid=cid, freq=freq, bw=bw, isBcst=True)
    
    #--------------------------------------------------------------------------
    def _buildChannels(self)->None:
        """
        Build a network registry of network channels.

        Notes
        -----
        Dynamically assigned during initialization based on access mode
        configurations.

        - accessMode='shared', assigns: _buildChannelsShared
        - accessMode='tdma', assigns: _buildChannelsTDMA  
        - accessMode='tdma_lf', assigns: _buildChannelsTDMALF
        - accessMode='fdma', assigns: _buildChannelsFDMA
        """
        
        raise NotImplementedError(
            "_buildChannels() method should be assigned during initialization")

    #--------------------------------------------------------------------------
    def _buildChannelsShared(self)->None:
        """
        Build a single operational network channel shared by all nodes.

        Notes
        -----
        Creates one channel at BASE_FREQ with id=BCAST_ADDR, isBcst=True.
        All nodes TX and RX on this channel. Clears existing channels first.
        """

        # Reset channels registry
        self.channels.clear()
        
        # Assign new channel iterator
        self._channelGen = self._genChannel()

        # Make new broadcast channel
        bcast = next(self._channelGen)
        bcast.cid = self.BCAST_ADDR
        bcast.isBcst = True

        # Make new channel registry
        self.channels = {self.BCAST_ADDR: bcast}

    #--------------------------------------------------------------------------
    def _buildChannelsTDMA(self)->None:
        """
        Build a single operational channel for TDMA using full bandwidth.

        Notes
        -----
        Similar to shared mode but uses entire BANDWIDTH instead of CHAN_MINBW.
        Channel id=BCAST_ADDR, isBcst=True. Time-domain multiplexing handled by
        protocol schedulers.
        """

        # Reset channels registry
        self.channels.clear()
        
        # Assign new channel iterator
        self._channelGen = self._genChannel()

        # Make new broadcast channel
        bcast = next(self._channelGen)
        bcast.cid = self.BCAST_ADDR
        bcast.bw = self.BANDWIDTH
        bcast.isBcst = True

        # Make new channel registry
        self.channels = {self.BCAST_ADDR: bcast}

    #--------------------------------------------------------------------------
    def _buildChannelsTDMALF(self)->None:
        """
        Build two-channel TDMA network for leader-follower coordination.
        
        Creates a dual-channel architecture optimized for leader-follower swarm
        communication where the leader broadcasts to all followers, and followers
        respond on a dedicated shared channel.
        
        Notes
        -----
        **Channel Architecture:**
        
        Channel 1 - Broadcast (BCAST_ADDR):

        - Purpose: Leader -> All Followers (one-to-many)
        - CID: BCAST_ADDR (typically 255)
        - Frequency: BASE_FREQ
        - Bandwidth: CHAN_MINBW
        - Flag: isBcst=True
        
        Channel 2 - Response (computed CID):

        - Purpose: All Followers -> Leader (many-to-one)
        - CID: Next available from channel generator (typically 1)
        - Frequency: BASE_FREQ + lane offset
        - Bandwidth: CHAN_MINBW
        - Flag: isBcst=False
        
        **Implementation Steps:**

        1. Clear existing channel registry
        2. Create new channel generator
        3. Generate broadcast channel
        4. Generate response channel
        5. Register both channels registry
        
        **Time-Domain Multiplexing:**

        Leader sends messages on broadcast channel during episode window.
        Followers respond on response channel in assigned time slots. This
        prevents collisions while using only two frequency channels.
        
        **Integration:**

        Used with _assignNodesTDMALF() which assigns:

        - Leader: TX on broadcast, RX on response
        - Followers: TX on response, RX on broadcast
        """

        # Reset channels registry
        self.channels.clear()
        
        # Assign new channel iterator
        self._channelGen = self._genChannel()

        # Create a broadcast channel for the leader transmissions
        bcast = next(self._channelGen)
        bcast.cid = self.BCAST_ADDR
        bcast.isBcst = True

        # Create a dedicated response channel for follower transmissions
        rspn = next(self._channelGen)
        
        # Make new channel registry
        self.channels = {
            self.BCAST_ADDR: bcast,
            rspn.cid: rspn,
        }

    #--------------------------------------------------------------------------
    def _buildChannelsFDMA(self)->None:
        """
        Build multiple channels for FDMA frequency-division network.
        
        Generates one unique transmission channel per node currently registered
        in self.nodes. Each node gets exclusive TX frequency to prevent
        collisions without time-domain multiplexing.
        
        Notes
        -----
        **Channel Allocation Strategy:**

        - One channel per node
        - Channel IDs match node addresses where possible
        - Avoids BCAST_ADDR (typically 255) as channel ID
        - Uses channel generator to assign frequencies
        
        **Rebuild Behavior:**

        This method performs a complete rebuild on each call:

        1. Clears all existing channels
        2. Creates new channel generator
        3. Generates channels for all current nodes
        
        Not incremental - regenerates entire channel set. This design ensures
        consistent frequency allocation but means calling during runtime
        rebuilds the full network.
        
        **Channel ID Assignment:**

        Assigns channel ID equal to node ID if node ID is not BCAST_ADDR and is
        not already assigned. Otherwise increments until those conditions are
        met.
        
        **Capacity Limit:**

        Maximum channels limited by CHAN_MAXN property. If capacity exceeded,
        logs warning and stops channel generation. Nodes without channels cannot
        transmit.
        
        **Frequency Assignment:**

        Channels generated in order by order mode strategy (center_out or
        edge_in). First registered node gets first generated frequency, second
        node gets second frequency, etc.
        
        **Integration:**

        Works with _assignNodesFDMA() which assigns:

        - TX: Single dedicated channel per node
        - RX: All channels except own TX channel
        """

        # Reset channels registry
        self.channels.clear()

        # Assign new channel iterator
        self._channelGen = self._genChannel()

        # Get next channel and register
        for nid in sorted(self.nodes.keys()):
            cid = int(nid)
            while cid == self.BCAST_ADDR or cid in self.channels:
                cid += 1
            try:
                channel = next(self._channelGen)
            except StopIteration:
                log.warning(
                    'FDMA CHANNEL ASSIGNMENT FAILURE: CHECK NETWORK SETTINGS')
                break
            channel.cid = cid
            self.channels[cid] = channel

    #--------------------------------------------------------------------------
    def _assignNodes(self, node:MuNode)->str:
        """
        Assign TX/RX channels to node according to network access mode.

        Parameters
        ----------
        node : MuNode
            Network node requesting channel assignment. Must have node.address
            and node.vehicle attributes.
        
        Returns
        -------
        status : str
            Assignment result: 'CONFIRMED' if successful, 'DENIED' if capacity
            exceeded or assignment failed.
        
        Notes
        -----
        Dynamically assigned during initialization based on access mode
        configuration:

        - accessMode='shared', assigns: _assignNodesShared
        - accessMode='tdma', assigns: _assignNodesTDMA
        - accessMode='tdma_lf', assigns: _assignNodesTDMALF  
        - accessMode='fdma', assigns: _assignNodesFDMA
        
        **Channel Sets:**

        Method modifies two node attributes:

        - node.tx_channels: set of channel IDs for transmission
        - node.rx_channels: set of channel IDs tuned for reception
        
        **Status Codes:**

        - 'CONFIRMED': Assignment successful, node ready for communication
        - 'DENIED': Assignment failed (typically FDMA capacity exceeded)
        
        **Side Effects:**

        May trigger channel building if channels not yet created. May update
        other nodes' RX channels (FDMA multi-node updates).
        """

        raise NotImplementedError(
            "_assignNodes() method should be assigned during initialization")

    #--------------------------------------------------------------------------
    def _assignNodesShared(self, node:MuNode)->str:
        """
        Assign node to single shared broadcast channel (TX and RX).

        Assigns all nodes to the single operational channel designated by the
        _buildChannelsShared channel planner.
        
        Parameters
        ----------
        node : MuNode
            Node requesting assignment.
        
        Returns
        -------
        str
            'CONFIRMED' (always successful).
        
        Notes
        -----
        Builds channel if none exists.
        """

        if (self.BCAST_ADDR not in self.channels):
            self._buildChannelsShared()
        node.tx_channels = {self.BCAST_ADDR}
        node.rx_channels = {self.BCAST_ADDR}
        return 'CONFIRMED'

    #--------------------------------------------------------------------------
    def _assignNodesTDMA(self, node:MuNode)->str:
        """
        Assign node to TDMA channel (TX and RX on same channel).

        Assigns all nodes to the operational channel designated by the _buildChannelsTDMA channel planner.
        
        Parameters
        ----------
        node : MuNode
            Node requesting assignment.
        
        Returns
        -------
        str
            'CONFIRMED' (always successful).
        
        Notes
        -----
        Builds TDMA channel if none exists.
        """

        if (self.BCAST_ADDR not in self.channels):
            self._buildChannelsTDMA()
        node.tx_channels = {self.BCAST_ADDR}
        node.rx_channels = {self.BCAST_ADDR}
        return 'CONFIRMED'

    #--------------------------------------------------------------------------
    def _assignNodesTDMALF(self, node:MuNode)->str:
        """
        Assign channels for leader-follower TDMA protocol.
    
        Assigns node to appropriate TX/RX channels based on vehicle role (leader
        or follower). Leader and followers use complementary channel assignments
        to enable bidirectional communication on two channels.
        
        Parameters
        ----------
        node : MuNode
            Network node requesting assignment. Requires node.vehicle.isLeader
            attribute for role determination.
        
        Returns
        -------
        status : str
            Always returns 'CONFIRMED' (no capacity limits in TDMA_LF).
        
        Notes
        -----
        **Channel Verification:**

        Checks that exactly 2 channels exist with proper broadcast
        configuration. Rebuilds channels via _buildChannelsTDMALF() if:

        - Channel count does not equal 2
        - BCAST_ADDR not in channels
        - No non-broadcast channel exists
        
        **Leader Assignment:**
        
        - TX: Broadcast channel (eg 255)    # Transmit on broadcast
        - RX: Response channel  (eg 1)      # Receive on dedicated

        Leader broadcasts messages to all followers simultaneously and listens
        for responses on a separate channel.
        
        **Follower Assignment:**

        - TX: Response channel  (eg 1)      # Transmit direct to leader
        - RX: Broadcast channel (eg 255)    # Receive on broadcast

        Followers listen for leader broadcasts and respond on dedicated channel.
        
        **Role Detection:**

        Uses node.vehicle.isLeader boolean:

        - True: Vehicle is formation leader (assigns leader channels)
        - False: Vehicle is follower (assigns follower channels)
        
        This attribute must be set on vehicle before registration.
        """

        # Verify channels exist
        if ((len(self.channels) != 2) or 
            (self.BCAST_ADDR not in self.channels) or
            (not any(not c.isBcst for c in self.channels.values()))):
            self._buildChannelsTDMALF()

        # Get non-broadcast channel id
        rspn_cid = next(cid for cid in self.channels if cid != self.BCAST_ADDR)

        # If Leader: assign (tx:bcast, rx:rspn)
        if (node.vehicle.isLeader):
            node.tx_channels = {self.BCAST_ADDR}
            node.rx_channels = {rspn_cid}
        else:
            node.tx_channels = {rspn_cid}
            node.rx_channels = {self.BCAST_ADDR}

        return 'CONFIRMED'

    #--------------------------------------------------------------------------
    def _assignNodesFDMA(self, node:MuNode)->str:
        """
        Assign exclusive TX channel and multi-RX for FDMA frequency division.
    
        Each node receives one dedicated TX channel (no other node transmits on
        that frequency) and listens on all channels except its own. This
        eliminates collisions via frequency separation rather than time-domain
        multiplexing.
        
        Parameters
        ----------
        node : MuNode
            Network node requesting channel assignment. Uses node.address as
            preferred TX channel ID.
        
        Returns
        -------
        status : str
            'CONFIRMED' if assignment successful, 'DENIED' if bandwidth capacity
            exceeded.
        
        Notes
        -----
        **Capacity Check:**

        If the number of channels exceeds the maximum capable by the network
        bandwidth (computed in CHAN_MAXN), the node is denied channel
        assignment.
        
        Maximum channels = (BANDWIDTH - 2*GUARD_BW) // (CHAN_MINBW + GUARD_BW).

        Default policy is to deny further assignment if FDMA overflow occurs. No
        alternatives are in development, but possibilities to consider are an
        RX-only fallback with no TX, or fallback to TX on a broadcast channel.
        
        **Channel ID Selection:**

        Prefers node.address as channel ID but adjusts if that ID is already
        taken or if is equal to BCAST_ADDR. If there is a conflict, then channel
        ID is incremented until first success.
        
        **Channel Generation:**

        Uses channel generator to get next frequency in sequence determined by
        order mode configuration (e.g. center-out or edge-in), and the channel
        ID is assigned. If the generator is exhausted (StopIteration), logs
        warning and returns 'DENIED'.
        
        **TX and RX Assignment:**

        - node.tx_channels = {cid}                    # Single dedicated channel
        - node.rx_channels = set(channels) - {cid}    # All except own TX
        
        **Multi-Node Update:**

        Updates all existing nodes when new channel is assigned, ensuring all
        nodes can receive from newly registered nodes.
        
        **Generator State:**

        Verifies channel generator exists before use and creates one if not
        found. The generator may be None after unpickling.

        **Error Handling:**

        Logs detailed error with bandwidth parameters if capacity exceeded:
        
        - Current BANDWIDTH setting
        - CHAN_MINBW requirement
        - GUARD_BW overhead
        """

        # Verify channel generator exists
        if (not ('_channelGen' in self.__dict__) or self._channelGen is None):
            self._channelGen = self._genChannel()

        # Verify network channel capacity
        if (len(self.channels) >= self.CHAN_MAXN):
            log.error(
                '%s: FDMA CHANNEL ASSIGNMENT FAILURE: ' +
                'CAPACITY REACHED - Increase BANDWIDTH (%.2f) or ' +
                'Reduce CHAN_MINBW (%.2f), GUARD_BW (%.2f)',
                node.vehicle.callSign, self.BANDWIDTH, 
                self.CHAN_MINBW, self.GUARD_BW
            )
            return 'DENIED'
        
        # Get and register next channel
        cid = int(node.address)
        while cid == self.BCAST_ADDR or cid in self.channels:
                cid += 1
        try:
            channel = next(self._channelGen)
        except StopIteration:
            log.warning(
                '%s: FDMA CHANNEL ASSIGNMENT FAILURE: CHECK NETWORK SETTINGS',
                node.vehicle.callSign)
            return 'DENIED'
        channel.cid = cid
        self.channels[cid] = channel

        # Assign single TX on channel and RX on all other channels
        node.tx_channels = {cid}
        node.rx_channels = set(self.channels.keys()) - {cid}

        # Update RX channel lists on existing nodes
        self._updateNodeRXFDMA()

        return 'CONFIRMED'

    #--------------------------------------------------------------------------
    def _updateNodeRXFDMA(self)->None:
        """
        Update all nodes' RX channels when new TX channel assigned in FDMA.
    
        Notes
        -----
        Sets each node's rx_channels to all channel IDs except its own TX
        channel.
        Called after new channel created in _assignNodesFDMA().
        """

        allCids = set(self.channels.keys()) - {self.BCAST_ADDR}
        for node in self.nodes.values():
            myTX = next(iter(node.tx_channels), None)
            node.rx_channels = allCids - ({myTX} if myTX is not None else set())

###############################################################################

@lru_cache(maxsize=5)
def getMsgStruct(msgType:str)->cst.Struct:
    """
    Return binary message structure for serialization/parsing.
    
    Uses Python construct library to define structured binary formats for three
    message types: REPORT, LEADER-REPORT, and BROADCAST/REQUEST. Structures
    specify field names, types, and byte layout for network transmission.
    
    Parameters
    ----------
    msgType : {'RPRT', 'LRPT', 'BCRQ', 'REPORT', 'RESPONSE', 'LEADER-REPORT', 'BROADCAST/REQUEST'}
        Message type identifier. Case-insensitive. Multiple aliases supported.
    
    Returns
    -------
    cst.Struct
        Construct library Struct object defining message format. Use
        .build(dict) to serialize and .parse(bytes) to deserialize.
    
    Notes
    -----
    **Message Structures:**
    
    .. code-block:: none

        RPRT (REPORT / RESPONSE) - 37 bytes:
        {
            'type': b'RPRT' or b'RSPN',      # 4 bytes  - Message type flag
            'group_id': bytes,               # 1 byte   - Group identifier
            'reporter_id': int,              # 4 bytes  - Sender vehicle ID
            'reporter_pos': [x, y, z],       # 12 bytes - Position (3x float32)
            'reporter_vel': [vx, vy, vz],    # 12 bytes - Velocity (3x float32)
            'time_stamp': float,             # 4 bytes  - Transmission time
        }
        
        LRPT (LEADER-REPORT) - 57 bytes:
        {
            'type': b'LRPT',                 # 4 bytes  - Const type flag
            'group_id': bytes,               # 1 byte   - Group identifier
            'leader_pos': [x, y, z],         # 12 bytes - Current position
            'leader_vel': [vx, vy, vz],      # 12 bytes - Current velocity
            'leader_next_pos': [x, y, z],    # 12 bytes - Next waypoint
            'leader_next_vel': [vx, vy, vz], # 12 bytes - Next velocity
            'time_stamp': float,             # 4 bytes  - Transmission time
        }
        
        BCRQ (BROADCAST/REQUEST) - Variable (353 bytes with 9 followers):
        {
            'type': b'BCRQ',                     # 4 bytes - Const type flag
            'group_id': bytes,                   # 1 byte
            'leader_pos': [x, y, z],             # 12 bytes
            'leader_vel': [vx, vy, vz],          # 12 bytes
            'leader_next_pos': [x, y, z],        # 12 bytes
            'leader_next_vel': [vx, vy, vz],     # 12 bytes
            'num_followers': int,                # 4 bytes - Array size = N
            'followers_pos': [[x,y,z], ...],     # 12*N bytes - Position arrays
            'followers_vel': [[vx,vy,vz], ...],  # 12*N bytes - Velocity arrays
            'followers_rspn': [t1, t2, ...],     # 4*N bytes - Response times
            'rspn_sched': [id1, id2, ...],       # 4*N bytes - Response order
            'rspn_start': float,                 # 4 bytes - Schedule start time
            'time_stamp': float,                 # 4 bytes
        }
        Total: 65 + 32xN bytes
    
    **Field Types:**

    - Float32l: Little-endian single-precision float (4 bytes)
    - Int32ul: Unsigned little-endian integer (4 bytes)
    - Bytes(n): Raw byte sequence of fixed length
    - Array[type]: Dynamic-length array based on num_followers field
    
    **Type Flag:**

    First 4 bytes identify message type for recvMsgCallback() dispatcher.
    RPRT/LRPT use Const() to enforce correct flag during build. REPORT allows
    variable flag (b'RPRT' or b'RSPN') for flexibility.
    
    **Transmission:**

    Only values are packed in serialization build, not keys. Keys are implicit in
    structure definition. Receiver must know message type to parse correctly.
    
    **LRU Caching:**

    Uses Least Recently Used (LRU) caching to store compiled structures in memory.
    Construct compilation has overhead; caching avoids recompilation on repeated
    calls. Maxsize=5 sufficient for 3 message types plus aliases.
    

    References
    ----------
    [1] construct library: https://construct.readthedocs.io/

    
    Examples
    --------
    ### Serialization:
    
    >>> msg_struct = getMsgStruct('REPORT')
    >>> bytes_msg = msg_struct.build({
    ...     'type': b'RPRT',
    ...     'group_id': b'A',
    ...     'reporter_id': 1,
    ...     'reporter_pos': [100.0, 200.0, 25.0],
    ...     'reporter_vel': [1.5, 0.5, 0.0],
    ...     'time_stamp': 123.45,
    ... })
    
    ### Deserialization:

    >>> msg_dict = msg_struct.parse(bytes_msg)
    >>> pos_1 = msg_dict['reporter_pos']  # [100.0, 200.0, 25.0]
    """

    # Define Field Formats
    fltType = cst.Float32l                  # single precision
    arrFltType = fltType[3]                 # array size to hold (x,y,z)
    intType = cst.Int32ul                   # standard integer
    chrType = cst.Bytes(1)                  # b'', single character
    flgType = cst.Bytes(4)                  # b'', four characters

    # Define Message Structures
    # REPORT  ::  37 bytes (as of 05/01/2024)
    RPRT = cst.Struct(
        "type"              / flgType,
        "group_id"          / chrType,
        "reporter_id"       / intType,
        "reporter_pos"      / arrFltType,
        "reporter_vel"      / arrFltType,
        "time_stamp"        / fltType,
    )

    # LEADER REPORT :: 57 bytes (as if 05/01/2024)
    LRPT = cst.Struct(
        "type"              / cst.Const(b'LRPT'),
        "group_id"          / chrType,
        "leader_pos"        / arrFltType,
        "leader_vel"        / arrFltType,
        "leader_next_pos"   / arrFltType,
        "leader_next_vel"   / arrFltType,
        "time_stamp"        / fltType,
    )

    # BROADCAST / REQUEST  ::  353 bytes with 9 Followers (as of 05/01/2024)
    BCRQ = cst.Struct(
        "type"              / cst.Const(b'BCRQ'),
        "group_id"          / chrType,
        "leader_pos"        / arrFltType,
        "leader_vel"        / arrFltType,
        "leader_next_pos"   / arrFltType,
        "leader_next_vel"   / arrFltType,
        "num_followers"     / intType,
        "followers_pos"     / arrFltType[cst.this.num_followers], #array of arrays
        "followers_vel"     / arrFltType[cst.this.num_followers], #array of arrays
        "followers_rspn"    / fltType[cst.this.num_followers],
        "rspn_sched"        / intType[cst.this.num_followers],
        "rspn_start"        / fltType,
        "time_stamp"        / fltType,
    )

    # Map Message Type to Message Structure
    msgStructures = {
        # RPRT
        'RPRT': RPRT,
        'REPORT': RPRT,
        'RSPN': RPRT,
        'RESPONSE': RPRT,
        # LRPT
        'LRPT': LRPT,
        'LREPORT': LRPT,
        'LEADER-REPORT': LRPT,
        'LEADERREPORT': LRPT,
        # BCRQ
        'BCRQ': BCRQ,
        'BROADCAST-REQUEST': BCRQ,
        'BROADCAST/REQUEST': BCRQ,
    }

    return msgStructures.get(msgType.upper())

###############################################################################

def recvMsg(vehicle:Vehicle)->None:
    """
    Spawn a new receiver thread for vehicle message listening.
    
    Parameters
    ----------
    vehicle : Vehicle
        Vehicle with CommNetwork attribute.
    
    Returns
    -------
    Thread
        Started daemon thread running network receiver loop.
    
    Notes
    -----
    - Only used by AquaNet communication network
    - Thread calls vehicle.CommNetwork.recv() with recvMsgCallback as handler.
    - Daemon thread terminates when main program exits.
    """

    # Start Receiving Server
    log.info('%s: LISTENING AT %s...', vehicle.callSign, vehicle.nodeAddr)
    t = Thread(target=vehicle.CommNetwork.recv,args=(recvMsgCallback,vehicle,))
    t.start()

    return t

###############################################################################

def recvMsgCallback(bytesMsg:bytes, vehicle:Vehicle)->None:
    """
    Dispatch incoming network message to appropriate handler by type flag.
    
    Extracts 4-byte message type identifier from header and routes message to
    corresponding parser (recvReport, recvLeaderReport, or recvBroadcastReq).
    Validates type flag for corruption before routing.
    
    Parameters
    ----------
    bytesMsg : bytes
        Complete received message including 4-byte type flag header.
    vehicle : Vehicle
        Vehicle receiving the message. Must have groupId and callSign
        attributes.
    
    Notes
    -----
    **Message Routing Table:**
    
    Type Flag -> Handler Function:

    - b'RPRT' -> recvReport(vehicle, bytesMsg)
    - b'RSPN' -> recvReport(vehicle, bytesMsg)  
    - b'LRPT' -> recvLeaderReport(vehicle, bytesMsg)
    - b'BCRQ' -> recvBroadcastReq(vehicle, bytesMsg)
    
    **Type Flag Extraction:**

    >>> msgType = bytesMsg[:4]  # First 4 bytes
    
    Type flags use fixed 4-byte ASCII encoding (e.g., b'RPRT', b'LRPT').
    
    **Validation Flow:**

    1. Extract type flag from message header
    2. Check flag against dataIsCorrupted('type') validator
    3. If corrupted: Attempt recovery via restoreCorrupt()
    4. If recovery fails: Log error and discard message
    5. If valid: Route to appropriate handler function
    
    **Error Handling:**
    
    Catches and logs multiple exception types:

    - UnicodeDecodeError: Corrupted type flag cannot decode to ASCII
    - IndexError: Message too short (< 4 bytes)
    - KeyError: Unknown message type flag
    - General Exception: Unexpected parsing errors
    
    All errors logged with vehicle callSign for debugging. Messages with
    unrecoverable errors are silently dropped (no delivery to handler).
    
    **Handler Responsibilities:**
    
    This callback performs only routing. Individual handlers are responsible
    for:
    
    - Full message parsing via getMsgStruct()
    - Group membership verification
    - Field-level data validation
    - State updates to vehicle/target/group objects
    """

    # Verify Message Integrity
    if (not bytesMsg or len(bytesMsg) == 0):
        log.warning('%s: CALLBACK SKIP - EMPTY MESSAGE', vehicle.callSign)
        return
    
    flagSize = 4
    if (len(bytesMsg) < flagSize):
        log.warning('%s: CALLBACK SKIP - SHORT MESSAGE', vehicle.callSign)
        return

    # Dispatch by Message Type
    try:
        # Validate Message Type
        flag = bytesMsg[:flagSize]
        if (dataIsCorrupted(flag, 'type')):
            log.warning('%s: CORRUPT DATA: FLAG', vehicle.callSign)
            flag, ok = restoreCorrupt(vehicle, 'type', flag)
            if (not ok):
                log.warning('%s: CALLBACK SKIP - CORRUPT FLAG', 
                            vehicle.callSign)
                return
            # Replace correpted flag with recovered flag
            bytesMsg = flag + bytesMsg[flagSize:]

        # Extract Message Type
        msgType = flag.decode('utf-8', errors='ignore')
        log.debug('%s: CALLBACK on %s', vehicle.callSign, msgType)

        # Map Message Type to Receiving Function
        msgHandlers = {
            'RPRT': recvReport,
            'RSPN': recvReport,
            'LRPT': recvLeaderReport,
            'BCRQ': recvBroadcastReq,
        }

        # Forward to Appropriate Function
        recvMessage = msgHandlers.get(msgType)
        if (recvMessage):
            recvMessage(vehicle, bytesMsg)
        else:
            log.warning('%s: CALLBACK SKIP - BAD FLAG: %s', 
                        vehicle.callSign, msgType)
    
    except UnicodeDecodeError:
        log.warning('%s: CALLBACK SKIP - FLAG DECODE ERROR', vehicle.callSign)

    except Exception as e:
        log.error('%s: CALLBACK SKIP - MESSAGE ERROR: %s', 
                  vehicle.callSign, str(e))

###############################################################################

def writeReport(vehicle:Vehicle, msgType:str='REPORT')->bytes:
    """
    Serialize vehicle state into REPORT message.
    
    Parameters
    ----------
    vehicle : Vehicle
        Vehicle with eta, velocity, clock, groupId, id attributes.

        - eta: vehicle position / attitude vector
        - velocity: vehicle velocity vector (END frame)
        - clock: simulation time
        - groupId: swarm group id
        - id: unique vehicle id

    msgType : {'REPORT', 'RESPONSE'}, default='REPORT'
        Message type determines flag: REPORT -> b'RPRT', RESPONSE -> b'RSPN'.
    
    Returns
    -------
    bytes
        Serialized RPRT message ready for transmission.
    
    Notes
    -----
    **REPORT vs RESPONSE:**
    
    Functionally identical message formats with different type flags:

    - REPORT (b'RPRT'): Standard vehicle state broadcast
    - RESPONSE (b'RSPN'): Reply in TDMA_LF scheduled response slot
    
    Type flag enables receivers to distinguish message context but both
    parsed identically by recvReport().
    
    **Message Payload (37 bytes):**

    Uses getMsgStruct('RPRT') for message format. Contains:

    .. code-block:: none

        {
            'type': b'RPRT' or b'RSPN',      # 4 bytes
            'group_id': bytes,               # 1 byte
            'reporter_id': int,              # 4 bytes
            'reporter_pos': [x,y,z],         # 12 bytes (3x float32)
            'reporter_vel': [vx,vy,vz],      # 12 bytes (3x float32)
            'time_stamp': float,             # 4 bytes
        }
    """

    # Map Message Type to Type Flag
    msgFlags = {
        'RPRT': b'RPRT',
        'RSPN': b'RSPN',
        'RESPONSE': b'RSPN',
    }
    
    # Build Message
    rprt = getMsgStruct(msgType)
    flag = msgFlags.get(msgType.upper(), b'RPRT')
    rprtMsg = rprt.build(dict(type          = flag,
                              group_id      = vehicle.groupId.encode(),
                              reporter_id   = vehicle.id,
                              reporter_pos  = vehicle.eta[:3].tolist(),
                              reporter_vel  = vehicle.velocity[:3].tolist(),
                              time_stamp    = vehicle.clock,
                              ))
    
    return rprtMsg

###############################################################################

def sendReport(vehicle:Vehicle,
               msgType:str='REPORT',
               address:Optional[int]=None,
               )->None:
    """
    Serialize and send vehicle state REPORT message over network.
    
    Packages current vehicle position, velocity, and timestamp into binary
    message format and transmits on assigned TX channel.
    
    Parameters
    ----------
    vehicle : Vehicle
        Vehicle sending report. Requires eta, velocity, clock, nodeAddr,
        groupId, callSign, CommNetwork, and txAddr attributes.

        - eta: vehicle position / attitude vector
        - velocity: vehicle velocity vector (END frame)
        - clock: simulation time
        - nodeAddr: unique address of network node
        - groupId: swarm group id
        - callSign: unique vehicle identification string
        - CommNetwork: a communication network manager instance
        - txAddr: transmission channel id

    msgType : {'REPORT', 'RESPONSE'}, default='REPORT'
        Message type flag written to header. 'REPORT' -> b'RPRT', 
        'RESPONSE' -> b'RSPN'. Both use same message structure.
    address : int, optional
        Override transmission address. If None, uses vehicle.txAddr.
        Determines which TX channel to send on.
    
    Notes
    -----
    **Message Construction Flow:**

    1. Call writeReport to serialize state
    2. Determine transmission address
    3. Call vehicle.CommNetwork.send(message, address)
    
    **REPORT vs RESPONSE:**
    
    Functionally identical message formats with different type flags:

    - REPORT (b'RPRT'): Standard vehicle state broadcast
    - RESPONSE (b'RSPN'): Reply in TDMA_LF scheduled response slot
    
    Type flag enables receivers to distinguish message context but both
    parsed identically by recvReport().
    
    **Message Payload (37 bytes):**

    .. code-block:: none

        {
            'type': b'RPRT' or b'RSPN',      # 4 bytes
            'group_id': bytes,               # 1 byte
            'reporter_id': int,              # 4 bytes
            'reporter_pos': [x,y,z],         # 12 bytes (3x float32)
            'reporter_vel': [vx,vy,vz],      # 12 bytes (3x float32)
            'time_stamp': float,             # 4 bytes
        }
    """
    
    # Write Message
    rprtMsg = writeReport(vehicle, msgType=msgType)

    #Send Message
    if (address is None):
        address = vehicle.txAddr
    log.debug('%s: %s', vehicle.callSign, msgType)
    vehicle.CommNetwork.send(rprtMsg, address)

###############################################################################

def recvReport(vehicle:Vehicle, bytesMsg:bytes)->None:
    """
    Parse and process received REPORT message from group member.
    
    Deserializes REPORT/RESPONSE message, validates all fields for corruption,
    attempts recovery where possible, and updates receiving vehicle's group
    member state with reporter position, velocity, and timestamp.
    
    Parameters
    ----------
    vehicle : Vehicle
        Vehicle receiving message. Requires groupId, group (list), and 
        callSign attributes.

        - groupId: swarm group id
        - group: list of swarm group neighbor data models
        - callSign: unique vehicle identification string

    bytesMsg : bytes
        Complete serialized REPORT message (37 bytes).
    
    Notes
    -----
    **Group Member Update:**
    
    If reporter is found in vehicle.group and all message fields valid, an
    internal data model of the reporter is updated with: 

    .. code-block:: none

        - reporter_pos                   # Reporter Position
        - reporter_vel                   # Reporter Velocity
        - time_stamp                     # Last message time
        - rprtRecv = True                # Flag: report received from this cycle
        - writeEtaVelLogs(reporter)      # Append to position/velocity log
    
    **Field Validation Sequence:**
    
    Each individual message field is checked for data corruption via
    dataIsCorrupted(). If corruption is detected, then data recovery is
    attempted via restoreCorrupt(). Data recovery strategies for each field can
    be defined in the vehicle.dataRepair dictionary.
    
    **Group Membership Verification:**
    
    Rejects messages from vehicles not in same group. Also rejects if
    reporter_id not found in vehicle.group list (may be former group member or
    incorrectly addressed message).
    
    **Message Parsing Flow:**

    1. Parse binary message via getMsgStruct('RPRT')
    2. Extract and validate reporter_id
    3. Extract and validate group_id
    4. Verify vehicle in same group (compare group IDs)
    5. Locate reporter in vehicle.group list by ID
    6. Validate reporter_pos, reporter_vel, time_stamp
    7. Update internal data model of group member if all validations pass
    """
    
    try:
        # Unpack Message
        rprt = getMsgStruct('RPRT')
        msg = rprt.parse(bytesMsg)

        # Validate message metadata
        reporter_id = msg.reporter_id
        if (dataIsCorrupted(reporter_id, 'reporter_id', uid=vehicle.id)):
            log.warning('%s: CORRUPT DATA: REPORTER ID', vehicle.callSign)
            reporter_id, ok = restoreCorrupt(vehicle, 'reporter_id', 
                                             reporter_id)
            if (not ok):
                return
        
        gid = msg.group_id
        if (dataIsCorrupted(gid, 'group_id')):
            log.warning('%s: CORRUPT DATA: GROUP ID', vehicle.callSign)
            gid, ok = restoreCorrupt(vehicle, 'group_id', gid, id=reporter_id)
            if (not ok):
                return
        group_id = gid.decode('utf-8', errors='ignore')
        
        # Get message title (type was validated in callback)
        title = (f"{msg.type.decode('utf-8', errors='ignore')}"
                 f"{group_id}{reporter_id}")

        # Verify Vehicle is Part of Message Group
        if (vehicle.groupId != group_id):
            log.warning('%s: SKIP %s - NOT IN GROUP', vehicle.callSign, title)
            return

        # Find Message Sender in Group List
        isSenderFound = False
        for m in vehicle.group:
            if (m.id == reporter_id):
                try:
                    # Validate data
                    rpos = msg.reporter_pos
                    if (dataIsCorrupted(rpos, 'position')):
                        log.warning('%s: %s - CORRUPT DATA: POSITION',
                                    vehicle.callSign, title)
                        rpos, ok = restoreCorrupt(vehicle, 'reporter_pos', rpos, 
                                                  id=reporter_id)
                        if (not ok):
                            return
                    reporter_pos = np.array(rpos)

                    rvel = msg.reporter_vel
                    if (dataIsCorrupted(rvel, 'velocity')):
                        log.warning('%s: %s - CORRUPT DATA: VELOCITY',
                                    vehicle.callSign, title)
                        rvel, ok = restoreCorrupt(vehicle, 'reporter_vel', rvel, 
                                                  id=reporter_id)
                        if (not ok):
                            return
                    reporter_vel = np.array(rvel)
                    
                    time_stamp = msg.time_stamp
                    if (dataIsCorrupted(time_stamp, 'timestamp')):
                        log.warning('%s: %s - CORRUPT DATA: TIME',
                                    vehicle.callSign, title)
                        time_stamp, ok = restoreCorrupt(vehicle, 
                                                        'time_stamp', 
                                                        time_stamp, 
                                                        id=reporter_id)
                        if (not ok):
                            return

                    # Update Data
                    m.eta = reporter_pos
                    m.velocity = reporter_vel
                    m.timeLastMsg = time_stamp
                    writeEtaVelLogs(m)
                    # Mark 'RPRT Received' flag to 'Yes'
                    m.rprtRecv = True

                    isSenderFound = True
                    log.debug('%s: RECEIVED %s', vehicle.callSign, title)
                    return
                
                except (ValueError, TypeError) as e:
                    log.warning('%s: DATA CORRUPTION %s: %s',
                                vehicle.callSign, title, str(e))
                    return
        
        if (not isSenderFound):
            log.warning('%s: SKIP %s - SENDER NOT IN GROUP',
                      vehicle.callSign, title)
            
    except Exception as e:
        log.error('%s: RPRT MESSAGE ERROR: %s', vehicle.callSign, str(e))

###############################################################################

def writeLeaderReport(vehicle:Vehicle)->bytes:
    """
    Serialize leader state into LRPT message.
    
    Parameters
    ----------
    vehicle : Vehicle
        Leader vehicle with eta, velocity, clock, groupId attributes.

        - eta: vehicle position / attitude vector
        - velocity: vehicle velocity vector (END frame)
        - clock: simulation time
        - groupId: swarm group id
    
    Returns
    -------
    bytes
        Serialized LRPT message ready for transmission.
    
    Notes
    -----
    **Message Payload (57 bytes):**
    
    Uses getMsgStruct('LRPT') for format. Contains:

    .. code-block:: none

        {
            'type': b'LRPT',                 # 4 bytes - Constant
            'group_id': bytes,               # 1 byte
            'leader_pos': [x,y,z],           # 12 bytes
            'leader_vel': [vx,vy,vz],        # 12 bytes
            'leader_next_pos': [x,y,z],      # 12 bytes
            'leader_next_vel': [vx,vy,vz],   # 12 bytes
            'time_stamp': float,             # 4 bytes
        }

    **Next Waypoint Prediction:**
    
    Uses guidance.predictNextEtaVel() to estimate leader's next position.
    Prediction based on current path following state, typically looking ahead to
    next waypoint in vehicle.waypoints queue.
    """

    # Prepare Message Contents
    vehicle.nextEta, vehicle.nextVel = guid.predictNextEtaVel(vehicle)

    # Write Message
    lrpt = getMsgStruct('LRPT')
    lrptMsg = lrpt.build(dict(group_id        = vehicle.groupId.encode(),
                              leader_pos      = vehicle.eta[:3].tolist(),
                              leader_vel      = vehicle.velocity[:3].tolist(),
                              leader_next_pos = vehicle.nextEta[:3].tolist(),
                              leader_next_vel = vehicle.nextVel[:3].tolist(),
                              time_stamp      = vehicle.clock,
                              ))

    return lrptMsg

###############################################################################

def sendLeaderReport(vehicle:Vehicle)->None:
    """
    Serialize and send leader state LEADER-REPORT message to followers.
    
    Packages leader position, velocity, next waypoint position, and next
    velocity into binary message and transmits on leader's TX channel.
    
    Parameters
    ----------
    vehicle : Vehicle
        Leader vehicle sending report. Requires eta, velocity, clock, nodeAddr,
        groupId, callSign, CommNetwork, txAddr, and navigation attributes for
        waypoint prediction.

        - eta: leader position / attitude vector
        - velocity: leader velocity vector (END frame)
        - clock: simulation time
        - nodeAddr: unique address of network node
        - groupId: swarm group id
        - callSign: unique vehicle identification string
        - CommNetwork: a communication network manager instance
        - txAddr: transmission channel id
    
    Notes
    -----
    **Message Construction Flow:**

    1. Predict next waypoint: guid.predictNextEtaVel(vehicle)
    2. Call writeLeaderReport(vehicle) to serialize complete leader state
    3. Transmit via vehicle.CommNetwork.send() on vehicle.txAddr
    
    **Message Payload (57 bytes):**
    
    .. code-block:: none

        {
            'type': b'LRPT',                 # 4 bytes - Constant
            'group_id': bytes,               # 1 byte
            'leader_pos': [x,y,z],           # 12 bytes
            'leader_vel': [vx,vy,vz],        # 12 bytes
            'leader_next_pos': [x,y,z],      # 12 bytes
            'leader_next_vel': [vx,vy,vz],   # 12 bytes
            'time_stamp': float,             # 4 bytes
        }
    
    **Next Waypoint Prediction:**
    
    Uses guidance.predictNextEtaVel() to estimate leader's next position.
    Prediction based on current path following state, typically looking ahead to
    next waypoint in vehicle.waypoints queue.
    """

    # Write Message
    msgType = 'LEADER-REPORT'
    lrptMsg = writeLeaderReport(vehicle)

    # Send Message
    log.debug('%s: %s', vehicle.callSign, msgType)
    vehicle.CommNetwork.send(lrptMsg, vehicle.txAddr)

###############################################################################

def recvLeaderReport(vehicle:Vehicle, bytesMsg:bytes)->None:
    """
    Parse and process LEADER-REPORT message from leader vehicle.
    
    Deserializes leader report, validates all fields, and updates follower's
    target tracking state with leader position, velocity, next waypoint, and
    message metadata.
    
    Parameters
    ----------
    vehicle : Vehicle
        Follower vehicle receiving message. Requires groupId, target, clock,
        and callSign attributes.

        - groupId: swarm group id
        - target: swarm group leader data model
        - clock: simulation time
        - callSign: unique vehicle identification string

    bytesMsg : bytes
        Complete serialized LRPT message (57 bytes).
    
    Notes
    -----
    **Target Object Updates:**
    
    If all message fields are valid, an internal data model of the target is
    updated with:

    .. code-block:: none

        - leader_pos                        # Leader position
        - leader_vel                        # Leader velocity
        - leader_next_pos                   # Next waypoint
        - leader_next_vel                   # Next leader velocity
        - time_stamp                        # Message timestamp
        - clock - timestamp                 # Age of message
        - writeEtaVelLogs(target)           # Append to position/velocity log
    
    **Field Validation:**
    
    Each individual message field is checked for data corruption via
    dataIsCorrupted(). If corruption is detected, then data recovery is
    attempted via restoreCorrupt(). Data recovery strategies for each field can
    be defined in the vehicle.dataRepair dictionary.
    
    **Message Parsing Flow:**

    1. Parse message via getMsgStruct('LRPT')
    2. Validate group_id and verify membership
    3. Validate leader_pos, leader_vel (current state)
    4. Validate leader_next_pos, leader_next_vel (predicted state)
    5. Validate time_stamp
    6. Update vehicle.target with all fields if valid
    7. Compute message delay: vehicle.clock - time_stamp
    """

    try:
        # Unpack Message
        lrpt = getMsgStruct('LRPT')
        msg = lrpt.parse(bytesMsg)

        # Validate message metadata
        gid = msg.group_id
        if (dataIsCorrupted(gid, 'group_id')):
            log.warning('%s: CORRUPT DATA: GROUP ID', vehicle.callSign)
            gid, ok = restoreCorrupt(vehicle, 'group_id', gid)
            if (not ok):
                return
        group_id = gid.decode('utf-8', errors='ignore')
        
        # Get message title (type was validated in callback)
        title = (f"{msg.type.decode('utf-8', errors='ignore')}{group_id}")

        # Verify Vehicle is Part of Message Group
        if (vehicle.groupId != group_id):
            log.warning('%s: SKIP %s - NOT IN GROUP', vehicle.callSign, title)
            return
        
        try:
            # Validate data
            lpos = msg.leader_pos
            if (dataIsCorrupted(lpos, 'position')):
                log.warning('%s: %s - CORRUPT DATA: POSITION',
                            vehicle.callSign, title)
                lpos, ok = restoreCorrupt(vehicle, 'leader_pos', lpos, 
                                          isLeader=True)
                if (not ok):
                    return
            leader_pos = np.array(lpos)

            lvel = msg.leader_vel
            if (dataIsCorrupted(lvel, 'velocity')):
                log.warning('%s: %s - CORRUPT DATA: VELOCITY',
                            vehicle.callSign, title)
                lvel, ok = restoreCorrupt(vehicle, 'leader_vel', lvel, 
                                          isLeader=True)
                if (not ok):
                    return
            leader_vel = np.array(lvel)

            lnpos = msg.leader_next_pos
            if (dataIsCorrupted(lnpos, 'position')):
                log.warning('%s: %s - CORRUPT DATA: NEXT POSITION',
                            vehicle.callSign, title)
                lnpos, ok = restoreCorrupt(vehicle, 'leader_next_pos', lnpos)
                if (not ok):
                    return
            leader_next_pos = np.array(lnpos)

            lnvel = msg.leader_next_vel
            if (dataIsCorrupted(lnvel, 'velocity')):
                log.warning('%s: %s - CORRUPT DATA: NEXT VELOCITY',
                            vehicle.callSign, title)
                lnvel, ok = restoreCorrupt(vehicle, 'leader_next_vel', lnvel)
                if (not ok):
                    return
            leader_next_vel = np.array(lnvel)
            
            time_stamp = msg.time_stamp
            if (dataIsCorrupted(time_stamp, 'timestamp')):
                log.warning('%s: %s - CORRUPT DATA: TIME',
                            vehicle.callSign, title)
                time_stamp, ok = restoreCorrupt(vehicle, 'time_stamp', 
                                                time_stamp, isLeader=True)
                if (not ok):
                    return

            # Update Target Data
            vehicle.target.eta      = leader_pos
            vehicle.target.velocity = leader_vel
            vehicle.target.nextEta  = leader_next_pos
            vehicle.target.nextVel  = leader_next_vel
            vehicle.target.timeLastMsg = time_stamp
            vehicle.target.delayLastMsg = vehicle.clock - time_stamp
            writeEtaVelLogs(vehicle.target)

        except (ValueError, TypeError) as e:
            log.warning('%s: DATA CORRUPTION %s: %s',
                        vehicle.callSign, title, str(e))
    
    except Exception as e:
        log.error('%s: LRPT MESSAGE ERROR: %s', vehicle.callSign, str(e))

###############################################################################

def writeBroadcastReq(vehicle:Vehicle)->bytes:
    """
    Serialize leader and all follower states into BROADCAST/REQUEST message.
    
    Packages complete swarm state (leader + all followers) into single broadcast
    message with response schedule. Used in protocols for coordinated
    leader-to-followers communication with scheduled response slots.
    
    Parameters
    ----------
    vehicle : Vehicle
        Leader vehicle sending broadcast. Requires eta, velocity, clock,
        groupId, group, rspnSched, nextEta, nextVel attributes.

        - eta: leader position / attitude vector
        - velocity: leader velocity vector (END frame)
        - clock: simulation time
        - groupId: swarm group id
        - group: list of swarm group follower data models
        - rspnSched: list of follower ids in scheduled response order
        - nextEta: leader next position
        - nextVel: leader next velocity
    
    Returns
    -------
    bytes
        Serialized BCRQ message ready for network transmission.
    
    Notes
    -----
    **Message Structure (Variable Size):**
    
    Base size: 65 bytes (leader data + metadata)
    Per-follower: 32 bytes (position + velocity + response time + ID)
    Total: 65 + (32 * num_followers) bytes
    
    Example with 9 followers: 65 + (32 * 9) = 353 bytes
    
    **Message Fields:**
    
    .. code-block:: none

     {
         'type': b'BCRQ',                     # 4 bytes - Constant
         'group_id': bytes,                   # 1 byte
         'leader_pos': [x,y,z],               # 12 bytes - Current position
         'leader_vel': [vx,vy,vz],            # 12 bytes - Current velocity
         'leader_next_pos': [x,y,z],          # 12 bytes - Next waypoint
         'leader_next_vel': [vx,vy,vz],       # 12 bytes - Next velocity
         'num_followers': int,                # 4 bytes - Array size
         'followers_pos': [[x,y,z], ...],     # 12*N bytes - Follower positions
         'followers_vel': [[vx,vy,vz], ...],  # 12*N bytes - Follower velocities
         'followers_rspn': [t1, t2, ...],     # 4*N bytes - Response times
         'rspn_sched': [id1, id2, ...],       # 4*N bytes - Response order
         'rspn_start': float,                 # 4 bytes - Schedule start time
         'time_stamp': float,                 # 4 bytes - Transmission time
     }
    
    **Next Waypoint Prediction:**
    
    Computes leader's next position for follower path prediction. Uses guidance
    system to estimate next waypoint based on current path following state.
    Enables followers to anticipate leader maneuvers for more stable following.
    
    **Follower Data Packaging:**
    
    When followers send REPORT messages to leader, leader stores follower states
    on internal data models in the group list. For each BCRQ message, leader
    iterates the group list to extract and repeat this information to all
    listeners:

    .. code-block:: none

        - follower.eta              # Follower position
        - follower.velocity         # Follower velocity
        - follower.timeLastMsg      # Time of follower report
        - follower.id               # Follower id
    
    **Response Schedule:**
    
    Two components define response timing:
    
    1. **rspn_sched**: Ordered list of follower IDs defining response sequence
    2. **rspn_start**: Absolute time when first response should occur
    
    Each follower calculates its response time:
    
    .. code-block:: none

        my_index = rspn_sched.index(my_id)
        my_response_time = rspn_start + (my_index * rspnDur)
        
    where rspnDur is stored on vehicle.
    """

    # Prepare Message Contents
    vehicle.nextEta, vehicle.nextVel = guid.predictNextEtaVel(vehicle)
    nFollowers = len(vehicle.group)
    idList, etaList, velList, rspnList = zip(*[(m.id, 
                                                m.eta[0:3].tolist(), 
                                                m.velocity[0:3].tolist(), 
                                                m.timeLastMsg) 
                                                for m in vehicle.group])
    vehicle.rspnSched = idList
    vehicle.rspnFrame = vehicle.clock + vehicle.bcrqDur

    # Write Message
    bcrq = getMsgStruct('BCRQ')
    bcrqMsg = bcrq.build(dict(group_id        = vehicle.groupId.encode(),
                              leader_pos      = vehicle.eta[:3].tolist(),
                              leader_vel      = vehicle.velocity[:3].tolist(),
                              leader_next_pos = vehicle.nextEta[:3].tolist(),
                              leader_next_vel = vehicle.nextVel[:3].tolist(),
                              num_followers   = nFollowers,
                              followers_pos   = etaList,
                              followers_vel   = velList,
                              followers_rspn  = rspnList,
                              rspn_sched      = vehicle.rspnSched,
                              rspn_start      = vehicle.rspnFrame,
                              time_stamp      = vehicle.clock,
                              ))    
    
    return bcrqMsg

###############################################################################

def sendBroadcastReq(vehicle:Vehicle)->None:
    """
    Serialize and send BROADCAST/REQUEST message from leader to all followers on
    broadcast channel.
    
    Constructs complete swarm state broadcast, resets follower response flags,
    and transmits on leader's TX channel (broadcast). Triggers follower response
    cycle in TDMA_LF protocol.
    
    Parameters
    ----------
    vehicle : Vehicle
        Leader vehicle. Requires all attributes for writeBroadcastReq() plus
        txAddr, CommNetwork, and callSign attributes.

        - txAddr: transmission channel id
        - CommNetwork: a communication network manager instance
        - callSign: unique vehicle identification string
    
    Notes
    -----
    **Message Contents Summary:**
    
    - Leader current and predicted state (position, velocity, next waypoint)
    - All follower positions and velocities (from last received reports)
    - Response schedule defining follower reply timing
    - Timestamp for message age calculation
    
    **Transmission Sequence:**
    
    1. Serialize complete swarm state via writeBroadcastReq()
    2. Reset all follower response flags: m.rprtRecv = False
    3. Transmit on broadcast channel via vehicle.CommNetwork.send()
    
    **Response Flag Reset:**
    
    Before transmission, clears rprtRecv flag for all followers. Leader uses
    these flags to track which followers have responded in current cycle. Flags
    set to True when follower responses received.
    
    **Message Cycle:**
    
    Typical TDMA_LF communication pattern:
    
    1. Leader sends BCRQ at t = m*epsdDur
    2. Follower 1 responds at t = rspn_start (e.g., t=1.0)
    3. Follower 2 responds at t = rspn_start + slot (e.g., t=1.5)
    4. ...
    5. Follower N responds at t = rspn_start + (N-1)*slot
    6. Leader sends next BCRQ at t = (m+1)*epsdDur
    7. Repeat cycle
    
    **Broadcast Channel:**
    
    Transmitted on vehicle.txAddr which for leader in TDMA_LF is BCAST_ADDR. All
    followers tuned to receive on this channel.
    """

    # Write Message
    msgType = 'BROADCAST/REQUEST'
    bcrqMsg = writeBroadcastReq(vehicle)
    
    # Reset 'RESPONSE Received' Flags to 'No'
    for m in vehicle.group:
        m.rprtRecv = False

    # Send Message
    log.debug('%s: %s', vehicle.callSign, msgType)
    vehicle.CommNetwork.send(bcrqMsg, vehicle.txAddr)

###############################################################################

def recvBroadcastReq(vehicle:Vehicle, bytesMsg:bytes)->None:
    """
    Parse BROADCAST/REQUEST message and update follower's target and group
    member states.
    
    Deserializes BCRQ message from leader, validates all fields with corruption
    checking, updates target (leader) tracking, updates all group member states,
    and computes response timing from schedule. Receiving this message triggers
    a RESPONSE message at the scheduled response time slot.
    
    Parameters
    ----------
    vehicle : Vehicle
        Follower vehicle receiving message. Requires groupId, target, group
        (list), id, clock, rspnDuration, and callSign attributes.

        - groupId: swarm group id
        - target: swarm group leader data model
        - group: list of group neighbor data models
        - id: unique vehicle id
        - clock: simulation time
        - rspnDur: duration of response time slot
        - callSign: unique vehicle identification string
    
    bytesMsg : bytes
        Complete serialized BCRQ message (variable size: 65 + 32*N bytes), where
        N is number of swarm followers.
    
    Notes
    -----
    **Target Updates (Leader Tracking):**
    
    If all message fields are valid, an internal data model of the target is
    updated with:

    .. code-block:: none

        - leader_pos                        # Leader position
        - leader_vel                        # Leader velocity
        - leader_next_pos                   # Next waypoint
        - leader_next_vel                   # Next leader velocity
        - time_stamp                        # Message timestamp
        - clock - timestamp                 # Age of message
        - writeEtaVelLogs(target)           # Append to position/velocity log
    
    **Group Member Updates:**
    
    For each follower found in vehicle.group and all message fields valid, an
    internal data model of the follower is updated with: 

    .. code-block:: none

        - reporter_pos                    # Reporter Position
        - reporter_vel                    # Reporter Velocity
        - time_stamp                      # Last message time
        - writeEtaVelLogs(reporter)       # Append to position/velocity log
    
    **Field Validation:**
    
    Each individual message field is checked for data corruption via
    dataIsCorrupted(). If corruption is detected, then data recovery is
    attempted via restoreCorrupt(). Data recovery strategies for each field can
    be defined in the vehicle.dataRepair dictionary.
    
    **Response Time Calculation:**
    
    Follower determines when to send response based on schedule position:
    
    .. code-block:: none

        my_index = rspn_sched.index(my_id)
        my_response_time = rspn_start + (my_index * rspnDuration)
        
    where rspnDuration is stored on vehicle. If vehicle.id not in schedule, logs
    warning (not expected to respond).
    
    **Performance Note:**
    
    Most computationally expensive message handler due to:

    - Variable-length arrays requiring iteration
    - Multiple validation passes
    - Group member lookups by ID
    - Log append operations for each member
    
    For large swarms (20+ followers), consider caching ID->index mappings.
    """

    try:
        # Unpack Message
        bcrq = getMsgStruct('BCRQ')
        msg = bcrq.parse(bytesMsg)

        # Validate message metadata
        gid = msg.group_id
        if (dataIsCorrupted(gid, 'group_id')):
            log.warning('%s: CORRUPT DATA: GROUP ID', vehicle.callSign)
            gid, ok = restoreCorrupt(vehicle, 'group_id', gid)
            if (not ok):
                return
        group_id = gid.decode('utf-8', errors='ignore')

        # Get message title (type was validated in callback)
        title = (f"{msg.type.decode('utf-8', errors='ignore')}{group_id}")

        # Verify Vehicle is Part of Message Group
        if (vehicle.groupId != group_id):
            log.warning('%s: SKIP %s - NOT IN GROUP', vehicle.callSign, title)
            return
    
        # Read Leader Data
        try:
            # Validate data
            lpos = msg.leader_pos
            if (dataIsCorrupted(lpos, 'position')):
                log.warning('%s: %s - CORRUPT DATA: TARGET POSITION',
                            vehicle.callSign, title)
                lpos, ok = restoreCorrupt(vehicle, 'leader_pos', lpos, 
                                          isLeader=True)
                if (not ok):
                    return
            leader_pos = np.array(lpos)

            lvel = msg.leader_vel
            if (dataIsCorrupted(lvel, 'velocity')):
                log.warning('%s: %s - CORRUPT DATA: TARGET VELOCITY',
                            vehicle.callSign, title)
                lvel, ok = restoreCorrupt(vehicle, 'leader_vel', lvel, 
                                          isLeader=True)
                if (not ok):
                    return
            leader_vel = np.array(lvel)

            lnpos = msg.leader_next_pos
            if (dataIsCorrupted(lnpos, 'position')):
                log.warning('%s: %s - CORRUPT DATA: TARGET NEXT POSITION',
                            vehicle.callSign, title)
                lnpos, ok = restoreCorrupt(vehicle, 'leader_next_pos', lnpos)
                if (not ok):
                    return
            leader_next_pos = np.array(lnpos)

            lnvel = msg.leader_next_vel
            if (dataIsCorrupted(lnvel, 'velocity')):
                log.warning('%s: %s - CORRUPT DATA: TARGET NEXT VELOCITY',
                            vehicle.callSign, title)
                lnvel, ok = restoreCorrupt(vehicle, 'leader_next_vel', lnvel)
                if (not ok):
                    return
            leader_next_vel = np.array(lnvel)
            
            time_stamp = msg.time_stamp
            if (dataIsCorrupted(time_stamp, 'timestamp')):
                log.warning('%s: %s - CORRUPT DATA: TIME',
                            vehicle.callSign, title)
                time_stamp, ok = restoreCorrupt(vehicle, 'time_stamp', 
                                                time_stamp, isLeader=True)
                if (not ok):
                    return

            # Update Target Data
            vehicle.target.eta      = leader_pos
            vehicle.target.velocity = leader_vel
            vehicle.target.nextEta  = leader_next_pos
            vehicle.target.nextVel  = leader_next_vel
            vehicle.target.timeLastMsg = time_stamp
            vehicle.target.delayLastMsg = vehicle.clock - time_stamp
            writeEtaVelLogs(vehicle.target)

        except (ValueError, TypeError) as e:
            log.warning('%s: TARGET DATA CORRUPTION %s: %s',
                        vehicle.callSign, title, str(e))

        # Validate message metadata on followers
        num_followers = msg.num_followers
        if (dataIsCorrupted(num_followers, 'num_followers')):
            log.warning('%s: %s - CORRUPT DATA: FOLLOWER COUNT', 
                        vehicle.callSign, title)
            num_followers, ok = restoreCorrupt(vehicle, 'num_followers', 
                                               num_followers, isLeader=True)
            if (not ok):
                return

        rspn_sched = msg.rspn_sched
        if (dataIsCorrupted(rspn_sched, 'rspn_sched', uid=vehicle.id)):
            log.warning('%s: %s - CORRUPT DATA: RESPONSE SCHEDULE', 
                        vehicle.callSign, title)
            rspn_sched, ok = restoreCorrupt(vehicle, 'rspn_sched', rspn_sched)
            if (not ok):
                return

        # Map Vehicle Ids to Indices in rspn_sched
        rspnSchedIdxs = {id: idx for idx, id in enumerate(rspn_sched)}

        # Update Group Data
        for m in vehicle.group:
            if (m.id in rspnSchedIdxs):
                try:
                    i = rspnSchedIdxs[m.id]

                    # Validate position data
                    fpos = msg.followers_pos[i]
                    if (dataIsCorrupted(fpos, 'position')):
                        log.warning('%s: %s:%d - CORRUPT DATA: POSITION',
                                    vehicle.callSign, title, m.id)
                        fpos, ok = restoreCorrupt(vehicle, 'followers_pos', 
                                                  fpos, id=m.id)
                        if (not ok):
                            continue
                    followers_pos = np.array(fpos)
                
                    # Validate velocity data
                    fvel = msg.followers_vel[i]
                    if (dataIsCorrupted(fvel, 'velocity')):
                        log.warning('%s: %s:%d - CORRUPT DATA: VELOCITY',
                                    vehicle.callSign, title, m.id)
                        fvel, ok = restoreCorrupt(vehicle, 'followers_vel', 
                                                  fvel, id=m.id)
                        if (not ok):
                            continue
                    followers_vel = np.array(fvel)

                    # Validate time data
                    followers_rspn = msg.followers_rspn[i]
                    if (dataIsCorrupted(followers_rspn, 'timestamp')):
                        log.warning('%s: %s:%d - CORRUPT DATA: TIME',
                                    vehicle.callSign, title, m.id)
                        followers_rspn, ok = restoreCorrupt(vehicle, 
                                                            'followers_rspn',
                                                            followers_rspn,
                                                            id=m.id)
                        if (not ok):
                            continue
                    
                    # Update Data
                    m.eta = followers_pos
                    m.velocity = followers_vel
                    m.timeLastMsg = followers_rspn
                    m.delayLastMsg = vehicle.clock - followers_rspn
                    writeEtaVelLogs(m)

                except (ValueError, TypeError) as e:
                    log.warning('%s: DATA CORRUPTION %s: %s',
                                vehicle.callSign, title, str(e))
                    continue

        # Validate data
        rspn_start = msg.rspn_start
        if (dataIsCorrupted(rspn_start, 'timestamp')):
            log.warning('%s: %s - CORRUPT DATA: RSPN START TIME',
                        vehicle.callSign, title)
            rspn_start, ok = restoreCorrupt(vehicle, 'rspn_start', rspn_start)
            if (not ok):
                return

        # Update Self
        vehicle.rspnSched = rspn_sched
        vehicle.rspnFrame = rspn_start
        if (vehicle.id in rspnSchedIdxs):
            slotIdx = rspnSchedIdxs[vehicle.id]
            vehicle.rspnSlot = rspn_start + (slotIdx * vehicle.rspnDur)
            # Set 'RSPN Requested' Flag to 'Yes'
            vehicle.rspnRqst = True
        else:
            log.warning('%s: NOT SCHEDULED: %s', vehicle.callSign, title)

        log.debug('%s: RECEIVED %s', vehicle.callSign, title)

    except Exception as e:
        log.error('%s: BCRQ MESSAGE ERROR: %s', vehicle.callSign, str(e))

###############################################################################

def sendResponse(vehicle:Vehicle)->None:
    """
    Sends a scheduled RESPONSE message from follower to leader.

    Wrapper function calling sendReport with RSPN message type. Used in TDMA
    Leader-Follower configuration during follower response window.

    Parameters
    ----------
    vehicle : Vehicle
        Follower vehicle sending report. Requires eta, velocity, clock,
        nodeAddr, groupId, callSign, CommNetwork, and txAddr attributes.

        - eta: vehicle position / attitude vector
        - velocity: vehicle velocity vector (END frame)
        - clock: simulation time
        - nodeAddr: unique address of network node
        - groupId: swarm group id
        - callSign: unique vehicle identification string
        - CommNetwork: a communication network manager instance
        - txAddr: transmission channel id

    Notes
    -----
    **REPORT vs RESPONSE:**
    
    Functionally identical message formats with different type flags:

    - REPORT (b'RPRT'): Standard vehicle state broadcast
    - RESPONSE (b'RSPN'): Reply in TDMA_LF scheduled response slot
    
    Type flag enables receivers to distinguish message context but both
    parsed identically by recvReport().
    
    **Message Payload (37 bytes):**

    .. code-block:: none

        {
            'type': b'RSPN',                 # 4 bytes
            'group_id': bytes,               # 1 byte
            'follower_id': int,              # 4 bytes
            'follower_pos': [x,y,z],         # 12 bytes (3x float32)
            'follower_vel': [vx,vy,vz],      # 12 bytes (3x float32)
            'time_stamp': float,             # 4 bytes
        }
    """

    # Send Message
    msgType = 'RESPONSE'
    address = vehicle.txAddr
    sendReport(vehicle, msgType, address)
    
    # Reset 'RSPN Requested' Flag to 'No'
    vehicle.rspnRqst = False

###############################################################################

def schedulerTdmaLF(vehicle:Vehicle)->None:
    """
    Compute and apply TDMA leader-follower transmission scheduling.
    
    Manages bidirectional group-synchronized messaging for TDMA_LF protocol.
    Leader broadcasts BCRQ messages at fixed intervals. Followers respond in
    assigned time slots determined by rspnSlot and rspnDur.
    
    Parameters
    ----------
    vehicle : Vehicle
        Leader or follower vehicle. Requires clock, epsdDur, rspnSlot,
        rspnDur, isLeader, and rspnRqst (follower) attributes.

        - clock: Simulation time
        - epsdDur: Duration of complete message frame cycle
        - rspnSlot: Start time of vehicles scheduled response time slot
        - rspnDur: Duration of response time slot
        - isLeader: Boolean flag if vehicle is swarm leader
        - rspnRqst: Boolean flag is RESPONSE message request is true

    Notes
    -----
    **Leader Behavior:**
    
    Broadcasts BCRQ message every epsdDur seconds. BCRQ contains complete swarm
    state (leader + all followers) with response schedule (rspn_sched,
    rspn_start). 
    
    **Follower Behavior:**
    
    Receipt of BCRQ by followers triggers a RSPN transmission. Waits until
    simulation time is between assigned start and end time of scheduled response
    slot, then sends RESPONSE message.
    
    **Response Timing:**
    
    Follower response time computed in recvBroadcastReq():
    
    >>> my_index = rspnSched.index(vehicle.id)
    >>> rspnSlot = rspn_start + (my_index * rspnDur)
    
    **Typical Timeline (3 followers, epsdDur=10s, rspnDur=1s):**
    
    - t=0.0:  Leader sends BCRQ
    - t=1.0:  Follower 0 responds
    - t=2.0:  Follower 1 responds
    - t=3.0:  Follower 2 responds
    - t=10.0: Leader sends next BCRQ (cycle repeats)
    
    **Coordination Requirements:**
    
    - Leader must send BCRQ before followers respond
    - Followers must receive and parse BCRQ to learn assignedResponseTime
    - Clock synchronization required
    - Response schedule prevents packet collisions via time-domain separation
    

    See Also
    --------
    sendBroadcastReq : Leader transmission function
    recvBroadcastReq : Follower handler computing response timing
    sendReport : Follower response transmission
    """

    # Parameters on vehicle
    t = vehicle.clock               # Simulation Time Since Start (s)
    epsdDur = vehicle.epsdDur       # Duration of One Episode Cycle (s)
    rspnSlot = vehicle.rspnSlot     # Individual Response-Slot Start Time (s)
    rspnDur = vehicle.rspnDur       # Duration of One Slot Cycle (s)

    # Leader: BROADCAST / REQUEST
    if (vehicle.isLeader):
        if (t % epsdDur == 0):
            sendBroadcastReq(vehicle)

    # Follower: RESPONSE
    else:
        if ((t >= rspnSlot) and 
            (t < (rspnSlot + rspnDur)) and
            (vehicle.rspnRqst)):
            sendResponse(vehicle)

###############################################################################

def schedulerFdmaLF(vehicle:Vehicle)->None:
    """
    Compute response and coordination timing for FDMA leader-follower messaging.
    
    Manages periodic state broadcasts for leader-follower formations using
    frequency-division channels. Leader sends LRPT messages at fixed intervals.
    Followers send RPRT messages at same intervals on dedicated channels.
    
    Parameters
    ----------
    vehicle : Vehicle
        Leader or follower vehicle. Requires clock, rspnSlot, epsdDur,
        sampleTime, and isLeader attributes.

        - clock: Simulation time
        - rspnSlot: Start time of vehicles scheduled response time slot
        - epsdDur: Duration of complete message frame cycle
        - sampleTime: Simulation iteration time step
        - isLeader: Boolean flag if vehicle is swarm leader
    
    Notes
    -----
    **Leader Behavior:**
    
    Sends Leader Report every epsdDur seconds. LRPT message contains leader
    current state and predicted next waypoint. Transmitted on leader's dedicated
    FDMA channel.
    
    **Follower Behavior:**
    
    Sends standard Report every epsdDur seconds. RPRT message contains follower
    position, velocity, timestamp. Transmitted on follower's dedicated FDMA
    channel. Vehicle transmit times typically initialized with offset from each
    other.
    
    **FDMA Channel Architecture:**
    
    - Each vehicle: TX on dedicated channel, RX on all other channels
    
    No collision risk because each vehicle has unique TX frequency. No response
    scheduling needed.

    **Typical Message Flow (txOffset=0.5s):**
    
    - t=0.0: Leader -> LRPT on channel 0 -> All members receive leader
    - t=0.5: Follower1 -> RPRT on channel 1 -> All members receive follower1
    - t=1.0: Follower2 -> RPRT on channel 2 -> All members receive follower2
    
    ...
    

    See Also
    --------
    sendLeaderReport : Leader transmission function
    recvLeaderReport : Follower handler for leader state
    sendReport : Follower transmission function
    recvReport : Leader handler for follower state
    """

    # Check if Time to Transmit
    if ((vehicle.clock - vehicle.rspnSlot) >= vehicle.sampleTime):

        # Update Next Transmission Time
        vehicle.rspnSlot += vehicle.epsdDur

        # Leader: LEADER-REPORT
        if (vehicle.isLeader):
            sendLeaderReport(vehicle)
        
        # Follower: REPORT
        else:
            sendReport(vehicle)

###############################################################################
            
def writeEtaVelLogs(model:Model)->None:
    """
    Appends reported position and velocity in vehicle model data logs.

    Parameters
    ----------
    model : Model
        Data model of vehicle that is getting data record updated.

    Notes
    -----
    Model eta and velocity logs are dynamically resized with resizeLog() if they
    are full.
    """

    if (model._logSize >= model._logCap):
        model._logCap *= 2
        model.etaLog = resizeLog(model.etaLog, model._logSize, model._logCap)
        model.velLog = resizeLog(model.velLog, model._logSize, model._logCap)
        log.debug("Resized Model %s Logs to %d", model.id, model._logCap)
    
    model.etaLog[model._logSize] = np.array([*model.eta[:3],model.timeLastMsg])
    model.velLog[model._logSize] = np.array([*model.velocity[:3], 
                                             nav.stateSpeed(model),
                                             model.timeLastMsg])
    model._logSize += 1

###############################################################################

def resizeLog(logArray:NPFltArr, logSize:int, newCap:int)->NPFltArr:
    """
    Expand position and velocity logs of data model to new capacity.
    
    Maintains chronological order while resizing log array. Pads with zero. Used
    by writeEtaVelLogs() to dynamically increase size of etaLog and velLog
    attributes of vehicle data models.
    
    Parameters
    ----------
    logArray : list
        Current log array containing position/velocity records.
    logSize : int
        Current number of valid entries in log.
    newCap : int
        New capacity of the resized log.
    
    Returns
    -------
    ndarray
        Resized log array with capacity=newCap, preserving most recent records.
    """

    newLog = np.zeros((newCap, logArray.shape[1]))
    newLog[:logSize] = logArray
    return newLog

###############################################################################

def readLastEtaLogs(model:Model, n:int=1)->NPFltArr:
    """
    Retrieve last n records from vehicle data model position (eta) log.
    
    Parameters
    ----------
    model : Model
        Vehicle model with etaLog attribute.
    n : int, default=1
        Number of most recent entries to retrieve.
    
    Returns
    -------
    ndarray, shape (n, 4)
        Last n eta records, or fewer if log shorter than n. Contains
        [x, y, z, time_stamp] per row.
    """

    if (model._logSize == 0):
        return np.array([]).reshape(0,model.etaLog.shape[1])
    nRecords = min(n, model._logSize)
    idx_start = model._logSize - nRecords
    return np.copy(model.etaLog[idx_start:model._logSize])
    
###############################################################################

def readLastVelLogs(model:Model, n:int=1)->NPFltArr:
    """
    Retrieve last n records from vehicle data model velocity log.
    
    Parameters
    ----------
    model : Model
        Vehicle model with velLog attribute.
    n : int, default=1
        Number of most recent entries to retrieve.
    
    Returns
    -------
    ndarray, shape (n, 5)
        Last n velocity records, or fewer if log shorter than n. Contains
        [vx, vy, vz, speed, time_stamp] per row.
    """

    if (model._logSize == 0):
        return np.array([]).reshape(0,model.velLog.shape[1])
    nRecords = min(n, model._logSize)
    idx_start = model._logSize - nRecords
    return np.copy(model.velLog[idx_start:model._logSize])

###############################################################################

def dataIsCorrupted(data:Any, expType:str, **kwargs)->bool:
    """
    Validate message field data for corruption using field-specific rules.
    
    Performs type checking, range validation, and structural verification on
    message fields. Designed for acoustic network error detection where bit
    errors and collisions corrupt data in unpredictable ways.
    
    Parameters
    ----------
    data : Any
        Data value to validate. Type depends on field parameter.
    expType : str, {'type', 'group_id', 'reporter_id', 'num_followers', 'int', 'rspn_sched', 'float', 'position', 'velocity', 'float_array', 'timestamp'}
        Field type identifier determining validation rules.
        
    **kwargs : dict
        Optional context for validation:

        - uid : int
            Receiving vehicle ID for self-check (reporter_id validation)
        - isLeader : bool
            True if validating leader data (affects recovery strategy)
        - id : int
            Specific vehicle ID being validated (for targeted recovery)
    
    Returns
    -------
    bool
        True if data appears corrupted, False if valid.
    
    Notes
    -----
    **Design Philosophy:**

    More comprehensive data validation is possible here. This framework provides
    basic validation, which can be extended for more robust or sophisticated
    requirements. 
    
    Validations employ range and threshold checks to reject unreasonable values,
    but these may not apply to your mission parameters. Check and adjust if
    needed.
    
    Validation designed for underwater acoustic channel characteristics:

    - High bit error rates
    - Collision-induced corruption (XOR of overlapping signals)
    
    Conservative validation prevents:

    - Physically impossible states (positions, velocities)
    - Type confusion (int as float, bytes as int)
    - Array structure damage (wrong size, wrong types)
    - Logic errors (negative time, self-reporting)
    
    **Limitations:**
    
    Cannot detect:

    - Semantically valid but factually wrong data
    - Corrupted data that remains within valid ranges
    - Multi-field corruption

    **Validation Rules by Field:**
    
    **type** (4-byte message flag):

    - Must be exactly 4 bytes in length
    - Must be one of the valid message types: RPRT, RSPN, LRPT, or BCRQ
    - Must decode properly as UTF-8 text
    
    **group_id** (1-byte group identifier):

    - Must be exactly 1 byte in length
    - Must decode as valid UTF-8 character
    - Must be printable and alphanumeric ASCII character
    
    **reporter_id** (4-byte unsigned integer):

    - Must be a valid positive integer type
    - Must be within reasonable ID range of own ID (+/-1000)
    
    **position** (3-element float array):

    - Must be a list or array type with exactly 3 elements
    - All elements must be finite numbers (no NaN or infinity)
    - Position values must be within reasonable geographic bounds (100 km)

    **velocity** (3-element float array):

    - Must be a list or array type with exactly 3 elements
    - All elements must be finite numbers (no NaN or infinity) 
    - Velocity values must be within realistic AUV limits (100 m/s)
    
    **timestamp** (4-byte float):

    - Must be a valid positive numeric type
    - Must be finite (no NaN or infinity) and less that 1,000,000
    
    **num_followers** (4-byte unsigned integer):

    - Must be a valid integer type
    - Must be positive and within reasonable swarm size limits (1-1000)
    
    **rspn_sched** (array of vehicle IDs):

    - Must be a list or array type
    - All elements must be valid positive integers
    - Must be within reasonable ID range of own ID (+/-1000)
    
    **rspn_start** (4-byte float):

    - Must be a valid numeric type
    - Must be finite (no NaN or infinity)
    - Should represent a future time for response scheduling

    **Data Recovery**
    
    - Calls restoreCorrupt() for recovery when corruption detected.
    - Data recovery strategies stored in dictionary at vehicle.dataRepair
    """

    try:
        # Message type: exactly 4 bytes, must strictly decode to known flags
        if (expType == 'type'):
            if (not isinstance(data, bytes) or len(data) != 4):
                return True
            try:
                decoded = data.decode('utf-8', errors='strict')
                return (decoded not in {'RPRT', 'RSPN', 'LRPT', 'BCRQ'})
            except UnicodeDecodeError:
                return True
        
        # Group ID: exactly 1 byte, accept any single printable char
        elif (expType == 'group_id'):
            """This won't catch flips to other single-byte chars. Could tighten
            requirements here if want to improve detection and are comitted to 
            implementation of group id specifications. Perhaps register valid 
            groups on vehicle."""
            if (not isinstance(data, bytes) or len(data) != 1):
                return True
            try:
                decoded = data.decode('utf-8', errors='strict')
                return (not (decoded.isprintable() and decoded.isalnum()))
            except UnicodeDecodeError:
                return True
            
        # Vehicle ID: valid positive integer, within range of own ID
        elif (expType == 'reporter_id'):
            """To make this more dynamic / less coupled, could store thresholds
            and ranges on vehicle"""
            try:
                # check range of positive Int32
                val = int(data)
                if (not (0 < val <= 2**31-1)):
                    return True
                # check proximity to caller vehicle ID
                if (uid := kwargs.get('uid')):
                    dv = 10**3
                    return (not (abs(val-uid) <= dv))
            except (ValueError, TypeError, OverflowError):
                return True
            
        # Num Followers: valid positive integer, within reasonable range
        elif (expType == 'num_followers'):
            try:
                # check range of positive integers less than max
                nMax = 10**3
                return (not (0 < int(data) <= nMax))
            except (ValueError, TypeError, OverflowError):
                return True
            
        # Generic int: valid integer
        elif (expType == 'int'):
            try:
                # check range of Int32
                return (not (-2**31 <= int(data) <= 2**31-1))
            except (ValueError, TypeError, OverflowError):
                return True
        
        # Response schedule: valid positive integers array, within range of 
        # own ID
        elif (expType == 'rspn_sched'):
            try:
                # check range of positive Int32
                arr = np.array(data, dtype=np.int32)
                if (not np.all((arr > 0) & (arr < 2**31-1))):
                    return True
                # check proximity to caller vehicle ID
                if (uid := kwargs.get('uid')):
                    dv = 10**3
                    return (not np.all(np.abs(arr-uid) <= dv))
            except (ValueError, TypeError, OverflowError) as e:
                return True
            
        # Generic float: is non-infinite float
        elif (expType == 'float'):
            return (not np.isfinite(float(data)))
        
        # Position vectors: valid floats, within reasonable range
        # leader_pos, reporter_pos, followers_pos, leader_next_pos
        elif (expType == 'position'):
            """Could try comparison to previous values to identify suddenly
            different values."""
            try:
                arr = np.array(data, dtype=np.float32)
                return ((not np.all(np.isfinite(arr))) or
                        (np.any(np.abs(arr)) > 1E5))
            except (ValueError, TypeError):
                return True
            
        # Velocity vectors: valid floats, within reasonable range
        # leader_vel, reporter_vel, followers_vel, leader_next_vel
        elif (expType == 'velocity'):
            try:
                arr = np.array(data, dtype=np.float32)
                return ((not np.all(np.isfinite(arr))) or
                        (np.any(np.abs(arr)) > 1E2))
            except (ValueError, TypeError):
                return True
            
        # Generic float array: valid float
        elif (expType == 'float_array'):
            return (not np.all(np.isfinite(np.array(data))))
        
        # Time stamp: valid positive float, less than max value
        # time_stamp, followers_rspn, rspn_start
        elif (expType == 'timestamp'):
            """Could try comparison to an expected range based on log
            history."""
            val = float(data)
            return (not (np.isfinite(val) and val >= 0 and val < 1e6))
        
    except Exception:
        return True
    
    return False

###############################################################################

def restoreCorrupt(vehicle:Vehicle, 
                   field:str, 
                   badData:Any, 
                   **kwargs)->Tuple[Any, bool]:
    """
    Attempt corrupted message field recovery by defined strategy with fallback.
    
    Applies field-specific recovery methods when dataIsCorrupted() detects
    corruption. Recovery strategies use historical data, vehicle state, or
    reasonable defaults depending on field type and available context.
    Strategies are defined per field in vehicle.dataRepair dictionary.
    
    Parameters
    ----------
    vehicle : Vehicle
        Vehicle receiving corrupted message. Requires groupId, group,
        target, and clock attributes.

        - groupId: swarm group id
        - group: list of swarm group neighbor data models
        - target: swarm group leader data model
        - clock: simulation time

    field : str
        Field type identifier matching dataIsCorrupted() field types.
    badData : Any
        Corrupted data.
    **kwargs : dict
        Optional context for targeted recovery:

        - id : int
            Vehicle ID for targeted log retrieval
        - isLeader : bool
            True if recovering leader data (uses vehicle.target logs)
    
    Returns
    -------
    goodData : Any
        Recovered data value. Type matches expected type for field.
    isGood : bool
        True if recovery successful, False if unrecoverable.
    
    Notes
    -----
    **Fallback:**

    Attempts data recovery strategy for each strategy listed by field in
    vehicle.dataRepair dictionary. When multiple strategies are listed, recovery
    is attempted in sequence until success or end of list. If no strategy is
    defined for field, falls back to default (skip entire message).

    **Strategies:**

    - **predict**: 

        Estimate field value using kinematic prediction, message history, or
        bitwise comparison for optimal matching.

    - **previous**: 

        Use last known valid value from vehicle logs, group member data, or
        target tracking state.

    - **neutral**: 

        Replace with safe default values (zero positions/velocities, current
        time for timestamps, origin values) for continued operation with reduced
        disruption.

    - **sentinel**: 

        Replace with obviously invalid marker values (-1, -999, 0xFF) for
        downstream handler.

    - **skip_field**: 

        Accept corrupted field as-is and continue processing remaining message
        fields.

    - **skip**: 

        Reject entire message.

    **Logging:**
    
    Recovery attempts logged at WARNING level. Failed recoveries logged at ERROR
    level.
    """

    # Strategy dispatch 
    strategyMap = {
        'predict': recoverPredict,
        'previous': recoverPrevious,
        'neutral': recoverNeutral,
        'sentinel': recoverSentinel,        
        'skip_field': recoverSkipField,
        'skip': recoverSkip,
    }

    # Default handler
    fallback = 'skip'

    # Get repair strategies from vehicle
    strategies = vehicle.dataRepair.get(field, fallback)
    if (not isinstance(strategies, list)):
        strategies = [strategies]

    # Attempt corrupt data recovery
    for strat in strategies:
        # Exit early if skipping
        if (strat == 'skip'):
            log.debug('%s: CORRUPT DATA RECOVERY ATTEMPT: SKIP on %s: %s', 
                     vehicle.callSign, field.upper(), badData)
            return badData, False
        
        # Run the next strategy in the list
        try:
            # Catch a bad input strategy name
            if ((handleBadData := strategyMap.get(strat)) is None):
                log.warning('%s: CORRUPT DATA RECOVERY FAILED: UNK %s on %s',
                            vehicle.callSign, strat.upper(), field.upper())
                return badData, False
            
            # Attempt the strategy
            log.debug('%s: CORRUPT DATA RECOVERY ATTEMPT: %s on %s: %s',
                        vehicle.callSign, strat.upper(), field.upper(), badData)
            goodData, isGood = handleBadData(vehicle, field, badData, **kwargs)

            # Return if successful
            if (isGood):
                log.debug('%s: CORRUPT DATA RECOVERED: %s on %s: %s',
                        vehicle.callSign, strat.upper(), field.upper(), 
                        goodData)
                return goodData, isGood

        # Report errors and keep moving through list   
        except Exception as e:
            log.warning('%s: CORRUPT DATA RECOVERY FAILED: %s on %s: %s',
                      vehicle.callSign, strat.upper(), field.upper(), e)
            continue

    # All recovery attempts failed
    log.debug('%s: CORRUPT DATA RECOVERY ATTEMPT: UNSUCCESSFUL on %s: %s', 
                     vehicle.callSign, field.upper(), badData)
    return badData, False

###############################################################################

def recoverPredict(vehicle:Vehicle, 
                   field:str, 
                   badData:Any, 
                   **kwargs)->Tuple[Any, bool]:
    """
    Recover corrupted field using predictive estimation from available data.
    
    Computes estimates for corrupted message fields using vehicle state, known
    data from same message, and guidance prediction algorithms. Provides a 
    forward-looking data recovery strategy.

    Parameters
    ----------
    vehicle : Vehicle
        Vehicle receiving corrupted message.
    field : str
        Message field identifier of corrupted data. (eg. 'reporter_pos')
    badData : Any
        Corrupted data.
    **kwargs : dict
        Additional parameters passed in for specific strategies.

        - id : int, Message source ID for group member fields
        - isLeader : bool, True if message is from leader
    
    Returns
    -------
    goodData : Any
        Predicted field value.
    isSuccess : bool
        True if data recovery was successful.
    
    Notes
    -----
    **Prediction Strategies by Field:**
    
    type: 

        Use bitwise comparison with valid flag types and select best match.
        Accept match if less than half of the bits are different (16bits).

    group_id:

        Use the vehicles own group id.

    reporter_id:

        Select the id from vehicle group list that has the longest time since
        receiving the last message.

    num_followers:

        Use the number of vehicles in own group list (+1 if is not leader). Uses
        kwarg 'isLeader'.

    rspn_start:

        Calculate from own last response frame start time and the episode
        duration.

    rspn_sched:

        Compare corrupted array to last stored response schedule. Uses bitwise
        comparison on unmatched reference ids to select the best match.
        Secondary criteria is index proximity. Accepts if less than half of the
        elements were predicted.

    time_stamp, followers_rspn:

        If source ID or isLeader is provided as keyword argument, adds average
        of time between eta log timestamps to the last timestamp entry.
        Otherwise subtracts one half episode duration from current time. Uses
        'isLeader' or 'id' kwargs.

    reporter_pos, followers_pos, leader_pos:

        Use last logged position, velocity, and time stored on model to predict
        position using constant velocity approximation. Uses 'id' or 'isLeader'
        kwargs.

    reporter_vel, followers_vel, leader_vel:

        Use last logged velocities to predict position using constant
        acceleration approximation. Uses 'id' or 'isLeader' kwargs.

    leader_next_pos, leader_next_vel:

        Not compatible with 'predict' strategy, would collapse to same recovery
        used in 'previous' strategy.
    """

    try:
        # Verify field is compatible with prediction recovery strategy
        fieldsNotCompatible = {'leader_next_pos', 'leader_next_vel',}
        if (field in fieldsNotCompatible):
            return badData, False
        
        # Get keyword args if supplied
        srcId = kwargs.get('id')
        isLeader = kwargs.get('isLeader', False)

        ## Mesage Type: try to match to closest valid type
        if (field == 'type'):
            # Enforce expected length: truncate if too long, pad if too short
            lenGood = 4
            bad = badData[:lenGood].ljust(lenGood, b'\x00')
            
            # XOR on validTypes to count differences
            validTypes = [b'RPRT', b'RSPN', b'LRPT', b'BCRQ']
            dBest = (lenGood * 8) + 1 # bigger than worst case
            good = bad
            for vt in validTypes:
                d = sum(bin(a^b).count('1') for a,b in zip(bad, vt))
                if (d < dBest):
                    dBest = d
                    good = vt
            
            # Use best match if less than half of bits are different
            return (good, True) if dBest < (lenGood * 8 // 2) else (bad, False)
        
        ## Group ID: use vehicles own group ID
        elif (field == 'group_id'):
            if ((gid := vehicle.groupId) is not None):
                return str(gid).encode(), True
            return badData, False
        
        ## Reporter ID: select group member with longest time since message sent
        elif (field == 'reporter_id'):
            if (group := vehicle.group):
                return min(group, key=lambda m: m.timeLastMsg).id, True
            return badData, False
        
        ## Number of Followers: use length of group list (+1)
        elif (field == 'num_followers'):
            if (group := vehicle.group):
                n = len(group)
                return (n if isLeader else n + 1), True
            return badData, False
        
        ## Response Frame Start Time: calculate from frame length and last start
        elif (field == 'rspn_start'):
            if ((frame := vehicle.rspnFrame) and (dur := vehicle.epsdDur)):
                return frame + dur, True
            return badData, False
        
        ## Respons Schedule: try to match with last stored response schedule
        elif (field == 'rspn_sched'):
            # Get stored value as reference
            if (not (refData := vehicle.rspnSched)):
                return badData, False

            # Convert badData
            try:
                # Check each element of the list    
                if (isinstance(badData, (list, tuple, np.ndarray))):
                    corrupt = []
                    for x in badData:
                        try:
                            val = int(x)
                            # Check value in plausible range of ids
                            if ((val < 0) or
                                (abs(val-vehicle.id) <= 10**3)):
                                # Mark index of bad values with 'None'
                                corrupt.append(None)  
                            else:
                                # Add good values into the list
                                corrupt.append(val)
                        except (ValueError, TypeError, OverflowError):
                            corrupt.append(None) # Track bad values with 'None'
                # Can't process if not a list
                else:
                    return badData, False
            except:
                return badData, False
            
            # Length mismatch: return the reference
            if (len(corrupt) != len(refData)):
                return badData, False
            
            # Repair corrupted positions
            refSet = set(refData)
            unused = list(refData)
            repaired = []
            nRepairs = 0
            pos = {value: index for index, value in enumerate(refData)}

            for i, (bad, ref) in enumerate(zip(corrupt, refData)):
                # Non-corrupt values: Keep if matches id from reference
                if ((bad is not None) and
                    (bad in refSet) and 
                    (bad in unused)):
                        repaired.append(bad)
                        unused.remove(bad)
                        continue
                # Corrupt values
                if (unused):
                    # Find closest bitwise value with index diff tiebreak
                    val = bad if bad is not None else ref
                    best = min(
                        unused, 
                        key=lambda r: (bin(val^r).count('1'),abs(pos[r] - i))
                    )
                    repaired.append(best)
                    unused.remove(best)
                # Fallback just grabs value at same index in reference
                else:
                    repaired.append(ref)
                    if (ref in unused):
                        unused.remove(ref)
                nRepairs += 1

            # Accept if reasonable and no duplicates
            if ((len(set(repaired)) == len(repaired)) and 
                (nRepairs <= len(refData) // 2)):
                return repaired, True
            else:
                return badData, False
        
        ## Timestamp: Use average time of entries in etaLog
        elif (field in {'time_stamp', 'followers_rspn'}):
            # Find relevant model
            model = None
            if ((srcId is not None) and vehicle.group):
                model = next((m for m in vehicle.group if m.id == srcId), None)
            elif (isLeader and vehicle.target):
                model = vehicle.target
            
            # model found
            if (model):
                logs = readLastEtaLogs(model,n=2)
                # Use average of eta log time entries
                if (len(logs) >= 2):
                    avgDiff = np.mean(np.diff(logs[:,3]))
                    return (logs[-1,3] + avgDiff), True
                # Only one entry: add episode duration to last logged timestamp
                elif (len(logs) == 1):
                    val = vehicle.epsdDur if vehicle.epsdDur is not None else 1
                    return (logs[-1,3] + val), True
                
            # Otherwise use average of frame response time (half episode len)
            val = vehicle.epsdDur if vehicle.epsdDur is not None else 1
            return vehicle.clock - (val / 2), True

        ## Position: use equations of motion with eta, vel log data
        # NOTE: Could add attempt to use data from message, eg reporter_vel,
        # but need to verify its also not corrupt
        elif (field in {'reporter_pos', 'followers_pos', 'leader_pos',}):
            # Find relevant model
            model = None
            if ((srcId is not None) and vehicle.group):
                model = next((m for m in vehicle.group if m.id == srcId), None)
            elif ((isLeader or field == 'leader_pos') and vehicle.target):
                model = vehicle.target

            # model found
            if (model is not None):
                # Check if values stored in model are acceptable for return
                if (model.delayLastMsg < 1.0):
                    return model.eta, True

                # Calculate time since last message
                dt = vehicle.clock - model.timeLastMsg

                # Try to use constant acceleration from velocity history
                # a = (v_end - v_start) / dt_total
                # x = x0 + v0*t + 0.5*a*t^2
                eLog = readLastEtaLogs(model)
                vLog = readLastVelLogs(model, n=2)
                if (len(vLog) >= 2):
                    a = (vLog[-1,:3] - vLog[0,:3]) / (vLog[-1,4] - vLog[0,4])
                    return (eLog[-1,:3] + vLog[-1,:3]*dt + 0.5*a*dt**2), True

                # If only one record, apply constant velocity prediction
                # x = x0 + v0*t
                elif (len(vLog) == 1):
                    return (eLog[-1,:3] + vLog[-1,:3] * dt), True

            # model not found or no meaningful kinematic data available
            return badData, False
        
        ## Velocity: use equations of motion with eta, vel log data
        # NOTE: Could add attempt to use data from message, eg time_stamp,
        # but need to verify its also not corrupt
        elif (field in {'reporter_vel', 'followers_vel', 'leader_vel',}):
            # Find relevant model
            model = None
            if ((srcId is not None) and vehicle.group):
               model = next((m for m in vehicle.group if m.id == srcId), None)
            elif ((isLeader or field == 'leader_vel') and vehicle.target):
                model = vehicle.target

            # model found
            if (model is not None):
                # Check if values stored in model are acceptable for return
                if (model.delayLastMsg < 1.0):
                    return model.velocity, True

                # Calculate time since last message
                dt = vehicle.clock - model.timeLastMsg

                # Try to use constant acceleration from velocity history
                # a = (v_end - v_start) / dt_total
                # v = v0 + a*t
                vLog = readLastVelLogs(model, n=2)
                if (len(vLog) >= 2):
                    a = (vLog[-1,:3] - vLog[0,:3]) / (vLog[-1,4] - vLog[0,4])
                    return (vLog[-1,:3] + a * dt), True

            # model not found or no meaningful kinematic data available
            return badData, False
        
        # Else attempt failed
        return badData, False

    except Exception as e:
        log.warning('%s: CORRUPT DATA RECOVERY FAILED: PREDICT - %s: %s',
                  vehicle.callSign, field.upper(), str(e))
        return badData, False

###############################################################################

def recoverPrevious(vehicle:Vehicle, 
                    field:str, 
                    badData:Any, 
                    **kwargs)->Tuple[Any, bool]:
    """
    Recover corrupted field using last known valid value.
    
    Retrieves most recent stored value, if available, for corrupted message
    field. Assumes recent values approximate current state.
    
    Parameters
    ----------
    vehicle : Vehicle
        Vehicle receiving corrupted message.
    field : str
        Message field identifier of corrupted data. (eg. 'reporter_pos')
    badData : Any
        Corrupted data.
    **kwargs : dict
        Additional parameters passed in for specific strategies.

        - id : int, Message source ID for group member fields
        - isLeader : bool, True if message is from leader
    
    Returns
    -------
    goodData : Any
        Last stored value for field.
    isSuccess : bool
        True if data recovery was successful.
    
    Notes
    -----
    **Recovery Strategies by Field:**
    
    type: 

        Not compatible with 'previous' strategy. No message type data stored.
        Try 'predict' strategy.

    group_id:

        With id keyword, finds model from group list and retrieves
        model.groupId. If no id keyword, attempts to retrieve target.groupId.

    reporter_id:

        Not compatible with 'previous' strategy. No reliable method to identify
        which previous reporter id. Try 'predict' strategy.

    num_followers:

        Not compatible with 'previous' strategy. No data stored on number of
        followers. Try 'predict' or 'skip' strategy.

    rspn_start:

        Use value stored in rspnFrame.

    rspn_sched:

        Use value stored in rspnSched.

    time_stamp:

        With id keyword, finds model from group list and retrieves
        model.timeLastMsg. If no id keyword, attempts to retrieve
        target.timeLastMsg.

    followers_rspn:

        With id keyword, finds model from group list and retrieves
        model.timeLastMsg.

    reporter_pos, followers_pos:

        With id keyword, finds model from group list and retrieves model.eta.

    reporter_vel, followers_vel:

        With id keyword, finds model from group list and retrieves
        model.velocity.

    leader_pos:

        Use value stored in target.eta.

    leader_vel:

        Use value stored in target.velocity.

    leader_next_pos:

        Use value stored in target.nextEta.

    leader_next_vel:

        Use value stored in target.nextVel.
    """

    try:
        # Verify field is compatible with previous data recovery strategy
        fieldsNotCompatible = {'type', 'reporter_id', 'num_followers',}
        if (field in fieldsNotCompatible):
            return badData, False
        
        # Get keyword args if supplied
        srcId = kwargs.get('id')
        
        # Check fields that need no additional information
        independentFields = {
            'rspn_sched': vehicle.rspnSched,
            'rspn_start': vehicle.rspnFrame,
        }
        if ((goodData := independentFields.get(field)) is not None):
            return goodData, True

        # Check fields that are only available on the target model
        elif (vehicle.target):
            targetFields = {
                'leader_pos': vehicle.target.eta,
                'leader_vel': vehicle.target.velocity,
                'leader_next_pos': vehicle.target.nextEta,
                'leader_next_vel': vehicle.target.nextVel,
            }
            if ((goodData := targetFields.get(field)) is not None):
                return goodData, True
            
        # Check fields that need an id for reference
        elif (vehicle.group and (srcId is not None)):
            if (model := 
                next((m for m in vehicle.group if m.id == srcId), None)):
                groupFields = {
                    'reporter_pos': model.eta,
                    'reporter_vel': model.velocity,
                    'followers_pos': model.eta,
                    'followers_vel': model.velocity,
                    'followers_rspn': model.timeLastMsg,
                    'group_id': model.groupId,
                    'time_stamp': model.timeLastMsg,
                }
                if ((goodData := groupFields.get(field)) is not None):
                    return goodData, True
            
        # Check ambiguous fields that are on vehicle.target model
        elif (vehicle.target): 
            ambiguousFields = {
                'group_id': vehicle.target.groupId, 
                'time_stamp': vehicle.target.timeLastMsg,
            }
            if ((goodData := ambiguousFields.get(field)) is not None):
                return goodData, True

        # Else attempt failed
        return badData, False
    
    except Exception as e:
        log.warning('%s: CORRUPT DATA RECOVERY FAILED: PREVIOUS - %s: %s',
                  vehicle.callSign, field.upper(), str(e))
        return badData, False

###############################################################################

def recoverNeutral(vehicle:Vehicle, 
                   field:str, 
                   badData:Any,
                   **kwargs)->Tuple[Any, bool]:
    """
    Handle corrupted field by replacing with zero or neutral value.

    Neutral values are chosen to be safe (mostly harmless) for continued
    operation.
    
    Parameters
    ----------
    vehicle : Vehicle
        Vehicle receiving corrupted message.
    field : str
        Message field identifier of corrupted data. (eg. 'reporter_pos').
    badData : Any
        Corrupted data.
    **kwargs:
        Additional parameters passed in for specific strategies

        - neutral: Custom neutral value to use instead of defaults (scalar)
    
    Returns
    -------
    goodData : Any
        Neutral default value for field type.
    isSuccess : bool
        True if the strategy was successful. In theory, always True as neutral
        value always available.
    
    Notes
    -----
    **Typical neutral values:**

    - Arrays: filled with appropriate scalar value
    - Scalars:

        - bytes: b'0'
        - integers: 0 or 1
        - times: current clock time
        - positions: zero (origin)
        - velocities: zero (stationary)
    """

    try:
        # Get keyword args if supplied
        neutral = kwargs.get('neutral')

        # Default policy values
        defaultScalars = {
            #bytes
            'type': b'0000',
            'group_id': b'0',
            #integers
            'reporter_id': 0,
            'num_followers': 1,
            #times
            'time_stamp': vehicle.clock,
            'rspn_start': vehicle.clock,
            'followers_rspn': vehicle.clock,
        }
        vectors = {
            'reporter_pos', 'leader_pos', 'leader_next_pos', 'followers_pos', 
            'reporter_vel', 'leader_vel', 'leader_next_vel', 'followers_vel',
        }
        defaultVectorValue = 0.0

        # Check simple scalar fields
        if ((lookup := defaultScalars.get(field)) is not None):
            return (neutral if neutral is not None else lookup), True
        
        # Check simple vector fields
        elif (field in vectors):
            value = (neutral if neutral is not None else defaultVectorValue)
            return np.full(3, value), True
        
        # Check fields with unique handling
        elif (field == 'rspn_sched'):
            if (group := vehicle.group):
                cnt = len(group)
                n = cnt if vehicle.isLeader else cnt + 1
                g = [neutral]*n if neutral is not None else list(range(2,n+2))
                return g, True
            return badData,False
        
        # Else attempt failed
        return badData, False

    except Exception as e:
        log.warning('%s: CORRUPT DATA RECOVERY FAILED: NEUTRAL - %s: %s',
                  vehicle.callSign, field.upper(), str(e))
        return badData, False

###############################################################################

def recoverSentinel(vehicle:Vehicle, 
                    field:str, 
                    badData:Any, 
                    **kwargs)->Tuple[Any, bool]:
    """
    Handle corrupted field by replacing with a sentinel value.

    Sentinel values are chosen to be obviously bad. Requires downstream handler
    to identify and prevent data consumption.

    Parameters
    ----------
    vehicle : Vehicle
        Vehicle receiving corrupted message.
    field : str
        Message field identifier of corrupted data. (eg. 'reporter_pos').
    badData : Any
        Corrupted data.
    **kwargs:
        Additional parameters passed in for specific strategies

        - sentinel: Custom sentinel value to use instead of default (scalar)

    Returns
    -------
    goodData : Any
        Sentinel value for field type.
    isSuccess : bool
        True if the strategy was successful. In theory, always True as sentinel
        value always available.
    
    Notes
    -----
    **Typical sentinel values:**

    - Arrays: filled with appropriate scalar sentinel
    - Scalars:

        - bytes: '\xFF'
        - integers: -1
        - times: -1.0
        - positions: -999999.0
        - velocities: -999.0
    """

    try:
        # Get keyword args if supplied
        sentinel = kwargs.get('sentinel')

        # Default policy values
        defaultScalars = {
            #bytes
            'type': b'\xFF\xFF\xFF\xFF',
            'group_id': b'\xFF',
            #integers
            'reporter_id': -1,
            'num_followers': -1,
            #times
            'time_stamp': -1.0,
            'rspn_start': -1.0,
            'followers_rspn': -1.0,
        }
        defaultVectors = {
            #positions
            'reporter_pos': -999999.0,
            'leader_pos': -999999.0,
            'leader_next_pos': -999999.0,
            'followers_pos': -999999.0,
            #velocities
            'reporter_vel': -999.0,
            'leader_vel': -999.0,
            'leader_next_vel': -999.0,
            'followers_vel': -999.0,
        }

        # Check simple scalar field
        if ((lookup := defaultScalars.get(field)) is not None):
            return (sentinel if sentinel is not None else lookup), True

        # Check simple vector fields
        elif ((lookup := defaultVectors.get(field)) is not None):
            value = (sentinel if sentinel is not None else lookup)
            return np.full(3, value), True

        # Check fields with unique handling
        elif (field == 'rspn_sched'):
            if (group := vehicle.group):
                cnt = len(group)
                n = cnt if vehicle.isLeader else cnt + 1
                return [sentinel if sentinel is not None else -1] * n, True
            return badData, False

        # Else attempt failed
        return badData, False

    except Exception as e:
        log.warning('%s: CORRUPT DATA RECOVERY FAILED: SENTINEL - %s: %s',
                  vehicle.callSign, field.upper(), str(e))
        return badData, False

###############################################################################

def recoverSkipField(vehicle:Vehicle, 
                     field:str, 
                     badData:Any, 
                     **kwargs)->Tuple[Any, bool]:
    """
    Skip recovery for corrupted field but continue processing rest of message.

    Allows continued message processing despite corrupted field by always
    returning isSuccess=True. Original corrupted data is returned.

    Parameters
    ----------
    vehicle : Vehicle
        Vehicle receiving corrupted message.
    field : str
        Message field identifier of corrupted data. (eg. 'reporter_pos').
    badData : Any
        Corrupted data.
    **kwargs:
        Additional parameters passed in for specific strategies.

    Returns
    -------
    badData : Any
        Original corrupted data.
    isSuccess : bool
        True if the strategy was successful. True for all cases.
    """
    
    return badData, True

###############################################################################

def recoverSkip(vehicle:Vehicle, 
                field:str, 
                badData:Any, 
                **kwargs)->Tuple[Any, bool]:
    """
    Handle corrupt data by skipping the entire message.

    Prevents continued message processing by always returning isSuccess=False.
    Original corrupted data is returned.

    Parameters
    ----------
    vehicle : Vehicle
        Vehicle receiving corrupted message.
    field : str
        Message field identifier of corrupted data. (eg. 'reporter_pos').
    badData : Any
        Corrupted data.
    **kwargs:
        Additional parameters passed in for specific strategies.

    Returns
    -------
    badData : Any
        Original corrupted data.
    isSuccess : bool
        True if the strategy was successful. False for all cases.
    """
    
    return badData, False

###############################################################################
