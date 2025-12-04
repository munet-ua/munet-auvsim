# aquanet_lib

Python library to interface with AquaNet stack of protocols.

## Usage

Clone `aquanet_lib` repository to your working directory:

    git clone https://github.com/dugdmitry/aquanet_lib.git

Import `aquanet_lib` module to your python program:

    import aquanet_lib

To initialize AquaNet stack and start sending/receiving messages, use the following code:

    ## Initialize aquanet-manager
	nodeAddr = 1    # your local node address
	destAddr = 2    # destination node address
	baseFolder = "/home/user/ros_catkin_ws/src/multi_auv_sim/scripts/backup/aquanet_lib"    # base folder of your scripting program
    # create the manager object
	aquaNetManager = aquanet_lib.AquaNetManager(nodeAddr, baseFolder)
    # initialize aquanet stack
    aquaNetManager.initAquaNet()

    # to send messages
    aquaNetManager.send(("hello".encode(), destAddr)

    # to receive messages
    aquaNetManager.recv(recv_callback)

Now, you can send/receive messages over aquanet-stack using `send()`, `publish()` and `recv()` methods. The basic send and receive examples can be found under `examples` folder.

## Example 1: run basic send-receive operation

    cd examples/

The following scripts will instantiate 2 AquaNet nodes with `1` and `2` network addresses. `Node 1` will be periodically sending `hello` messages to `Node 2`.

Run receive-example script:

    ./receiver_example.py

You should see the following lines at the end of the output:

```
----------------------------------------------------------
		AQUANET-SOCKET-INTERFACE 
The socket-interface application for Aqua-Net
Developed by Dmitrii Dugaev
All rights reserved
----------------------------------------------------------

X:2:2-24210126>[socket-interface]	Starting AQUANET-SOCKET-INTERFACE.
X:2:2-24210126>[     stack]	Application Module Connected.
X:2:2-24210126>[socket-interface]	unix-domain socket created and listening for incoming connections
receiving messages from AquaNet
```

In another terminal, run sender-example script:

    ./sender_example.py

The script will start sending `hello` messages periodically:

```
X:1:1-24210132>[socket-interface]	Starting AQUANET-SOCKET-INTERFACE.
X:1:1-24210132>[socket-interface]	unix-domain socket created and listening for incoming connections
X:1:1-24210132>[     stack]	Application Module Connected.
sending messages to AquaNet
X:1:1-24210133>[socket-interface]	Client connected. Receiving data...
Message sent: b'hello: 0'
X:1:1-24210133>[socket-interface]	received from unix-socket: hello: 0
X:1:1-24210133>[socket-interface]	sending 8 bytes from 1 to 2
X:1:1-24210133>[     stack]	Got 1 connection.
X:1:1-24210133>[     stack]	Received 8 bytes from app layer
X:1:1-24210133>[       tra]	received 8 bytes
X:1:1-24210133>[     stack]	Got 1 connection.
X:1:1-24210133>[     stack]	Received 12 bytes from tra layer
X:1:1-24210133>[    sroute]	Received 12 bytes
X:1:1-24210133>[    sroute]	Next Hop for 2: Node 2
X:1:1-24210133>[     stack]	Got 1 connection.
X:1:1-24210133>[     stack]	Received 22 bytes from net layer
X:1:1-24210133>[     bcmac]	Node 1 : Get Packet from upper layer & cache it
X:1:1-24210133>Node 1 : Send Packet in 0 seconds 13400 microseconds
X:1:1-24210133>[     stack]	Got 1 connection.
X:1:1-24210133>[     stack]	Received 32 bytes from mac layer
X:1:1-24210133>[      vmdc]	Got 1 connection
X:1:1-24210133>[      vmdc]	Got packets from the protocol stack
X:1:1-24210133>[      vmdc]	received 664 bytes from protocol stack:hello: 0
X:1:1-24210133>[      vmdc]	modem_send: sent 664 bytes
```

At the receiver-end, the messages should start appearing in the console output:

```
X:2:2-24210127>[socket-interface]	Client connected. Receiving data...
X:2:2-24210130>[      vmds]	New connection from 127.0.0.1 on socket 5
X:2:2-24210133>[      vmds]	received 664 bytes from socket 5
X:2:2-24210133>[      vmds]	sent 664 bytes to socket 4
X:2:2-24210133>[      vmdc]	Got 1 connection
X:2:2-24210133>[      vmdc]	Got packets from the virtual modem
X:2:2-24210133>[      vmdc]	modem_recv: received 664 bytes: hello: 0
X:2:2-24210133>[      vmdc]	received 664 bytes from serial port:hello: 0
X:2:2-24210133>[     stack]	Got 1 connection.
X:2:2-24210133>[     stack]	Received 664 bytes from phy layer
X:2:2-24210133>Node 2 : Get Packet from node 1 & send it to upper layer
X:2:2-24210133>[     stack]	Got 1 connection.
X:2:2-24210133>[     stack]	Received 22 bytes from mac layer
X:2:2-24210133>[    sroute]	Received 22 bytes
X:2:2-24210133>[    sroute]	It's for me, receive it
X:2:2-24210133>[     stack]	Got 1 connection.
X:2:2-24210133>[     stack]	Received 22 bytes from net layer
X:2:2-24210133>[       tra]	received 12 bytes
X:2:2-24210133>[     stack]	Got 1 connection.
X:2:2-24210133>[     stack]	Received 12 bytes from tra layer

X:2:2-24210133>[socket-interface]	received from 1 to 2: hello: 0
Callback on received msg: b'hello: 0'
```

## Example 2: run swarm-tracing algoithm in Gazebo+UUV simulator

To run this example, Gazebo and ROS+UUV simulator should be installed. Also, make sure that `uuv_plume_simulator` and `multi_auv_sim` packages are installed in ROS environment. See more details here:

ROS installation:

http://wiki.ros.org/noetic/Installation/Ubuntu

UUV+Gazebo instllation:

https://uuvsimulator.github.io/installation/

`uuv_plume_simulator` package info:

https://uuvsimulator.github.io/packages/uuv_plume_simulator/intro/

In `examples/uuv` folder, you can find examples of the modified `leader1.py` and `node2.py` scripts alongisde with necessary launch files to run a plume-tracing scenario. The communication between `leader1` and `node2` is conducted over AquaNet stack using `aquanet_lib` library.

Step 1: Run ROV simulation:

    roslaunch multi_auv_sim multi_rov_test.launch

RViz window should appear, visualizing 5 ROVs. The communication over AquaNet is established in 1-way unicast way from `leader1` to `node2`.

Step 2: Run plume-tracing script:

    roslaunch multi_auv_sim start_mbplume.launch

You should see the following output in the terminal, indicating that `aquanet_lib` has published a ROS message from `leader1` towards `node2`:

```
Publishing ROS message:
header: 
  seq: 0
  stamp: 
    secs: 80
    nsecs: 990000000
  frame_id: "world"
point: 
  x: 20
  y: 25
  z: -3.34947309796
max_forward_speed: 0.75
heading_offset: 0.0
use_fixed_heading: False
radius_of_acceptance: 0.0
```    

and successfully received/deserialized the ROS-message at `node2`:

```
user-gazebo:2:2-26085825>[socket-interface]	received from 1 to 2: 
('Received msg:', '\x00\x00\x00\x00\x99\x00\x00\x00\x00\xab\x87\x04\x05\x00\x00\x00world\x1c2\xf5w\x1c\x90Q\xc0\xfa\xb0\x8e\xfb\xd5pP\xc0I`%_\x1d\xd1\t\xc0\x00\x00\x00\x00\x00\x00\xe8?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
('Deserialized msg:', header: 
  seq: 0
  stamp: 
    secs: 153
    nsecs:  76000000
  frame_id: "world"
point: 
  x: -70.2517375845
  y: -65.7630604642
  z: -3.22710680325
max_forward_speed: 0.75
heading_offset: 0.0
use_fixed_heading: False
radius_of_acceptance: 0.0)
Algorithm state: Find
```

## Example 3: Basic Broadcast Example

This example shows basic broadcast operation.

Under `examples/broadcast` folder, you find 4 scripts: `broadcast_sender.py`, `receiver_1.py`, `receiver_2.py` and `receiver_3.py`.

`broadcast_sender.py` generates periodic broadcast `hello` messages and send them to all the nodes connected to the network, i.e. receiver 1, 2 and 3.

To run the example, run `receiver_1.py`, `receiver_2.py` and `receiver_3.py` in three different terminals:

```
cd examples/broadcast
./receiver_1.py
./receiver_2.py
./receiver_3.py
```

From each terminal, you should see the output, similar to the Example 1.

Then, in a separate terminal, run the sender script:

```
./broadcast_sender.py
```

The sender should start broadcasting messages by executing `send()` function with the special `broadcast` address specified (see line 31). The defualt aquanet broadcast address is `255`.

You should see the following output:

```
----------------------------------------------------------
		AQUANET-SOCKET-INTERFACE 
The socket-interface application for Aqua-Net
Developed by Dmitrii Dugaev
All rights reserved
----------------------------------------------------------

X:1:1-11203851>[socket-interface]	Starting AQUANET-SOCKET-INTERFACE.
X:1:1-11203851>[socket-interface]	unix-domain socket created and listening for incoming connections
X:1:1-11203851>[     stack]	Application Module Connected.
broadcasting messages to AquaNet
X:1:1-11203851>[socket-interface]	Client connected. Receiving data...
X:1:1-11203851>[socket-interface]	dest_addr: 255
Message sent: b'broadcast hello: 0'
X:1:1-11203851>[socket-interface]	received from unix-socket: broadcast hello: 0
X:1:1-11203851>[socket-interface]	sending 18 bytes from 1 to 255
X:1:1-11203851>[     stack]	Got 1 connection.
X:1:1-11203851>[     stack]	Received 18 bytes from app layer
X:1:1-11203851>[       tra]	received 18 bytes
X:1:1-11203851>[     stack]	Got 1 connection.
X:1:1-11203851>[     stack]	Received 22 bytes from tra layer
X:1:1-11203851>[    sroute]	Received 22 bytes
X:1:1-11203851>[    sroute]	Next Hop for 255: Node 255
X:1:1-11203851>[     stack]	Got 1 connection.
X:1:1-11203851>[     stack]	Received 32 bytes from net layer
X:1:1-11203851>[     bcmac]	Node 1 : Get Packet from upper layer & cache it
X:1:1-11203851>Node 1 : Send Packet in 0 seconds 16840 microseconds
X:1:1-11203851>[     stack]	Got 1 connection.
X:1:1-11203851>[     stack]	Received 42 bytes from mac layer
X:1:1-11203851>[      vmdc]	Got 1 connection
X:1:1-11203851>[      vmdc]	Got packets from the protocol stack
X:1:1-11203851>[      vmdc]	received 664 bytes from protocol stack:broadcast hello: 0
X:1:1-11203851>[      vmdc]	modem_send: sent 664 bytes
Message sent: b'broadcast hello: 1'
...
```

Each receiver should output the following message when the broadcast message is received:

```
...
X:3:3-11203900>[socket-interface]	received from 1 to 255: broadcast hello: 9
Callback on received msg: b'broadcast hello: 9'
...
```

## Example 4: Swarm-tracing algorithm in Gazebo+UUV with 1 leader-node broadcasting to 4 receiver-nodes.

This example is similar to Example 2. The only difference is that the `leader1.py` **broadcasts** messages to **all** 4 nodes in the simulation scenario, i.e. to `node2.py`, `node3.py`, `node4.py` and `node5.py`.

The corresponding modified scripts for `leader1.py`, `node2.py`, `node3.py`, `node4.py` and `node5.py` can be found under `examples/broadcast_uuv/` folder.

To run the Gazebo+UVV scenario please follow the steps from Example 2. I.e. run `multi_rov_test.launch` first, followed by `start_mbplume.launch`.

You should observe the output, similar to Example 2. With the only difference, that the `Waypoint` message is received by all 4 recepients simultaneously:

```
...
('Received msg:', '\x00\x00\x00\x00W\x00\x00\x00\x80,\xea\x18\x05\x00\x00\x00world\xf0;\x0c\x8af\x93Q\xc0\xb8\x978\x14;yP\xc03\xbf\xbe>\xe0\xfc*\xc0\x00\x00\x00\x00\x00\x00\xe8?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
user-gazebo:5:5-10103057>[     stack]	Received 84 bytes from net layer
user-gazebo:4:4-10103057>[     stack]	Got 1 connection.
user-gazebo:4:4-10103057>[     stack]	Received 664 bytes from phy layer
user-gazebo:5:5-10103057>[       tra]	received 74 bytes
user-gazebo:5:5-10103057>[     stack]	Got 1 connection.
user-gazebo:5:5-10103057>[     stack]	Received 74 bytes from tra layer
user-gazebo:4:4-10103057>Node 4 : Get Packet from node 1 & send it to upper layer

user-gazebo:4:4-10103057>[     stack]	Got 1 connection.
user-gazebo:5:5-10103057>[socket-interface]	received from 1 to 255: 
user-gazebo:4:4-10103057>[     stack]	Received 84 bytes from mac layer
('Received msg:', '\x00\x00\x00\x00W\x00\x00\x00\x80,\xea\x18\x05\x00\x00\x00world\xf0;\x0c\x8af\x93Q\xc0\xb8\x978\x14;yP\xc03\xbf\xbe>\xe0\xfc*\xc0\x00\x00\x00\x00\x00\x00\xe8?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
user-gazebo:4:4-10103057>[    sroute]	Received 84 bytes
user-gazebo:4:4-10103057>[    sroute]	It's for me, receive it
user-gazebo:4:4-10103057>[     stack]	Got 1 connection.
user-gazebo:4:4-10103057>[     stack]	Received 84 bytes from net layer
user-gazebo:4:4-10103057>[       tra]	received 74 bytes
user-gazebo:4:4-10103057>[     stack]	Got 1 connection.
user-gazebo:4:4-10103057>[     stack]	Received 74 bytes from tra layer

user-gazebo:4:4-10103057>[socket-interface]	received from 1 to 255: 
('Deserialized msg:', header: 
  seq: 0
  stamp: 
    secs: 87
    nsecs: 418000000
  frame_id: "world"
point: 
  x: -70.3031334991
  y: -65.8942308953
  z: -13.4938983543
max_forward_speed: 0.75
heading_offset: 0.0
use_fixed_heading: False
radius_of_acceptance: 0.0)
('Received msg:', '\x00\x00\x00\x00W\x00\x00\x00\x80,\xea\x18\x05\x00\x00\x00world\xf0;\x0c\x8af\x93Q\xc0\xb8\x978\x14;yP\xc03\xbf\xbe>\xe0\xfc*\xc0\x00\x00\x00\x00\x00\x00\xe8?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
('Deserialized msg:', header: 
  seq: 0
  stamp: 
    secs: 87
    nsecs: 418000000
  frame_id: "world"
point: 
  x: -70.3031334991
  y: -65.8942308953
  z: -13.4938983543
max_forward_speed: 0.75
heading_offset: 0.0
use_fixed_heading: False
radius_of_acceptance: 0.0)
('Deserialized msg:', header: 
  seq: 0
  stamp: 
    secs: 87
    nsecs: 418000000
  frame_id: "world"
point: 
  x: -70.3031334991
  y: -65.8942308953
  z: -13.4938983543
max_forward_speed: 0.75
heading_offset: 0.0
use_fixed_heading: False
radius_of_acceptance: 0.0)
user-gazebo:2:2-10103057>[     stack]	Got 1 connection.
user-gazebo:2:2-10103057>[     stack]	Received 664 bytes from phy layer
user-gazebo:2:2-10103057>Node 2 : Get Packet from node 1 & send it to upper layer
user-gazebo:2:2-10103057>[     stack]	Got 1 connection.
user-gazebo:2:2-10103057>[     stack]	Received 84 bytes from mac layer
user-gazebo:2:2-10103057>[    sroute]	Received 84 bytes
user-gazebo:2:2-10103057>[    sroute]	It's for me, receive it
user-gazebo:2:2-10103057>[     stack]	Got 1 connection.
user-gazebo:2:2-10103057>[     stack]	Received 84 bytes from net layer
user-gazebo:2:2-10103057>[       tra]	received 74 bytes
user-gazebo:2:2-10103057>[     stack]	Got 1 connection.
user-gazebo:2:2-10103057>[     stack]	Received 74 bytes from tra layer

user-gazebo:2:2-10103057>[socket-interface]	received from 1 to 255: 
('Received msg:', '\x00\x00\x00\x00W\x00\x00\x00\x80,\xea\x18\x05\x00\x00\x00world\xf0;\x0c\x8af\x93Q\xc0\xb8\x978\x14;yP\xc03\xbf\xbe>\xe0\xfc*\xc0\x00\x00\x00\x00\x00\x00\xe8?\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00')
('Deserialized msg:', header: 
  seq: 0
  stamp: 
    secs: 87
    nsecs: 418000000
  frame_id: "world"
point: 
  x: -70.3031334991
  y: -65.8942308953
  z: -13.4938983543
max_forward_speed: 0.75
heading_offset: 0.0
use_fixed_heading: False
radius_of_acceptance: 0.0)
Algorithm state: Find
...
```

## Channel emulation configuration

`AquaNet-Lib` now provides basic support for channel emulation functionality. This includes three parameters: `packet loss rate`, `channel delay` and `channel jitter`.

`Packet loss rate` determines a percentage of packets that get randomly dropped while traversing over the emulated channel. `Channel delay` emulates channel latency, and `channel jitter` adds random variation to the channel delay.

To control the emulation parameters, `emulation_config.py` file should be modified with the desired values. The default parameters in `emulation_config.py` define 1% of packet losses, 500ms of delay with 10% of random jitter:

```
"""
Define channel emulation parameters here:

1) PLR - Packet Loss Ratio:
Defines a fraction of packets that are randomly dropped in the channel.
Range: [0.0, 1.0], where 0.0 - no packets dropped; 1.0 - all packets dropped

2) Channel delay, ms:
Defines for how long a packet stays in the channel, emulating the network latency, in milliseconds.

3) Channel jitter:
Adds randomness to the channel delay.
Range: [0.0, 0.1]
E.g.:
If channel jitter is 0.1 and delay is 500ms, then the actual delay will be randomly picked wihtin [500 - 50, 500 + 50] range.

"""

PLR = 0.01                      # one percent
CHANNEL_DELAY_MS = 500          # milliseconds
CHANNEL_JITTER = 0.1            # add 10% of randomness to channel delay

```

Make sure that `emulation_config.py` is created and initialized with proper values before initializing the aquanet stack via `initAquaNet()`.

## Current limitations

### Limited topology and MAC/routing protocol support:

Current implementation has been tested **only** with `Static` routing at L3, and `Broadcast MAC` at L2. The `Static` routing uses a pre-defined routing table (see `configs/config_net.cfg` file). The `Broadcast MAC` is similar to `Pure ALOHA` protocol that sends packets with no channel sensing, and with some `jitter` to mitigate simultaneous transmissions from multiple nodes.

The other potential L3/L2 protocols are either not supported or should be tested later. This includes `ALOHA`, `SFAMA` and `TRUMAC` at L2, and `Dynamic` routing at L3.

In addition, the network topology is limited to `1-hop` network **only**, meaning that all nodes are directly connected with each other. This means, if a node sends a `unicast` message, the message will go directly to the recepient, with no relays in-between. When a node sends a `broadcast` message, the message is received directly by all the nodes, and the Routing layer **does not relay** this message back to the network (e.g., for `Flooding` purposes).

## Troubleshooting

The software is still very experimental and in the early stage of development. There might be significant changes done to both the program design and user interfaces in the future.

Thus, please be aware that the software may crash or become unresponsive. When this happens, please follow the following general algorithm:

1) Stop all the sender/receiver scripts that you're running. E.g., `sender_example.py` script, `leader1.py`, `node2.py`, `start_mbplume.launch` and `multi_rov_test.launch` files, etc.

2) In a separate terminal, kill all the AquaNet-related processes by executing `scripts/stack-stop.sh` script.

3) Wait at least 30 seconds before launching the programs a second time. This is needed to give time to a Linux system to unlink the unix- and tcp- sockets used by the previous AquaNet processes. Otherwise, a socket connection to AquaNet VMDS server might fail.

4) Restart your experiments. Repeat.

In the event of a crash or non-functional behavior, please save all the console outputs and send a bug-report to the developer.
