# Poisson traffic application

`poisson_traffic.py` implements an application that generates and sends messages according to Poisson distribution.

The script also tracks all transmission `t` and reception `r` events and saves them into a trace-file `.tr` for further processing. Each `t/r` event is traced with additional information, such as timestamp, message sequence number, CRC checksum, etc.

An example of a trace file format is presented below:

```
>>> cat 1-0.5-11.tr 
['./poisson_traffic.py', '1', '2', '0.5', '11', 'TRUMAC', '2', '100', '10']
TX-RX : NODE_ID : SOURCE_ID : DEST_ID : SEQ_NO : FRAME_SIZE : DELAY_MS : CRC_FAILED : TIMESTAMP
t:1:1:2:0:26:0:0:1690328728.1154952
t:1:1:2:1:26:0:0:1690328728.8365922
r:1:2:1:0:26:619:0:1690328729.294102
t:1:1:2:2:26:0:0:1690328729.9480557
t:1:1:2:3:26:0:0:1690328730.9193537
r:1:2:1:1:26:631:0:1690328730.9792767
r:1:2:1:2:26:617:0:1690328731.4555762
r:1:2:1:3:26:551:0:1690328732.441459
t:1:1:2:4:26:0:0:1690328736.49274
```

## Program Format

`poisson_traffic.py` takes the following arguments:

```
./poisson_traffic <src_addr> <dst_addr> <lambda_rate> <message_size_bytes> <MAC protocol>
```

where:

`src_addr` - source address of the local node.

`dst_addr` - address of a destination node.

`lambda_rate` - lambda parameter of Poisson distribution: e.g. if `lambda_rate = 0.5`, the app will be sending 0.5pkts/sec in average.

`msg_size_bytes` - size of a generated message. Together with the control fields (i.e. src/dst addresses, CRC checks, timestamps, etc.), this constructs a final "frame size" that is sent down to AquaNet.

`MAC protocol` - define MAC protocol to use at L2 of AquaNet. Available options are: BCMAC (by default), ALOHA and TRUMAC.

## Installation and configuration

Clone `aquanet_lib` to your node:

```
git clone https://github.com/dugdmitry/aquanet_lib.git
```

**Important:** To configure `aquanet_lib` to work with real modems on ARM-based platforms (i.e. on Raspberry Pis), please make sure that `ARM_PLATFORM` and `USE_GATECH` are set to `True` on lines 29-30.

Also, please make sure to modify `AQUANET_BASE_FOLDER` providing a valid path to `aquanet_lib` folder.


To run a simple example, go to `applications` folder and execute `poisson_traffic.py`:

```
./poisson_traffic.py 2 1 0.5 11 ALOHA
```

This will run the Poisson app from `node-2` to `node-1` with `0.5pkts/sec`, `11-byte` message payload and `ALOHA` at L2.

While the script runs, it will keep updating a trace-file with the following name: `2-0.5-11.tr`.

## Running example

This example assumes two nodes: `RPi-1` and `RPi-2`, with acoustic modems attached.

At `RPi-1`, run `poisson_traffic.py` in background with `ALOHA` at L2:

```
>>> nohup ./poisson_traffic.py 1 2 0.05 10 ALOHA &
```

At `RPi-2`, run `poisson_traffic.py` in background with `ALOHA` at L2:

```
>>> nohup ./poisson_traffic.py 2 1 0.05 10 ALOHA &
```

As a result, the nodes will generate two trace files with `1-0.05-10.tr` and `2-0.05-10.tr` names, accordingly.

To stop `poisson_traffic.py` running in the background, you may use the following command:

```
killall poisson_traffic.py
```

**Important:** After that, please make sure that no AquaNet proccesses are left in the background. You can check that using `ps aux | grep aqua` command.

If there are AquaNet processes left in the background you can terminate them by executing `stack-stop.sh` script:

```
>>> cd scripts/
>>> ./stack-stop.sh
```

## Processing trace-files

Given a trace-file format (see line 2 of the trace-file), it is now possible to gather basic performance metrics of the performed communication test.

`calc_metrics.py` takes the trace-files and calculates the following metrics:

```
- Packet Delivery Ratio (PDR)
- Total Network Throughput
- E2E Delay
- Number of corrupted packets
```

To run the script, modify line-20 with the actual trace-file names that needs to be processed, e.g.:

```
TRACE_FILES = ["1-0.5-10.tr",  "2-0.5-10.tr"]
```

You should be able to see the following output:

```
>>> ./calc_metrics.py 
#Nodes	PDR	Throughput_bps	Delay_ms	Corrupted_pkts
2	0.59	74.77	722.37	0
```
