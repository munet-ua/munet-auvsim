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

PLR = 0.00                      # default = 0.01 (one percent)
CHANNEL_DELAY_MS = 0            # milliseconds
CHANNEL_JITTER = 0.0            # add 10% of randomness to channel delay
