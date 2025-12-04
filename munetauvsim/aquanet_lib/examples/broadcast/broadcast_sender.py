#!/usr/bin/python3

"""
@package aquanet-lib
Created on Jun 17, 2023

@author: Dmitrii Dugaev


Example of broadcast send operation
"""

import time
# Import aquanet_lib module
from __init__ import *

# this is a broadcast address for aquanet
AQUANET_BCAST_ADDR = 255    


def main():
    # Initialize aquanet-stack
    nodeAddr = 1
    baseFolder = "/home/dmitrii/aquanet_lib"
    aquaNetManager = AquaNetManager(nodeAddr, baseFolder, macProto="TRUMAC", trumacContentionTimeoutMs=100, trumacGuardTimeMs=10)
    aquaNetManager.initAquaNet()

    # Send message to AquaNet
    print("broadcasting messages to AquaNet")
    for i in range(10):
        print("Sending: " + "broadcast hello: " + str(i))
        aquaNetManager.send(("broadcast hello: " + str(i)).encode(), AQUANET_BCAST_ADDR)
        time.sleep(1)

    # stop aquanet stack at the end
    aquaNetManager.stop()


if __name__ == '__main__':
    main()
