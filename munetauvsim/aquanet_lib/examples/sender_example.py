#!/usr/bin/python3

"""
@package aquanet-lib
Created on Jun 17, 2023

@author: Dmitrii Dugaev


This is an example how to initialize and send user messages over aquanet-lib and AquaNet stack
"""

import time
# Import aquanet_lib module
from __init__ import *


def main():
    # Initialize aquanet-stack
    nodeAddr = 1
    destAddr = 2
    baseFolder = "/home/dmitrii/aquanet_lib"
    aquaNetManager = AquaNetManager(nodeAddr, baseFolder)
    aquaNetManager.initAquaNet()

    # Send message to AquaNet
    print("sending messages to AquaNet")
    for i in range(10):
        aquaNetManager.send(("hello: " + str(i)).encode(), destAddr)
        time.sleep(1)

    # stop aquanet stack at the end
    aquaNetManager.stop()


if __name__ == '__main__':
    main()
