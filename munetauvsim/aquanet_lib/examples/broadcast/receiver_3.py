#!/usr/bin/python3

"""
@package aquanet-lib
Created on Jun 17, 2023

@author: Dmitrii Dugaev


This is a receiver example
"""

# Import aquanet_lib module
from __init__ import *


# call this function when receiving messages
def callback(msg):
    print("Callback on received msg:", msg)


def main():
    # Initialize aquanet-stack
    nodeAddr = 4
    baseFolder = "/home/dmitrii/aquanet_lib"
    aquaNetManager = AquaNetManager(nodeAddr, baseFolder)
    aquaNetManager.initAquaNet()

    try:
        # Receive messages from AquaNet
        print("receiving messages from AquaNet")
        aquaNetManager.recv(callback)

    except KeyboardInterrupt:
        print("Keyboard interrupt received. Exiting gracefully.")
        aquaNetManager.stop()


if __name__ == '__main__':
    main()
