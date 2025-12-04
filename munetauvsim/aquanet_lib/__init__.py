#!/usr/bin/python3

"""
@package aquanet-lib Created on Jun 17, 2023

@author: Dmitrii Dugaev


This module provides methods to interact with AquaNet communication stack via
Unix Domain Socket interface.
-------------------------------------------------------------------------------

This is a modified copy of the original '__init__.py'.

JP Crawford
University of Alabama
July 2023

Modifications:
    --+ Added '.' to 'import emulation_config'.
    --+ Removed anything related to ROS: importing uuv-simulator Waypoint, any
        code relating to 'publish()', and any code dealing with serialization /
        deserialization.
    --+ Adjusted overall text to a maximum line width of 80 characters.
    --+ Added '*args' in function paramaters for 'recv()' and 'callback()'.
    --+ Moved 'callback()' out of 'else' block.
    --+ Added 'logToFile' parameter to wrap control of printing to stdout vs a 
        logFile
    --+ Changed defaults: trumacMaxNode=10, trumacContentionTimeoutMS=100, 
        trumacGuardTimeMS=10
    --> Changed default MAC protocol to "TRUMAC". (May return to BCMAC when
        'routing loop' issue has been resolved.)

Notes:
    -- BCMAC is the original default. This MAC layer is based on a broadcast
       method and lets the routing layer handle the destination address.
       Because of this, when attempting to unicast with more than 2 nodes there
       is a "routing loop" created whereby the other nodes bounce the packet
       between layers and the destination will repeatedly receive the unicast.
       -- BCMAC broadcast address: 255

    -- ALOHA supports unicast but does not support broadcast.

    -- TRUMAC is a 'Token-Ring Based' MAC layer and does support both broadcast
       and unicast while checking the destination address before passing to the
       routing layer.
       -- Max Node: set this equal to the total number of nodes you will use
       -- Contention Timeout: switches from "collision-free" to 
          "contention based" operation. Setting to 100ms or lower will force
          TRUMAC to work as basic ALOHA protocol with minimal access delay, but
          with broadcast capability.
       -- Guard Time: time between two consecutive transmission slots. Should 
          be less than Contention Timeout. Set to 10ms to minimize delay
"""

# Import necessary python modules from the standard library
import subprocess
import socket
import time
import struct
import logging

# channel emulation parameters
from .emulation_config import *

# Global parameters
VMDS_ADDR = "127.0.0.1"
VMDS_PORT = "2021"
LOG_NAME = "aquanet.log"

## Class for handling send/recv communication with underlying AquaNet stack
class AquaNetManager:
    ## Constructor
    def __init__(self, nodeId, baseFolder, 
                 arm=False, 
                 gatech=False, 
                 macProto="TRUMAC", 
                 trumacMaxNode=10, 
                 trumacContentionTimeoutMs=100, 
                 trumacGuardTimeMs=10,
                 logToFile = True):
        self.nodeId = nodeId
        self.baseFolder = baseFolder
        self.workingDir = baseFolder + "/tmp" + "/node" + str(self.nodeId)
        self.socketSendPath = self.workingDir + "/socket_send"
        self.socketRecvPath = self.workingDir + "/socket_recv"
        self.send_socket = 0
        self.recv_socket = 0
        self.publishAddr = 0      # store default publishing address when using
                                  # publish() method
        self.macProto = macProto  # BCMAC by default
        
        # set logFile option
        self.logFile = None
        self.logToFile = logToFile
        if (self.logToFile):
            self.logFilePath = self.workingDir + "/" + LOG_NAME

        # set logger
        self.log = logging.getLogger('AquaNet')
        
        # default TRUMAC params
        self.trumacMaxNode = trumacMaxNode
        self.trumacContentionTimeoutMs = trumacContentionTimeoutMs
        self.trumacGuardTimeMs = trumacGuardTimeMs

        # check if ARM platform or not
        self.armFolder = ""
        if arm:
            self.armFolder = "arm/"
    
        # decide whether to use VMDS emulation or GATECH driver
        self.gatech = gatech

        # refresh working directory from previous sessions
        subprocess.Popen("rm -r " + self.workingDir, shell=True).wait()
        subprocess.Popen("mkdir -p " + self.workingDir, shell=True).wait()

        # create aquanet config files
        subprocess.Popen("touch " + self.workingDir + 
                         "/config_add.cfg", shell=True).wait()
        subprocess.Popen("echo " + str(self.nodeId) + " : " + 
                         str(self.nodeId) + " > " + self.workingDir + 
                         "/config_add.cfg", shell=True).wait()
        
        # copy arp, conn and route configurations
        subprocess.Popen("cp " + baseFolder + "/configs/" + "config_arp.cfg" + 
                         " " + self.workingDir, shell=True).wait()
        subprocess.Popen("cp " + baseFolder + "/configs/" + "config_conn.cfg" +
                         " " + self.workingDir, shell=True).wait()
        subprocess.Popen("cp " + baseFolder + "/configs/" + "config_net.cfg" + 
                         " " + self.workingDir, shell=True).wait()
        
        # set mac protocols
        if (self.macProto == "BCMAC"):
            # copy bc-mac configuration file
            subprocess.Popen("cp " + baseFolder + "/configs/" + 
                             "aquanet-bcmac.cfg" + " " + self.workingDir, 
                             shell=True).wait()
        elif (self.macProto == "ALOHA"):
            # aloha has no configuration file
            pass
        elif (self.macProto == "TRUMAC"):
            subprocess.Popen("touch " + self.workingDir + 
                             "/aquanet-trumac.cfg", shell=True).wait()
            # put max node_id : contention_timeout_ms : guard_time_ms parameters
            subprocess.Popen("echo " + str(self.trumacMaxNode) + " : " + 
                             str(self.trumacContentionTimeoutMs) + " : " + 
                             str(self.trumacGuardTimeMs) + " > " + 
                             self.workingDir + "/aquanet-trumac.cfg", 
                             shell=True).wait()
        else:
            self.log.error("ERROR! Unkown MAC protocol provided." + 
                           " Using BCMAC instead.")
            # copy bc-mac configuration file
            subprocess.Popen("cp " + baseFolder + "/configs/" +
                             "aquanet-bcmac.cfg" + " " + self.workingDir, 
                             shell=True).wait()
            self.macProto = "BCMAC"

        # copy GATech serial interface configuration
        if self.gatech:
            subprocess.Popen("cp " + baseFolder + "/configs/" + 
                             "config_ser.cfg" + " " + self.workingDir, 
                             shell=True).wait()

    ## Initialize AquaNet processes
    def initAquaNet(self):
        # create a recv_socket for receiving incoming connections from AquaNet
        self.recv_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.recv_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            # Bind the socket to the specified path
            self.recv_socket.bind(self.socketRecvPath)
            self.log.info("socket file created and bound to the Unix domain" + 
                          " socket.")
        except socket.error as e:
            self.log.error("error binding the socket: %s", e)
            self.recv_socket.close()
            return
        self.recv_socket.listen(5)

        # create log-file descriptor
        if (self.logToFile):
            self.logFile = open(self.logFilePath, "w")

        # start AquaNet stack
        if not self.isPortTaken(VMDS_PORT) and not self.gatech:
            self.log.info("starting local VMDS server")
            subprocess.Popen(["../../bin/" + self.armFolder + "aquanet-vmds", 
                              VMDS_PORT], cwd=self.workingDir, 
                              stdout=self.logFile, stderr=self.logFile)
            time.sleep(0.5)

        self.log.info("starting protocol stack...")
        subprocess.Popen(["../../bin/" + self.armFolder + "aquanet-stack"], 
                         cwd=self.workingDir, stdout=self.logFile, 
                         stderr=self.logFile)
        time.sleep(0.5)

        if not self.gatech:
            self.log.info("starting VMDM client...")
            subprocess.Popen(["../../bin/" + self.armFolder + "aquanet-vmdc", 
                              VMDS_ADDR, VMDS_PORT, str(self.nodeId), "0", 
                              "0", "0", str(PLR), str(CHANNEL_DELAY_MS), 
                              str(CHANNEL_JITTER)], cwd=self.workingDir, 
                              stdout=self.logFile, stderr=self.logFile)
            time.sleep(0.5)
        else:
            # start interface to real modem
            self.log.info("starting GATech driver...")
            subprocess.Popen(["../../bin/" + self.armFolder + 
                              "aquanet-gatech"], cwd=self.workingDir, 
                              stdout=self.logFile, stderr=self.logFile)
            time.sleep(0.5)

        if (self.macProto == "BCMAC"):
            self.log.info("starting BCMAC MAC protocol...")
            subprocess.Popen(["../../bin/" + self.armFolder + "aquanet-bcmac"],
                             cwd=self.workingDir, stdout=self.logFile, 
                             stderr=self.logFile)
            time.sleep(0.5)
        if (self.macProto == "ALOHA"):
            self.log.info("starting ALOHA MAC protocol...")
            subprocess.Popen(["../../bin/" + self.armFolder + 
                              "aquanet-uwaloha"], cwd=self.workingDir, 
                              stdout=self.logFile, stderr=self.logFile)
            time.sleep(0.5)
        if (self.macProto == "TRUMAC"):
            self.log.info("starting TRUMAC MAC protocol...")
            subprocess.Popen(["../../bin/" + self.armFolder + 
                              "aquanet-trumac"], cwd=self.workingDir, 
                              stdout=self.logFile, stderr=self.logFile)
            time.sleep(0.5)

        self.log.info("starting routing protocol...")
        subprocess.Popen(["../../bin/" + self.armFolder + "aquanet-sroute"], 
                         cwd=self.workingDir, stdout=self.logFile, 
                         stderr=self.logFile)
        time.sleep(0.5)

        self.log.info("starting transport layer...")
        subprocess.Popen(["../../bin/" + self.armFolder + "aquanet-tra"], 
                         cwd=self.workingDir, stdout=self.logFile, 
                         stderr=self.logFile)
        time.sleep(0.5)

        self.log.info("starting application layer...")
        subprocess.Popen(["../../bin/" + self.armFolder + 
                          "aquanet-socket-interface " + str(self.nodeId) + 
                          " " + self.socketSendPath + " " +
                          self.socketRecvPath], cwd=self.workingDir, 
                          stdout=self.logFile, stderr=self.logFile, shell=True)
        time.sleep(0.5)

        # Connect to unix socket for sending data
        self.send_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.send_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            self.send_socket.connect(self.socketSendPath)
        except socket.error as e:
            self.log.error("Error connecting to the socket: %s", e)

    ## Send to AquaNet
    def send(self, message, destAddr):
        if (self.macProto == "ALOHA" and destAddr == 255):
            self.log.error("Error! ALOHA does not support broadcast " + 
                           "transmission. Skip sending.")
            return
        try:
            # set the destAddr first
            self.send_socket.sendall(struct.pack("<h", destAddr))
            # send the message right after
            self.send_socket.sendall(message)
            self.log.info("Message sent: %s", message)
        except socket.error as e:
            self.log.error("Error sending data: %s", e)

    ## Receive from AquaNet
    def recv(self, callback, *args):
        # Accept a client connection
        client_sock, client_addr = self.recv_socket.accept()

        # Receive data from the client
        while True:
            data = client_sock.recv(1024)
            # If empty bytes object is received, the sender has closed the 
            # connection
            if not data:  
                self.log.info("Sender has closed the connection.")
                break
            # Process the received data
            callback(data, *args)

        # Close the client socket
        client_sock.close()

    ## Check if specific TCP port taken to ensure that VMDS is already running
    def isPortTaken(self, port):
        # Create a socket object
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # Try to bind the socket to the given port
            sock.bind(('localhost', int(port)))
            return False  # Port is available
        except socket.error as e:
            if e.errno == socket.errno.EADDRINUSE:
                return True  # Port is already in use
            else:
                raise e
        finally:
            sock.close()

    ## Stop the AquaNet stack
    def stop(self):
        self.log.info("stopping aquanet...")
        subprocess.Popen(self.baseFolder + "/scripts/stack-stop.sh", 
                         shell=True)
        if (self.logToFile):
            self.logFile.close()