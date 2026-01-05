# XPOW.py
#
# Library for defining XPOW class and routines for communicating with the XPOW breadboard. Based on Greg Sercel's script "D:\Greg's DSF Attempt\XPOW.py".
# Modified by: Anusha, 12/3/2025 

#### IMPORTS ####

import sys
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from numpy.polynomial import Polynomial as poly
import glob
import serial
import matplotlib.pyplot as plt
import time
import warnings

import logging
_log = logging.getLogger('XPOW')

warnings.filterwarnings('ignore')

ATTEN_CHANNEL = 39
ATTEN_MAXVOLTAGE = 5
XPOWPorts = ["COM4", "COM6", "COM8"] # change based on device manager
XPOWBaudRate = 115200
XPOWVoltageMax = 17
XPOWVoltageControlMin = 0
XPOWVoltageControlMax = 17
XPOWCurrentMax = 100
XPOWVoltageDelay = 0.5    # used to be 0.5        # (sec) Delay after each XPOW voltage setting
XPOWErrorDelay = 1                              # (sec) Delay between re-tries of sending XPOW commands of error encountered
XOPWCommandDelay = 0.05
XPOWResetInterval = 0
XPOWCommandTimeout = 15
XPOWSerialLines = []
XPOWErrorNum = 0

#### XPOW CLASS ####

class XPOW:
   
    def __init__(self):
        ''' Upon initialization of class instance:
            1. Serial connection to XPOW breadboard.
            2. Resets all channels.

            Parameters
            ----------
            None

            Returns
            -------
            None
        '''
        self.open()
        return

    def open(self):
        ''' Opens connection to XPOW and resets all channels to 0 V.

            Parameters
            ----------
            None

            Returns
            -------
            None
        '''
        XPOW._XPOWOpen()
        XPOW._XPOWResetAllChannels()
        _log.info('Connected to XPOW.')
        return  

    def apply_voltage(self, channel, voltage):
        ''' Applies voltage to a channel.

            Parameters
            ----------
            channel : int
                Channel of interest
            voltage : float
                Voltage to assign channel of interest

            Returns
            -------
            None
        '''
        if (channel == ATTEN_CHANNEL) and (voltage > ATTEN_MAXVOLTAGE):
            raise ValueError(f'Voltage applied to optical attenuator cannot exceed 5 V. Received: {voltage} V')
        XPOW._XPOWChannelAdjust(channel, voltage)
        _log.debug(f"Channel {channel} set to {voltage} V.")
        return

    def read_XPOW(self, channel):
        ''' Readout voltage, current, and power of a channel.

            Parameters
            ----------
            channel : int
                Channel of interest

            Returns
            -------
            actualvoltage : float
                Readout voltage of channel
            current : float
                Readout current of channel
            power : float
                Readout power of channel
        '''
        actualvoltage, current, power = XPOW._XPOWChannelGetData(channel)
        return actualvoltage, current, power

    def close(self):
        ''' Resets all channels to 0 V and closes connection to XPOW.

            Parameters
            ----------
            None

            Returns
            -------
            None
        '''
        XPOW._XPOWResetAllChannels()
        XPOW._XPOWClearPorts()
        _log.debug('Connection to XPOW closed.')
        return 

    #### XPOW STATIC METHODS ####

    @staticmethod
    def _XPOWChannelGetData(channel):

        """
        Function for acquiring the voltage and current readings of a given XPOW channel.
                
        Args:
            channel: int    - XPOW channel
        """
        
        global XPOWErrorNum
        
        error = True
        resetTracker = 0

        while (error == True):
            try:
                if (resetTracker >= XPOWResetInterval):
                    # XPOWResetPorts()
                    resetTracker = 0
                else:
                    resetTracker += 1

                portIdx = None
                modChannel = None

                if (channel >= 1) and (channel <= 40):
                    portIdx = 0
                    modChannel = channel
                elif (channel >= 41) and (channel <= 80):
                    portIdx = 1
                    modChannel = channel - 40
                elif (channel >= 81) and (channel <= 120):
                    portIdx = 2
                    modChannel = channel - 80
                else:
                    raise ValueError("ERROR: Invalid XPOW channel " + str(channel) + "!")

                reply = XPOW._XPOWSendCommandSingle("CH:" + str(modChannel) + ":VAL?", portIdx, XPOWCommandTimeout)
            
                if (reply[0] == '<'):
                    reply = reply[1:]
                if (reply[len(reply) - 1] == '>'):
                    reply = reply[:-1]

                replyParts = reply.split(":")

                replyParts[0] = replyParts[0].rstrip(replyParts[0][14])
                replyParts[1] = replyParts[1].rstrip(replyParts[1][-1])
                replyParts[2] = replyParts[2].rstrip(replyParts[2][-1])
                # _log.debug("channel: ", replyParts[0])
                # _log.debug("voltage: ", replyParts[1])
                # _log.debug("current: ", replyParts[2])

                voltage = float(replyParts[1])
                current = float(replyParts[2])
                power = voltage * current
                error = False

            except:
                XPOWErrorNum += 1

                _log.error("\n-- ERROR " + str(XPOWErrorNum) + " ENCOUNTERED, TRYING AGAIN... --")

                time.sleep(XPOWErrorDelay)
        

        return voltage, current, power

    @staticmethod
    def _XPOWChannelAdjust(channel, voltage):
        """
        Function for setting a certain XPOW channel to a certain voltage.
                
        Args:
            channel: int    - XPOW channel
            voltage: float  - voltage to set XPOW channel to
        """
        
        global XPOWErrorNum

        error = True
        resetTracker = 0

        if (voltage < 0):
            voltage = 0
        if (voltage > XPOWVoltageMax):
            voltage = XPOWVoltageMax

        while (error == True):
            try:
                if (resetTracker >= XPOWResetInterval):
                    # XPOWResetPorts()
                    resetTracker = 0
                else:
                    resetTracker += 1

                portIdx = None
                modChannel = None

                if (channel >= 1) and (channel <= 40):
                    portIdx = 0
                    modChannel = channel
                elif (channel >= 41) and (channel <= 80):
                    portIdx = 1
                    modChannel = channel - 40
                elif (channel >= 81) and (channel <= 120):
                    portIdx = 2
                    modChannel = channel - 80
                else:
                    raise ValueError("ERROR: Invalid XPOW channel " + str(channel) + "!")

                XPOW._XPOWSendCommandSingle("CH:" + str(modChannel) + ":VOLT:" + str(round(voltage, 2)), portIdx, XPOWCommandTimeout)
                time.sleep(XPOWVoltageDelay)
                XPOW._XPOWSendCommandSingle("CH:" + str(modChannel) + ":VAL?", portIdx, XPOWCommandTimeout)

                error = False

            except:
                XPOWErrorNum += 1

                _log.error("\n-- ERROR " + str(XPOWErrorNum) + "for channel " + str(modChannel) + " ENCOUNTERED, TRYING AGAIN... --")

                time.sleep(XPOWErrorDelay)

    @staticmethod
    def _XPOWWaitSerial(portName, serialData, maxIterations):
        iteration = 0

        while iteration < maxIterations:
            if (serialData.inWaiting() > 0):        
                myData = serialData.readline().decode('utf-8')[5:-3]
                msg = portName + "answer = {" + myData + "}"

                return msg
            else:
                iteration += 1
                time.sleep(0.1)

        return portName + " is not answered"
    
    @staticmethod
    def _XPOWCheckKey(xpowKey, maxIterations):
        command = "*key?\n"

        for j in range(len(XPOWSerialLines)):
            XPOWSerialLines[j].write(command.encode())
            value = XPOW._XPOWWaitSerial(XPOWPorts[j], XPOWSerialLines[j], maxIterations)

            if (xpowKey in value):
                _log.debug(value + " | XPOW key MATCHED")
            else:
                _log.debug(value + " | XPOW key NOT MATCHED")

    @staticmethod
    def _XPOWSendCommandAll(cmd, maxIterations):
        _log.debug("Sent XPOW command: \"" + cmd + "\" to all ports.")
        command = cmd + "\n"

        for j in range(len(XPOWSerialLines)):
            XPOWSerialLines[j].write(command.encode())
            time.sleep(XOPWCommandDelay)

            _log.debug(XPOW._XPOWWaitSerial(XPOWPorts[j], XPOWSerialLines[j], maxIterations))

    @staticmethod
    def _XPOWSendCommandSingle(cmd, portIdx, maxIterations):
        # _log.debug("XPOWSerialLines: ", XPOWSerialLines)
        if (portIdx >= len(XPOWSerialLines)):
            raise ValueError("ERROR: XPOW port index " + str(portIdx) + " is too large!")

        _log.debug("Sent XPOW command: \"" + cmd + "\" to port \"" + XPOWPorts[portIdx] + "\".")
        command = cmd + "\n"

        XPOWSerialLines[portIdx].write(command.encode())
        time.sleep(XOPWCommandDelay)

        reply = ""

        if (maxIterations > 0):
            reply = XPOW._XPOWWaitSerial(XPOWPorts[portIdx], XPOWSerialLines[portIdx], maxIterations)
            _log.debug(reply)

        return reply

    @staticmethod
    def _XPOWClearPorts():
        XPOW._XPOWClosePorts()

        XPOWSerialLines.clear()
        time.sleep(XOPWCommandDelay)

        _log.debug("XPOW ports cleared!")

    @staticmethod
    def _XPOWOpenPorts():
        for i in range(len(XPOWSerialLines)):
            XPOWSerialLines[i].open()
            time.sleep(XOPWCommandDelay)

        _log.debug("XPOW ports opened!")

    @staticmethod
    def _XPOWClosePorts():
        for i in range(len(XPOWSerialLines)):
            XPOWSerialLines[i].close()
            time.sleep(XOPWCommandDelay)

        _log.debug("XPOW ports closed!")

    @staticmethod
    def _XPOWCreatePorts():
        for i in range(len(XPOWPorts)):
            XPOWSerialLines.append(serial.Serial(XPOWPorts[i], baudrate = XPOWBaudRate, timeout = 3.0, writeTimeout = 0))
            time.sleep(XOPWCommandDelay)

        _log.debug("XPOW ports created!")

    @staticmethod
    def _XPOWResetPorts():
        XPOW._XPOWClosePorts()
        XPOW._XPOWOpenPorts()

    @staticmethod
    def _XPOWResetAllChannels():
        channel = 1

        while (channel <= 120):
            modChannel = channel
            
            portIdx = 0

            if (channel >= 41) and (channel <= 80):
                modChannel = channel - 40
                portIdx = 1
            elif (channel >= 81) and (channel <= 120):
                modChannel = channel - 80
                portIdx = 2

            XPOW._XPOWSendCommandSingle("CH:" + str(modChannel) + ":VOLT:0", portIdx, XPOWCommandTimeout)
            time.sleep(XPOWVoltageDelay)
            XPOW._XPOWSendCommandSingle("CH:" + str(modChannel) + ":CUR:" + str(XPOWCurrentMax), portIdx, XPOWCommandTimeout)

            channel += 1

    @staticmethod
    def _XPOWOpen():
        _log.debug("------------------------------------------\nChosen XPOW Ports: " + str(XPOWPorts) + "\n------------------------------------------\n")

        error = True

        while (error == True):
            try:
                XPOW._XPOWCreatePorts()
                XPOW._XPOWSendCommandAll("board?", XPOWCommandTimeout)
                # XPOWResetAllChannels()

                error = False

            except:
                _log.error("\n-- XPOW ERROR ENCOUNTERED, TRYING AGAIN... --")

                time.sleep(XPOWErrorDelay)

    @staticmethod
    def _XPOWResetAndOpen():
        _log.debug("------------------------------------------\nChosen XPOW Ports: " + str(XPOWPorts) + "\n------------------------------------------\n")

        error = True

        while (error == True):
            try:
                XPOW._XPOWCreatePorts()
                XPOW._XPOWSendCommandAll("board?", XPOWCommandTimeout)
                XPOW._XPOWResetAllChannels()

                error = False

            except:
                _log.debug("\n-- XPOW ERROR ENCOUNTERED, TRYING AGAIN... --")

                time.sleep(XPOWErrorDelay)

