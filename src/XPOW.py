# XPOW.py
#
# Library for defining XPOW class and routines for communicating with the XPOW breadboard. Based on Greg Sercel's script "D:\Greg's DSF Attempt\XPOW.py".
# Modified by: Yoo Jung, 1/17/2026
# remove "global"s and integrate all constants into class

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

#### XPOW CLASS ####

class XPOW:

    ATTEN_CHANNEL = 60
    ATTEN_MAXVOLTAGE = 4.8
    # XPOWPorts = ["COM4", "COM6", "COM8"] # change based on device manager
    XPOWBaudRate = 115200
    XPOWVoltageMax = 16 # changed from 17 
    XPOWVoltageControlMin = 0
    XPOWVoltageControlMax = 17
    XPOWCurrentMax = 200 # changed from 100
    XPOWVoltageDelay = 0.05    # used to be 0.5        # (sec) Delay after each XPOW voltage setting
    XPOWErrorDelay = 1                              # (sec) Delay between re-tries of sending XPOW commands of error encountered
    XOPWCommandDelay = 0.05
    XPOWResetInterval = 0
    XPOWCommandTimeout = 15
    XPOWSerialLines = []
    XPOWErrorNum = 0
    XPOWVoltagePrecision = 4
    check = False

    def __init__(self, reset = True, ports = ['COM4', 'COM6', 'COM8'],
                 selected_channels = None):
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
        self.XPOWPorts = ports
        self.open(reset = reset,
                  selected_channels = selected_channels)
        self.selected_channels = selected_channels
        return

    def open(self, reset = True, selected_channels = None):
        ''' Opens connection to XPOW and resets all channels to 0 V.

            Parameters
            ----------
            None

            Returns
            -------
            None
        '''
        self._XPOWOpen()
        if reset: self._XPOWResetAllChannels(selected_channels=selected_channels)
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
        if (channel == self.ATTEN_CHANNEL) and (voltage > self.ATTEN_MAXVOLTAGE):
            raise ValueError(f'Voltage applied to optical attenuator cannot exceed {self.ATTEN_MAXVOLTAGE} V. Received: {voltage} V')
        reply = self._XPOWChannelAdjustVoltage(channel, voltage)
        _log.debug(f"Channel {channel} set to {voltage} V.")
        return reply
    
    def apply_current(self, channel, current):
        ''' Applies current to a channel.

            Parameters
            ----------
            channel : int
                Channel of interest
            current : float
                Current to assign channel of interest

            Returns
            -------
            None
        '''
        if (channel == self.ATTEN_CHANNEL) and (current > self.XPOWCurrentMax):
            raise ValueError(f'Voltage applied to optical attenuator cannot exceed {self.XPOWCurrentMax} mA. Received: {current} mA')
        self._XPOWChannelAdjustCurrent(channel, current)
        _log.debug(f"Channel {channel} set to {current} mA.")
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
        actualvoltage, current, power = self._XPOWChannelGetData(channel)
        return actualvoltage, current, power

    def close(self, reset = True, selected_channels = None):
        ''' Resets all channels to 0 V and closes connection to XPOW.

            Parameters
            ----------
            None

            Returns
            -------
            None
        '''
        if reset: self._XPOWResetAllChannels(selected_channels=selected_channels)
        self._XPOWClearPorts()
        _log.info('Connection to XPOW closed.')
        return 

    #### XPOW Internal METHODS ####

    def _XPOWChannelGetData(self, channel):

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
                if (resetTracker >= self.XPOWResetInterval):
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

                reply = self._XPOWSendCommandSingle("CH:" + str(modChannel) + ":VAL?", portIdx, self.XPOWCommandTimeout)
            
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

                time.sleep(self.XPOWErrorDelay)
        

        return voltage, current, power

    def _XPOWChannelAdjustVoltage(self, channel, voltage):
        """
        Function for setting a certain XPOW channel to a certain voltage.
                
        Args:
            channel: int    - XPOW channel
            voltage: float  - voltage to set XPOW channel to
        """
        
        # global XPOWErrorNum

        error = True
        resetTracker = 0

        if (voltage < 0):
            voltage = 0
        if (voltage > self.XPOWVoltageMax):
            voltage = self.XPOWVoltageMax

        while (error == True):
            try:
                if (resetTracker >= self.XPOWResetInterval):
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

                self._XPOWSendCommandSingle("CH:" + str(modChannel) + ":VOLT:" + str(round(voltage, self.XPOWVoltagePrecision)), portIdx, self.XPOWCommandTimeout)
                time.sleep(self.XPOWVoltageDelay)
                if self.check:
                    reply = self._XPOWSendCommandSingle("CH:" + str(modChannel) + ":VAL?", portIdx, self.XPOWCommandTimeout)

                error = False

                if self.check:
                    return reply

            except:
                self.XPOWErrorNum += 1

                _log.error("\n-- ERROR " + str(self.XPOWErrorNum) + "for channel " + str(modChannel) + " ENCOUNTERED, TRYING AGAIN... --")

                time.sleep(self.XPOWErrorDelay)

    def _XPOWChannelAdjustCurrent(self, channel, current):
        """
        Function for setting a certain XPOW channel to a certain current.
                
        Args:
            channel: int    - XPOW channel
            current: float  - current to set XPOW channel to
        """
        
        # global XPOWErrorNum

        error = True
        resetTracker = 0

        if (current < 0):
            current = 0
        if (current > self.XPOWCurrentMax):
            current = self.XPOWCurrentMax

        while (error == True):
            try:
                if (resetTracker >= self.XPOWResetInterval):
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

                self._XPOWSendCommandSingle("CH:" + str(modChannel) + ":CUR:" + str(round(current, 2)), portIdx, self.XPOWCommandTimeout)
                time.sleep(self.XPOWVoltageDelay)
                self._XPOWSendCommandSingle("CH:" + str(modChannel) + ":VAL?", portIdx, self.XPOWCommandTimeout)

                error = False

            except:
                XPOWErrorNum += 1

                _log.error("\n-- ERROR " + str(XPOWErrorNum) + "for channel " + str(modChannel) + " ENCOUNTERED, TRYING AGAIN... --")

                time.sleep(self.XPOWErrorDelay)


    def _XPOWWaitSerial(self, portName, serialData, maxIterations):
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
    

    def _XPOWCheckKey(self, xpowKey, maxIterations):
        command = "*key?\n"

        for j in range(len(self.XPOWSerialLines)):
            self.XPOWSerialLines[j].write(command.encode())
            value = self._XPOWWaitSerial(self.XPOWPorts[j], self.XPOWSerialLines[j], maxIterations)

            if (xpowKey in value):
                _log.debug(value + " | XPOW key MATCHED")
            else:
                _log.debug(value + " | XPOW key NOT MATCHED")

    def _XPOWSendCommandAll(self, cmd, maxIterations):
        _log.debug("Sent XPOW command: \"" + cmd + "\" to all ports.")
        command = cmd + "\n"

        for j in range(len(self.XPOWSerialLines)):
            self.XPOWSerialLines[j].write(command.encode())
            time.sleep(self.XOPWCommandDelay)

            _log.debug(self._XPOWWaitSerial(self.XPOWPorts[j], self.XPOWSerialLines[j], maxIterations))


    def _XPOWSendCommandSingle(self, cmd, portIdx, maxIterations):
        # _log.debug("XPOWSerialLines: ", XPOWSerialLines)
        if (portIdx >= len(self.XPOWSerialLines)):
            raise ValueError("ERROR: XPOW port index " + str(portIdx) + " is too large!")

        _log.debug("Sent XPOW command: \"" + cmd + "\" to port \"" + self.XPOWPorts[portIdx] + "\".")
        command = cmd + "\n"

        self.XPOWSerialLines[portIdx].write(command.encode())
        time.sleep(self.XOPWCommandDelay)

        reply = ""

        if (maxIterations > 0):
            reply = self._XPOWWaitSerial(self.XPOWPorts[portIdx], self.XPOWSerialLines[portIdx], maxIterations)
            _log.debug(reply)

        return reply


    def _XPOWClearPorts(self):
        self._XPOWClosePorts()

        self.XPOWSerialLines.clear()
        time.sleep(self.XOPWCommandDelay)

        _log.info("XPOW ports cleared!")


    def _XPOWOpenPorts(self):
        for i in range(len(self.XPOWSerialLines)):
            self.XPOWSerialLines[i].open()
            time.sleep(self.XOPWCommandDelay)

        _log.info("XPOW ports opened!")


    def _XPOWClosePorts(self):
        for i in range(len(self.XPOWSerialLines)):
            self.XPOWSerialLines[i].close()
            time.sleep(self.XOPWCommandDelay)

        _log.info("XPOW ports closed!")


    def _XPOWCreatePorts(self):
        for i in range(len(self.XPOWPorts)):
            self.XPOWSerialLines.append(serial.Serial(self.XPOWPorts[i], baudrate =self.XPOWBaudRate, timeout = 3.0, writeTimeout = 0))
            time.sleep(self.XOPWCommandDelay)

        _log.info("XPOW ports created!")

    def _XPOWResetPorts(self):
        self._XPOWClosePorts()
        self._XPOWOpenPorts()

    def _XPOWResetAllChannels(self, selected_channels = None):
        channel = 1

        if selected_channels is None:
            selected_channels = np.arange(1, 121)

        while (channel <= 120):
            modChannel = channel
            
            portIdx = 0

            if (channel >= 41) and (channel <= 80):
                modChannel = channel - 40
                portIdx = 1
            elif (channel >= 81) and (channel <= 120):
                modChannel = channel - 80
                portIdx = 2

            if channel not in selected_channels:
                channel += 1
                continue

            self._XPOWSendCommandSingle("CH:" + str(modChannel) + ":VOLT:0", portIdx, self.XPOWCommandTimeout)
            time.sleep(self.XPOWVoltageDelay)
            self._XPOWSendCommandSingle("CH:" + str(modChannel) + ":CUR:" + str(self.XPOWCurrentMax), portIdx, self.XPOWCommandTimeout)

            channel += 1

    def _XPOWOpen(self):
        _log.debug("------------------------------------------\nChosen XPOW Ports: " + str(self.XPOWPorts) + "\n------------------------------------------\n")

        error = True

        while (error == True):
            try:
                self._XPOWCreatePorts()
                self._XPOWSendCommandAll("board?", self.XPOWCommandTimeout)
                # XPOWResetAllChannels()

                error = False

            except:
                _log.error("\n-- XPOW ERROR ENCOUNTERED, TRYING AGAIN... --")

                time.sleep(self.XPOWErrorDelay)

    def _XPOWResetAndOpen(self):
        _log.debug("------------------------------------------\nChosen XPOW Ports: " + str(self.XPOWPorts) + "\n------------------------------------------\n")

        error = True

        while (error == True):
            try:
                self._XPOWCreatePorts()
                self._XPOWSendCommandAll("board?", self.XPOWCommandTimeout)
                self._XPOWResetAllChannels()

                error = False

            except:
                _log.error("\n-- XPOW ERROR ENCOUNTERED, TRYING AGAIN... --")

                time.sleep(self.XPOWErrorDelay)

