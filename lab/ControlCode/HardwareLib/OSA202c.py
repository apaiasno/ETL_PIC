# script from "D:\Greg's DSF Attempt\OSA202c.py"
#### IMPORTS ####
import sys
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from numpy.polynomial import Polynomial as poly
import clr
import glob
import serial
import matplotlib.pyplot as plt
import time
import warnings

warnings.filterwarnings('ignore')

clr.AddReference("C:\\Program Files\\Thorlabs\\ThorSpectra\\ThorlabsOSAWrapper.dll")
clr.AddReference("System.Collections")

from ThorlabsOSAWrapper import *
from ThorlabsOSAWrapper import InstrumentModel
from ThorlabsOSAWrapper import AcquisitionUpdateFlag
from ThorlabsOSAWrapper import SpectrumStruct
from ThorlabsOSAWrapper import FileIOInterface
from ThorlabsOSAWrapper import XUnit
from ThorlabsOSAWrapper import YUnit
from System.Collections.Generic import List

#### GLOBAL VARIABLES ####

OSASensitivity = 0
OSAResoltion = 0
OSASpectrumLength = 22021
OSANoiseLevel = -90
OSABaseUnits = True
OSADiscardedSpectrumNum = 3
OSASpectrumCheckDelay = 1        # (sec) Interval between checks for an acquired OSA spectrum
OSASpectrumTimeout = 10


#### DO NOT EDIT ####

OSA = None
OSALocator = None
OSAFileInterface = FileIOInterface()
OSACurrSpectrum = None
OSASaveSpectrum = True
OSAWaitingSpectrum = False
StartTime = 0
#### OSA FUNCTIONS ####

def OnSingleAcquisition(sender, event_args):
    global OSACurrSpectrum
    global OSAWaitingSpectrum

    if (event_args.CallbackMessage.LastDataTypeUpdateFlag == AcquisitionUpdateFlag.Spectrum):
        if (OSASaveSpectrum == True):
            delay = round(time.time() - StartTime, 2)

            OSACurrSpectrum = SpectrumStruct(sender.ChannelInterfaces[0].GetLastSpectrumLength(), True)
            
            OSACurrSpectrum.XAxisUnit = int(XUnit.nmAir)
            OSACurrSpectrum.YAxisUnit = int(YUnit.dB)
            
            sender.ChannelInterfaces[0].GetLastSpectrum(OSACurrSpectrum)

            print("Spectrum Acquired! (" + str(delay) + " sec)")

        OSAWaitingSpectrum = False

def OSASaveSpectraFile(spectra, filePath, fileFormat):
    global OSAFileInterface

    fileExt = None

    if (fileFormat == 1):
        fileExt = ".txt"
    elif (fileFormat == 2):
        fileExt = ".spf2"
    else:
        raise ValueError("ERROR: Unknown OSA spectra format supplied!")

    OSAFileInterface.WriteSpectra(spectra, filePath + fileExt, fileFormat)

def OSASaveSpectrumFile(spectrum, filePath, fileFormat):
    global OSAFileInterface

    fileExt = None

    if (fileFormat == 1):
        fileExt = ".txt"
    elif (fileFormat == 2):
        fileExt = ".spf2"
    else:
        raise ValueError("ERROR: Unknown OSA spectrum format supplied!")

    OSAFileInterface.WriteSpectrum(spectrum, filePath + fileExt, fileFormat)

def OSAGetSpectralData(spectrum, baseUnits):
    dataX = np.empty(shape = OSASpectrumLength, dtype = float)
    dataY = np.empty(shape = OSASpectrumLength, dtype = float)

    for i in range(OSASpectrumLength):
        if (baseUnits == True):
            dataX[(OSASpectrumLength - 1) - i] = 10000000 / spectrum.GetXAtIndex(i)
            dataY[(OSASpectrumLength - 1) - i] = 10 * np.log10(spectrum.GetValueAtIndex(i))
        else:
            dataX[i] = spectrum.GetXAtIndex(i)
            dataY[i] = spectrum.GetValueAtIndex(i)

    return [dataX, dataY]

def OSAGetSpectrum(discardedNum):
    global OSA
    global OSACurrSpectrum
    global StartTime
    global OSAWaitingSpectrum

    for i in range(discardedNum):
        OSAGetSpectrum(0)

    OSAWaitingSpectrum = True
    elapsedTime = OSASpectrumTimeout

    while (OSAWaitingSpectrum == True):
        if (elapsedTime >= OSASpectrumTimeout):
            OSA.AcquireSingleSpectrum()

            StartTime = time.time()
            elapsedTime = 0

        time.sleep(OSASpectrumCheckDelay)
        elapsedTime += OSASpectrumCheckDelay

    return OSACurrSpectrum

def OSAGetMultipleSpectrum(discardedNum, effectiveNum = 1):
    global OSA
    global OSACurrSpectrum
    global StartTime
    global OSAWaitingSpectrum

    for i in range(discardedNum):
        OSAGetSpectrum(0)

    return_list = []
    print(effectiveNum)
    for i in range(effectiveNum):
        OSAWaitingSpectrum = True
        elapsedTime = OSASpectrumTimeout

        while (OSAWaitingSpectrum == True):
            if (elapsedTime >= OSASpectrumTimeout):
                OSA.AcquireSingleSpectrum()

                StartTime = time.time()
                elapsedTime = 0

            time.sleep(OSASpectrumCheckDelay)
            elapsedTime += OSASpectrumCheckDelay
        return_list.append(OSACurrSpectrum)

    return return_list

def OSAOpen():
    global OSA
    global OSALocator

    OSALocator = DeviceLocator()
    osaNum = OSALocator.InitializeSpectrometers()
    
    print("OSAs found: " + str(osaNum))
    
    if (osaNum > 0):
        print("Opening first OSA...")

        OSA = LibDeviceInterface(0)

    else:
        raise ValueError("ERROR: No OSA found!")

    OSA.SetSensitivityMode(OSASensitivity) 
    OSA.SetResolutionMode(OSAResoltion) 
    OSA.AcquisitionSettings.AutomaticGain = True
    OSA.OnSingleAcquisition += OnSingleAcquisition

def OSAClose():
    global OSA

    OSA.CloseSpectrometer


    """ Lists serial port names

        :raises EnvironmentError:
            On unsupported or unknown platforms
        :returns:
            A list of the serial ports available on the system
    """
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result