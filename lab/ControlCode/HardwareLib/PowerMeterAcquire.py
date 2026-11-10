import os
import socket
from TLPM import TLPM
import threading
import numpy as np
from astropy.io import fits
from ctypes import c_int16, c_uint32, byref, create_string_buffer, c_bool, c_char_p, c_double
import time
from datetime import datetime
import matplotlib.pyplot as plt

#### GLOBAL VARIABLES ####  
runtime = 7200
CurrPath = os.path.dirname(os.path.realpath(__file__)) + "\\"


# MeterName = "P0015560" #"P0022602"#"P0045632"
MeterName = "P0015519" # student lab console
MeterID = "0x8078"  
TLPMDllPath = "C:\\Program Files\\IVI Foundation\\VISA\\Win64\\Bin\\"
MeterWavelength = 1590.0
#PowerReadingsPerSec = 100
PowerReadingsAvgNum = 1
PowerFeedDataNum = 1000

#### DO NOT EDIT ####
SwitchFinished = False
MonitorPause = False
TLPMChannel = None
#### IMPORTS ####

#### SUPPORT FUNCTIONS ####

def SaveDataFITS(datas, formats, labels, headerData, filePath):
    cols  = []
    
    for i in range(len(datas)):
        cols.append(fits.Column(name = labels[i], format = formats[i], array = datas[i]))

    table = fits.BinTableHDU.from_columns(cols)

    hdr = fits.Header()

    for key in headerData:
        hdr[key] = headerData[key]

    primaryHdu = fits.PrimaryHDU(header = hdr)

    hdu = fits.HDUList([primaryHdu, table])
    hdu.writeto(filePath)

def WriteToFile(filePath, data):
    file = None

    if not (os.path.isfile(filePath)):
        file = open(filePath, 'w')
    else:
        file = open(filePath, 'a')

    file.write(data)

    file.close()

#### METER FUNCTIONS ####

def MeterReadPower():
    #time.sleep(1)

    #readNum = PowerReadingsPerSec * PowerReadingsTime
    #readDelay = 1 / PowerReadingsPerSec
    reads = np.empty(shape = PowerReadingsAvgNum, dtype = float)

    #startTime = time.time()

    for i in range(PowerReadingsAvgNum):
        power = c_double()
        TLPMChannel.measPower(byref(power))
        reads[i] = power.value

        #time.sleep(readDelay)
    
    #print(time.time() - startTime)
    #print(readDelay)

    return [np.average(reads), 0]

def MeterGetCalibDate():
    buff = create_string_buffer(1024)
    TLPMChannel.getCalibrationMsg(buff)

    return buff

def MeterInit(name, id):
    global TLPMChannel

    TLPMInit(TLPMDllPath)

    print("Connected devices: ")
    devices = ListAllDevices()
    foundDevice = None

    for device in devices:
        deviceName = str(device.value)
        print("\t\"" + deviceName + "\"")
        
        if (deviceName.find("::" + name + "::") != -1) and (deviceName.find("::" + id + "::") != -1) and (deviceName.find("::INSTR") != -1):
            foundDevice = device

    print(name, id)

    if (foundDevice is None):
        raise ValueError("ERROR: Power meter \"" + name + "\" with ID \"" + id + "\" cannot be found!")

    TLPMChannel.open(foundDevice, c_bool(True), c_bool(True))
    TLPMChannel.setWavelength(c_double(1000.0))
    TLPMChannel.setWavelength(c_double(MeterWavelength))

    wl = c_double()
    TLPMChannel.getWavelength(c_int16(0), byref(wl))
    print("Internal wavelength:", wl.value)


    #TLPMChannel.setAvgTime(c_double(5))
    #TLPMChannel.setAvgCnt(c_int16(500))
    #TLPMChannel.setTimeoutValue(c_uint32(10000))

def ListAllDevices():
    global TLPMChannel

    deviceCount = c_uint32()
    TLPMChannel.findRsrc(byref(deviceCount))
    names = []

    nameBuff = create_string_buffer(1024)

    for i in range(0, deviceCount.value):
        TLPMChannel.getRsrcName(i, nameBuff)
        names.append(nameBuff)

    return names

def TLPMInit(dllPath):
    global TLPMChannel

    TLPMChannel = TLPM(dllPath)

def MonitorSwitch():
    global SwitchFinished
    global MonitorPause
    global SwitchConnected

    timeReadings = []
    stageHReadings = []
    stageVReadings = []
    powerAvgReadings = []
    powerStdReadings = []
    connectedReadings = []
    graphTimes = []
    graphPowers = []  
    startTime = time.time()
    time_start = time.perf_counter()
    while time.perf_counter() < (time_start + runtime):
        if (MonitorPause is False):
        
            power = MeterReadPower()
            
            elapsedTime = time.time() - startTime
            #timesFile.write(str(elapsedTime) + "\n")
            timeReadings.append(elapsedTime)
            WriteToFile(CurrPath + "timeReadings.txt", str(elapsedTime) + "\n")

           
            #powerAvgFile.write(str(power[0]) + "\n")
            #powerStdFile.write(str(power[1]) + "\n")
            WriteToFile(CurrPath + "powerAvgReadings.txt", str(power[0]) + "\n")
            WriteToFile(CurrPath + "powerStdReadings.txt", str(power[1]) + "\n")
            powerAvgReadings.append(power[0])
            powerStdReadings.append(power[1])

            '''WriteToFile(CurrPath + "connectedReadings.txt", str(int(SwitchConnected)) + "\n")
            connectedReadings.append(int(SwitchConnected))'''

            

            if (len(graphTimes) > PowerFeedDataNum):
                del graphTimes[0]
                del graphPowers[0]

            graphTimes.append(time.time() - startTime)
            graphPowers.append(power)

            plt.plot(graphTimes, graphPowers, linewidth = 1, color = "dodgerblue")
            plt.pause(0.01)
            plt.clf()

    SaveDataFITS([np.array(timeReadings), np.array(stageHReadings), np.array(stageVReadings), np.array(powerAvgReadings), np.array(powerStdReadings), np.array(connectedReadings)], 
                 ["D", "D", "D", "D", "D", "D"],
                 ["Time (s)", "H Stage Position", "V Stage Position", "Power (Average) (W)", "Power (St Deviation)", "Connection Status"],
                 {"Operator" : "Svarun", "Date" : "8/3/2023"},
                 CurrPath + "data.fits")
    plt.show()

def StartMeter(meter_name=MeterName, meter_id=MeterID): 
    # Initialize power meter
    print("\n-- INITIALIZING METER --\n")

    print("\nConnecting to power meter...")
    MeterInit(meter_name, meter_id)
    # MonitorSwitch()

def CloseMeter():
    global TLPMChannel

    TLPMChannel.close()
# StartMeter()