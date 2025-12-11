# XenicsCam.py
#
# Library for defining XenicsCam class and routines for communicating with the Xenics camera. Based on Greg Sercel's script.
# Modified by: Anusha, 12/3/2025 

import sys
import numpy as np

import time

from xenics.xeneth import *
from xenics.xeneth import discovery, XDeviceStates
from xenics.xeneth.errors import XenethAPIException
from xenics.xeneth.xcamera import XCamera

# cam = XCamera()
# url = "cam://0"
# if len(sys.argv) > 1:
#     url = sys.argv[1]

### Camera functions ###
IDX = 0
TEMP_LIM = 225
PIXEL_SIZE = 30 # microns
NAVG = 5 # number of frames
T_EXP = 350 # microseconds

class XENICSCAM:

    def __init__(self, idx=IDX, t_exp=T_EXP):
        ''' Upon initialization of class instance:
            1. Finds camera.
            2. Opens serial communication with camera.
            3. Sets exposure time.

            Parameters
            ----------
            idx : int
                Camera index, e.g. 0 if there's only one camera.
            t_exp : float
                Exposure time in microseconds. Default: T_EXP.

            Returns
            -------
            None
        '''
        self.find(idx)
        self.open()
        self.set_exposure(t_exp)
        return

    def find(self, idx=IDX):
        ''' Finds camera and stores relevant properties to class instance.
        
            Parameters
            ----------
            idx: int
                Device index. Default: 0.

            Returns
            -------
            None
        '''
        try:
            devices = discovery.enumerate_devices()
            if len(devices) == 0:
                print("No devices found")
                return

            states = {XDeviceStates.XDS_Available : "Available",
                    XDeviceStates.XDS_Busy : "Busy",
                    XDeviceStates.XDS_Unreachable : "Unreachable"}

            dev = devices[idx]
            self.url = dev.url
            self.dev = dev

            print(f"Device[{idx}] {dev.name} @ {dev.address} ({dev.transport})")
            print(f"URL: {dev.url}")
            print(f"State: {states[dev.state]} ({dev.state})\n")
            return     
           
        except XenethAPIException as e:
            print(f"Error occurred during device discovery: {e.message}")
        return

    def open(self):
        ''' Opens connection to camera and store relevant properties to class instance.
        
            Parameters
            ----------
            None

            Returns
            -------
            None
        '''
        try:
            self.cam = XCamera()
            self.cam.open(self.url)
            if self.cam.is_initialized:
                print("Checking camera temperature")
                temp = self.cam.get_property_value("Temperature")
                while temp > TEMP_LIM:
                    temp = self.cam.get_property_value("Temperature")
                    print ("CCD is cooling down...Please wait a few moments")
                    time.sleep(5)
                print("CCD is cold! Now available to take images!")
                self.cam.start_capture()
                return
            else:
                print("Initialization failed")  
        except XenethAPIException as e:
            print(e.message)   
        return

    def take_image(self, navg=NAVG, filename=False):
        ''' Captures an image.
        
            Parameters
            ----------
            navg : int
                Number of frames to average together. Default: NAVG.
            filename : False or str
                Filename to save image; if False, doesn't save image.

            Returns
            -------
            x, y : 1D numpy arrays
                Image x- and y-axis in microns.
            data : 2D numpy array
                Image data averaged over all frames.
            timestamp : float
                Average timestamp (seconds since UNIX epoch) across all frames.
        '''
        data = []
        timestamps = []
        for i in range(navg):
            buffer = self.cam.create_buffer()
            if self.cam.get_frame(buffer, flags=XGetFrameFlags.XGF_Blocking):
                data.append(buffer.image_data)
            timestamps.append(time.time())
        data = np.nanmean(data, axis=0)
        timestamp = np.mean(timestamps)
        
        y_dim, x_dim = np.shape(data)
        x = np.linspace(-x_dim/2 * PIXEL_SIZE, (x_dim/2 - 1) * PIXEL_SIZE, x_dim)
        y = np.linspace(-y_dim/2 * PIXEL_SIZE,(y_dim/2 - 1) * PIXEL_SIZE, y_dim)
        if filename:
            print(filename)
            np.savetxt(filename, data, delimiter=",")
        return x, y, data, timestamp
    
    def set_exposure(self, t_exp=T_EXP):
        ''' Sets exposure time.
        
            Parameters
            ----------
            t_exp : float
                Exposure time (in microseconds). Default: T_EXP.

            Returns
            -------
            None
        '''
        t_exp_current = self.cam.get_property_value('IntegrationTime')
        print("Exposure time was " + str(t_exp_current) + " μs")
        self.cam.set_property_value('IntegrationTime', t_exp)
        t_exp_new = self.cam.get_property_value('IntegrationTime')
        print("Exposure time is now " + str(t_exp_new) + " μs")
        return
        
    def close(self):
        ''' Closes connection to camera.
        
            Parameters
            ----------
            None

            Returns
            -------
            None
        '''
        if self.cam.is_capturing:
            try:
                self.cam.stop_capture()
                print("Stopped streaming frames!")
                self.cam.close()
                print("Camera closed!")
            except XenethAPIException as e:
                print(e.message)
        return
    
    @property
    def t_exp(self):
        ''' Property for camera's exposure time (in microseconds).'''
        return self.cam.get_property_value('IntegrationTime')
    

