# XenicsCam.py
#
# Library for defining XenicsCam class and routines for communicating with the Xenics camera. Based on Greg Sercel's script.
# Modified by: Anusha, 12/3/2025 

import sys
import numpy as np
import time
import struct

import logging
_log = logging.getLogger('XenicsCam')

from xenics.xeneth import *
from xenics.xeneth import discovery, XDeviceStates
from xenics.xeneth.errors import XenethAPIException
from xenics.xeneth.xcamera import XCamera
from xenics.xeneth.capi.enums import XFrameType

# cam = XCamera()
# url = "cam://0"
# if len(sys.argv) > 1:
#     url = sys.argv[1]

### Camera functions ###
IDX = 0
TEMP_LIM = 225
PIXEL_SIZE = 30 # microns
NAVG = 50 # number of frames
T_EXP = 100 # microseconds

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
        self.fb = self.cam.create_buffer(XFrameType.FT_NATIVE)
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
                _log.error("No devices found")
                return

            states = {XDeviceStates.XDS_Available : "Available",
                    XDeviceStates.XDS_Busy : "Busy",
                    XDeviceStates.XDS_Unreachable : "Unreachable"}

            dev = devices[idx]
            self.url = dev.url
            self.dev = dev

            _log.info(f"Device[{idx}] {dev.name} @ {dev.address} ({dev.transport})")
            _log.info(f"URL: {dev.url}")
            _log.info(f"State: {states[dev.state]} ({dev.state})\n")
            return     
           
        except XenethAPIException as e:
            _log.error(f"Error occurred during device discovery: {e.message}")
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
                _log.info("Checking camera temperature")
                temp = self.cam.get_property_value("Temperature")
                while temp > TEMP_LIM:
                    temp = self.cam.get_property_value("Temperature")
                    _log.info("CCD is cooling down...Please wait a few moments")
                    time.sleep(5)
                _log.info("CCD is cold! Now available to take images!")
                self.cam.start_capture()
                return
            else:
                _log.error("Initialization failed")  
        except XenethAPIException as e:
            _log.error(e.message)   
        return

    # def take_image(self, navg=NAVG, filename=False, stack=True):
    #     ''' Captures an image.
        
    #         Parameters
    #         ----------
    #         navg : int
    #             Number of frames to average together. Default: NAVG.
    #         filename : False or str
    #             Filename to save image; if False, doesn't save image.
    #         stack : boolean
    #             If True, mean-stack all navg frames; else, don't combine frames.

    #         Returns
    #         -------
    #         x, y : 1D numpy arrays
    #             Image x- and y-axis in microns.
    #         data : numpy array
    #             Image data averaged over all frames if stack == True; else, collection of all frames.
    #         timestamp : float or numpy array
    #             Average timestamp (seconds since UNIX epoch) across all frames if stack == True; else, collection 
    #             of timestamps for each frame.
    #     '''
    #     data = []
    #     timestamp = []
    #     for i in range(navg):
    #         buffer = self.cam.create_buffer()
    #         if self.cam.get_frame(buffer, flags=XGetFrameFlags.XGF_Blocking):
    #             data.append(buffer.image_data)
    #         timestamp.append(time.time())
    #     data, timestamp = np.array(data), np.array(timestamp)
    #     y_dim, x_dim = np.shape(data[0])
    #     x = np.linspace(-x_dim/2 * PIXEL_SIZE, (x_dim/2 - 1) * PIXEL_SIZE, x_dim)
    #     y = np.linspace(-y_dim/2 * PIXEL_SIZE,(y_dim/2 - 1) * PIXEL_SIZE, y_dim)

    #     if stack:
    #         data = np.nanmean(data, axis=0)
    #         timestamp = np.mean(timestamp)
                
    #     if filename:
    #         _log.debug(filename)
    #         if stack:
    #             np.savetxt(filename, data, delimiter=",")
    #         else:
    #             np.save(filename, data)
    #     return x, y, data, timestamp
    
    def take_image(self, navg=NAVG, timeout=None, filename=False, stack=True):
        """ Acquire images, returning frames and timestamps.

        Parameters
        ----------
        cam : XCamera
            Opened Xenics camera
        navg : int
            Number of frames to average
        timeout : float or None
            Optional timeout in seconds
        filename : False or str
                Filename to save image; if False, doesn't save image.
            stack : boolean
                If True, mean-stack all navg frames; else, don't combine frames.

        Returns
        -------
        x, y : 1D numpy arrays
            Image x- and y-axis in mm.
        frames : numpy array
            Image data averaged over all frames if stack == True; else, collection of all frames.
        timestamps : np.ndarray
            Shape (navg,)
            Camera or host timestamps (see notes)
        """     

        height = self.cam.height
        width = self.cam.width

        # Determine dtype
        bits = self.cam.bitsize
        if bits <= 8:
            dtype = np.uint8
        elif bits <= 16:
            dtype = np.uint16
        else:
            dtype = np.uint32

        frames = np.empty((navg, height, width), dtype=dtype)
        timestamps = np.empty(navg, dtype=np.float64)

        count = 0
        t0 = time.perf_counter()

        while count < navg:
            if timeout is not None and (time.perf_counter() - t0) > timeout:
                raise TimeoutError("Timed out while acquiring frames")

            if not self.cam.get_frame(self.fb):
                continue

            frame_view = np.frombuffer(
                self.fb.data,
                dtype=dtype,
                count=height * width
            ).reshape(height, width)

            frames[count] = frame_view
            timestamps[count] = time.perf_counter()

            count += 1
        
        frames = frames.astype(np.float32)
        y_dim, x_dim = np.shape(frames[0])
        x = np.linspace(-x_dim/2 * PIXEL_SIZE, (x_dim/2 - 1) * PIXEL_SIZE, x_dim)/1000 # in mm
        y = np.linspace(-y_dim/2 * PIXEL_SIZE,(y_dim/2 - 1) * PIXEL_SIZE, y_dim)/1000 # in mm

        if stack:
            frames = np.nanmean(frames, axis=0)
            timestamps = np.mean(timestamps)
                
        if filename:
            _log.debug(filename)
            np.save(filename, frames)
        return x, y, frames, timestamps
    
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
        self.cam.stop_capture()
        t_exp_current = self.cam.get_property_value('IntegrationTime')
        _log.info("Exposure time was " + str(t_exp_current) + " μs")
        self.cam.set_property_value('IntegrationTime', t_exp)
        t_exp_new = self.cam.get_property_value('IntegrationTime')
        _log.info("Exposure time is now " + str(t_exp_new) + " μs")
        self.cam.start_capture()
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
                _log.info("Stopped streaming frames!")
                self.cam.close()
                _log.info("Camera closed!")
            except XenethAPIException as e:
                _log.error(e.message)
        return
    
    @property
    def t_exp(self):
        ''' Property for camera's exposure time (in microseconds).'''
        return self.cam.get_property_value('IntegrationTime')
    
    @property
    def navg(self):
        ''' Default number of frames to average'''
        return NAVG
    

