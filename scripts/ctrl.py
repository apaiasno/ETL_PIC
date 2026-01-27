import time, datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import collections
import argparse
import logging
from configparser import ConfigParser

sys.path.append('../src/')

import XPOW
import XenicsCam as XCam
from xenics.xeneth import *
import PIC_lib as PIC
import TLX2

# Configure logging

log_filename = datetime.datetime.now().strftime("../log/%Y-%m-%d.log")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=log_filename,
    filemode='a'  
)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

class Bench:

    def __init__(self, laser=True, xpow=True, cam=True, conf_name = '../conf/conf.txt'):
        
        self.tlx = None
        self.cam = None
        self.xpow = None

        self.load_conf(conf_name)

        if laser:
            try:
                logging.info("Opening TLX2")
                self.tlx = TLX2.TLX()
            except Exception as e:
                logging.info(f"Failed to open TLX2: {e}")

        if cam:
            try:
                logging.info("Opening Xenics Camera")
                self.cam = XCam.XENICSCAM()
            except Exception as e:
                logging.info(f"Failed to open Xenics Camera: {e}")

        if xpow:
            try:
                logging.info("Opening XPOW")
                self.xpow = XPOW.XPOW(selected_channels=self.channels.tolist())
            except Exception as e:
                logging.info(f"Failed to open XPOW: {e}")

        logging.info("Setup complete")
    
    def take_image(self, nframes=1, stack=True, subtract_dark = True):
        _, _, frames, _ = self.cam.take_image(navg=nframes, stack=stack)
        if subtract_dark: return frames - self.dark
        else: return frames
    
    
    def load_conf(self, conf_name = '../conf/conf.txt'):
        conf = np.genfromtxt(conf_name, delimiter=',', dtype=int)

        self.channels = conf[:,0]
        self.refs = conf[:,1]
        self.current_voltages = np.zeros_like(self.channels)

        logging.info(f"Loaded configuration from {conf_name}")
        logging.info(f"Channels: {self.channels}")
        logging.info(f"References: {self.refs}")


    def get_dark(self, take_new = False, num_frames=100, save = True):

        if not take_new:
            try:
                dark = np.load('dark.npy')
                self.dark = dark
                logging.info("Loaded existing dark frame from dark.npy")
                return dark
            except FileNotFoundError:
                logging.info("No existing dark frame found, taking new one.")
        
        else:

            if self.cam is None:
                logging.warning("Camera not initialized, cannot take dark frame.")
                return None

            logging.info(f"Taking dark frame with {num_frames} frames")

            laser_status = self.tlx.laser_status()

            if laser_status == 'on':
                self.tlx.laser_off()
                time.sleep(1)  

            dark = self.take_image(nframes=num_frames, stack=True, subtract_dark=False)
            self.dark = dark

            if laser_status == 'on':
                self.tlx.laser_on()
                time.sleep(3)

            if save:
                np.save('dark.npy', dark)
                logging.info("Dark frame saved to dark.npy")
            return dark
    
    def apply_v(self, channel, voltage):
        if self.xpow is None:
            logging.warning("XPOW not initialized, cannot apply voltage.")
            return

        try:
            self.xpow.apply_voltage(self.channels[channel], voltage)
            logging.info(f"Applied {voltage} V to channel {channel} (xpow channel {self.channels[channel]})")

            self.current_voltages[channel] = voltage
        except Exception as e:
            logging.error(f"Failed to apply voltage to channel {channel}: {e}")
        return
    
    def shift_v(self, channel, delta_v):
        if self.xpow is None:
            logging.warning("XPOW not initialized, cannot shift voltage.")
            return

        try:
            new_voltage = self.current_voltages[channel] + delta_v
            self.xpow.apply_voltage(self.channels[channel], new_voltage)
            logging.info(f"Shifted voltage by {delta_v} V on channel {channel} (xpow channel {self.channels[channel]}), new voltage: {new_voltage} V")

            self.current_voltages[channel] = new_voltage
        except Exception as e:
            logging.error(f"Failed to shift voltage on channel {channel}: {e}")
        return

    def make_masks(self, num_apertures = 5, radius = 8, save = True):
        for (c, v) in zip(self.channels, self.refs):
            self.apply_v(c, v)
        spots = PIC.find_spots(self.take_image(nframes=50, stack=True), num_apertures=num_apertures)
        self.masks = PIC.make_circular_masks(np.shape(self.dark), spots, radius)
        if save:
            np.save('masks.npy', self.masks)
            logging.info("Masks saved to masks.npy")

        return self.masks
    
    def close(self, reset_all = False):
        if self.xpow is not None:
            if reset_all:
                self.xpow.close()
            else:
                self.xpow.close(selected_channels=self.channels.tolist())
            logging.info("XPOW connection closed.")

        if self.cam is not None:
            self.cam.close()
            logging.info("Camera connection closed.")

        if self.tlx is not None:
            self.tlx.laser_off()
            self.tlx.close()
            logging.info("TLX2 connection closed.")

        logging.info("Bench closed.")

if '__name__' == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-laser', action='store_true', help='Do not initialize laser')
    parser.add_argument('--no-xpow', action='store_true', help='Do not initialize XPOW')
    parser.add_argument('--no-cam', action='store_true', help='Do not initialize camera')
    args = parser.parse_args()

    bench = Bench(laser=not args.no_laser, xpow=not args.no_xpow, cam=not args.no_cam)