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
from multiprocessing import shared_memory
import atexit

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
logging.getLogger('asyncio').setLevel(logging.WARNING)

class Bench:

    reader_delay = 0.01
    reader_nframes = 1

    def __init__(self, laser=True, xpow=True, cam=True, conf_name = '../conf/conf.txt'):
        
        self.tlx = None
        self.cam = None
        self.xpow = None

        # --- Data & Threading Attributes ---
        self.stop_event = threading.Event()
        self.data_queues = [] 
        self.masks = None
        self.num_spots = 0
        self.monitor_thread = None
        self.camera_lock = threading.Lock() 
        self.ani = None 
        # -----------------------------------

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
        self.current_voltages = np.zeros(len(self.channels))

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
        for (c, v) in zip(range(len(self.channels)), self.refs):
            self.apply_v(c, v)
        spots = PIC.find_spots(self.take_image(nframes=50, stack=True), num_apertures=num_apertures)
        self.masks = PIC.make_circular_masks(np.shape(self.dark), spots, radius)
        self.num_spots = len(self.masks)
        if save:
            np.save('masks.npy', self.masks)
            logging.info("Masks saved to masks.npy")

        return self.masks
    
    def load_masks(self, filename = 'masks.npy'):
        self.masks = np.load(filename)
        self.num_spots = len(self.masks)
    
    def close(self, reset_all = False):

        self.stop_reader()
        self.cleanup_shm()

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

    def make_queues(self, maxlen = 600):

        self.data_queues = [collections.deque(maxlen=maxlen) for _ in range(self.num_spots)]
        print(f"Masks created for {self.num_spots} spots.")

    # # --- PART 1: The Reader (Background Data Collection) ---
    # def _reader_task(self):
    #     logging.info("Reader thread started")
    #     while not self.stop_event.is_set():
    #         try:
    #             im = self.take_image(nframes=1)
    #             if self.masks is not None:
    #                 values = PIC.phot(im, self.masks)
    #                 logging.info('reading photometry, %.1f' % (values[0]))
    #                 for i, val in enumerate(values):
    #                     # if i < len(self.data_queues):
    #                     self.data_queues[i].append(val)
    #             time.sleep(0.01)
    #         except Exception as e:
    #             logging.error(f"Reader Error: {e}")
    #             time.sleep(0.5)
    #     logging.info("Reader thread finished")

    # def start_reader(self):
    #     """Starts the background thread if it's not already running."""
    #     if self.masks is None:
    #         print("No masks found. Running make_masks() first...")
    #         self.make_masks()

    #     if self.monitor_thread is None or not self.monitor_thread.is_alive():
    #         self.stop_event.clear()
    #         self.monitor_thread = threading.Thread(target=self._reader_task)
    #         self.monitor_thread.daemon = True
    #         self.monitor_thread.start()
    #         print(">> Background Reader Started (filling queues).")
    #     else:
    #         print(">> Reader is already running.")

    # def stop_reader(self):
    #     self.stop_event.set()
    #     if self.monitor_thread:
    #         self.monitor_thread.join()
    #     print(">> Reader Stopped.")

    def init_shared_memory(self, shm_name = 'phot.shm', history_len =600):
        """Creates a memory block in RAM accessible by other scripts."""
        self.history_len = history_len
        self.shm_name = shm_name
        try:
            # Try to create new memory
            # Size = 5 spots * 600 floats * 8 bytes (float64)
            size = self.num_spots * self.history_len * 8 
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=size)
            print(f"Shared Memory '{self.shm_name}' created.")
        except FileExistsError:
            # If it exists (from a previous crash), connect to it
            self.shm = shared_memory.SharedMemory(name=self.shm_name)
            print(f"Connected to existing Shared Memory '{self.shm_name}'.")

        # Create a Numpy array backed by this memory
        self.shm_arr = np.ndarray((self.num_spots, self.history_len), dtype='float64', buffer=self.shm.buf)
        
        # Initialize with zeros if new
        if np.sum(self.shm_arr) == 0:
             self.shm_arr[:] = 0

        # Register cleanup so memory is freed when script exits
        # atexit.register(self.cleanup_shm)

    def cleanup_shm(self):
        if self.shm:
            # 1. Close the connection (Detaches this script from memory)
            try:
                self.shm.close()
            except Exception as e:
                logging.info(e) # Already closed, ignore
            
            # # 2. Unlink (Deletes the memory block from the OS)
            # try:
            #     self.shm.unlink()
            #     print("Shared Memory unlinked successfully.")
            # except FileNotFoundError:
            #     # This is GOOD. It means the memory is already gone.
            #     # Common on Windows or if cleanup ran twice.
            #     pass 
            # except Exception as e:
            #     print(f"Warning during SHM cleanup: {e}")
            
            # Reset variable so we don't try again
            self.shm = None

    # ... (Your Hardware methods: apply_v, take_image, get_dark, make_masks) ...

    def start_reader(self):

        if self.masks is None: self.make_masks()
        
        # 1. RESET THE STOP SIGNAL (Crucial Step)
        self.stop_event.clear()

        # FIX: Assign to self.monitor_thread so we can reference it later
        self.monitor_thread = threading.Thread(target=self._reader_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        print(">> Background Reader Writing to Shared Memory...")
    
    def stop_reader(self):
        """Signals the background thread to stop and waits for it to finish."""
        # Check if thread exists and is actually running
        if self.monitor_thread and self.monitor_thread.is_alive():
            print("Stopping reader thread...")
            
            # 1. Signal the loop to stop
            self.stop_event.set()
            
            # 2. Wait for the thread to finish current loop and exit
            # timeout=2.0 prevents infinite hang if thread is stuck
            self.monitor_thread.join(timeout=2.0)
            
            if self.monitor_thread.is_alive():
                logging.warning("Reader thread did not stop gracefully!")
            else:
                print("Reader thread stopped.")
        else:
            print("Reader was not running.")

    def phot(self, nframes=1):

        im = self.take_image(nframes=nframes)
        values = PIC.phot(im, self.masks)
        return values

    def _reader_loop(self):
        while not self.stop_event.is_set():
            try:
                # 1. Get Data
                # im = self.take_image(nframes=self.reader_nframes)

                values = self.phot(self.reader_nframes) #PIC.phot(im, self.masks) # Returns list of 5 floats
                
                # 2. Write to Shared Memory (Ring Buffer Logic)
                # Shift everything left by 1
                self.shm_arr[:, :-1] = self.shm_arr[:, 1:]
                # Set last column to new values
                self.shm_arr[:, -1] = values

                # logging.info('reading photometry, %.1f' % (values[0]))
                    
                time.sleep(0.01)
            except Exception as e:
                print(e)
                time.sleep(1)
# if '__name__' == '__main__':

import routines as routines
import importlib

parser = argparse.ArgumentParser()
parser.add_argument('--no-laser', action='store_true', help='Do not initialize laser')
parser.add_argument('--no-xpow', action='store_true', help='Do not initialize XPOW')
parser.add_argument('--no-cam', action='store_true', help='Do not initialize camera')
args = parser.parse_args()

b = Bench(laser=not args.no_laser, xpow=not args.no_xpow, cam=not args.no_cam)
b.get_dark()
b.load_masks()
b.init_shared_memory()

