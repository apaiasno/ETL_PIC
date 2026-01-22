import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import collections
from matplotlib.widgets import Button 

sys.path.append('../src/')

import XPOW
import XenicsCam as XCam
from xenics.xeneth import *
import PIC_lib as PIC
import argparse
import logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(name)-12s: %(levelname)-8s %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)],
                   )

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dark', type = str, help='path to dark file (npy)')
parser.add_argument('-n', '--num_spots', type= int, help = 'number of spots',
                    default = 5)
parser.add_argument('-r', '--radius', help='photometry radius in pixels',
                    default = 8)

args = parser.parse_args()
dark = np.load(args.dark)

_xpow = XPOW.XPOW(reset = False)
_xpow.check = True

class MockXPOW:
    def __init__(self, channel = 63, ref = 7):
        self.xpow = _xpow
        self.voltage = ref
        self.channel = channel

        self.xpow.apply_voltage(self.channel, self.voltage)

    def change_voltage(self, amount):
        
        try:
            self.xpow.apply_voltage(self.channel, self.voltage + amount)
            print(f"HARDWARE: Voltage is now {(self.voltage+amount):.2f} V")
            self.voltage += amount
            
        except:
            print(f"Cannot change voltage!")

xpow = MockXPOW(channel = 42)

# Xenics camera
cam = XCam.XENICSCAM()

print('ready to take images')

_xpow.apply_voltage(42, 0)
_xpow.apply_voltage(63, 8)
_xpow.apply_voltage(64, 8)

_,_,image,_ = cam.take_image(navg=50)
print('took images')

# find spots

spots = PIC.find_spots(image - dark, num_apertures=args.num_spots)
masks = PIC.make_circular_masks(np.shape(dark), spots, args.radius)
print(masks)

# --- Configuration ---
MAX_DATA_POINTS = 600 # Number of data points to display on the plot
DATA_POLL_INTERVAL_S = 0 # How often to read from the device (in seconds)
PLOT_REFRESH_INTERVAL_MS = 200 # How often to refresh the plot (in milliseconds)
NAVG = 1
VERBOSE = False

# data_points = collections.deque(maxlen=MAX_DATA_POINTS)
# ### CHANGED: Create a list of deques, one for each spot
data_queues = [collections.deque(maxlen=MAX_DATA_POINTS) for _ in range(args.num_spots)]
# --- Threading Control ---
# This event will be used to signal the data reading thread to stop
stop_event = threading.Event()

def reader():
    """
    This function runs in a separate thread.
    It continuously reads data from the device and adds it to our deque.
    """
    print("reader thread started.")

    
    # Loop until the main thread signals us to stop

    discard_num = 10
    it = 0

    while not stop_event.is_set():
        # it += 1
        # _, _, im, _ = cam.take_image(navg=NAVG, stack=True)
        # value = PIC.phot(im - dark, masks)
        
        # if VERBOSE: print(value)
        # if it > discard_num:
        #     data_points.append((value[2]))
        #     time.sleep(DATA_POLL_INTERVAL_S)
        it += 1
        _, _, im, _ = cam.take_image(navg=NAVG, stack=True)
        
        # Calculate photometry for all spots
        # Assuming values is a list/array of length 'num_spots'
        values = PIC.phot(im - dark, masks)
        
        if VERBOSE: print(values)

        if it > discard_num:
            # ### CHANGED: Append each spot's value to its corresponding deque
            for i in range(args.num_spots):
                # Ensure we don't crash if PIC.phot returns fewer values than expected
                if i < len(values):
                    data_queues[i].append(values[i])
            
            time.sleep(DATA_POLL_INTERVAL_S)
    print("reader thread finished.")

# --- Plotting Functions ---

# Create the figure and axes for the plot
fig, ax = plt.subplots(figsize=(10, 5))

lines = []
colors = ['C%d' % i for i in range(args.num_spots)] #plt.cm.jet(np.linspace(0, 1, args.num_spots)) # generate distinct colors

for i in range(args.num_spots):
    # Initialize empty lines with labels
    ln, = ax.plot([], [], label=f'Spot {i+1}', color=colors[i])
    lines.append(ln)

ax.legend(loc='upper right')
# line, = ax.plot(range(len(data_points)), np.zeros(len(data_points))) # Start with an empty line
# line, = ax.plot([], []) # Start with an empty line

def setup_plot():
    """Sets the initial properties of the plot."""
    # ax.set_ylim(0, 1e-7) # Set Y-axis limits (adjust as needed)

    ax.set_xlim(0, MAX_DATA_POINTS) # X-axis will show the last N points
    # ax.set_ylabel("Power (W)")
    ax.set_xlabel("Time (recent ->)")
    # ax.grid(True)


# Status Text
status_text = ax.text(0.02, 0.95, f'Voltage: {xpow.voltage:.2f} V', 
                      transform=ax.transAxes, fontsize=12, 
                      bbox=dict(facecolor='white', alpha=0.8))

# ==========================================
# BUTTON CONFIGURATION
# ==========================================

# 1. Define where the buttons go [left, bottom, width, height]
ax_down = plt.axes([0.3, 0.05, 0.15, 0.075])
ax_up   = plt.axes([0.55, 0.05, 0.15, 0.075])

# 2. Create the Button objects (KEEP REFERENCES to these variables!)
btn_down = Button(ax_down, 'Decrease (-)', color='lightblue', hovercolor='0.975')
btn_up   = Button(ax_up,   'Increase (+)', color='lightgreen', hovercolor='0.975')

# 3. Define what happens when clicked

increment = 0.5
def increase(event):
    xpow.change_voltage(increment)
    status_text.set_text(f'Voltage: {xpow.voltage:.2f} V')
    # Use draw_idle to refresh the text without freezing the animation
    fig.canvas.draw_idle() 

def decrease(event):
    xpow.change_voltage(-increment)
    status_text.set_text(f'Voltage: {xpow.voltage:.2f} V')
    fig.canvas.draw_idle()

# 4. Connect the clicks
btn_up.on_clicked(increase)
btn_down.on_clicked(decrease)
# ==========================================

def update_plot(frame):
    """This function is called by the animation to update the plot."""
    # # Set the plot data to the current contents of our deque
    # line.set_data(range(len(data_points)), data_points)
    global_min = np.inf
    global_max = -np.inf
    has_data = False
    # # --- NEW: Auto-adjust Y-axis ---
    # if data_points: # Check if the deque is not empty
    #     # min_val = min(data_points)
    #     max_val = max(data_points)
    #     min_val = 0

    #     ax.set_ylim(min_val, max_val)

    # return line,

    # ### CHANGED: Loop through every spot and update its specific line
    for i, line in enumerate(lines):
        # Get the data for this spot
        y_data = list(data_queues[i])
        x_data = range(len(y_data))
        
        line.set_data(x_data, y_data)
        
        # Track min/max for auto-scaling
        if y_data:
            has_data = True
            global_min = min(global_min, min(y_data))
            global_max = max(global_max, max(y_data))

    # --- Auto-adjust Y-axis based on ALL lines ---
    if has_data and global_max > -np.inf:
        # Add a 10% buffer for nicer visualization
        buffer = (global_max - global_min) * 0.1
        if buffer == 0: buffer = 1.0 # Prevent crash if line is flat
        
        ax.set_ylim(global_min - buffer, global_max + buffer)
    
    return lines +  [status_text]

if __name__ == "__main__":
    # try:
    #     scripts.connect_PM()
    # except:
    #     print("cannot connect to power meter")
    #     sys.exit(1)


    # 1. Set up the plot aesthetics
    setup_plot()
    
    # 2. Create and start the background thread for reading data
    print("Starting data acquisition thread...")
    reader_thread = threading.Thread(target=reader)
    reader_thread.daemon = True # Allows main program to exit even if thread is running
    reader_thread.start()
    
    # 3. Set up the plot animation
    # FuncAnimation will repeatedly call the 'update_plot' function
    ani = FuncAnimation(fig, 
                                  update_plot, 
                                  interval=PLOT_REFRESH_INTERVAL_MS, 
                                  )#blit=True) # blit=True improves performance
    
    try:
        # 4. Show the plot. This is a blocking call and will run until the window is closed.
        plt.show()
        
        # This code runs after the plot window is closed manually
        print("Plot window closed.")
        
    except KeyboardInterrupt:

        print("\nKeyboardInterrupt detected. Shutting down...")
        
    finally:
        # This block runs regardless of how the try block was exited
        # (manual close or Ctrl+C)
        
        # 5. Signal the data reader thread to stop
        stop_event.set()
        
        # 6. Wait for the thread to finish its current task
        reader_thread.join()
        
        # 7. Perform the final device shutdown
        # scripts.disconnect_PM() 
        
        print("resetting XPOW")
        _xpow.close(reset = True)

        print("Script terminated.")
        sys.exit(0)