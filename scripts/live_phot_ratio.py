import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import collections
import argparse
import logging

sys.path.append('../src/')
import XenicsCam as XCam
from xenics.xeneth import *
import PIC_lib as PIC

logging.basicConfig(level=logging.DEBUG,
                    format='%(name)-12s: %(levelname)-8s %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)],
                   )
logging.getLogger('matplotlib').setLevel(logging.WARNING)

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dark', type = str, help='path to dark file (npy)')
parser.add_argument('-m', '--mask', type = str, help= 'path to mask file (npy)')
args = parser.parse_args()

# --- Load Hardware & Config ---
try:
    dark = np.load(args.dark)
    masks = np.load(args.mask)
    num_spots = len(masks)
    print(f"Loaded {num_spots} spots from mask file.")
    
    # Validation: We need at least 4 spots (Indices 0, 1, 2, 3) to do the math requested
    # Port 1=idx0, Port 2=idx1, Port 3=idx2, Port 4=idx3
    if num_spots < 4:
        print("ERROR: You requested ratios involving Port 4, but the mask file has fewer than 4 spots.")
        sys.exit(1)

    cam = XCam.XENICSCAM()
    print('Camera initialized.')

except Exception as e:
    print(f"Initialization Error: {e}")
    sys.exit(1)

# --- Configuration ---
MAX_DATA_POINTS = 600 
DATA_POLL_INTERVAL_S = 0 
PLOT_REFRESH_INTERVAL_MS = 200 
NAVG = 1
VERBOSE = False

# List of deques to store RAW data (we calculate ratios later)
data_queues = [collections.deque(maxlen=MAX_DATA_POINTS) for _ in range(num_spots)]
stop_event = threading.Event()

# --- Reader Thread ---
def reader():
    print("reader thread started.")
    discard_num = 10
    it = 0

    while not stop_event.is_set():
        it += 1
        # Take Image
        _, _, im, _ = cam.take_image(navg=NAVG, stack=True)
        
        # Photometry
        values = PIC.phot(im - dark, masks)
        
        if VERBOSE: print(values)

        if it > discard_num:
            # Append raw values to queues
            for i in range(num_spots):
                if i < len(values):
                    val = float(values[i])
                    # Protect against NaNs
                    if np.isnan(val): val = 0.0
                    data_queues[i].append(val)
            
            time.sleep(DATA_POLL_INTERVAL_S)
    print("reader thread finished.")

# --- Plotting Setup ---

# Create 3 subplots vertically
fig, axs = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
plt.subplots_adjust(hspace=0.3) # Add space between plots

# Initialize lines for each subplot
# We only need 1 line per subplot based on your request
line1, = axs[0].plot([], [], color='blue', label='Port 2 Ratio')
line2, = axs[1].plot([], [], color='green', label='Port 3 Ratio')
line3, = axs[2].plot([], [], color='red', label='Port 4 Ratio')

# Setup Titles and Labels
titles = [
    "Ratio: P2 / (P2 + P3 + P4)",
    "Ratio: P3 / (P2 + P3 + P4)",
    "Ratio: P4 / (P2 + P3 + P4)"
]

for i, ax in enumerate(axs):
    ax.set_xlim(0, MAX_DATA_POINTS)
    ax.set_ylabel("Ratio")
    ax.grid(True, alpha=0.5)
    ax.set_title(titles[i], fontsize=10)
    # Start with reasonable Y limits (0 to 1 since it is a ratio)
    ax.set_ylim(0, 1.0) 

axs[2].set_xlabel("Time (Frames)")

def update_plot(frame):
    # 1. Convert relevant deques to lists (Snapshots)
    # Python indices: Port 1 = 0, Port 2 = 1, Port 3 = 2, Port 4 = 3
    try:
        p2 = list(data_queues[1])
        p3 = list(data_queues[2])
        p4 = list(data_queues[3])
    except IndexError:
        return [line1, line2, line3]

    # 2. Ensure all lists are same length (sync check)
    min_len = min(len(p2), len(p3), len(p4))
    if min_len == 0:
        return [line1, line2, line3]

    # Trim to shortest length
    p2 = np.array(p2[:min_len])
    p3 = np.array(p3[:min_len])
    p4 = np.array(p4[:min_len])

    # 3. Calculate Denominator
    denom = p2 + p3 + p4
    
    # Avoid divide by zero (replace 0 with 1 temporarily to keep math safe)
    # This just ensures we don't crash; the ratio will be 0
    safe_denom = np.where(denom == 0, 1.0, denom)

    # 4. Calculate Ratios
    r1 = p2 / safe_denom
    r2 = p3 / safe_denom
    r3 = p4 / safe_denom

    x_data = range(min_len)

    # 5. Update Lines
    line1.set_data(x_data, r1)
    line2.set_data(x_data, r2)
    line3.set_data(x_data, r3)

    # 6. Auto-Scale Y Axis (Optional but recommended)
    # We loop through the 3 plots and adjust Y limits dynamically
    for ax, data in zip(axs, [r1, r2, r3]):
        if len(data) > 0:
            ymin, ymax = np.min(data), np.max(data)
            # Add small buffer so line isn't touching the edge
            margin = (ymax - ymin) * 0.1 if ymax != ymin else 0.1
            ax.set_ylim(max(0, ymin - margin), min(1.0, ymax + margin))

    return [line1, line2, line3]

if __name__ == "__main__":
    print("Starting data acquisition thread...")
    reader_thread = threading.Thread(target=reader)
    reader_thread.daemon = True
    reader_thread.start()
    
    ani = FuncAnimation(fig, 
                        update_plot, 
                        interval=PLOT_REFRESH_INTERVAL_MS, 
                        blit=False) 
    
    try:
        plt.show()
        print("Plot window closed.")
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt detected.")
    finally:
        stop_event.set()
        reader_thread.join()
        sys.exit(0)