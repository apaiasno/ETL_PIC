import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from multiprocessing import shared_memory
import time
import sys

# --- CONFIG (Must match ctrl.py) ---
SHM_NAME = 'phot.shm'
NUM_SPOTS = 5
HISTORY_LEN = 600

def connect_memory():
    """Connects to the memory block created by the Bench script."""
    try:
        existing_shm = shared_memory.SharedMemory(name=SHM_NAME)
        # Create numpy array wrapper around the shared buffer
        arr = np.ndarray((NUM_SPOTS, HISTORY_LEN), dtype='float64', buffer=existing_shm.buf)
        return existing_shm, arr
    except FileNotFoundError:
        print(f"Error: Could not find Shared Memory '{SHM_NAME}'.")
        print("Make sure 'ctrl.py' is running and 'b.start_reader()' has been called.")
        sys.exit(1)

def run_normalized_plotter():
    print("Connecting to Bench memory...")
    shm, data_arr = connect_memory()
    
    # Setup Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    lines = []
    # Distinct colors
    colors = ['C%d' % i for i in range(NUM_SPOTS)]
    
    for i in range(NUM_SPOTS):
        ln, = ax.plot([], [], label=f'Spot {i+1} (Norm)', color=colors[i], linewidth=1.5)
        lines.append(ln)

    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.subplots_adjust(right=0.85) # Make room for legend
    
    ax.set_xlim(0, HISTORY_LEN)
    ax.set_ylim(0, 1.0) # Normalized data is always between 0 and 1
    ax.grid(True, linestyle='--')
    ax.set_title(f"Normalized Photometry (Port / Sum)")
    ax.set_ylabel("Ratio")
    ax.set_xlabel("Time (Frames)")

    def update(frame):
        # 1. READ RAW DATA (Snapshot)
        raw_data = data_arr.copy()
        
        # 2. CALCULATE SUM
        # Sum along axis 0 (down the column) for every time point
        total_intensity = np.sum(raw_data[1:4], axis=0)
        
        # Avoid divide by zero: replace 0 with 1 temporarily
        # (If sum is 0, the ratio will just be 0/1 = 0)
        total_intensity[total_intensity == 0] = 1.0
        
        # 3. NORMALIZE
        norm_data = raw_data / total_intensity
        
        # 4. PLOT
        global_min, global_max = 1.0, 0.0
        
        for i, line in enumerate(lines):
            y = norm_data[i, :]
            
            # Simple check to avoid plotting empty arrays
            if len(y) > 0:
                line.set_data(range(len(y)), y)
                
                # Track Min/Max for zoom
                # We ignore 0s if they are just artifacts of initialization
                valid_vals = y[y > 0]
                if len(valid_vals) > 0:
                    c_min, c_max = np.min(valid_vals), np.max(valid_vals)
                    if c_min < global_min: global_min = c_min
                    if c_max > global_max: global_max = c_max

        # Auto Scale Y-Axis (Zoom in if the ratios are small, e.g. 0.2)
        if global_max > global_min:
            margin = (global_max - global_min) * 0.1
            if margin == 0: margin = 0.05
            
            # Clamp limits to 0.0 - 1.0 so we don't zoom out too far
            new_min = max(0.0, global_min - margin)
            new_max = min(1.0, global_max + margin)
            
            ax.set_ylim(new_min, new_max)

        return lines

    # Blit=False is safer for dynamic resizing
    ani = FuncAnimation(fig, update, interval=100, blit=False)
    plt.show()
    
    shm.close()

if __name__ == "__main__":
    run_normalized_plotter()