# plotter.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from multiprocessing import shared_memory
import time
import sys

# MUST MATCH CONFIG IN CTRL.PY
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

def run_plotter():
    print("Connecting to Bench memory...")
    shm, data_arr = connect_memory()
    
    # Setup Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    lines = []
    colors = ['C%d' % i for i in range(NUM_SPOTS)]
    
    for i in range(NUM_SPOTS):
        ln, = ax.plot([], [], label=f'Spot {i+1}', color=colors[i])
        lines.append(ln)

    ax.legend(loc='upper right')
    ax.set_xlim(0, HISTORY_LEN)
    ax.grid(True)
    ax.set_title(f"Live View (Reading from RAM: {SHM_NAME})")

    def update(frame):
        # 1. READ (The array updates automatically in background!)
        # We perform a copy just to be safe during plotting logic
        current_data = data_arr.copy()
        
        global_min, global_max = np.inf, -np.inf
        
        for i, line in enumerate(lines):
            y = current_data[i, :]
            x = range(len(y))
            line.set_data(x, y)
            
            # Min/Max tracking
            c_min, c_max = np.min(y), np.max(y)
            if c_min < global_min: global_min = c_min
            if c_max > global_max: global_max = c_max

        # Auto Scale
        if global_max > -np.inf:
            margin = (global_max - global_min) * 0.1
            if margin == 0: margin = 1.0
            ax.set_ylim(global_min - margin, global_max + margin)

        return lines

    ani = FuncAnimation(fig, update, interval=100, blit=False)
    plt.show()
    
    # Cleanup when plot closes
    shm.close()

if __name__ == "__main__":
    run_plotter()