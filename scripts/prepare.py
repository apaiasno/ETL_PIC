import sys
sys.path.append('../src/')
import XenicsCam as XCam

import PIC_lib as PIC
import TLX2
import XPOW

import numpy as np

N_STACK = 100
N_SPOT = 5
RADIUS = 8
SKIP_DARK = False

cam = XCam.XENICSCAM()

if SKIP_DARK:
    dark = np.load('dark.npy')
else:
    tlx = TLX2.TLX()

    tlx.laser_off()
    _,_,dark,_ = cam.take_image(navg=N_STACK, stack=True)
    tlx.laser_on()

    np.save('dark.npy', dark)

xpow = XPOW.XPOW()

xpow.apply_voltage(42, 0)
xpow.apply_voltage(63, 8)
xpow.apply_voltage(64, 8)

_,_,im,_ = cam.take_image(navg=50)

spots = PIC.find_spots(im - dark,num_apertures = N_SPOT)
masks = PIC.make_circular_masks(np.shape(dark), spots, RADIUS)

np.save('masks.npy', masks)

xpow.close()