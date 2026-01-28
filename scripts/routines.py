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


def voltage_scan(b, channel, vs, nframes = 10, niter=1,
                 plot = True, normalize = True):

    _voltages = b.current_voltages.copy()
    _nf = b.reader_nframes

    was_running = False
    if b.monitor_thread and b.monitor_thread.is_alive():
        b.stop_reader()
        was_running = True

    # b.reader_nframes = nframes
    # b.start_reader()
    outputs = np.zeros((niter, len(vs), b.num_spots))

    for ni in range(niter):

        for i, v in enumerate(vs):
            b.apply_v(channel, v)
            values = b.phot(nframes = nframes)
            outputs[ni, i]  = values
        
    b.reader_nframes = _nf

    if was_running:
        b.start_reader()
    
    if plot:
        # fig = plt.figure()
        for ni in range(niter):
            for i in range(b.num_spots):
                if not normalize:
                    plt.plot(vs, outputs[ni,:,i], color='C%d' % i, alpha=0.3)
                else:
                    plt.plot(vs, outputs[ni,:,i] / np.sum(outputs[ni,:,1:4],axis=1), color='C%d' % i, alpha=0.3)

        plt.show()
    
    for i, v in enumerate(_voltages):
        b.apply_v(i, v)

    return outputs




def voltage_scan2d(b, channel1, channel2, vs1, vs2, nframes = 10, niter=1,
                 plot = True, normalize = True):

    _voltages = b.current_voltages.copy()
    _nf = b.reader_nframes

    was_running = False
    if b.monitor_thread and b.monitor_thread.is_alive():
        b.stop_reader()
        was_running = True

    # b.reader_nframes = nframes
    # b.start_reader()
    outputs = np.zeros((niter, len(vs1), len(vs2), b.num_spots))

    for ni in range(niter):

        for i1, v1 in enumerate(vs1):
            b.apply_v(channel1, v1)

            for i2, v2 in enumerate(vs2):
                b.apply_v(channel2, v2)

                values = b.phot(nframes = nframes)
                outputs[ni, i1, i2]  = values
        
    b.reader_nframes = _nf

    if was_running:
        b.start_reader()
    
    if plot:
        # Average over iterations for cleaner plotting if niter > 1
        avg_output = np.mean(outputs, axis=0) 
        
        fig, axs = plt.subplots(ncols=b.num_spots, figsize=(3*b.num_spots, 4))
        if b.num_spots == 1: axs = [axs] # Handle single spot case
        
        # Determine Extent for imshow
        # Extent = (left, right, bottom, top)
        # We mapped vs2 to Inner Loop (Columns/X) and vs1 to Outer Loop (Rows/Y)
        # Note: imshow defaults to origin='upper'. We usually want 'lower' for voltage plots.
        extent = (min(vs2), max(vs2), min(vs1), max(vs1))

        for i in range(b.num_spots):
            # Select Data for this spot
            data_slice = avg_output[:, :, i]
            
            if normalize:
                # Normalize by total power in all spots at each pixel
                total_power = np.sum(avg_output[:,:,1:4], axis=2)
                # Avoid divide by zero
                total_power[total_power == 0] = 1 
                data_slice = data_slice / total_power

            # Plot
            # origin='lower' puts index (0,0) at bottom-left
            im = axs[i].imshow(data_slice, 
                               origin='lower',
                               aspect='auto',
                               extent=extent)
            
            axs[i].set_title(f'Spot {i+1}')
            axs[i].set_xlabel(f'Ch{channel2} (V)')
            if i == 0: axs[i].set_ylabel(f'Ch{channel1} (V)')
            
        plt.tight_layout()
        plt.show()
    
    for i, v in enumerate(_voltages):
        b.apply_v(i, v)

    return outputs




def optimize_null(b, ind_null=2,norm_inds=[1,2,3],
                    maxiter = 100,
                    navg = 10):
    
    norm_inds = np.array(norm_inds)

    def fit_func(voltages):
        
        for i, v in enumerate(voltages):
            b.apply_v(i, v)

        outs = b.phot(nframes = navg)

        return outs[ind_null] / np.sum(outs[norm_inds])
    
    from numpy.random import default_rng
    rng = default_rng(seed=4524)
    sig_v = 0.2 # [V]
    n_v = len(b.current_voltages)
    initial_voltages = b.current_voltages.copy()
    initial_simplex = initial_voltages[np.newaxis,:] + sig_v*rng.normal(size=(n_v+1,n_v))

    xatol = 0.05 # [V] should be bigger than resolution of xpow  FIXME  check this value
    bounds = [(0., 16.), (4., 15.), (4., 15.)]

    # _log.info(initial_simplex)
    # _log.info(xatol)
    # _log.info(bounds)
    # _log.info(fit_func(initial_voltages))
    from scipy.optimize import minimize
    options = {'initial_simplex': initial_simplex, 
               'xatol':xatol,
               'maxiter': maxiter}
    res = minimize(fit_func, None, method='Nelder-Mead', options=options, bounds=bounds)
    null_depth = fit_func(res.x)
    return res, null_depth
