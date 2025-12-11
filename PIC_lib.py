# PIC_lib.py
#
# Library of routines for PIC testbed.
# Modified by: Anusha, 12/4/2025 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
import scipy.ndimage

import logging
_log = logging.getLogger('PIC_lib')

import glob, time

from Zaber import UNIT, UNIT_VELOCITY
from XenicsCam import NAVG, T_EXP


MAXVOLTAGE = 13
XPOW_DELAY = 1 # seconds
LASER_ON_DELAY = 10 # seconds
LASER_OFF_DELAY = 2 # seconds
CAM_X, CAM_Y, CAM_FOCUS = 0, 2, 1 # Zaber axes
VOA_MIN, VOA_MAX = 1, 20 # dB

# CAM_Y: camera vertical axis, positive corresponds to camera moving down (spot on image moves up by factor of 10?)

### Hardware control routines for setting up experiments and data collection ###

def find_focus(mask_region, step, tlx, zaber, cam, num_iter=20, step_unit=UNIT, navg=NAVG):
    ''' Iteratively searches for camera focus position by maximizing flux in specified region.

        Parameters
        ----------
        mask_region : 2x2 numpy array
            x and y limits of region in camera image to add up flux for optimizing focus. Format: [[x_min, x_max], [y_min, y_max]]
        step : float
            Initial step size for moving camera focus position with Zaber to search for optimal focus
        tlx : TLX object
            Object for programmatically controlling TLX2 laser as defined in TLX2.py
        zaber: ZABER object
            Object for programmatically controlling Zaber actuators/stages as defined in Zaber.py
        cam : XENICSCAM object
            Object for programmatically controlling Xenics camera as defined in XenicsCam.py
        num_iter : int
            Number of iterations over which to optimize focus. Default: 20. 
        step_unit : Units enum from zaber_motion library
            Unit of step size for moving camera focus position with Zaber. Default: UNIT.
        navg : int
            Number of camera frames to average together when collecting data. Default: NAVG

        Returns:
        --------
        zaber : ZABER object
            Zaber control object after moving to optimize focus position        
    '''

    # Make mask
    tlx.laser_off()
    time.sleep(LASER_OFF_DELAY)
    _, _, dark, _ = cam.take_image(navg=navg)
    tlx.laser_on()
    time.sleep(LASER_ON_DELAY)
    x, y, data, _ = cam.take_image(navg=navg)
    x_dim, y_dim = len(x), len(y)

    mask = np.zeros_like(data, dtype=bool)
    if (mask_region[0][1] <= mask_region[0][0]) or (mask_region[1][1] <= mask_region[1][0]):
        raise ValueError(f"Mask region array must define lower limits in first column and upper limits in second column. Received: \n {mask_region}")
    ix = np.abs(x - np.mean(mask_region[0])) < mask_region[0][1] - mask_region[0][0]
    iy = np.abs(y - np.mean(mask_region[1])) < mask_region[1][1] - mask_region[1][0]
    mask[iy[:, None] & ix[None, :]] = True

    # Plot mask
    yellow_cmap = colors.ListedColormap(['yellow'])

    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    cbar_norm = colors.Normalize(vmin=0, vmax=10000)  
    im = ax.pcolormesh(x, y, data - dark, norm=cbar_norm, zorder=0)
    masked_overlay = np.ma.masked_where(~mask, np.ones_like(mask))
    ax.pcolormesh(x, y, masked_overlay, cmap=yellow_cmap, alpha=0.2, zorder=1)
    fig.colorbar(im, ax=ax, fraction=0.05*y_dim/x_dim)
    ax.set_xlabel('x (microns)', fontsize=18)
    ax.set_ylabel('y (microns)', fontsize=18)
    ax.set_aspect('equal')
    plt.title('Masked region')
    plt.show()

    # Find focus
    data_reduce = data[mask].flatten() - dark[mask].flatten()
    metric = np.sum(data_reduce[np.argsort(data_reduce)[-5:]])
    metric_prior, metric_first = np.copy(metric), np.copy(metric)
    device_ind = CAM_FOCUS
    _log.info('Starting metric:', metric)

    for i in range(20):
        zaber.move_relative(device_ind, step)
        time.sleep(0.5)
        _, _, data, _ = cam.take_image(navg=navg)
        data_reduce = data[mask].flatten() - dark[mask].flatten()
        metric = np.sum(data_reduce[np.argsort(data_reduce)[-5:]])

        if metric < metric_prior:
            step = -0.8 * step
        _log.info('Change in metric:', metric - metric_prior)
        metric_prior = np.copy(metric)
        _log.info('\n')

    _log.info("Final metric: ", metric)
    _log.info(f"Final change in metric from start: {metric/metric_first * 100:0.2f}%")
    return zaber

def voltage_scan(channels, ind_null, voltages, outputs_masks, filename_root, cam, xpow, tlx, navg=NAVG, dark_frame=None, dark_threshhold=3, bright_threshhold=10000, delta_VOA=3):
    ''' Performs a voltage scan over multiple active components with the power meter. Saves each image and the hardware settings to a table.

        Parameters
        ----------
        channels : n x 1 numpy array of integers
            List of channels for each active component to set
        ind_null : integer
            Index of channel corresponding to null port
        voltages : List of n sequences of floats
            All sequences of voltages to scan through for each active component. This routine will go through all possible combinations of voltages for 
            each active component based on their corresponding sequences of voltages to cycle through.
        outputs_masks : list of n numpy slice objects
            List of slice objects to specify each output "aperture"
        dark_threshhold : float
            If (bright) null port flux falls below dark_threshhold * dark flux in null port, laser attenuation is reduced. Default: 3
        bright_threshhold : float
            If (bright) any pixel flux falls above bright_threshhold counts, laser attenuation is increased. Default: 10,000
        filename_root : string
            Path and prefix for each file produced by this routine
        cam : XENICSCAM object
            Object for programmatically controlling Xenics camera as defined in XenicsCam.py
        xpow : XPOW object
            Object for programmatically controlling XPOW breadboard using class definition in XPOW.py
        tlx : TLX object
            Object for programmatically controlling TLX2 laser as defined in TLX2.py
        navg : int
            Number of camera frames to average together when collecting data. Default: NAVG
        dark_frame : None or 2D numpy array
            If None, take a dark frame for each voltage setting; else use provided dark frame.
        delta_VOA : float (in dB)
            Amount to change VOA by whenever a dark or bright threshhold is crossed

        Returns
        -------
        dats : list
            Table of data log containing XPOW and laser readout settings 
    '''
    # Construct all voltage combos
    voltagecombos = np.meshgrid(*voltages)
    voltagecombos = np.stack(voltagecombos, axis=-1).reshape(-1, len(voltagecombos))

    assert np.min(voltagecombos) >= 0, "voltage should be non-negative"
    assert np.max(voltagecombos) < MAXVOLTAGE, "exceeds max voltage allowed"

    dats = []
    for voltages in voltagecombos:
        table_row = []
        channel_str = f''

        # Set XPOW
        for i, voltage in enumerate(voltages):
            # Set XPOW
            xpow.apply_voltage(channels[i], voltage)
            time.sleep(XPOW_DELAY)
            # Readout settings for data log table
            actualvoltage, current, component_power = xpow.read_XPOW(channels[i])
            table_row = np.hstack([table_row, voltage, actualvoltage, current, component_power])
            # Build filename extension for image data
            v_str = "{:.2f}".format(round(voltage, 2)).replace('.', 'p')
            channel_str = channel_str + f'_Ch{channels[i]}_V{v_str}'
        
        # Take bright image
        ### Add temperature if we can stream that info ###
        filename = filename_root + f"{channel_str}.csv"
        x, y, data, timestamp = cam.take_image(navg=navg, filename=filename)
        dats.append(np.hstack([1, tlx.voa, table_row, timestamp]))

        # Take dark image
        if dark_frame is None:
            tlx.laser_off()
            time.sleep(LASER_OFF_DELAY)
            filename = filename_root + f"{channel_str}_dark.csv"
            x, y, dark, timestamp = cam.take_image(navg=navg, filename=filename)
            dats.append(np.hstack([0, tlx.voa, table_row, timestamp]))
            tlx.laser_on()
            time.sleep(LASER_ON_DELAY)
        else:
            dark = np.copy(dark_frame)

        # Subtract stripes
        mask_stripes = outputs_masks.sum(axis=0)
        inds = np.argwhere(mask_stripes==True)
        inds_rows = list(set(inds[:,0]))
        ind_min, ind_max = np.min(inds_rows), np.max(inds_rows)
        inds_rows = np.arange(ind_min - 2*(ind_max-ind_min), ind_max + 2*(ind_max-ind_min))
        for ind in inds_rows:
            mask_stripes[ind,:] = True

        data_stripes, dark_stripes = np.copy(data), np.copy(dark)
        data_stripes[mask_stripes] = np.nan
        dark_stripes[mask_stripes] = np.nan

        data = data - np.nanmean(data_stripes, axis=0)
        dark = dark - np.nanmean(dark_stripes, axis=0)

        # Report intensities
        intensities = []
        for mask in outputs_masks:
            intensities.append(np.sum(data[mask] - dark[mask]))
        _log.info("Output intensities: ", intensities)
        _log.info("\n")

        # Dark threshhold condition
        null_bright = np.max(data[outputs_masks[ind_null]] - dark[outputs_masks[ind_null]])
        null_dark = np.std(dark[outputs_masks[ind_null]])
        if null_bright/null_dark < dark_threshhold:
            if tlx.voa - delta_VOA > VOA_MIN:
                tlx.set_voa(tlx.voa - delta_VOA)
            else:
                tlx.set_voa(VOA_MIN)
        else:
            # Bright threshhold condition
            for mask in outputs_masks:
                if np.max(data[mask]) > bright_threshhold:
                    if tlx.voa + delta_VOA < VOA_MAX:
                        tlx.set_voa(tlx.voa + delta_VOA)
                    else:
                        tlx.set_voa(VOA_MAX)
                    break        
        

    # Construct filename extension and table header for data log
    channel_str = f''
    header_str = f'laserStatus,laserVOA,'
    for ch in channels:
        channel_str = channel_str + f'_Ch{ch}'
        header_str = header_str + f'inputVoltage{ch},actualVoltage{ch},current{ch},componentPower{ch},'
    header_str = header_str + 'timestamp'
    output_filename = filename_root + channel_str + '.log'
    x_filename = filename_root + channel_str + '_x.csv'
    y_filename = filename_root + channel_str + '_y.csv'
    np.savetxt(output_filename, dats, header=header_str, delimiter=",")
    np.savetxt(x_filename, x, delimiter=",")
    np.savetxt(y_filename, y, delimiter=",")    
    return np.array(dats)

def optimize_null(initial_voltages, channels, outputs_masks, ind_null, cam, xpow, dark_frame=None):
    def fit_func(voltages):
        
        # Set XPOW
        for i, voltage in enumerate(voltages):
            # Set XPOW
            xpow.apply_voltage(channels[i], voltage)
            time.sleep(XPOW_DELAY)
            # # Readout settings for data log table
            # actualvoltage, current, component_power = xpow.read_XPOW(channels[i])
            # table_row = np.hstack([table_row, voltage, actualvoltage, current, component_power])
            # # Build filename extension for image data
            # v_str = "{:.2f}".format(round(voltage, 2)).replace('.', 'p')
            # channel_str = channel_str + f'_Ch{channels[i]}_V{v_str}'
        
        # Take bright image
        # tlx.laser_on()
        # laser_voa = tlx.voa
        # time.sleep(LASER_ON_DELAY)
        # ### Add temperature if we can stream that info ###
        # filename = filename_root + f"{channel_str}.csv"
        x, y, data, timestamp = cam.take_image(navg=NAVG)
        if dark_frame is None:
            data_reduce = np.copy(data)
        else:
            data_reduce = data - dark_frame

        # Report intensities
        B_intensity = np.sum(data_reduce[outputs_masks[1]])
        null_intensity = np.sum(data_reduce[outputs_masks[2]])
        D_intensity = np.sum(data_reduce[outputs_masks[3]])
        _log.info("voltages:", voltages)
        _log.info("intensities:", B_intensity, null_intensity, D_intensity)
        _log.info("\n")
        return null_intensity/(B_intensity + D_intensity)
    
    from numpy.random import default_rng
    rng = default_rng(seed=4524)
    sig_v = 0.2 # [V]
    n_v = len(initial_voltages)
    initial_simplex = initial_voltages[np.newaxis,:] + sig_v*rng.normal(size=(n_v+1,n_v))

    xatol = 0.05 # [V] should be bigger than resolution of xpow  FIXME  check this value
    bounds = [(0., 12.)]*n_v

    # _log.info(initial_simplex)
    # _log.info(xatol)
    # _log.info(bounds)
    # _log.info(fit_func(initial_voltages))
    from scipy.optimize import minimize
    options = {'initial_simplex': initial_simplex, 
               'xatol':xatol}
    res = minimize(fit_func, None, method='Nelder-Mead', options=options, bounds=bounds)
    null_depth = fit_func(res.x)
    return res, null_depth


def optimize_null_dynamicVOA(initial_voltages, channels, outputs_masks, ind_null, cam, xpow, tlx, dark_frame=None, dark_threshhold=3, bright_threshhold=10000, delta_VOA=3):
    def fit_func(voltages):
        # Set XPOW
        for i, voltage in enumerate(voltages):
            xpow.apply_voltage(channels[i], voltage)
            time.sleep(XPOW_DELAY)
        
        # Take bright image
        x, y, data, timestamp = cam.take_image(navg=NAVG)
        # data_reduce = np.copy(data)# * 10**((tlx.voa-20) / 10)
        if dark_frame is None:
            tlx.laser_off()
            time.sleep(LASER_OFF_DELAY)
            x, y, dark, timestamp = cam.take_image(navg=navg, filename=filename)
            tlx.laser_off()
            time.sleep(LASER_ON_DELAY)
        else:
            dark = np.copy(dark_frame)

        # Subtract stripes
        mask_stripes = outputs_masks.sum(axis=0)
        inds = np.argwhere(mask_stripes==True)
        inds_rows = list(set(inds[:,0]))
        ind_min, ind_max = np.min(inds_rows), np.max(inds_rows)
        inds_rows = np.arange(ind_min - 2*(ind_max-ind_min), ind_max + 2*(ind_max-ind_min))
        for ind in inds_rows:
            mask_stripes[ind,:] = True

        data_stripes, dark_stripes = np.copy(data), np.copy(dark)
        data_stripes[mask_stripes] = np.nan
        dark_stripes[mask_stripes] = np.nan

        data = data - np.nanmean(data_stripes, axis=0)
        dark = dark - np.nanmean(dark_stripes, axis=0)
        
        data_reduce = data - dark

        # Report intensities
        B_intensity = np.sum(data_reduce[outputs_masks[1]])
        null_intensity = np.sum(data_reduce[outputs_masks[2]])
        D_intensity = np.sum(data_reduce[outputs_masks[3]])
        _log.info("voltages:", voltages)
        _log.info("raw intensities:", np.sum(data[outputs_masks[1]]), np.sum(data[outputs_masks[2]]), np.sum(data[outputs_masks[3]]))
        _log.info("intensities:", B_intensity, null_intensity, D_intensity)        

        # Dark threshhold condition
        null_bright = np.max(data[outputs_masks[ind_null]] )
        null_dark = np.std(dark[outputs_masks[ind_null]])
        _log.info("null_bright, null_dark:", null_bright, null_dark)
        if null_bright/null_dark < dark_threshhold:
            tlx.set_voa(tlx.voa - delta_VOA)
        else:
            # Bright threshhold condition
            for mask in outputs_masks:
                if np.max(data[mask]) > bright_threshhold:
                    tlx.set_voa(tlx.voa + delta_VOA)
                    break 
        _log.info("\n")
        return null_intensity/(B_intensity + D_intensity)
    
    from numpy.random import default_rng
    rng = default_rng(seed=4524)
    sig_v = 0.2 # [V]
    n_v = len(initial_voltages)
    initial_simplex = initial_voltages[np.newaxis,:] + sig_v*rng.normal(size=(n_v+1,n_v))

    xatol = 0.05 # [V] should be bigger than resolution of xpow  FIXME  check this value
    bounds = [(0., 12.)]*n_v

    # _log.info(initial_simplex)
    # _log.info(xatol)
    # _log.info(bounds)
    # _log.info(fit_func(initial_voltages))
    from scipy.optimize import minimize
    options = {'initial_simplex': initial_simplex, 
               'xatol':xatol}
    res = minimize(fit_func, None, method='Nelder-Mead', options=options, bounds=bounds)
    null_depth = fit_func(res.x)
    return res, null_depth


def optimize_null_twoAtten(initial_voltages, channels, outputs_masks, ind_null, cam, xpow, tlx, dark_frame=None, null_VOA=1, bright_VOA=20):
    
    all_voltages = []
    null_metrics = []
    def fit_func(voltages):
        # Set XPOW
        for i, voltage in enumerate(voltages):
            xpow.apply_voltage(channels[i], voltage)
            time.sleep(XPOW_DELAY)
        
        # Take bright image
        tlx.set_voa(bright_VOA)
        x, y, data_bright, timestamp = cam.take_image(navg=NAVG)
        # data_reduce = np.copy(data)# * 10**((tlx.voa-20) / 10)
        if dark_frame is None:
            tlx.laser_off()
            time.sleep(LASER_OFF_DELAY)
            x, y, dark, timestamp = cam.take_image(navg=navg, filename=filename)
            tlx.laser_off()
            time.sleep(LASER_ON_DELAY)
        else:
            dark = np.copy(dark_frame)

        # Take null image
        tlx.set_voa(null_VOA)

        x, y, data_null, timestamp = cam.take_image(navg=NAVG)

        # # Subtract stripes
        # mask_stripes = outputs_masks.sum(axis=0)
        # inds = np.argwhere(mask_stripes==True)
        # inds_rows = list(set(inds[:,0]))
        # ind_min, ind_max = np.min(inds_rows), np.max(inds_rows)
        # inds_rows = np.arange(ind_min - 2*(ind_max-ind_min), ind_max + 2*(ind_max-ind_min))
        # for ind in inds_rows:
        #     mask_stripes[ind,:] = True

        # data_stripes, dark_stripes = np.copy(data), np.copy(dark)
        # data_stripes[mask_stripes] = np.nan
        # dark_stripes[mask_stripes] = np.nan

        # data = data - np.nanmean(data_stripes, axis=0)
        # dark = dark - np.nanmean(dark_stripes, axis=0)
        
        data_reduce_bright = data_bright - dark
        data_reduce_null = data_null - dark

        # Report intensities
        B_intensity = np.sum(data_reduce_bright[outputs_masks[1]]) * 10**((bright_VOA-20) / 10)
        null_intensity = np.sum(data_reduce_null[outputs_masks[2]]) * 10**((null_VOA-20) / 10)
        D_intensity = np.sum(data_reduce_bright[outputs_masks[3]]) * 10**((bright_VOA-20) / 10)
        null_value = null_intensity/(B_intensity + D_intensity)
        _log.info("voltages:", voltages)
        # _log.info("raw intensities:", np.sum(data[outputs_masks[1]]), np.sum(data[outputs_masks[2]]), np.sum(data[outputs_masks[3]]))
        _log.info("intensities:", B_intensity, null_intensity, D_intensity)   
        _log.info("null:", 10*np.log10(null_value), " dB")

        # # Dark threshhold condition
        # null_bright = np.max(data[outputs_masks[ind_null]] )
        # null_dark = np.std(dark[outputs_masks[ind_null]])
        # _log.info("null_bright, null_dark:", null_bright, null_dark)
        # if null_bright/null_dark < dark_threshhold:
        #     tlx.set_voa(tlx.voa - delta_VOA)
        # else:
        #     # Bright threshhold condition
        #     for mask in outputs_masks:
        #         if np.max(data[mask]) > bright_threshhold:
        #             tlx.set_voa(tlx.voa + delta_VOA)
        #             break 
        _log.info("\n")
        all_voltages.append(voltages)
        null_metrics.append(null_value)
        return null_value
    
    from numpy.random import default_rng
    rng = default_rng(seed=4524)
    sig_v = 0.1 # [V]
    n_v = len(initial_voltages)
    initial_simplex = initial_voltages[np.newaxis,:] + sig_v*rng.normal(size=(n_v+1,n_v))

    xatol = 0.05 # [V] should be bigger than resolution of xpow  FIXME  check this value
    bounds = [(0., 12.)]*n_v

    # _log.info(initial_simplex)
    # _log.info(xatol)
    # _log.info(bounds)
    # _log.info(fit_func(initial_voltages))
    from scipy.optimize import minimize
    options = {'initial_simplex': initial_simplex, 
               'xatol':xatol}
    res = minimize(fit_func, None, method='Nelder-Mead', options=options, bounds=bounds)
    null_depth = fit_func(res.x)
    all_voltages = np.array(all_voltages)
    null_metrics = np.array(null_metrics)
    return res, null_depth, all_voltages, null_metrics

### Data reduction routines ###

def build_output_mask(filename_root, voltages_ref, num_outputs, filename_dark=None, mask_range=2, filt_size=5):
    ''' Builds a collection of boolean masks identifying each output in the reference image.
    
        Parameters
        ----------
        filename_root : str
            Path and beginning identifier of each file in image collection for which the outputs masks are being constructed
        voltages_ref : list of floats
            Voltages of image to use as reference for building a mask
        num_outputs : int
            Number of outputs to identify in reference image
        filename_dark : None or str
            If None, uses voltage-specific filename for reading dark image; else uses provided string as dark filename
        mask_range : int
            Mask width and height ranges from +/- mask_range (in pixels) around center of output. Default: 2
        filt_size : int
            Size of maximum filter (in pixels) for isolating local maxima to identify outputs. Default: 5

        Returns
        -------
        outputs_masks : 3D numpy array
            Collection of 2D boolean masks for each output in reference image    
    '''

    filename_log = glob.glob(filename_root+"*.log")[0]
    channel_str = filename_log.split(filename_root+'_')[-1].split('.log')[0]
    channel_str = channel_str.split('_')
    channels = [int(channel.split('Ch')[-1]) for channel in channel_str]

    v_str = ''
    for i in range(len(channels)):
        v_i = f"{voltages_ref[i]:.2f}".replace('.', 'p')
        v_str += f"*_V{v_i}"
    filename_ref = glob.glob(filename_root + v_str + ".csv")[0]

    # Read in data
    data = np.loadtxt(filename_ref, delimiter=',')
    if filename_dark is None:
        filename_dark = filename_ref.split('.csv')[0] + '_dark.csv'
        dark = np.loadtxt(filename_dark, delimiter=',')
    else:
        dark = np.loadtxt(filename_dark, delimiter=',')
    data_reduce = data - dark
    x = np.loadtxt(glob.glob(filename_root+"*_x.csv")[0], delimiter=',')
    y = np.loadtxt(glob.glob(filename_root+"*_y.csv")[0], delimiter=',')

    # Determine coordinates of outputs
    filtered = scipy.ndimage.maximum_filter(data_reduce, size=filt_size)
    mask = (data_reduce == filtered)
    coords = np.column_stack(np.nonzero(mask))
    coords = coords[np.argsort(data_reduce[coords[:,0], coords[:,1]])][-num_outputs:]
    coords = coords[np.argsort(coords[:, 1])] # important to sort in ABCDE order

    # Make 2D masks of the outputs
    outputs_masks = []
    for coord in coords:
        mask = np.zeros_like(data, dtype=bool)
        mask[coord[0]-mask_range:coord[0]+mask_range+1, coord[1]-mask_range:coord[1]+mask_range+1] = True 
        outputs_masks.append(mask)

    # Plot the reference data with the output mask
    yellow_cmap = ListedColormap(['yellow'])
    vmax = 3000

    fig, ax = plt.subplots(1, 1, figsize=(10,6))
    cbar_norm = colors.Normalize(vmin=0, vmax=vmax)  

    data_reduce = data - dark
    cbar_norm = colors.Normalize(vmin=0, vmax=vmax)
    im = ax.pcolormesh(x, y, data_reduce, norm=cbar_norm, zorder=0)
    for mask in outputs_masks:
        masked_overlay = np.ma.masked_where(~mask, np.ones_like(mask))
        ax.pcolormesh(x, y, masked_overlay, cmap=yellow_cmap, alpha=0.2, zorder=1)
    fig.colorbar(im, ax=ax, fraction=0.05*len(y)/len(x))
    ax.set_xlabel('x (microns)', fontsize=18)
    ax.set_ylabel('y (microns)', fontsize=18)
    ax.set_title('Data - Dark')
    ax.set_aspect('equal')
    plt.show()

    return outputs_masks

def extract_outputs(filename_root, outputs_masks, filename_dark=None):
    ''' Builds a collection of boolean masks identifying each output in the reference image.
    
        Parameters
        ----------
        filename_root : str
            Path and beginning identifier of each file in image collection for which the outputs masks are being constructed
        outputs_masks : 3D numpy array
            Collection of 2D boolean masks for each output in reference image   
        filename_dark : None or str
            If None, uses voltage-specific filename for reading dark image; else uses provided string as dark filename

        Returns
        -------
        voltages : list of floats
            Voltages scanned by each channel
        outputs : list of numpy arrays

    '''
    # Get channels
    filename_log = glob.glob(filename_root+"*.log")[0]
    channel_str = filename_log.split(filename_root+'_')[-1].split('.log')[0]
    channel_str = channel_str.split('_')
    channels = [int(channel.split('Ch')[-1]) for channel in channel_str]

    # Read voltages from data log
    log = np.loadtxt(filename_log, delimiter=',')
    with open(filename_log) as f:
        colnames = f.readline().strip().strip('# ').split(',')
    log = np.core.records.fromarrays(log.T, names=colnames)

    voltages = []
    for channel in channels:
        voltages.append(np.sort(list(set(log['inputVoltage'+str(channel)]))))
    
    # Read outputs for each voltage setting
    voltages_mesh = np.meshgrid(*voltages, indexing="ij")
    coords = np.column_stack([voltage.ravel() for voltage in voltages_mesh])

    outputs = []
    shape = [len(voltage) for voltage in voltages]
    for output_i, mask in enumerate(outputs_masks):
        _log.info(f"Output {output_i}...")
        output = []
        for coord in coords:
            v_str = ''
            for i in range(len(channels)):
                v_i = f"{coord[i]:.2f}".replace('.', 'p')
                v_str += f"*_V{v_i}"
            filename = glob.glob(filename_root + v_str + ".csv")[0]
            data = np.loadtxt(filename, delimiter=',')
            if filename_dark is None:
                filename_dark = filename.split('.csv')[0] + '_dark.csv'
                dark = np.loadtxt(filename_dark, delimiter=',')
            else:
                dark = np.loadtxt(filename_dark, delimiter=',')

            # Subtract stripes
            mask_stripes = np.copy(mask)
            inds = np.argwhere(mask_stripes==True)
            inds_rows = list(set(inds[:,0]))
            ind_min, ind_max = np.min(inds_rows), np.max(inds_rows)
            inds_rows = np.arange(ind_min - 2*(ind_max-ind_min), ind_max + 2*(ind_max-ind_min))
            for ind in inds_rows:
                mask_stripes[ind,:] = True

            data_stripes, dark_stripes = np.copy(data), np.copy(dark)
            data_stripes[mask_stripes] = np.nan
            dark_stripes[mask_stripes] = np.nan

            data = data - np.nanmean(data_stripes, axis=0)
            dark = dark - np.nanmean(dark_stripes, axis=0)
            
            # Determine VOA scaling
            cols = [f"inputVoltage{channel}" for channel in channels]
            row = np.all([log[name] == val for name, val in zip(cols, coord)], axis=0)
            voa = log[row]['laserVOA']

            # Subtract darks from data
            data_reduce = (data - dark) * 10**((voa-20) / 10)
            output.append(np.sum(data_reduce[mask]))

        output = np.array(output)
        output = output.reshape(shape)
        outputs.append(output)
            
    return voltages, outputs
