# Bench.py
#
# Library for defining Bench class and routines for operating PIC bench in either photometry or OSA data acquisitions modes.
# Modified by: Anusha, 08/13/2026 

import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import scipy.ndimage
import os, time
import logging
_log = logging.getLogger('Bench')

import TEC
import Zaber

from TEC import COM_PORT as TEC_COM_PORT, BAUDRATE as TEC_BAUDRATE, TIMEOUT as TEC_TIMEOUT

try:
    from Zaber import COM_PORT as ZABER_COM_PORT, UNIT as ZABER_UNIT
    from Zaber import CAM_X, CAM_Y, CAM_FOCUS
    # Use package_with_problem functionality here
except ImportError:
    print("Zaber not available, functionality will be disabled.")
    ZABER_COM_PORT, ZABER_UNIT, CAM_X, CAM_Y, CAM_FOCUS = None, None, None, None, None

from TLX2 import COM_PORT as TLX_COM_PORT, VOA_MAX as TLX_VOA_MAX, VOA_MIN as TLX_VOA_MIN 

from XenicsCam import IDX as CAM_IDX, T_EXP as CAM_T_EXP
try:
    from XenicsCam import NAVG as CAM_NAVG
except:
    print("XenicsCam not available, functionality will be disabled.")
    CAM_NAVG = None

EXTERNALVOA_POWER = ['D:\\Active_PIC\\ControlCode\\CalibrationData\\2026-11-04-Calibration\\ExternalVOA_1\\ExternalVOA_1_voltageScan_power.npy', 
                     'D:\\Active_PIC\\ControlCode\\CalibrationData\\2026-10-27-Calibration\\ExternalVOA_2\\ExternalVOA_2_voltageScan_power.npy'] #'D:\\Active_PIC\\ControlCode\\CalibrationData\\2026-05-15-Calibration\\ExternalVOA\\ExternalVOA_voltageScan_power.npy'
EXTERNALVOA_VOLTAGE = ['D:\\Active_PIC\\ControlCode\\CalibrationData\\2026-11-04-Calibration\\ExternalVOA_1\\ExternalVOA_1_voltageScan_voltage.npy',
                       'D:\\Active_PIC\\ControlCode\\CalibrationData\\2026-10-27-Calibration\\ExternalVOA_2\\ExternalVOA_2_voltageScan_voltage.npy'] #'D:\\Active_PIC\\ControlCode\\CalibrationData\\2026-05-15-Calibration\\ExternalVOA\\ExternalVOA_voltageScan_voltage.npy'

# --- adaptive step control constants ---
ADAPT_MAX_STEP = 0.5                # largest step the grow factor may reach
ADAPT_MIN_STEP_DECREASING = 0.02    # attenuation falling: output fading toward the null
ADAPT_MIN_STEP_INCREASING = 0.10    # floor for the growing step
ADAPT_MIN_STEP_REWIND = 0.002        # step used during a fine rescan
ADAPT_ESCAPE_STEP = 0.20            # kick out of a long run of minimum steps
ADAPT_MAX_MIN_STEP_STREAK = 15
ADAPT_GROW_FACTOR = 1.5
ADAPT_ATTEN_FLOOR = 1.0             # value attenuation_adjust pins at when underexposed
 
MASK_RADIUS = 10                    # half-width in pixels of the box around each output

class BENCH:

    def __init__(self, tlx=True, cam=True, osa=False, xpow=False, xdac=True, keysight=True,
                 tec_com_port=TEC_COM_PORT, tec_baudrate=TEC_BAUDRATE, tec_timeout=TEC_TIMEOUT, 
                 zaber_com_port=ZABER_COM_PORT,
                 tlx_com_port=TLX_COM_PORT, 
                 cam_idx=CAM_IDX, cam_t_exp=CAM_T_EXP):
        ''' Upon initialization of class instance:
            1. Connects to XPOW for voltage control of chip and external VOA
            2. Connects to TEC Arduino controller for temperature readout and control
            3. Connects to Zaber for chip and camera translation

            If tlx = True:
            4. Connects to TLX2 monochromatic laser for on/off, attenuation, and wavelength control

            If cam = True:
            5. Connects to Xenics Camera for photometric data acquisition
            Else if osa = True:
            5. Connects to OSA for spectroscopic data acquisition
            *** NOTE: Both cam and osa cannot be True simultaneously ***

            If xpow = True:
            6. Connects to Nicslab XPOW multichannel voltage/current source/measurement
            Else if xdac = True:
            6. Connects to Nicslab XDAC multichannel voltage/current source/measurement
            *** NOTE: Both xpow and xdac cannot be True simultaneously ***

            If keysight = True:
            7. Connects to Keysight E36312A power supply.

            8. Saves a dark image automatically if using both TLX2 laser and camera.

            Parameters
            ----------
            tlx : boolean
                If True, connect to TLX2 monochromatic laser.
            cam : boolean
                If True, connect to Xenics camera
            osa : boolean
                If True, connect to OSA. Note: either cam or osa must be True, but not both simultaneously.
            xpow : boolean
                If True, connect to XPOW
            xdac : boolean
                If True, connect to XDAC
            keysight : boolean
                If True, connect to Keysight E36312A
            tec_com_port: str
                Name of COM port for communication with TEC.
            tec_baudrate : int
                Baudrate for serial communication with TEC.
            tec_timeout : float
                Time in seconds for serial communication timeout.
            zaber_com_port
                Name of COM port for communication with Zabers.
            tlx_com_port
                Name of COM port for communication with laser
            cam_idx
                Camera index, e.g. 0 if there's only one camera.
            cam_t_exp : float
                Camera exposure time in microseconds.

            Returns
            -------
            None
        '''
        self.open(tlx, cam, osa, xpow, xdac, keysight, tec_com_port, tec_baudrate, tec_timeout, zaber_com_port, tlx_com_port, cam_idx, cam_t_exp)
        self.take_dark()
        return

    def open(self, tlx, cam, osa, xpow, xdac, keysight, tec_com_port, tec_baudrate, tec_timeout, zaber_com_port, tlx_com_port=TLX_COM_PORT, cam_idx=CAM_IDX, cam_t_exp=CAM_T_EXP):
        ''' Opens the following connections:
            1. Connects to XPOW for voltage control of chip and external VOA
            2. Connects to TEC Arduino controller for temperature readout and control
            3. Connects to Zaber for chip and camera translation

            If tlx = True:
            4. Connects to TLX2 monochromatic laser for on/off, attenuation, and wavelength control

            If cam = True:
            5. Connects to Xenics Camera for photometric data acquisition
            Else if osa = True:
            5. Connects to OSA for spectroscopic data acquisition
            *** NOTE: Both cam and osa cannot be True simultaneously ***

            If xpow = True:
            6. Connects to Nicslab XPOW multichannel voltage/current source/measurement
            Else if xdac = True:
            6. Connects to Nicslab XDAC multichannel voltage/current source/measurement
            *** NOTE: Both xpow and xdac cannot be True simultaneously ***

            If keysight = True:
            7. Connects to Keysight E36312A power supply.

            8. Saves a dark image automatically if using both TLX2 laser and camera.
        
            Parameters
            ----------
            tlx : boolean
                If True, connect to TLX2 monochromatic laser.
            cam : boolean
                If True, connect to Xenics camera
            osa : boolean
                If True, connect to OSA. Note: either cam or osa must be True, but not both simultaneously.
            xpow : boolean
                If True, connect to XPOW
            xdac : boolean
                If True, connect to XDAC
            keysight : boolean
                If True, connect to Keysight E36312A
            tec_com_port: str
                Name of COM port for communication with TEC.
            tec_baudrate : int
                Baudrate for serial communication with TEC.
            tec_timeout : float
                Time in seconds for serial communication timeout.
            zaber_com_port
                Name of COM port for communication with Zabers.
            tlx_com_port
                Name of COM port for communication with laser
            cam_idx
                Camera index, e.g. 0 if there's only one camera.
            cam_t_exp : float
                Camera exposure time in microseconds.

            Returns
            -------
            None
        '''
        if not (cam ^ osa):
            raise ValueError(f'Either camera (photometric) or OSA (spectroscopic) data acquisition mode must be selected. Received: cam={cam}, osa={osa}')   

        if not (xpow ^ xdac):
            raise ValueError(f'Either XPOW or XDAC multichannel controller must be selected. Received: xpow={xpow}, xdac={xdac}')        

        if tlx:
            import TLX2
            self.tlx = TLX2.TLX(com_port=tlx_com_port)
        else:
            self.tlx = None

        if cam:
            import XenicsCam as XCam
            self.cam = XCam.XENICSCAM(idx=cam_idx, t_exp=cam_t_exp)
        else:
            self.cam = None

        if osa:
            import OSA
            self.osa = OSA.OSA()  
        else:
            self.osa = None

        if xpow:
            import XPOW
            from XPOW import ATTEN_CHANNEL
            from XPOW import XPOWVoltageControlMax as XCOM_MAXVOLTAGE
            self.xcom = XPOW.XPOW()
            self.xcom.atten_channel = ATTEN_CHANNEL
            self.xcom.xcom_maxvoltage = XCOM_MAXVOLTAGE
        else:
            self.xcom = None

        if xdac:
            import XDAC
            from XDAC import ATTEN_CHANNEL
            from XDAC import XDACVoltageControlMax as XCOM_MAXVOLTAGE
            self.xcom = XDAC.XDAC()
            self.xcom.atten_channel = ATTEN_CHANNEL
            self.xcom.xcom_maxvoltage = XCOM_MAXVOLTAGE
        else:
            self.xcom = None

        if keysight:
            import KeysightE36312A
            from KeysightE36312A import VOLTAGE_MAX as KEYSIGHT_MAXVOLTAGE
            self.keysight = KeysightE36312A.KEYSIGHTE36312A()
            self.keysight_maxvoltage = KEYSIGHT_MAXVOLTAGE
        else:
            self.keysight = None
            self.keysight_maxvoltage = {}

        self.tec = TEC.TEC(com_port=tec_com_port, baudrate=tec_baudrate, timeout=tec_timeout)
        self.zaber = Zaber.ZABER(com_port=zaber_com_port)
        self.voltages = {}
        
        return
    
    ### Set-up hardware control routines ###
    def take_dark(self):
        ''' Save a dark image to BENCH object.
        
            Parameters
            ----------
            None

            Returns
            -------
            None
        '''
        if (self.tlx is not None)*(self.cam is not None):
            self.tlx.laser_off()
            x, y, dark, _ = self.cam.take_image()
            self.tlx.laser_on()
            
            self.dark = dark 
            self.x = x
            self.y = y
            self.x_dim = len(x)
            self.y_dim = len(y)
        elif self.cam is not None:
            logging.warning("Do not forget to save a dark image to bench.dark!")
        return

    def set_exposure(self, t_exp=CAM_T_EXP):
        ''' Sets camera exposure time.
        
            Parameters
            ----------
            t_exp : float
                Exposure time (in microseconds). Default: T_EXP.

            Returns
            -------
            None
        '''
        self.cam.set_exposure(t_exp)
        # Retake dark image
        self.take_dark()
        return

    
    def find_focus(self, mask_region, step, num_iter=20, step_unit=ZABER_UNIT):
        ''' Iteratively searches for camera focus position by maximizing flux in specified region.

            Parameters
            ----------
            mask_region : 2x2 numpy array
                x and y limits of region in camera image to add up flux for optimizing focus. Format: [[x_min, x_max], [y_min, y_max]]
            step : float
                Initial step size for moving camera focus position with Zaber to search for optimal focus
            num_iter : int
                Number of iterations over which to optimize focus. Default: 20. 
            step_unit : Units enum from zaber_motion library
                Unit of step size for moving camera focus position with Zaber. Default: UNIT.

            Returns:
            --------
            None      
        '''

        # Make mask  
        _, _, data, _ = self.cam.take_image()
        
        mask = np.zeros_like(data, dtype=bool)
        if (mask_region[0][1] <= mask_region[0][0]) or (mask_region[1][1] <= mask_region[1][0]):
            raise ValueError(f"Mask region array must define lower limits in first column and upper limits in second column. Received: \n {mask_region}")
        ix = np.abs(self.x - np.mean(mask_region[0])) < (mask_region[0][1] - mask_region[0][0])/2
        iy = np.abs(self.y - np.mean(mask_region[1])) < (mask_region[1][1] - mask_region[1][0])/2
        mask[iy[:, None] & ix[None, :]] = True

        # Plot mask
        yellow_cmap = colors.ListedColormap(['yellow'])

        fig, ax = plt.subplots(1, 1, figsize=(10,6))
        cbar_norm = colors.Normalize(vmin=0, vmax=10000)  
        im = ax.pcolormesh(self.x, self.y, data - self.dark, norm=cbar_norm, zorder=0)
        masked_overlay = np.ma.masked_where(~mask, np.ones_like(mask))
        ax.pcolormesh(x, y, masked_overlay, cmap=yellow_cmap, alpha=0.2, zorder=1)
        fig.colorbar(im, ax=ax, fraction=0.05*self.y_dim/self.x_dim)
        ax.set_xlabel('x (microns)', fontsize=18)
        ax.set_ylabel('y (microns)', fontsize=18)
        ax.set_aspect('equal')
        plt.title('Masked region')
        plt.show()

        # Find focus
        data_reduce = data[mask].flatten() - self.dark[mask].flatten()
        metric = np.sum(data_reduce[np.argsort(data_reduce)[-5:]])
        metric_prior, metric_first = np.copy(metric), np.copy(metric)
        device_ind = CAM_FOCUS
        _log.info('Starting metric:'+str(metric))

        for i in range(num_iter):
            self.zaber.move_relative(device_ind, step, unit=step_unit)
            _, _, data, _ = self.cam.take_image()
            data_reduce = data[mask].flatten() - self.dark[mask].flatten()
            metric = np.sum(data_reduce[np.argsort(data_reduce)[-5:]])

            if metric < metric_prior:
                step = -0.8 * step
            _log.info('Change in metric:'+str(float(metric - metric_prior)))
            metric_prior = np.copy(metric)
            _log.info('\n')

        _log.info("Final metric: "+str(metric))
        _log.info(f"Final change in metric from start: {metric/metric_first * 100:0.2f}%")
        return 
    
    def center_frame(self, num_outputs):
        ''' Centers camera on central output along x-axis. (Centering on y-axis isn't worth it because the Zabers won't 
            raise without a slip/stall error beyond a certain point and there's dust on the camera.)

            Parameters
            ----------
            num_outputs : int
                Number of outputs to identify in reference image

            Returns
            -------
            data : 2D numpy array
                Image data with camera centered

        '''
        _, _, data, _ = self.cam.take_image()
        data_reduce = data - self.dark

        # Find central output and shift
        coords = self.__class__.find_output_centers(data_reduce, num_outputs)
        coords_center = (np.max(coords[:, 1]) + np.min(coords[:, 1]))/2
        shift = np.interp(coords_center, np.arange(len(self.x)), self.x) # in mm
        self.zaber.move_relative(CAM_X, shift, camera=True)

        # Data
        _, _, data, _ = self.cam.take_image()
        data_reduce = data - self.dark

        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(24,5))
        norm = colors.Normalize(vmin=0, vmax=10000) 
        im0 = ax.pcolormesh(self.x, self.y, data_reduce, norm=norm, cmap='RdBu_r')
        fig.colorbar(im0, ax=ax, fraction=0.05*self.y_dim/self.x_dim)
        ax.set_xlabel('x (microns)', fontsize=18)
        ax.set_ylabel('y (microns)', fontsize=18)
        ax.set_aspect('equal')
        plt.show()
        return data
    

    ### Hardware calibrated properties ###
    @staticmethod
    def _as_list(x):
        ''' Wraps a single path/channel in a list; passes lists/tuples through. '''
        if isinstance(x, (str, os.PathLike)) or np.isscalar(x):
            return [x]
        return list(x)

    @classmethod
    def _externalVOA_calibration(cls, voa_index=0):
        ''' Loads calibration for one external VOA.

            Returns
            -------
            external_voltage : np.ndarray
                Calibration voltages (V)
            atten_measured : np.ndarray
                Attenuation (positive dB) at each calibration voltage, relative to the first point
        '''
        power_paths = cls._as_list(EXTERNALVOA_POWER)
        voltage_paths = cls._as_list(EXTERNALVOA_VOLTAGE)
        if len(voltage_paths) == 1:  # one shared voltage sweep for all VOAs
            voltage_paths = voltage_paths * len(power_paths)
        if len(voltage_paths) != len(power_paths):
            raise ValueError(f'{len(power_paths)} power calibration files but {len(voltage_paths)} voltage files.')

        data = np.load(power_paths[voa_index])[:, 1:, :]
        external_voltage = np.load(voltage_paths[voa_index])[1:]
        ratio = np.abs(np.mean(data[:, :, 0] / data[:, 0, 0][:, None], axis=0))
        atten_measured = np.abs(10 * np.log10(ratio))
        return external_voltage, atten_measured

    @classmethod
    def _n_externalVOAs(cls):
        return len(cls._as_list(EXTERNALVOA_POWER))

    def _externalVOA_max_atten(self):
        ''' Max attenuation (dB) of each external VOA, one per calibration file. '''
        return [np.max(self._externalVOA_calibration(i)[1]) for i in range(self._n_externalVOAs())]

    def _atten_channels(self):
        ''' xcom attenuation channels, one per VOA, in calibration-file order. '''
        channels = self._as_list(self.xcom.atten_channel)
        if len(channels) != self._n_externalVOAs():
            raise ValueError(f'{self._n_externalVOAs()} calibration files but {len(channels)} xcom attenuation channels.')
        return channels

    @classmethod
    def externalVOA_voltage_to_dB(cls, voltage, voa_index=0):
        ''' Converts voltage applied to one external VOA to dB of attenuation.

            Parameters
            ----------
            voltage : float
                Voltage in V applied to the external VOA
            voa_index : int
                Index of the VOA (matches its position in EXTERNALVOA_POWER). Default: 0

            Returns
            -------
            dB : float
                Decibels of attenuation applied by that VOA
        '''
        external_voltage, atten_measured = cls._externalVOA_calibration(voa_index)
        return np.interp(voltage, external_voltage, atten_measured)

    @classmethod
    def externalVOA_dB_to_voltage(cls, dB, voa_index=0):
        ''' Converts dB of attenuation to the voltage needed on one external VOA.

            Parameters
            ----------
            dB : float
                Decibels of attenuation to be applied by that VOA
            voa_index : int
                Index of the VOA (matches its position in EXTERNALVOA_POWER). Default: 0

            Returns
            -------
            voltage : float
                Voltage in V to apply to the external VOA
        '''
        external_voltage, atten_measured = cls._externalVOA_calibration(voa_index)
        return np.interp(dB, atten_measured, external_voltage)

    ### Channel routing ###
    #
    # A channel is either a plain int, meaning the XCOM device (XDAC or XPOW),
    # or a ('keysight', n) tuple. So existing calls with bare ints are
    # unchanged, and Keysight channels are opt-in:
    #
    #   channels = [68, 69, ('keysight', 2)]

    def _resolve_channel(self, spec):
        ''' Resolves a channel specification to the instrument that owns it.

            Parameters
            ----------
            spec : int or tuple
                Channel number for the XCOM device, or (device_name, channel)
                where device_name is 'xcom' or 'keysight'

            Returns
            -------
            device_name : str
                'xcom' or 'keysight'
            device : object
                The instrument instance
            channel : int
                Channel number on that instrument
        '''
        if isinstance(spec, (tuple, list)):
            device_name, channel = spec
            device_name = str(device_name).lower()
        else:
            device_name, channel = 'xcom', spec

        if device_name in ('xcom', 'xdac', 'xpow'):
            device_name, device = 'xcom', self.xcom
        elif device_name in ('keysight', 'psu'):
            device_name, device = 'keysight', self.keysight
        else:
            raise ValueError(f"Device must be 'xcom' or 'keysight'. Received: {device_name}")

        if device is None:
            raise ValueError(f"Channel {spec} refers to {device_name}, which is not connected.")

        return device_name, device, int(channel)

    def channel_label(self, spec):
        ''' Short label for a channel, used in filenames and log headers.

            Parameters
            ----------
            spec : int or tuple
                Channel specification

            Returns
            -------
            label : str
                'Ch68' for XCOM channels, 'KCh2' for Keysight channels
        '''
        device_name, _, channel = self._resolve_channel(spec)
        return f'Ch{channel}' if device_name == 'xcom' else f'KCh{channel}'

    def channel_maxvoltage(self, spec):
        ''' Maximum voltage allowed on a channel.

            Parameters
            ----------
            spec : int or tuple
                Channel specification

            Returns
            -------
            maxvoltage : float
                Ceiling in V for that channel
        '''
        device_name, _, channel = self._resolve_channel(spec)
        if device_name == 'xcom':
            return self.xcom.xcom_maxvoltage
        return self.keysight_maxvoltage[channel]

    def read_channel(self, spec):
        ''' Readout voltage, current, and power of a channel on either instrument.

            Keysight readings are converted from A and W to mA and mW so that
            both instruments report the same units, since these values share
            columns in the saved log tables.

            Parameters
            ----------
            spec : int or tuple
                Channel specification

            Returns
            -------
            voltage : float
                Measured voltage in V
            current : float
                Measured current in mA
            power : float
                Measured power in mW
        '''
        device_name, device, channel = self._resolve_channel(spec)
        voltage, current, power = device.read_channel(channel)

        # if device_name == 'keysight':
        #     current, power = current * 1e3, power * 1e3

        return voltage, current, power
    
    ### Experimental hardware control routines ###
    def apply_voltage(self, spec, voltage):
        ''' Applies voltage to a channel on either instrument and logs it.

            Parameters
            ----------
            spec : int or tuple
                Channel specification: an int for the XCOM device, or
                ('keysight', n) for a power supply channel
            voltage : float
                Voltage to assign channel of interest

            Returns
            -------
            None        
        '''
        _, device, channel = self._resolve_channel(spec)
        device.apply_voltage(channel, voltage)
        self.voltages[self.channel_label(spec)] = voltage
        return

    def set_VOA(self, dB):
        ''' Applies attenuation with external VOAs in series by setting voltages
            (and laser if tlx is not None).

            External attenuation is allocated in the order of EXTERNALVOA_POWER: each VOA
            is filled to its maximum before the next is used. Unused VOAs are set to 0 V.

            Parameters
            ----------
            dB : float
                Decibels of attenuation to apply.

            Returns
            -------
            None
        '''
        channels = self._atten_channels()
        voa_maxes = self._externalVOA_max_atten()
        ext_max = float(np.sum(voa_maxes))

        if self.tlx is not None:
            atten_max, atten_min = ext_max + TLX_VOA_MAX, TLX_VOA_MIN
        else:
            atten_max, atten_min = ext_max, 0

        if (dB > atten_max) or (dB < atten_min):
            raise ValueError(f'Attenuation limited between {atten_min:0.3f} dB to {atten_max:0.3f} dB. Received: {dB} dB')

        if self.tlx is not None:
            tlx_dB = min(dB, 20.0)
            self.tlx.set_voa(tlx_dB)
            remaining = dB - tlx_dB
        else:
            remaining = dB

        for i, (channel, voa_max) in enumerate(zip(channels, voa_maxes)):
            alloc = min(remaining, voa_max)
            remaining -= alloc
            voltage = 0 if alloc <= 0 else self.__class__.externalVOA_dB_to_voltage(alloc, voa_index=i)
            self.xcom.apply_voltage(channel, voltage)

        time.sleep(3)
        return

    def get_VOA(self):
        ''' Reads out total attenuation from all external VOAs (and optionally the laser).

            Returns
            -------
            dB : float
                Decibels of attenuation applied by external VOAs (and optionally laser)
        '''
        dB = 0.0
        for i, channel in enumerate(self._atten_channels()):
            voltage = self.xcom.read_channel(channel)[0]
            dB += self.__class__.externalVOA_voltage_to_dB(voltage, voa_index=i)
        if self.tlx is not None:
            dB += self.tlx.voa
        return dB

    def attenuation_adjust(self, coord, atten, dB_step=3, R=5, navg=CAM_NAVG, stack=False, output_name=''):
        ''' Dynamically adjust attenuation to ensure a given output is imaged within the linear regime of the camera.

            Parameters
            ----------
            coord : 1x2 numpy array 
                Approximate coordinate of output port pixel
            atten : float
                Initial attenuation (in dB) setting
            dB_step : float
                Increment (in dB) by which to adjust dB when output is outside linear regime of camera. Default: 3
            R : integer
                Pixel radius for mask around output coordinate to test for linearity. Default: 5
            navg : int
                Number of camera frames to average together when collecting final data image.
            stack : boolean
                If True, stack all camera exposures together in final data image, else don't.
            output_name : string
                Name of output port associated with provided coordinate. Default: ''

            Returns
            -------
            atten : float
                Final attenuation (in dB) adopted for final data image with camera's linear regime
            data : numpy array
                Image data averaged over all frames if stack == True; else, collection of all frames.
            timestamps : np.ndarray
                Shape (navg,)
                Camera or timestamps       
        '''
        self.set_VOA(atten)        
        _, _, data, _ = self.cam.take_image(navg=1, stack=True) 
        mask = np.zeros(np.shape(self.dark), bool)
        r, c =  coord
        r, c = int(r), int(c)
        mask[max(0,r-R):r+R+1, max(0,c-R):c+R+1] = 1

        while np.max(data*mask) > 12000:
            try:
                atten = atten + dB_step
                self.set_VOA(atten)
            except:
                raise ValueError(f'Output {output_name} saturating at max attenuation limit of VOA.')
            _, _, data, _ = self.cam.take_image(navg=1, stack=True)   
        while np.max((data-self.dark)*mask) < 5*np.std(self.dark):
            if atten >= 4.0:
                atten = atten - dB_step
                self.set_VOA(atten)
            else:
                atten = 1.0
                print(f'Output {output_name} underexposed at min attenuation limit of VOA.')
            _, _, data, _ = self.cam.take_image(navg=1, stack=True)
        _, _, data, timestamp = self.cam.take_image(navg=navg, stack=stack)
        return atten, data, timestamp

    ### Data acquisition routines ###
    def _scan_setup(self, channels, voltages, coords, filename_root, mask_radius=MASK_RADIUS):
        ''' Prepares a scan: validates the voltage grid, works out the crop
            region covering every output, and saves the axis and dark arrays.
 
            Parameters
            ----------
            channels : list
                Channel specifications being scanned
            voltages : list of sequences of float
                Voltage values to scan for each channel
            coords : Nx2 numpy array or list of integers
                Approximate coordinates of output port pixels
            filename_root : string
                Path and prefix for each file produced by the scan
            mask_radius : int
                Half-width in pixels of the box around each output. Default: MASK_RADIUS
 
            Returns
            -------
            voltagecombos : 2D numpy array
                Every combination of the requested voltages
            crop : tuple of int
                (r0, r1, c0, c1) inclusive bounds of the region saved per image
        '''
        if len(channels) == 0:
            # No channels on a fixed grid: one empty combination, so the caller's
            # loop body still runs once. np.meshgrid of nothing returns an empty
            # tuple, which np.stack rejects.
            voltagecombos = np.empty((1, 0))
        else:
            voltagecombos = np.meshgrid(*voltages)
            voltagecombos = np.stack(voltagecombos, axis=-1).reshape(-1, len(voltagecombos))

            assert np.min(voltagecombos) >= 0, "voltage should be non-negative"
            for i, spec in enumerate(channels):
                ceiling = self.channel_maxvoltage(spec)
                assert np.max(voltagecombos[:, i]) < ceiling, \
                    f"channel {spec} exceeds its max voltage of {ceiling} V"
 
        # One box mask per output, then a single crop region spanning all of
        # them, so every saved image covers the same part of the detector and
        # the saved x, y and dark arrays describe it.
        masks = []
        for coord in coords:
            mask = np.zeros(np.shape(self.dark), bool)
            r, c = int(coord[0]), int(coord[1])
            R = mask_radius
            mask[max(0, r-R):r+R+1, max(0, c-R):c+R+1] = True
            masks.append(mask)
 
        rows, cols = np.where(np.any(masks, axis=0))
        crop = (rows.min(), rows.max(), cols.min(), cols.max())
        r0, r1, c0, c1 = crop
 
        np.savetxt(filename_root + '_x.csv', self.x[c0:c1+1], delimiter=",")
        np.savetxt(filename_root + '_y.csv', self.y[r0:r1+1], delimiter=",")
        np.savetxt(filename_root + '_dark.csv', self.dark[r0:r1+1, c0:c1+1], delimiter=",")
 
        return voltagecombos, crop
 
    def _set_channels(self, channels, voltages, table_row=None, channel_str=''):
        ''' Sets a group of channels, reads them back, and builds the log row
            and filename extension describing them.
 
            Parameters
            ----------
            channels : list
                Channel specifications to set
            voltages : sequence of float
                Voltage for each channel, in the same order
            table_row : numpy array or None
                Existing log row to extend. If None, starts a new one.
            channel_str : string
                Existing filename extension to extend
 
            Returns
            -------
            table_row : numpy array
                Log row with four values appended per channel
            channel_str : string
                Filename extension with one part appended per channel
        '''
        if table_row is None:
            table_row = []
 
        for i, voltage in enumerate(voltages):
            self.apply_voltage(channels[i], voltage)
            actualvoltage, current, component_power = self.read_channel(channels[i])
            table_row = np.hstack([table_row, voltage, actualvoltage, current, component_power])
            v_str = "{:.3f}".format(round(voltage, 3)).replace('.', 'p')
            channel_str = channel_str + f'_{self.channel_label(channels[i])}_V{v_str}'
 
        return table_row, channel_str
 
    def _measure_and_save(self, coords, attens, filename_root, channel_str, table_row,
                          crop, dats, navg=CAM_NAVG, stack=False, extra=None):
        ''' Images every output at the current settings, saves the cropped
            stacks, and appends a log row for each.
 
            Parameters
            ----------
            coords : Nx2 numpy array or list of integers
                Approximate coordinates of output port pixels
            attens : 1xN numpy array
                Current attenuation per output, updated in place
            filename_root : string
                Path and prefix for each file produced by the scan
            channel_str : string
                Filename extension describing the current settings
            table_row : numpy array
                Per-channel log values for the current settings
            crop : tuple of int
                (r0, r1, c0, c1) region of each frame to save
            dats : list
                Log rows, appended to in place
            navg : int
                Number of camera frames to average together
            stack : boolean
                If True, stack all camera exposures in the saved image
            extra : sequence or None
                Values inserted after the external attenuator readout, for
                example the achieved wavelength in a laser scan
 
            Returns
            -------
            None
        '''
        r0, r1, c0, c1 = crop
 
        for i, coord in enumerate(coords):
            output_name = chr(i + 65)
            atten, data, timestamp = self.attenuation_adjust(
                coord, attens[i], navg=navg, stack=stack, output_name=output_name)
            attens[i] = atten
            atten_str = f"{attens[i]:.3f}".replace('.', 'p')
            filename = filename_root + f"{channel_str}_output{output_name}_{atten_str}"
            np.save(filename, data[:, r0:r1+1, c0:c1+1])
 
            temp = self.tec.read_temperature()
            row = [self.xcom.read_channel(self.xcom.atten_channel)]
            if extra is not None:
                row += list(extra)
            if self.tlx is not None:
                row.append(self.tlx.voa)
            row += [table_row, np.mean(timestamp), temp]
            dats.append(np.hstack(row))
 
        return
 
    def _scan_header(self, channels, extra_cols=()):
        ''' Builds the log filename extension and column header for a scan.
 
            Parameters
            ----------
            channels : list
                Channel specifications in scan order
            extra_cols : sequence of str
                Column names inserted after the external attenuator columns,
                for example ('laserWav',) for a laser scan
 
            Returns
            -------
            channel_str : string
                Filename extension listing the scanned channels
            header_str : string
                Comma separated column names
        '''
        channel_str = ''
        header_str = 'externalAttenActualVoltage,externalAttenCurrent,externalAttenPower,'
        header_str = header_str + ''.join(f'{col},' for col in extra_cols)
 
        if self.tlx is not None:
            header_str = header_str + 'laserVOA,'
 
        for ch in channels:
            label = self.channel_label(ch)
            channel_str = channel_str + f'_{label}'
            header_str = header_str + f'inputVoltage{label},actualVoltage{label},current{label},componentPower{label},'
 
        header_str = header_str + 'timestamp,temperature'
        return channel_str, header_str
 
    ### Scans ###
 
    def voltage_scan(self, channels, voltages, filename_root, coords, attens, navg=CAM_NAVG):
        ''' Performs a voltage scan over multiple active components over a regular grid of voltages. Saves each image and the hardware settings to a table.
 
            Parameters
            ----------
            channels : n x 1 numpy array of integers
                List of channels for each active component to set
            voltages : List of n sequences of floats
                All sequences of voltages to scan through for each active component. This routine will go through all possible combinations of voltages for 
                each active component based on their corresponding sequences of voltages to cycle through.
            filename_root : string
                Path and prefix for each file produced by this routine
            coords : Nx2 numpy array or list of integers
                Approximate coordinates of output port pixels
            attens : 1xN numpy array
                Initial attenuations for each of the outputs
            navg : int
                Number of camera frames to average together when collecting data. 
 
            Returns
            -------
            dats : list
                Table of data log containing XCOM device and laser readout settings 
        '''
        voltagecombos, crop = self._scan_setup(channels, voltages, coords, filename_root)
        dats = []
 
        for voltagecombo in voltagecombos:
            table_row, channel_str = self._set_channels(channels, voltagecombo)
            self._measure_and_save(coords, attens, filename_root, channel_str,
                                   table_row, crop, dats, navg=navg)
 
        channel_str, header_str = self._scan_header(channels)
        np.savetxt(filename_root + channel_str + '.log', dats, header=header_str, delimiter=",")
        return np.array(dats)
 
    # def laser_scan(self, channels, voltages, wavs, filename_root, coords, attens, navg=CAM_NAVG):
    #     ''' Performs a wavelength and voltage scan of multiple active components over a grid of voltages. Saves each image and the hardware settings to a table.
 
    #         Note that the filename records the requested wavelength while the log
    #         records the wavelength actually reached, which differ because the
    #         laser tunes on a 50 GHz channel grid. Both are written to the log so
    #         that a reduction can match filenames to rows.
 
    #         Parameters
    #         ----------
    #         channels : n x 1 numpy array of integers
    #             List of channels for each active component to set
    #         voltages : List of n sequences of floats
    #             All sequences of voltages to scan through for each active component. This routine will go through all possible combinations of voltages for 
    #             each active component based on their corresponding sequences of voltages to cycle through.
    #         wavs : List of float
    #             Wavelengths (nm) to scan through for each voltage setting
    #         filename_root : string
    #             Path and prefix for each file produced by this routine
    #         coords : Nx2 numpy array or list of integers
    #             Approximate coordinates of output port pixels
    #         attens : 1xN numpy array
    #             Initial attenuations for each of the outputs
    #         navg : int
    #             Number of camera frames to average together when collecting data. 
 
    #         Returns
    #         -------
    #         dats : list
    #             Table of data log containing XCOM device and laser readout settings 
    #     '''
    #     voltagecombos, crop = self._scan_setup(channels, voltages, coords, filename_root)
    #     dats = []
 
    #     for wav in wavs:
    #         wav_true = self.tlx.laser_tune(wav)
    #         wav_str = "{:.3f}".format(round(wav, 3)).replace('.', 'p')
 
    #         for voltagecombo in voltagecombos:
    #             table_row, channel_str = self._set_channels(
    #                 channels, voltagecombo, channel_str=f'_wav{wav_str}')
    #             self._measure_and_save(coords, attens, filename_root, channel_str,
    #                                    table_row, crop, dats, navg=navg,
    #                                    extra=[wav, wav_true])
 
    #     channel_str, header_str = self._scan_header(
    #         channels, extra_cols=('laserWavRequested', 'laserWav'))
    #     np.savetxt(filename_root + channel_str + '.log', dats, header=header_str, delimiter=",")
    #     return np.array(dats)
 
    def voltage_scan_adaptive(self, channels_fix, voltages_fix, channel_adapt, adapt_lim,
                              adapt_step_init, filename_root, coords, ind_null, attens,
                              navg=CAM_NAVG):
        ''' Performs a voltage scan over multiple active components, adaptively adjusting step size for one of the channels. Saves each image and the hardware settings to a table.
 
            The adaptive channel steps coarsely where the null port's attenuation
            is unchanging and finely where it falls, since a falling attenuation
            means the output is fading toward the null. When a local minimum in
            attenuation is passed, the region around it is rescanned at
            ADAPT_MIN_STEP_REWIND. Each region is rescanned at most once.
 
            Parameters
            ----------
            channels_fix : n x 1 numpy array of integers
                List of channels for each active component to set according to a fixed grid
            voltages_fix : List of n sequences of floats
                All sequences of voltages to scan through for each active component set to a fixed grid. This routine will go through all possible combinations of voltages for 
                each active component based on their corresponding sequences of voltages to cycle through.
            channel_adapt : integer
                Channel to modulate adaptively
            adapt_lim : 1 x 2 numpy array or list
                [min, max] of voltage range over which to modulate the adaptive channel
            adapt_step_init : float
                Initial step size for adaptive channel
            filename_root : string
                Path and prefix for each file produced by this routine
            coords : Nx2 numpy array or list of integers
                Approximate coordinates of output port pixels
            ind_null : integer
                Index of null port
            attens : 1xN numpy array
                Initial attenuations for each of the outputs
            navg : int
                Number of camera frames to average together when collecting data.
 
            Returns
            -------
            dats : list
                Table of data log containing XCOM device and laser readout settings 
        '''
        channels_fix = list(channels_fix)
        voltagecombos, crop = self._scan_setup(channels_fix, voltages_fix, coords, filename_root)
        dats = []

        # The adaptive channel is not part of the fixed grid, so _scan_setup does
        # not check it. With no fixed channels it is the only one being driven.
        ceiling = self.channel_maxvoltage(channel_adapt)
        if min(adapt_lim) < 0 or max(adapt_lim) > ceiling:
            raise ValueError(
                f'adapt_lim {adapt_lim} is outside 0-{ceiling} V for channel {channel_adapt}.')
 
        for voltagecombo in voltagecombos:
            if len(channels_fix):
                table_row, channel_str = self._set_channels(channels_fix, voltagecombo)
            else:
                # Nothing held fixed: start the log row and filename empty for the
                # adaptive channel to extend.
                table_row, channel_str = [], ''
 
            # First point of the sweep
            adapt_voltage = adapt_lim[0]
            table_row_all, channel_str_all = self._set_channels(
                [channel_adapt], [adapt_voltage], table_row=table_row, channel_str=channel_str)
            self._measure_and_save(coords, attens, filename_root, channel_str_all,
                                   table_row_all, crop, dats, navg=navg)
 
            # Mutable step-control state, reset for each fixed-channel combination
            adapt_step = float(adapt_step_init)
            min_step_streak = 0
            in_rewind = False
            rewind_end_voltage = None
            rescanned = []                  # (lo, hi) intervals already fine-scanned
            atten_history = []
            atten_prev = attens[ind_null]   # seeded from the first measurement
 
            while adapt_voltage < adapt_lim[-1]:
                adapt_voltage = min(adapt_voltage + adapt_step, adapt_lim[-1])
                print(f'Phase: {adapt_voltage} V')
 
                table_row_all, channel_str_all = self._set_channels(
                    [channel_adapt], [adapt_voltage], table_row=table_row, channel_str=channel_str)
                self._measure_and_save(coords, attens, filename_root, channel_str_all,
                                       table_row_all, crop, dats, navg=navg)
 
                atten_now = attens[ind_null]
                atten_delta = atten_now - atten_prev
                atten_prev = atten_now
 
                if atten_history and atten_history[-1][2] == atten_now:
                    # extend the current plateau: (v_first, v_last, atten)
                    atten_history[-1] = (atten_history[-1][0], adapt_voltage, atten_now)
                else:
                    atten_history.append((adapt_voltage, adapt_voltage, atten_now))
                    if len(atten_history) > 3:
                        atten_history.pop(0)

                if not in_rewind and len(atten_history) == 3:
                    (_, vb_last, a_before), (vm_first, vm_last, a_min), (va_first, _, a_after) = atten_history
                    already_rescanned = any((vb_last < hi) and (va_first > lo)
                                            for lo, hi in rescanned)
                    if a_min < a_before and a_min < a_after and not already_rescanned:
                        adapt_voltage = vb_last          # last point before the minimum plateau
                        rewind_end_voltage = va_first
                        rescanned.append((vb_last, va_first))
                        in_rewind = True
 
                at_min_step = False
 
                if in_rewind:
                    adapt_step = ADAPT_MIN_STEP_REWIND
                    if adapt_voltage >= rewind_end_voltage:
                        in_rewind = False
                        # Stale entries from the fine pass would otherwise
                        # satisfy the triple test again on the way forward.
                        atten_history.clear()
                elif atten_delta < 0 or atten_now <= ADAPT_ATTEN_FLOOR:
                    # Falling attenuation means the output is fading toward the
                    # null. A pinned attenuation means it has bottomed out, so
                    # growing the step would stride over the null and no local
                    # minimum would be recorded to trigger a rewind. Both want
                    # the fine step.
                    adapt_step = ADAPT_MIN_STEP_DECREASING
                    at_min_step = True
                else:
                    # Attenuation rising or steady: moving away from the null, so
                    # let the step keep growing.
                    adapt_step = adapt_step * ADAPT_GROW_FACTOR
                    adapt_step = min(max(adapt_step, ADAPT_MIN_STEP_INCREASING), ADAPT_MAX_STEP)
 
                if not in_rewind:
                    min_step_streak = min_step_streak + 1 if at_min_step else 0
 
                    if min_step_streak >= ADAPT_MAX_MIN_STEP_STREAK:
                        adapt_step = ADAPT_ESCAPE_STEP
                        min_step_streak = 0
 
        channel_str, header_str = self._scan_header(list(channels_fix) + [channel_adapt])
        np.savetxt(filename_root + channel_str + '.log', dats, header=header_str, delimiter=",")
        return np.array(dats)
    
    def optimize_null(bench, initial_voltages, channels, coords, ind_null, inds_norm=None, atten_null=40, atten_norm=40, dB_step=3,
                      mask_range=5, filt_size=5, plot=False, r_max=20, r_lin=5, bounds=None, sig_v=0.5, xatol=0.5, max_iter=100):
    
        all_voltages = []
        null_metrics = []
        temps = [] 

        null_coords = coords[ind_null].astype(int)
        if inds_norm is not None:
            norm_coords = [coords[ind].astype(int) for ind in inds_norm]
            attens_norm = atten_norm * np.ones(len(inds_norm))

        def fit_func(voltages):
            nonlocal atten_null, attens_norm
            # Set XCOM
            for i, voltage in enumerate(voltages):
                bench.apply_voltage(channels[i], voltage)

            # Take null mode image
            atten_null, data_null, _ = bench.attenuation_adjust(null_coords, atten_null, dB_step=dB_step, R=mask_range, navg=1, stack=True, output_name=f'chr{65+ind_null}')
            output_null = BENCH.extract_outputs(data_null, 1, r_max=r_max, r_lin=r_lin, 
                                        filt_size=filt_size, plot=plot, coords=np.array([null_coords]), find_centers=False) * 10**(atten_null/10)

            # Take norm images
            if inds_norm is not None:
                outputs_norm = 0
                for i, coord in enumerate(norm_coords):
                    atten, data_norm, _ = bench.attenuation_adjust(norm_coords[i], attens_norm[i], dB_step=dB_step, R=mask_range, navg=1, stack=True, output_name=f'chr{65+inds_norm[i]}')
                    attens_norm[i] = atten
                    outputs_norm += BENCH.extract_outputs(data_norm, 1, r_max=r_max, r_lin=r_lin, 
                                                filt_size=filt_size, plot=plot, coords=np.array([norm_coords[i]]), find_centers=False) * 10**(attens_norm[i]/10)
            else:
                outputs_norm = 1

            # Report intensities
            null_value = output_null/outputs_norm
            _log.info("voltages:"+str(voltages))
            _log.info("Null mode intensities: "+str(output_null))   
            _log.info("Side mode intensities: "+str(outputs_norm))   
            _log.info("Null: "+str(10*np.log10(null_value))+" dB")
            _log.info("\n")
            all_voltages.append(voltages)
            null_metrics.append(null_value)
            temps.append(bench.tec.read_temperature())
            return null_value
        
        from numpy.random import default_rng
        rng = default_rng(seed=4524)
        n_v = len(initial_voltages)
        initial_simplex = initial_voltages[np.newaxis,:] + sig_v*rng.normal(size=(n_v+1,n_v))

        if bounds is None:
            bounds = [(0., bench.channel_maxvoltage(spec)) for spec in channels]

        from scipy.optimize import minimize
        options = {'initial_simplex': initial_simplex, 
                'xatol':xatol,
                'maxiter':max_iter}
        res = minimize(fit_func, None, method='Nelder-Mead', options=options, bounds=bounds)
        null_depth = fit_func(res.x)
        all_voltages = np.array(all_voltages)
        null_metrics = np.array(null_metrics)
        temps = np.array(temps)
        return res, null_depth, all_voltages, null_metrics, temps  

    def laser_scan(self, channels, channel_ps, wav_ref, wavs, filename_root,
               coords, attens, ind_null, initial_voltages,
               min_range=0.2, min_step=0.001,
               max_low=10.0, max_up=13.0, max_step=0.2,
               navg=CAM_NAVG, optimize=True, **optimize_kwargs):
        ''' Measures null degradation and null phase shifter voltage against wavelength.

            The null is found at the reference wavelength with optimize_null. The
            other channels are then held at those settings for every wavelength, and
            only channel_ps is swept, so a change in null depth reflects wavelength
            rather than re-tuning.

            At each wavelength the phase shifter is swept over two windows:

                minimum   v_ref - min_range to v_ref + min_range, in min_step,
                        where v_ref is the null voltage found at wav_ref.
                        Resolves the null itself and the voltage it sits at.
                maximum   max_low to max_up, in max_step, as absolute voltages.
                        Supplies the maximum the null is normalised against.
                        Set these to bracket the fringe peak, which sits about a
                        quarter of a fringe from the null.

            Images and log rows are saved for every point; the null depth and the
            null voltage come out of the reduction rather than being computed here,
            so they use the same photometry as the rest of the analysis.

            The two windows are disjoint in phase shifter voltage, so a reduction
            can take the null as the minimum and the normalisation as the maximum
            over all points at a wavelength without needing to know which window
            each point came from.

            Filenames carry the requested wavelength rather than the achieved one,
            since the laser tunes on a 50 GHz grid. Both go in the log.

            Parameters
            ----------
            channels : n x 1 numpy array of integers
                TOPM channels held fixed at their reference-wavelength settings
            channel_ps : integer
                Phase shifter channel swept at every wavelength
            wav_ref : float
                Reference wavelength (nm) at which the null is found
            wavs : List of float
                Wavelengths (nm) to measure the null at. wav_ref is measured first
                whether or not it appears here.
            filename_root : string
                Path and prefix for each file produced by this routine
            coords : Nx2 numpy array or list of integers
                Approximate coordinates of output port pixels
            attens : 1xN numpy array
                Initial attenuations for each of the outputs
            ind_null : integer
                Index of the null port in coords
            initial_voltages : 1D numpy array
                Starting voltages for channels then channel_ps
            min_range : float
                Half-width in V of the minimum window around v_ref. Default: 0.2
            min_step : float
                Step in V within the minimum window. Default: 0.001
            max_low, max_up : float
                Absolute voltages bounding the maximum window. Default: 10.0 and 13.0
            max_step : float
                Step in V within the maximum window. Default: 0.2
            navg : int
                Number of camera frames to average together when collecting data
            optimize : boolean
                Whether to run optimize_null at wav_ref. If False, initial_voltages
                is used as the null setting directly.
            optimize_kwargs : dict
                Further keyword arguments for optimize_null

            Returns
            -------
            dats : 2D numpy array
                Table of data log containing XCOM device and laser readout settings
        '''

        import logging
        _log = logging.getLogger('Bench')
        
        channels = list(channels)
        all_channels = channels + [channel_ps]
        initial_voltages = np.asarray(initial_voltages, dtype=float)

        if len(initial_voltages) != len(all_channels):
            raise ValueError(
                f'initial_voltages needs {len(all_channels)} entries, one per channel '
                f'in {channels} plus the phase shifter {channel_ps}. '
                f'Got {len(initial_voltages)}.')

        if max_low >= max_up:
            raise ValueError(f'max_low {max_low} must be below max_up {max_up}.')

        dats = []

        # --- find the null at the reference wavelength ---
        if optimize:
            wav_ref_true = self.tlx.laser_tune(wav_ref)
            _log.info(f'Optimizing null at {wav_ref} nm (reached {wav_ref_true:.4f} nm)...')

            res, null_depth, all_voltages, null_metrics, temps = self.optimize_null(initial_voltages, 
                                                                                        all_channels, coords, ind_null, **optimize_kwargs)
            while null_metrics[-1] > null_metrics[0]:
                res, null_depth, all_voltages, null_metrics, temps = self.optimize_null(initial_voltages, 
                                                                                        all_channels, coords, ind_null, **optimize_kwargs)
            null_voltages = np.asarray(res.x, dtype=float)
            i_best = np.argmin(null_metrics)
            
            print(res.message, res.nit, res.nfev)
            
            plt.figure(figsize=(10,6))
            for i in range(len(all_channels)):
                plt.plot(all_voltages[:,i])
            plt.axvline(i_best, color='grey', linestyle='dashed')
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel('Iterations', fontsize=20)
            plt.ylabel('Voltage (V)', fontsize=20)
            # plt.legend(fontsize=16)
            plt.show()
        
            plt.figure(figsize=(10,6))
            plt.plot(null_metrics, 'k')
            plt.axvline(i_best, color='grey', linestyle='dashed')
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel('Iterations', fontsize=20)
            plt.ylabel('Flux (arbitrary units)', fontsize=20)
            plt.show()
        else:
            null_voltages = initial_voltages

        # channels only: the phase shifter is the last entry and gets swept, so it
        # must not be included in the settings held fixed.
        topm_voltages = null_voltages[:len(channels)]
        v_ref = float(null_voltages[-1])
        _log.info(f'Reference null: {channels} at {topm_voltages}, '
                f'{channel_ps} at {v_ref:.3f} V')

        # --- phase shifter windows ---
        ceiling = self.channel_maxvoltage(channel_ps)

        ps_min = np.arange(v_ref - min_range, v_ref + min_range + 0.5*min_step, min_step)
        ps_max = np.arange(max_low, max_up + 0.5*max_step, max_step)

        ps_min = ps_min[(ps_min >= 0.0) & (ps_min <= ceiling)]
        ps_max = ps_max[(ps_max >= 0.0) & (ps_max <= ceiling)]

        if len(ps_min) == 0 or len(ps_max) == 0:
            raise ValueError(
                f'A phase shifter window lies outside 0-{ceiling} V for channel '
                f'{channel_ps} with v_ref {v_ref:.3f} V.')

        _log.info(f'Minimum window: {ps_min[0]:.3f} to {ps_min[-1]:.3f} V, {len(ps_min)} points')
        _log.info(f'Maximum window: {ps_max[0]:.3f} to {ps_max[-1]:.3f} V, {len(ps_max)} points')

        # Voltage grid for the ceiling checks and the crop
        voltages_for_setup = [[v] for v in topm_voltages]
        voltages_for_setup.append(np.concatenate([ps_min, ps_max]))
        _, crop = self._scan_setup(all_channels, voltages_for_setup, coords, filename_root)

        # wav_ref first, then the rest, without measuring it twice
        wav_list = [wav_ref] + [w for w in wavs if not np.isclose(w, wav_ref)]

        for wav in wav_list:
            wav_true = self.tlx.laser_tune(wav)
            wav_str = "{:.3f}".format(round(wav, 3)).replace('.', 'p')
            _log.info(f'--- {wav} nm (reached {wav_true:.4f} nm) ---')

            # TOPMs back to the reference setting: the previous wavelength left the
            # phase shifter elsewhere, and re-applying guards against drift in the
            # tracked state.
            table_row_fix, channel_str_fix = self._set_channels(
                channels, topm_voltages, channel_str=f'_wav{wav_str}')

            for v_ps in np.concatenate([ps_min, ps_max]):
                table_row, channel_str = self._set_channels(
                    [channel_ps], [v_ps], table_row=table_row_fix,
                    channel_str=channel_str_fix)
                self._measure_and_save(coords, attens, filename_root, channel_str,
                                    table_row, crop, dats, navg=navg,
                                    extra=[wav, wav_true])

        channel_str, header_str = self._scan_header(
            all_channels, extra_cols=('laserWavRequested', 'laserWav'))
        np.savetxt(filename_root + channel_str + '.log', dats,
                header=header_str, delimiter=",")

        return np.array(dats)     

    def live_view_outputs(self, num_outputs, coords=None, mask_range=5,
                      interval=0.2, vmin=0, vmax=12000):
        """
        Live streaming viewer:
        - one panel per output (masked)
        - horizontal crosscut per panel
        - max pixel readout
        """

        if coords is None:
            _, _, frame, _ = self.cam.take_image()
            coords = self.find_output_centers(frame - self.dark, num_outputs)

        masks = self.build_output_mask(frame, num_outputs,
                                    coords=coords,
                                    mask_range=mask_range,
                                    find_centers=False,
                                    plot=False)

        fig, axes = plt.subplots(num_outputs, 2, figsize=(8, 3*num_outputs))
        if num_outputs == 1:
            axes = np.array([axes])

        ims = []
        cross_lines = []
        cross_axes = []
        text_boxes = []

        # setup plots once
        for i in range(num_outputs):
            ax_img = axes[i, 0]
            ax_prof = axes[i, 1]

            im = ax_img.imshow(frame, vmin=vmin, vmax=vmax, cmap="viridis")
            ims.append(im)

            ax_img.set_title(f"Output {chr(65+i)}")
            ax_img.set_axis_off()

            line, = ax_img.plot([], [], 'r-', lw=1)
            cross_lines.append(line)

            ax_prof.set_title("Crosscut (row)")
            prof_line, = ax_prof.plot([], [], lw=1)
            ax_prof.set_ylim(vmin, vmax)
            cross_axes.append((ax_prof, prof_line))

            txt = ax_img.text(5, 5, "", color="white")
            text_boxes.append(txt)

        plt.tight_layout()

        while True:
            _, _, frame, _ = self.cam.take_image()
            frame = frame - self.dark

            for i in range(num_outputs):
                masked = np.where(masks[i], frame, np.nan)

                # update image
                ims[i].set_data(masked)

                # max pixel
                max_val = np.nanmax(masked)
                text_boxes[i].set_text(f"max: {max_val:.1f}")

                # cross-section (middle row of mask bbox)
                ys, xs = np.where(masks[i])
                y0 = int(np.mean(ys))

                profile = frame[y0, :]

                cross_axes[i][1].set_data(np.arange(len(profile)), profile)
                cross_axes[i][0].relim()
                cross_axes[i][0].autoscale_view(scaley=True)

            plt.pause(interval)     

    ### Photometric extraction methods ###
    @staticmethod
    def find_output_centers(frame, num_outputs, init_coords=None, filt_size=5, plot=True):
        ''' Finds the coordinates of each output using the pixel of maximum intensity for each output.
        
            Parameters
            ----------
            frame : 2D numpy array
                Image data
            num_outputs : int
                Number of outputs to identify in reference image
            init_coords : 2D numpy array
                Initial guess for coordinates of each output 
            filt_size : int
                Size of maximum filter (in pixels) for isolating local maxima to identify outputs. Default: 5
            plot : boolean
                Flag for plotting masks overlayed on image

            Returns
            -------
            coords : 2D numpy array
                Coordinates of each output    
        '''

        if init_coords is None:
            # Find local maxima
            filtered = scipy.ndimage.maximum_filter(frame, size=filt_size)
            masks = (frame == filtered)
            # Dilated as a 2D mask. Iterating over a 2D array yields rows, so
            # the per-row version dilated horizontally only: each labelled
            # region ended up one pixel tall and its centroid in y could only
            # be that row's index, never a weighted position.
            masks = scipy.ndimage.binary_dilation(masks, iterations=2)
            # Label each local maximum region
            labeled, num_features = scipy.ndimage.label(masks)
            # Compute centroid of each labeled region
            centroids = np.array(scipy.ndimage.center_of_mass(frame, labeled, range(1, num_features+1)))

            # Compute total intensity in each region
            region_sums = np.array([frame[labeled==i+1].sum() for i in range(num_features)])
            top_idx = np.argsort(region_sums)[-num_outputs:]
            coords = centroids[top_idx]

            # Sort by x-coordinate to keep ABCDE order
            coords = coords[np.argsort(coords[:, 1])]#.astype(int)    
        
        else:
            coords = []

            R = filt_size

            for r, c in init_coords:

                r, c = int(r), int(c)

                r0 = max(0, r - R)
                r1 = min(frame.shape[0], r + R + 1)

                c0 = max(0, c - R)
                c1 = min(frame.shape[1], c + R + 1)

                # Crop local search region
                subframe = frame[r0:r1, c0:c1]
                subframe = np.clip(subframe, 0, None)

                # Compute local center of mass only within this region
                rr, cc = scipy.ndimage.center_of_mass(subframe)

                coords.append([r0 + rr, c0 + cc])

            coords = np.array(coords)       

        
        if plot:
            y_dim, x_dim = np.shape(frame)

            fig, ax = plt.subplots(1, 1, figsize=(10,6))
            im = ax.pcolormesh(frame, zorder=0)
            for coord in coords:
                ax.scatter(coord[1]+0.5, coord[0]+0.5, marker='+', color='r', alpha=0.5)
            fig.colorbar(im, ax=ax, fraction=0.05*y_dim/x_dim)
            ax.set_xlabel('x (pixels)', fontsize=18)
            ax.set_ylabel('y (pixels)', fontsize=18)
            ax.set_aspect('equal')
            plt.show()
        return np.asarray(coords)
    
    @staticmethod
    def build_output_mask(frame, num_outputs, coords=None, find_centers=True, mask_range=5, filt_size=5, plot=True):
        ''' Builds a collection of boolean masks identifying each output in the reference image.
        
            Parameters
            ----------
            frame : 2D numpy array
                Image data
            num_outputs : int
                Number of outputs to identify in reference image
            coords : 2D numpy array or None
                Coordinate pairs of output centers; initial guess if find_centers=True, else used as fixed output centers.
            find_centers : boolean
                True if need to find output centers; else False. Cannot be False if coords is None.
            mask_range : int
                Mask radius (in pixels) around center of output. Default: 5
            filt_size : int
                Size of maximum filter (in pixels) for isolating local maxima to identify outputs. Default: 5
            plot : boolean
                Flag for plotting masks overlayed on image

            Returns
            -------
            outputs_masks : 3D numpy array
                Collection of 2D boolean masks for each output in reference image    
        '''
        if coords is None and find_centers == False:
            raise ValueError("Cannot set find_centers to False if coords is None.")

        # Determine coordinates of outputs
        if find_centers:
            coords = BENCH.find_output_centers(frame, num_outputs, init_coords=coords, filt_size=filt_size, plot=False)

        # Create 2D grid
        height, width = np.shape(frame)
        Y, X = np.ogrid[:height, :width]  # Y: (height,1), X: (1,width)
        
        # Reshape coords for broadcasting: (N,1,1)
        Xc = coords[:, 1][:, None, None]
        Yc = coords[:, 0][:, None, None]
        
        # Compute squared distance for each coordinate
        dist2 = (X - Xc)**2 + (Y - Yc)**2  # shape: (N, height, width)
        
        # Create masks
        outputs_masks = dist2 <= mask_range**2

        # Plot the reference data with the output mask
        if plot:
            yellow_cmap = colors.ListedColormap(['red'])
            y_dim, x_dim = np.shape(frame)

            fig, ax = plt.subplots(1, 1, figsize=(10,6))
            im = ax.pcolormesh(frame, zorder=0)
            for mask in outputs_masks:
                masked_overlay = np.ma.masked_where(~mask, np.ones_like(mask))
                ax.pcolormesh(masked_overlay, cmap=yellow_cmap, alpha=0.3, zorder=1)
            fig.colorbar(im, ax=ax, fraction=0.05*y_dim/x_dim)
            ax.set_xlabel('x (pixels)', fontsize=18)
            ax.set_ylabel('y (pixels)', fontsize=18)
            ax.set_aspect('equal')
            plt.show()

        return np.array(outputs_masks)

    @staticmethod
    def plot_masked(frame, mask, vmin=0, vmax=12000):
        ''' Builds a collection of boolean masks identifying each output in the reference image.
        
            Parameters
            ----------
            frame : 2D numpy array
                Image data
            mask : 2D numpy array
                2D boolean mask of region to plot
            vmin, vmax: float
                Minimum and maximum of plot colorbar

            Returns
            -------
            None
        '''
        masked_data = np.where(mask, frame, np.nan)

        ys, xs = np.where(mask)

        pad = 2  # extra pixels around mask

        xmin = xs.min() - pad
        xmax = xs.max() + pad
        ymin = ys.min() - pad
        ymax = ys.max() + pad

        fig, ax = plt.subplots(1, 1, figsize=(6,6))

        im = ax.pcolormesh(masked_data, zorder=0, vmin=vmin, vmax=vmax)

        ax.set_xlim(xmin, xmax + 1)
        ax.set_ylim(ymax + 1, ymin)   # invert y-axis to match image orientation

        fig.colorbar(im, ax=ax)

        ax.set_xlabel('x (pixels)', fontsize=18)
        ax.set_ylabel('y (pixels)', fontsize=18)
        ax.set_aspect('equal')

        plt.show()
        return

    @staticmethod
    def remove_stripes(frame, outputs_masks=None):
        ''' Removes stripes from Xenics camera images.

            Parameters
            ----------
            frame : 2D numpy array
                Image data
            outputs_masks : None or 3D numpy array
                Collection of 2D boolean masks for each output in reference image

            Returns
            -------
            frame_reduced : 2D numpy array
                Image data with stripes removed
        '''
        if outputs_masks is not None:
            mask_stripes = np.sum(outputs_masks, axis=0)
            inds = np.argwhere(mask_stripes==True)
            inds_rows = list(set(inds[:,0]))
            ind_min, ind_max = np.min(inds_rows), np.max(inds_rows)
            inds_rows = np.arange(ind_min - 5*(ind_max-ind_min), ind_max + 5*(ind_max-ind_min))
            for ind in inds_rows:
                mask_stripes[ind,:] = True

            frame_stripes = np.copy(frame)
            frame_stripes[mask_stripes] = np.nan
        else:
            frame_stripes = np.copy(frame)

        frame_reduced = frame - np.nanmean(frame_stripes, axis=0)
        return frame_reduced

    @staticmethod
    def extract_outputs_frame(frame, num_outputs, r_max=15, r_lin=5, filt_size=5,
                              plot=False, coords=None, find_centers=True,
                              ax=None, ax_img=None):
        ''' Get output intensities from data frame using Nem's photometry extraction method.

            Parameters
            ----------
            frame : 2D numpy array
                Image data
            num_outputs : int
                Number of outputs to identify in reference image
            r_max : int
                Maximum pixel radius for photometry extraction method. Default: 15
            r_lin : int
                Lower limit of pixel radius at which photometry extraction method behaves linearly. Default: 5
            filt_size : int
                Size of maximum filter (in pixels) for isolating local maxima to identify outputs. Default: 5
            plot : boolean
                Flag for plotting masks overlayed on image and photometry extraction
            coords : 2D numpy array or None
                Coordinate pairs of output centers; if None, finds output centers
            find_centers : boolean
                True if need to find output centers; else False. Cannot be False if coords is None.
            ax : matplotlib Axes or None
                If given, the photometry fit is drawn onto this axis instead of a
                new figure, and the figure is not shown. Lets a caller composite
                the fit into a larger figure, such as a movie frame.
            ax_img : matplotlib Axes or None
                If given, the aperture bounds of the fitted band are drawn onto
                this axis as dashed circles, with a red cross at each center.
                Drawn from the same centers the photometry used, so they cannot
                disagree with it.

            Returns
            -------
            outputs : list of numpy arrays
        '''
        import string

        # Found once here rather than inside build_output_mask at every radius:
        # the centers are the same for all radii, so repeating the search is
        # wasted work, and having them here means the aperture circles are drawn
        # from exactly the centers the photometry used.
        # The mask and centre overlays open their own figures, which cannot be
        # composited into a caller's axis, so they are suppressed when an axis
        # is given.
        plot_masks = plot and ax is None and ax_img is None

        if coords is None:
            if not find_centers:
                raise ValueError('find_centers cannot be False when coords is None.')
            # plot=False explicitly: find_output_centers defaults to plot=True,
            # so omitting it opens a figure for every frame reduced.
            coords = BENCH.find_output_centers(frame, num_outputs,
                                               filt_size=filt_size, plot=plot_masks)

        radii = np.arange(1, r_max)
        enclosed_fluxes = []

        for r in radii:
            outputs_masks = BENCH.build_output_mask(
                frame, num_outputs, mask_range=r, filt_size=filt_size,
                plot=plot_masks and (r == np.max(radii)),
                coords=coords, find_centers=False)
            enclosed_fluxes.append(np.sum(outputs_masks * frame, axis=(1, 2)))
        enclosed_fluxes = np.array(enclosed_fluxes).T

        enclosed_pixels = np.pi*radii**2
        fit = radii > r_lin

        # The band actually fitted: radii runs 1 to r_max - 1 and the fit uses
        # radii > r_lin, so the first and last fitted radii are these.
        r_fit_lo, r_fit_hi = radii[fit][0], radii[fit][-1]

        if ax_img is not None:
            from matplotlib.patches import Circle
            for c in coords:
                for r_ap in (r_fit_lo, r_fit_hi):
                    ax_img.add_patch(Circle((c[1], c[0]), r_ap, fill=False,
                                            linestyle='--', color='0.7', linewidth=0.8))
                # x is the column and y is the row, and imshow centres pixel i on
                # coordinate i, so no half pixel offset here. The pcolormesh plot
                # in find_output_centers needs one because that artist draws cell
                # edges rather than centres.
                ax_img.plot(c[1], c[0], marker='+', color='r', markersize=8,
                            markeredgewidth=1.2, linestyle='none')

        drawing = plot or ax is not None
        if drawing:
            target = ax if ax is not None else plt.figure().gca()

        outputs = []
        for i in range(num_outputs):
            enclosed_flux = enclosed_fluxes[i]
            m, intercept = np.polyfit(enclosed_pixels[fit], enclosed_fluxes[i][fit], 1)
            flux = enclosed_flux[-1] ### REMOVE THIS

            if drawing:
                target.plot(enclosed_pixels, enclosed_flux, 'o', color=f"C{i}",
                            label=f"Output {string.ascii_uppercase[i]}")
                # Drawn with the fitted intercept, not with flux: the line above
                # replaces flux with the last data point, so using it here would
                # draw the fit shifted up to meet that point rather than through
                # the data it was fitted to.
                target.plot(enclosed_pixels, m*enclosed_pixels + intercept,
                            linestyle='dashed', color=f"C{i}")
            outputs.append(flux)
        outputs = np.array(outputs)

        if drawing:
            # Same bounds as the circles, in enclosed-pixel units, so the fitted
            # span is readable on both panels.
            for r_ap in (r_fit_lo, r_fit_hi):
                target.axvline(np.pi*r_ap**2, linestyle='--', color='0.7', linewidth=0.8)

            target.set_xlim(0, np.max(enclosed_pixels))
            target.set_xlabel('Enclosed pixels', fontsize=10)
            target.set_ylabel('Enclosed counts', fontsize=10)
            target.legend(fontsize=8)
            if ax is None:
                plt.show()

        return outputs

    @staticmethod
    def extract_outputs(frames, num_outputs, r_max=15, r_lin=5, filt_size=5, plot=False,
                        coords=None, find_centers=True, stack=True, ax=None, ax_img=None):
        ''' Get output intensities from data frame using Nem's photometry extraction method.

            Parameters
            ----------
            frames : 2D or 3D numpy array
                Image data
            num_outputs : int
                Number of outputs to identify in reference image
            r_max : int
                Maximum pixel radius for photometry extraction method. Default: 15
            r_lin : int
                Lower limit of pixel radius at which photometry extraction method behaves linearly. Default: 5
            filt_size : int
                Size of maximum filter (in pixels) for isolating local maxima to identify outputs. Default: 5
            plot : boolean
                Flag for plotting masks overlayed on image and photometry extraction
            coords : 2D numpy array or None
                Coordinate pairs of output centers; if None, finds output centers
            find_centers : boolean
                True if need to find output centers; else False. Cannot be False if coords is None.
            stack : boolean
                If True, mean-stack all frames; else, don't combine frames.
            ax : matplotlib Axes or None
                If given, the photometry fit is drawn onto this axis. For a 3D
                stack only the middle frame draws, so a 50 frame stack does not
                overplot 50 fits onto one axis.
            ax_img : matplotlib Axes or None
                If given, the aperture bounds are drawn onto this axis. As with
                ax, only the middle frame of a stack draws.

            Returns
            -------
            outputs : list of numpy arrays
        '''
        if len(np.shape(frames)) == 2:
            outputs = BENCH.extract_outputs_frame(frames, num_outputs, coords=coords,
                                                  find_centers=find_centers, r_max=r_max,
                                                  r_lin=r_lin, filt_size=filt_size,
                                                  plot=plot, ax=ax, ax_img=ax_img)
        elif len(np.shape(frames)) == 3:
            outputs = []
            for i in range(len(frames)):
                middle = (i == int(len(frames)/2))
                outputs.append(BENCH.extract_outputs_frame(
                    frames[i], num_outputs, coords=coords, find_centers=find_centers,
                    r_max=r_max, r_lin=r_lin, filt_size=filt_size,
                    plot=plot if middle else False,
                    ax=ax if middle else None,
                    ax_img=ax_img if middle else None))
            outputs = np.array(outputs)
            if stack:
                outputs = np.mean(outputs, axis=0)
        return outputs
    
    # Close object
    def close(self):
        ''' Close connections to all bench hardware.

            Each instrument is closed independently, so a failure in one does
            not prevent the rest from shutting down.

            Parameters
            ----------
            None

            Returns
            -------
            None        
        '''
        if self.tlx is not None:
            try:
                self.tlx.laser_off()
                self.tlx.close()
            except Exception as e:
                _log.warning(f'Could not close TLX2 laser: {type(e).__name__}: {e}')

        try:
            self.tec.off()
            self.tec.close()
        except Exception as e:
            _log.warning(f'Could not close TEC: {type(e).__name__}: {e}')

        try:
            self.zaber.close()
        except Exception as e:
            _log.warning(f'Could not close Zabers: {type(e).__name__}: {e}')

        if self.cam is not None:
            try:
                self.cam.close()
            except Exception as e:
                _log.warning(f'Could not close camera: {type(e).__name__}: {e}')

        if self.osa is not None:
            try:
                self.osa.close()
            except Exception as e:
                _log.warning(f'Could not close OSA: {type(e).__name__}: {e}')

        if self.xcom is not None:
            try:
                self.xcom.close()
            except Exception as e:
                _log.warning(f'Could not close XCOM: {type(e).__name__}: {e}')

        if self.keysight is not None:
            try:
                self.keysight.close()
            except Exception as e:
                _log.warning(f'Could not close Keysight power supply: {type(e).__name__}: {e}')

        _log.info('Bench hardware closed.')
        return
    
    @property
    def voa(self):
        ''' Property for bench's variable optical attenuation.'''
        return self.get_VOA()
