# PM.py 
#
# Library for defining PM class and routines for reading out optical power from a Thorlabs power meter console.
# Modified: Anusha, 2/25/26

import PowerMeterAcquire as pma
import numpy as np
import time

import logging
_log = logging.getLogger('PM')

# METER_NAME = "P0015519" # student lab console
# METER_NAME = "P0015560" # HISPEC console
# METER_NAME = "M01294960
METER_NAME = "M01307960" # Photonics PM
METER_ID = '0x8099'


class PM:
    def __init__(self, meter_name=METER_NAME, meter_id=METER_ID):
        ''' Upon initialization of class instance, opens a connection to the Thorlabs power meter

            Parameters
            ----------
            meter_name : str
                Name of Thorlabs power meter console, depends on which one you use. Default: meter name of student lab console = "P0015519"

            Returns
            -------
            None
        '''
        self.meter_name = meter_name
        self.meter_id = meter_id
        self.open()
        return

    def open(self):
        ''' Open connection to Thorlabs power meter.

            Parameters
            ----------
            None

            Returns
            -------
            None        
        '''
        print(self.meter_name)
        pma.StartMeter(meter_name=self.meter_name, meter_id=self.meter_id)
        _log.info('Connected to power meter.')
        return

    def close(self):
        ''' Close connection to Thorlabs power meter.

            Parameters
            ----------
            None

            Returns
            -------
            None        
        '''
        pma.CloseMeter()
        _log.info('Disconnected from power meter.')
        return

    def read_power(self, iter=10, std=False):
        ''' Read optical power from Thorlabs power meter.
        
            Parameters
            ----------
            iter : int
                Number of readings to take

            std : boolean
                True if include standard deviation of measurements in output, else False

            Returns
            -------
            power : float
                Average of power readings
            sigma : float, optional 
                Standard deviation of power readings
        '''
        powers=[]
        for it in range(iter):
            powers.append(pma.MeterReadPower()[0])
        power = np.average(powers)
        if std == False:
            return power
        else:
            sigma = np.std(powers)
            return power, sigma
        
    def stream_power(self, duration=10, iter=10, std=False):
        ''' Stream power readings from Thorlabs power meter for a fixed duration of time.
        
            Parameters
            ----------
            duration : float
                Time (in seconds) over which to readout power measurements
            iter : int
                Number of readings to take

            std : boolean
                True if include standard deviation of measurements in output, else False

            Returns
            -------
            times : 1D array of floats
                Timestamps of measurements
            powers : 1D array of floats
                Power measurements
            sigmas : 1D array of floats, optional
                Standard deviations of power readings
        '''
        times = [time.time()]
        power = self.read_power(iter=iter, std=std)
        
        if std:
            powers, sigmas = [power[0]], [power[1]]
        else:
            powers = [power]
        time_end = times[0] + duration

        while time.time() < time_end:
            power = self.read_power(iter=iter, std=std)
            times.append(time.time())
            if std:
                powers.append(power[0])
                sigmas.append(power[1])
            else:
                powers.append(power)

        if std:
            return times, powers, sigmas
        else:
            return times, powers
