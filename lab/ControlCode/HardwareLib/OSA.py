# OSA.py 
#
# Library for defining OSA class and routines for reading out spectra from a Thorlabs OSA 202C.
# Modified: Anusha, 2/26/26

import OSA202c
import matplotlib.pyplot as plt
import numpy as np
import clr
import os, sys

import logging
_log = logging.getLogger('OSA')

clr.AddReference("C:\\Program Files\\Thorlabs\\ThorSpectra\\ThorlabsOSAWrapper.dll")
clr.AddReference("System.Collections")
from ThorlabsOSAWrapper import SpectrumStruct
from System.Collections.Generic import List
sys.path.insert(1, os.path.normpath(os.getcwd() + os.sep + os.pardir))

class OSA:
    def __init__(self):
        ''' Upon initialization of class instance, opens a connection to the Thorlabs power meter

            Parameters
            ----------
            None

            Returns
            -------
            None
        '''
        self.open()
        return

    def open(self):
        ''' Open connection to Thorlabs OSA 202c.

            Parameters
            ----------
            None

            Returns
            -------
            None        
        '''
        OSA202c.OSAOpen()
        return

    def close(self):
        ''' Close connection to Thorlabs OSA 202c.

            Parameters
            ----------
            None

            Returns
            -------
            None        
        '''
        OSA202c.OSAClose()
        _log.info('Disconnected from OSA.')
        return

    def read_spectrum(self, navg=5, discard=0, plot=False, filename=None):
        ''' Read spectra from Thorlabs OSA 202c.

            Parameters
            ----------
            None

            Returns
            -------
            None        
        '''
        spectrum_list = OSA202c.OSAGetMultipleSpectrum(discard, 
                                                effectiveNum = navg)
        
        if filename is not None:
            for it, spectrum in enumerate(spectrum_list):
                OSA202c.OSASaveSpectrumFile(spectrum, filename + '_%d' % (it), 1)
        
        all_data = []
        for it, spectrum in enumerate(spectrum_list):
            data = OSA202c.OSAGetSpectralData(spectrum, True)
            all_data.append(data)
        sigma = np.std(all_data, axis=0)
        all_data = np.average(all_data, axis=0)

        if plot:
            plt.figure(figsize=(10,6))
            plt.errorbar(all_data[0], all_data[1], sigma)
            plt.xlabel('Wavelength (nm)')
            plt.ylabel('dB')
            plt.show()

        return all_data, sigma
    
