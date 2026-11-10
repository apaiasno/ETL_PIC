# TEC.py 
#
# Library for defining TEC class and routines for managing temperature control with the Arduino+TEC system.
# Modified: Anusha, 1/12/26

import time
import serial

import logging
_log = logging.getLogger('TEC')

COM_PORT = 'COM7'
BAUDRATE = 9600
TIMEOUT = 1

class TEC:
    def __init__(self, com_port=COM_PORT, baudrate=BAUDRATE, timeout=TIMEOUT):
        ''' Upon initialization of class instance:
            1. Sets serial connection attributes.
            2. Serial connection to Arduino controller is opened. 

            Parameters
            ----------
            com_port : str
                Name of COM port for serial communication with TEC. Default: COM_PORT
            baudrate : int
                Baudrate for serial communication with TEC.
            timeout : float
                Time in seconds for serial communication timeout.

            Returns
            -------
            None
        '''
        self.com_port = com_port
        self.baudrate = baudrate
        self.timeout = timeout
        self.open()
        return

    def open(self):
        ''' Open serial connection to Arduino controller for TEC.

            Parameters
            ----------
            None

            Returns
            -------
            None        
        '''
        self.ser = serial.Serial(self.com_port, self.baudrate, timeout=self.timeout)
        time.sleep(2)  # wait for Arduino reset
        self.ser.reset_input_buffer()   # clear stale bytes
        self.ser.reset_output_buffer()
        _log.info("Arduino port {self.ser.name} opened successfully")
        _log.info(self.ser.readline().decode().strip())
        _log.info(self.ser.readline().decode().strip())
        return

    def close(self):
        ''' Close serial connection to Arduino controller for TEC.

            Parameters
            ----------
            None

            Returns
            -------
            None        
        '''
        if self.ser and self.ser.is_open:
            self.off()
            self.ser.close()
            print("Connection to Arduino closed.")
        return

    def on(self):
        ''' Turn on PID loop for temperature control stabilized to 18 degrees Celsius.
        
            Parameters
            ----------
            None

            Returns
            -------
            None
        '''
        self._write("ON")
        _log.info("Temperature control loop on.")
        return

    def off(self):
        ''' Turn off PID loop for temperature control stabilized to 18 degrees Celsius.
        
            Parameters
            ----------
            None

            Returns
            -------
            None
        '''
        self._write("OFF")
        _log.info("Temperature control loop off.")
        return

    def read_temperature(self):
        ''' Read current temperature value from thermistor input to Arduino.
        
            Parameters
            ----------
            None

            Returns
            -------
            temp : float
                Current temperature reading
        '''
        if self.ser and self.ser.is_open:
            try:     
                self.ser.reset_input_buffer()       
                temp = self.ser.readline().decode().strip()
                temp = float(temp)
                return temp
            except ValueError:
                raise ValueError(f"Received: {temp}")

    def _write(self, message):
        ''' Helper for serial communication with Arduino.

            Parameters
            ----------
            message : str
                Command for Arduino

            Returns
            -------
            None      
        '''
        if self.ser and self.ser.is_open:
            self.ser.write((message + "\n").encode())
        return
