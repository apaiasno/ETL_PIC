# TLX2.py
# 
# Library for defining XPOW class and routines for communicating with the TLX2 1.55 micron laser. Based on Yoo Jung's original class definition.
# Modified by: Anusha, 12/3/2025 

import serial
import time

COM_PORT = 'COM1'
dict_status = {'1\r': 'on', '0\r': 'off'}
VOA_MAX, VOA_MIN = 20, 1

class TLX:

    def __init__(self, open=True, com_port=COM_PORT):
        ''' Upon initialization of class instance, opens serial communication with laser if open is set to True.

            Parameters
            ----------
            open : bool
                Opens serial communication with laser if True, else just initializes instance of class. Default: True.
            com_port : str
                Name of COM port for communication with laser. Default: COM_PORT

            Returns
            -------
            None
        '''
        if open:
            self.open(com_port)
            self.voa_on()
            self.set_voa(VOA_MAX)
        return
            
    def open(self, com_port):
        ''' Opens serial communication with laser.
            
            Parameters
            ----------
            com_port : str
                Name of COM port for communication with laser. Default: COM_PORT

            Returns
            -------
            None
        '''
        self.ser =  serial.Serial(
                    port=com_port,      # Change to your COM port
                    baudrate=115200,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=3         # seconds
                )
        print('Connected to TLX2 laser.')
        return
    
    def send_command(self, cmd):
        ''' Sends a SCPI-style serial command and reads the response.

            Parameters
            ----------
            cmd : str
                Command to send laser.

            Returns
            -------
            resp : str
                Response from laser.        
        '''
        full = cmd + '\r\n'
        self.ser.write(full.encode('ascii'))
        time.sleep(0.1)  # small delay; adjust as needed
        resp = self.ser.read_all().decode('ascii')
        return resp
    
    def laser_on(self):
        ''' Turns laser on.

            Parameters
            ----------
            None

            Returns
            -------
            None
        '''
        resp = self.send_command('LASer:POWer: 1')
        time.sleep(3)
        print(f"Laser set to {self.laser_status()}. VOA is {self.voa}.")
        return

    def laser_off(self):
        ''' Turns laser off.

            Parameters
            ----------
            None

            Returns
            -------
            None
        '''
        resp = self.send_command('LASer:POWer: 0')
        time.sleep(1.5)
        print(f"Laser set to {self.laser_status()}.")
        return

    def laser_status(self):
        ''' Returns on/off status of laser.

            Parameters
            ----------
            None

            Returns
            -------
            resp : str
                'on' or 'off' to indicate laser status.
        '''
        resp = self.send_command('LASer:POWer?')
        resp = dict_status[resp]
        return resp
    
    def voa_on(self):
        ''' Turns VOA on.

            Parameters
            ----------
            None

            Returns
            -------
            None
        '''
        resp = self.send_command("VOA:POWer: 1")
        time.sleep(1)
        print(f"VOA set to {self.voa} dB.")
        return

    def set_voa(self, atten):
        ''' Sets laser attenuation.

            Parameters
            ----------
            atten : float
                Attenuation in dB. Must be between 0.5 and 20.0.

            Returns
            -------
            resp : str
                'on' or 'off' to indicate laser status.
        '''
        if (atten >= 0.5) * (atten <= 20):
            resp = self.send_command(f"VOA:ATTen: {atten}")
            time.sleep(1)
            print(f"VOA set to {self.voa} dB.")
        else:
            raise ValueError(f"Attenuation must be between 0.5 and 20. Received: {atten}.")
        return
    
    def close(self):
        ''' Closes serial communication with laser.
            
            Parameters
            ----------
            None

            Returns
            -------
            None
        '''
        self.laser_off()
        self.ser.close()
        print('Connection to TLX2 laser closed.')
        return

    @property
    def voa(self):
        ''' Property for laser's variable optical attenuation.'''
        resp = self.send_command('VOA:ATTen?')
        return float(resp)

