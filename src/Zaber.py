# Zaber.py 
#
# Library for defining Zaber class and routines for controlling Zaber translation motors.
# Modified: Anusha, 12/3/25

import numpy as np
import time

import logging
_log = logging.getLogger('Zaber')

from zaber_motion import Units, UnitTable
from zaber_motion.ascii import Connection

COM_PORT = "COM5"
UNIT = Units.LENGTH_MILLIMETRES
UNIT_VELOCITY = Units.VELOCITY_MILLIMETRES_PER_SECOND
BACKLASH_OVERSHOOT = 0.6 # linear motion of 1 motor revolution for X-NA08A25-E09/X-NA08A25-SE09

class ZABER:
   
    def __init__(self, com_port=COM_PORT):
        ''' Upon initialization of class instance:
            1. Serial connection to Zaber devices is opened. Sets connection and devices attributes.
            2. Prints current positions of devices.

            Parameters
            ----------
            com_port : str
                Name of COM port for communication with Zabers. Default: COM_PORT

            Returns
            -------
            None
        '''
        self.unit = UNIT
        self.unit_velocity = UNIT_VELOCITY
        self.open(com_port)
        self.get_positions(print_flag=True)
        return

    def open(self, com_port=COM_PORT):
        ''' Opens serial connection to Zaber devices.

            Parameters
            ----------
            com_port : str
                Name of COM port for communication with Zabers. Default: COM_PORT

            Returns
            -------
            None        
        '''            
        self.connection = Connection.open_serial_port(com_port)
        self.connection.enable_alerts()
        self.devices = self.connection.detect_devices()
        _log.info("Found {} devices".format(len(self.devices)))
        return
    
    def home(self, device_ind):
        ''' Homes the specified device.
            
            Parameters
            ----------
            device_ind : int
                Integer index addressing device to be homed.

            Returns
            -------
            None
        '''
        device = self.devices[device_ind]
        axis = device.get_axis(1)
        if not axis.is_homed():
            _log.info(f"Homing device {device_ind}...")
            axis.home()
            _log.info(f"Done homing device {device_ind}.")
        else:
             _log.info(f"Device {device_ind} is already homed.")
        return
    
    def get_positions(self, unit=UNIT, print_flag=False):
        ''' Returns the position of all Zaber devices connected.
            
            Parameters
            ----------
            unit : Units object
                Unit in which to return device positions.
            print : bool
                True to print positions of each device; else False. Default: False

            Returns
            -------
            positions : 1D numpy array
        '''
        positions = []
        for device_ind, device in enumerate(self.devices):
            axis = device.get_axis(1)
            position = axis.get_position(unit=unit)
            positions.append(position)
        if print_flag:
            _log.info("Positions:")
            [_log.info(f"Device {device_ind}: {position} {UnitTable.get_symbol(unit)}") for device_ind, position in enumerate(positions)];
        return np.array(positions)
    
    def move_absolute(self, device_ind, position, unit=UNIT, velocity=0, velocity_unit=UNIT_VELOCITY):
        ''' Moves a device to an absolute position. Anti-backlash correction applied to all (absolute) positions greater than BACKLASH_OVERSHOOT.
            
            Parameters
            ----------
            device_ind : int
                Integer index addressing device to be moved.
            position : float
                Target absolute position for the device.
            unit : Units object
                Unit in which absolute position is given. Default: UNIT
            velocity : float
                Velocity at which to move device. Default: 0 (corresponds to Zaber default velocity)
            velocity_unit : Units object
                Unit in which velocity is given. Default: UNIT_VELOCITY

            Returns
            -------
            None
        '''
        # Convert position to default units
        position = UnitTable.convert_units(position, unit, self.unit)

        # Get axis
        axis = self.devices[device_ind].get_axis(1)           
        previous = self.positions[device_ind]

        # Move device, overshooting for anti-backlash if moving backwards
        _log.debug(f"Moving device {device_ind}...")
        if (position < previous): # anti-backlash
            try:
                axis.move_absolute(position-BACKLASH_OVERSHOOT, self.unit, velocity=velocity, velocity_unit=velocity_unit)
                axis.move_absolute(position, self.unit, velocity=velocity, velocity_unit=velocity_unit)
            except:
                axis.move_absolute(position, self.unit, velocity=velocity, velocity_unit=velocity_unit)
        else:
            axis.move_absolute(position, self.unit, velocity=velocity, velocity_unit=velocity_unit)

        new = self.positions[device_ind]        
        _log.debug(f"Done moving device {device_ind} from {UnitTable.convert_units(previous, self.unit, unit)} {UnitTable.get_symbol(unit)} to {UnitTable.convert_units(new, self.unit, unit)} {UnitTable.get_symbol(unit)}.")
        return
    
    def move_relative(self, device_ind, position, unit=UNIT, velocity=0, velocity_unit=UNIT_VELOCITY):
        ''' Moves a device by a relative amount. Anti-backlash correction applied to all (absolute) positions greater than BACKLASH_OVERSHOOT.
            
            Parameters
            ----------
            device_ind : int
                Integer index addressing device to be moved.
            position : float
                Target relative position for the device.
            unit : Units object
                Unit in which relative position is given.
            velocity : float
                Velocity at which to move device. Default: 0 (corresponds to Zaber default velocity)
            velocity_unit : Units object
                Unit in which velocity is given. Default: UNIT_VELOCITY

            Returns
            -------
            None
        '''
        # Convert position to default units
        position = UnitTable.convert_units(position, unit, self.unit)

        # Get axis
        axis = self.devices[device_ind].get_axis(1)           
        previous = self.positions[device_ind]
        
        # Move device, overshooting for anti-backlash if moving backwards
        _log.debug(f"Moving device {device_ind}...")
        if (position < previous): # anti-backlash
            try:
                axis.move_relative(position-BACKLASH_OVERSHOOT, self.unit, velocity=velocity, velocity_unit=velocity_unit)
                axis.move_relative(BACKLASH_OVERSHOOT, self.unit, velocity=velocity, velocity_unit=velocity_unit)
            except:
                axis.move_relative(position, self.unit, velocity=velocity, velocity_unit=velocity_unit)
        else:
            axis.move_relative(position, self.unit, velocity=velocity, velocity_unit=velocity_unit)
        
        new = self.positions[device_ind]
        _log.debug(f"Done moving device {device_ind} from {UnitTable.convert_units(previous, self.unit, unit)} {UnitTable.get_symbol(unit)} to {UnitTable.convert_units(new, self.unit, unit)} {UnitTable.get_symbol(unit)}.")
        return
        
    # def move_velocity(self, device_ind, position, velocity, wait, unit=UNIT, unit_velocity=UNIT_VELOCITY):
    #     ''' Moves a device by a specific velocity to an absolute position.
            
    #         Parameters
    #         ----------
    #         device_ind : int
    #             Integer index addressing device to be moved.
    #         position : float
    #             Target absolute position for device 
    #         velocity : float
    #             Velocity at which to move device, can be positive or negative 
    #         wait : float
    #             Wait time in seconds for assessing device position as it moves
    #         unit : Units object
    #             Unit in which absolute position is given. Default: UNIT
    #         unit_velocity : Units object
    #             Unit in which velocity is given. Default: UNIT_VELOCITY

    #         Returns
    #         -------
    #         positions : 1D numpy array
    #             Array of positions for each device.
    #     '''

    #     previous = self.positions[device_ind]
    #     axis = self.devices[device_ind].get_axis(1)           
        
    #     _log.debug(f"Moving device {device_ind}...")
    #     tracker = 0
    #     if velocity > 0:
    #         axis.move_velocity(velocity, unit_velocity)
    #         while self.positions[device_ind] < position:
    #             time.sleep(wait)
    #             tracker += 1
    #             if tracker == 10:
    #                 _log.debug(f"Device {device_ind}: {self.positions[device_ind]} {unit.name}")
    #                 tracker = 0
    #         axis.stop()
    #     elif velocity < 0:
    #         axis.move_absolute(velocity, unit_velocity)
    #         while self.positions[device_ind] > position:
    #             time.sleep(wait)
    #             tracker += 1
    #             if tracker == 10:
    #                 _log.debug(f"Device {device_ind}: {self.positions[device_ind]} {unit.name}")
    #                 tracker = 0
    #         axis.stop()
    #     else:
    #         raise ValueError(f"Inputs must be a nonzero, real value. Received: {position} {unit.name}, {velocity} {unit_velocity.name}")
        
    #     new = self.positions[device_ind]
    #     _log.debug(f"Done moving device {device_ind} from {previous} {self.unit.name} to {new} {self.unit.name}.")
    #     return

            # # Move by an additional 5mm
            # axis.move_relative(5, Units.LENGTH_MILLIMETRES)

    def close(self):
        ''' Closes connection to Zabers.

            Parameters
            ----------
            None

            Returns
            -------
            None        
        '''
        self.connection.close()
        _log.info('Connection to Zabers closed.')
        return

    @property
    def positions(self):
        '''Property for the position of every connected device.'''
        return self.get_positions(unit=self.unit)
