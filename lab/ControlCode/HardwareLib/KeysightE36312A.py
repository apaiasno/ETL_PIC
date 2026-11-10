# KeysightE36312A.py
# 
# Library for defining KEYSIGHTE36312A class and routines for communicating with
# the Keysight E36312A power supply.
# Modified by: Anusha, 6/24/2026
#
# Method names follow XDAC.py where the operation is the same, so that the two
# instrument libraries read alike:
#
#   apply_voltage(channel, voltage)     set one channel  (NOTE: channel first)
#   apply_voltage_all(voltages)         set every channel
#   read_channel(channel)               -> voltage, current, power
#   read_all()                          -> three lists, one entry per channel
#   readout_voltages / _currents / _powers    live properties, read on access
#   get_readout_voltage(channel)        single-channel measurement
#   get_set_voltage(channel)            what was last commanded
#   turn_on / turn_off / turn_on_all / turn_off_all
#   reset_all_channels()                zero every channel
#   close()
#
# set_voltage(voltage, channel), read_voltage, read_current, output_on and
# output_off are kept as aliases so existing Bench code keeps working. Note the
# argument order differs: apply_voltage takes the channel first, matching XDAC,
# while set_voltage takes the voltage first, as this class always has.

import pyvisa
import time

import logging
_log = logging.getLogger('KeysightE36312A')
logging.getLogger("pyvisa").setLevel(logging.WARNING)

VOLTAGE_MIN, VOLTAGE_MAX_1, VOLTAGE_MAX_2, VOLTAGE_MAX_3 = 0, 6, 16, 16
CURRENT_MIN, CURRENT_MAX = 0, 0.5
INSTRUMENT_ADDRESS = 'USB0::0x2A8D::0x1102::MY61003343::INSTR'
CHANNEL = 2
CHANNELS = (1, 2, 3)
CHANNELS_ON = [2, 3]

# Per-channel voltage ceilings, keyed by channel number.
VOLTAGE_MAX = {1: VOLTAGE_MAX_1, 2: VOLTAGE_MAX_2, 3: VOLTAGE_MAX_3}

# Over-voltage protection is armed above the working range, not at it. With
# OVP set equal to the maximum allowed setpoint, any overshoot on the way to
# that setpoint trips the protection, which latches and shuts the output off
# until it is explicitly cleared.
OVP_HEADROOM = 1.0                  # (V) How far above the channel maximum to arm OVP
VOLTAGE_TOLERANCE = 0.1             # (V) Mismatch that apply_voltage will warn about


class KEYSIGHTE36312A :

    def __init__(self, instrument_address=INSTRUMENT_ADDRESS, channels_on=CHANNELS_ON):
        ''' Upon initialization of class instance:
            1. Opens VISA communication with the power supply.
            2. Sets current and voltage limits on every channel.
            3. Zeroes every channel, so nothing is energized by a setpoint left
               over from a previous session.
            4. Enables the outputs listed in channels_on.

            Parameters
            ----------
            instrument_address : str
                Instrument address for communication with power supply. Default: INSTRUMENT_ADDRESS
            channels_on : list of int
                Channels whose outputs to enable on connection. Default: CHANNELS_ON

            Returns
            -------
            None
        '''
        self.open(instrument_address)

        for channel in CHANNELS:
            self.set_current_max(current_max=CURRENT_MAX, channel=channel)
            self.set_voltage_max(voltage_max=VOLTAGE_MAX[channel], channel=channel)

        # The supply remembers setpoints across power cycles, so zero every
        # channel before enabling any output. Otherwise turning an output on
        # would immediately apply whatever voltage was left set last session.
        self.reset_all_channels()

        for channel in channels_on:
            self.turn_on(channel)

        return

    ### Functions for enabling communication and channel setting ###

    def open(self, instrument_address):
        ''' Opens VISA communication with power supply.

            Parameters
            ----------
            instrument_address : str
                Instrument address for communication with power supply.

            Returns
            -------
            None
        '''
        # Keep the resource manager alive for the lifetime of the connection:
        # if it is garbage collected the session can be closed underneath us.
        self.rm = pyvisa.ResourceManager()
        self.visa = None

        # Voltage most recently commanded on each channel, and the current limit
        # most recently set. These record what was sent, which is not
        # necessarily what the hardware is doing: a tripped or current-limited
        # channel will measure something else entirely.
        self.set_voltages = {channel: 0.0 for channel in CHANNELS}
        self.set_current_limits = {channel: 0.0 for channel in CHANNELS}

        try:
            self.visa = self.rm.open_resource(instrument_address)

            self.visa.baud_rate = 9600
            self.visa.data_bits = 8
            self.visa.stop_bits = pyvisa.constants.StopBits.one
            self.visa.parity = pyvisa.constants.Parity.none

            self.visa.read_termination = '\n'
            self.visa.write_termination = '\n'
            self.visa.timeout = 5000          # ms

            idn = self.visa.query('*IDN?').strip()

        except Exception:
            # Do not leave a half-open session behind, and do not report success.
            if self.visa is not None:
                try:
                    self.visa.close()
                except Exception:
                    pass
                self.visa = None
            _log.error(f'Could not connect to power supply at {instrument_address}.')
            raise

        self.idn = idn
        _log.info(f'Connected to Keysight E36312A power supply: {idn}')
        return

    def turn_on(self, channel=CHANNEL):
        ''' Enables the output of a channel.

            Without this a voltage setpoint depends on the front-panel output
            state and can silently do nothing.

            Parameters
            ----------
            channel : int
                Channel to enable. Default: CHANNEL

            Returns
            -------
            None
        '''
        self._check_channel(channel)
        self.visa.write(f'OUTP ON,(@{channel})')
        _log.debug(f'Channel {channel} output enabled.')
        return

    def turn_off(self, channel=CHANNEL):
        ''' Disables the output of a channel.

            The setpoint is left alone, so turn_on restores whatever was set.

            Parameters
            ----------
            channel : int
                Channel to disable. Default: CHANNEL

            Returns
            -------
            None
        '''
        self._check_channel(channel)
        self.visa.write(f'OUTP OFF,(@{channel})')
        _log.debug(f'Channel {channel} output disabled.')
        return

    def turn_on_all(self):
        ''' Enables the output of every channel.

            Parameters
            ----------
            None

            Returns
            -------
            None
        '''
        for channel in CHANNELS:
            self.turn_on(channel)
        return

    def turn_off_all(self):
        ''' Disables the output of every channel.

            Setpoints are left alone, so turn_on_all restores what was set.

            Parameters
            ----------
            None

            Returns
            -------
            None
        '''
        for channel in CHANNELS:
            self.turn_off(channel)
        return

    def is_output_on(self, channel=CHANNEL):
        ''' Checks whether a channel's output is enabled.

            Parameters
            ----------
            channel : int
                Channel to query. Default: CHANNEL

            Returns
            -------
            enabled : bool
                Whether the channel output is on
        '''
        self._check_channel(channel)
        enabled = bool(int(self.visa.query(f'OUTP? (@{channel})')))
        return enabled

    ### Protection functions ###

    def is_tripped(self, channel=CHANNEL):
        ''' Checks whether a channel's over-voltage protection has tripped.

            A trip latches: the output shuts off and stays off, while voltage
            writes are still accepted and report no error. This is the check
            that distinguishes that state from a channel that is simply set
            to zero.

            Parameters
            ----------
            channel : int
                Channel to query. Default: CHANNEL

            Returns
            -------
            tripped : bool
                Whether protection is latched on the channel
        '''
        self._check_channel(channel)
        tripped = bool(int(self.visa.query(f'VOLT:PROT:TRIP? (@{channel})')))
        return tripped

    def clear_protection(self, channel=CHANNEL):
        ''' Clears a latched over-voltage protection trip on a channel.

            Raising the OVP level does not clear an existing trip, so this has
            to be called explicitly after one occurs.

            Parameters
            ----------
            channel : int
                Channel to clear. Default: CHANNEL

            Returns
            -------
            None
        '''
        self._check_channel(channel)
        self.visa.write(f'OUTP:PROT:CLE (@{channel})')
        _log.info(f'Cleared protection trip on channel {channel}.')
        return

    ### Readout functions ###

    def get_readout_voltage(self, channel=CHANNEL):
        ''' Readout voltage of channel.

            Parameters
            ----------
            channel : int
                Channel to readout voltage. Default: CHANNEL

            Returns
            -------
            voltage : float
                Measured voltage of channel, in V
        '''
        self._check_channel(channel)
        voltage = float(self.visa.query(f'MEAS:VOLT? (@{channel})'))
        return voltage    

    def get_readout_current(self, channel=CHANNEL):
        ''' Readout current of channel.

            Parameters
            ----------
            channel : int
                Channel to readout current. Default: CHANNEL

            Returns
            -------
            current : float
                Measured current of channel, in A
        '''
        self._check_channel(channel)
        current = float(self.visa.query(f'MEAS:CURR? (@{channel})'))
        return current

    def get_set_voltage(self, channel=CHANNEL):
        ''' Returns the voltage most recently commanded on a channel.

            This is what was asked for, not what the hardware measures. Use
            get_readout_voltage for a measurement.

            Parameters
            ----------
            channel : int
                Channel of interest. Default: CHANNEL

            Returns
            -------
            voltage : float
                Commanded voltage of the channel, in V
        '''
        self._check_channel(channel)
        return self.set_voltages[channel]

    def read_channel(self, channel=CHANNEL):
        ''' Readout voltage, current, and power of a channel.

            Parameters
            ----------
            channel : int
                Channel of interest. Default: CHANNEL

            Returns
            -------
            voltage : float
                Measured voltage of channel, in V
            current : float
                Measured current of channel, in A
            power : float
                Measured power of channel, in W
        '''
        voltage = self.get_readout_voltage(channel)
        current = self.get_readout_current(channel)
        power = voltage * current
        return voltage, current, power

    def read_all(self):
        ''' Readout voltage, current, and power of every channel.

            Parameters
            ----------
            None

            Returns
            -------
            voltages : list of float
                Measured voltage of each channel in V, ordered channel 1 to 3
            currents : list of float
                Measured current of each channel in A, ordered channel 1 to 3
            powers : list of float
                Measured power of each channel in W, ordered channel 1 to 3
        '''
        voltages = [self.get_readout_voltage(channel) for channel in CHANNELS]
        currents = [self.get_readout_current(channel) for channel in CHANNELS]
        powers = [v * c for v, c in zip(voltages, currents)]
        return voltages, currents, powers

    ### Live readout properties ###
    #
    # These query the supply every time they are accessed, one VISA round-trip
    # per channel.

    @property
    def readout_voltages(self):
        ''' Measured voltage of every channel, read fresh from the supply.

            Returns
            -------
            voltages : list of float
                Measured voltage of each channel in V, ordered channel 1 to 3
        '''
        return [self.get_readout_voltage(channel) for channel in CHANNELS]

    @property
    def readout_currents(self):
        ''' Measured current of every channel, read fresh from the supply.

            Returns
            -------
            currents : list of float
                Measured current of each channel in A, ordered channel 1 to 3
        '''
        return [self.get_readout_current(channel) for channel in CHANNELS]

    @property
    def readout_powers(self):
        ''' Measured power of every channel, read fresh from the supply.

            Returns
            -------
            powers : list of float
                Measured power of each channel in W, ordered channel 1 to 3
        '''
        voltages = self.readout_voltages
        currents = self.readout_currents
        return [v * c for v, c in zip(voltages, currents)]

    ### Setting functions ###

    def set_current_max(self, current_max=CURRENT_MAX, channel=CHANNEL):
        ''' Set maximum current of channel.

            Parameters
            ----------
            current_max : float
                Maximum current to set. Default: CURRENT_MAX
            channel : int
                Channel to set maximum current. Default: CHANNEL

            Returns
            -------
            None       
        '''
        self._check_channel(channel)

        if not CURRENT_MIN <= current_max <= CURRENT_MAX:
            raise ValueError(f'Current limit must be between {CURRENT_MIN} A and {CURRENT_MAX} A. Received: {current_max} A')

        self.visa.write(f'SOUR:CURR {current_max},(@{channel})')
        self.set_current_limits[channel] = float(current_max)
        _log.debug(f"Channel {channel} current limit set to {current_max} A.")
        return 

    def set_voltage_max(self, voltage_max=VOLTAGE_MAX_2, channel=CHANNEL):
        ''' Set maximum voltage of channel.

            Over-voltage protection is armed OVP_HEADROOM above voltage_max,
            not at it: arming OVP at the highest allowed setpoint means any
            overshoot while settling trips the protection, which latches and
            shuts the output off until clear_protection() is called.

            Parameters
            ----------
            voltage_max : float
                Maximum voltage to set. Default: VOLTAGE_MAX_2
            channel : int
                Channel to set maximum voltage. Default: CHANNEL

            Returns
            -------
            None       
        '''
        self._check_channel(channel)

        ovp_level = voltage_max + OVP_HEADROOM
        self.visa.write(f'VOLT:PROT {ovp_level},(@{channel})')
        self.visa.write(f'VOLT:PROT:STAT ON,(@{channel})')
        _log.debug(f"Channel {channel} voltage limit set to {voltage_max} V, OVP armed at {ovp_level} V.")
        return 

    def apply_voltage(self, channel=CHANNEL, voltage=0.0, verify=True):
        ''' Applies voltage to a channel.

            Note the argument order: channel first, matching XDAC.apply_voltage.
            The set_voltage alias takes the voltage first.

            Parameters
            ----------
            channel : int
                Channel to set voltage. Default: CHANNEL
            voltage : float
                Voltage to set, in V
            verify : bool
                Whether to read the voltage back and warn if it does not match
                the setpoint, which catches a latched protection trip, a
                disabled output, or the channel running in constant-current
                mode. Default: True

            Returns
            -------
            None       
        '''
        self._check_channel(channel)
        self._check_voltage(voltage, channel)

        self.visa.write(f'SOUR:VOLT {voltage},(@{channel})')
        self.set_voltages[channel] = float(voltage)

        if not verify:
            _log.debug(f"Channel {channel} voltage set to {voltage} V.")
            return

        measured = self.get_readout_voltage(channel)

        if abs(measured - voltage) > VOLTAGE_TOLERANCE:
            # Work out why, so the log says something useful rather than just
            # reporting a number that does not match what was asked for.
            reasons = []

            if self.is_tripped(channel):
                reasons.append('over-voltage protection is latched, call clear_protection()')
            if not self.is_output_on(channel):
                reasons.append('output is disabled, call turn_on()')

            current = self.get_readout_current(channel)
            limit = self.set_current_limits[channel] or CURRENT_MAX
            if current >= limit * 0.98:
                reasons.append(f'channel is current limited at {current:.4f} A')

            detail = '; '.join(reasons) if reasons else 'cause unknown'
            _log.warning(
                f"Channel {channel} set to {voltage} V but measures "
                f"{measured:.4f} V ({detail})."
            )
        else:
            _log.debug(f"Channel {channel} voltage set to {measured} V.")

        return    

    def apply_voltage_all(self, voltages, verify=True):
        ''' Applies voltages to every channel.

            Parameters
            ----------
            voltages : list of float or float
                Voltage for each channel, ordered channel 1 to 3. A single
                number applies that voltage to every channel.
            verify : bool
                Whether to read each voltage back and warn on a mismatch.
                Default: True

            Returns
            -------
            None
        '''
        if isinstance(voltages, (int, float)):
            voltages = [float(voltages)] * len(CHANNELS)

        voltages = [float(v) for v in voltages]

        if len(voltages) != len(CHANNELS):
            raise ValueError(f'Expected {len(CHANNELS)} values, got {len(voltages)}')

        # Validate everything before sending anything, so a bad entry does not
        # leave some channels set and others not.
        for channel, voltage in zip(CHANNELS, voltages):
            self._check_voltage(voltage, channel)

        for channel, voltage in zip(CHANNELS, voltages):
            self.apply_voltage(channel=channel, voltage=voltage, verify=verify)

        return

    def set_voltage(self, voltage, channel=CHANNEL, verify=True):
        ''' Sets the voltage of a channel. Alias for apply_voltage that takes
            the voltage first, kept for existing Bench code.

            Parameters
            ----------
            voltage : float
                Voltage to set, in V
            channel : int
                Channel to set voltage. Default: CHANNEL
            verify : bool
                Whether to read the voltage back and warn on a mismatch.
                Default: True

            Returns
            -------
            None
        '''
        return self.apply_voltage(channel=channel, voltage=voltage, verify=verify)

    def reset_all_channels(self):
        ''' Sets every channel to 0 V.

            Outputs and current limits are left alone.

            Parameters
            ----------
            None

            Returns
            -------
            None
        '''
        # verify=False: on a channel whose output is off, or one that has
        # tripped, a read-back of 0 V is expected and not worth warning about.
        for channel in CHANNELS:
            self.apply_voltage(channel=channel, voltage=0.0, verify=False)

        _log.info('All power supply channels set to 0 V.')
        return

    def close(self):
        ''' Sets every channel to 0 V and closes VISA communication with the
            Keysight E36312A power supply.

            Parameters
            ----------
            None

            Returns
            -------
            None
        '''
        try:
            self.reset_all_channels()
        except Exception as e:
            _log.warning(f'Could not zero channels before closing: {type(e).__name__}: {e}')

        self.visa.close()
        _log.info('Connection to Keysight E36312A power supply closed.')
        return

    ### Validation ###

    @staticmethod
    def _check_channel(channel):
        ''' Validates a channel number.

            Without this, a bad channel is written straight to the supply:
            SOUR:VOLT 5,(@7) is accepted by the SCPI parser and does nothing.

            Parameters
            ----------
            channel : int
                Channel to validate

            Returns
            -------
            None
        '''
        if channel not in CHANNELS:
            raise ValueError(f'Channel must be one of {list(CHANNELS)}. Received: {channel}')
        return

    @staticmethod
    def _check_voltage(voltage, channel):
        ''' Validates a voltage against the ceiling for a given channel.

            NaN is rejected explicitly: every comparison against NaN is False,
            so it would otherwise pass the range checks and be written to the
            supply as "SOUR:VOLT nan".

            Parameters
            ----------
            voltage : float
                Voltage to validate, in V
            channel : int
                Channel the voltage is destined for

            Returns
            -------
            None
        '''
        if voltage != voltage:
            raise ValueError(f'Voltage cannot be NaN. Channel {channel}.')

        ceiling = VOLTAGE_MAX[channel]

        if voltage < VOLTAGE_MIN or voltage > ceiling:
            raise ValueError(
                f'Channel {channel} voltage must be between {VOLTAGE_MIN} V and '
                f'{ceiling} V. Received: {voltage} V'
            )
        return

    @property
    def voltage(self):
        ''' Property for voltage of modulating channel.'''
        return self.get_readout_voltage()

    @property
    def current(self):
        ''' Property for current of modulating channel.'''
        return self.get_readout_current()