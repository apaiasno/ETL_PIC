# XDAC.py
#
# Library for defining XDAC class and routines for communicating with the XDAC
# breadboard over ZMQ. Successor to XPOW.py; the public interface deliberately
# mirrors it so that existing Bench code needs minimal changes.
#
# CHANNELS ARE 1-INDEXED (1 to 120), matching XPOW. The wire protocol is also
# 1-indexed, so channel numbers pass through unchanged.
#
# Differences from XPOW worth knowing:
#   * All 120 channels can be set in a single command (~0.37 s) instead of one
#     at a time (~146 ms each, ~17 s for a full pass). See apply_voltage_all().
#   * All 120 readings arrive on a broadcast socket in ~262 ms instead of
#     ~146 ms per channel. See read_all().
#   * Hardware ranges (SETRV/SETRC) MUST be configured before setpoints take
#     effect. open() does this on every connection.
#   * ZERO clears a channel's current setpoint as well as its voltage, so a
#     channel that has been turned off will acknowledge a voltage command and
#     source nothing until a current is re-applied. turn_on/turn_on_all
#     restore both.
#   * Failures raise instead of retrying forever. XPOW's `while error:` loops
#     could spin indefinitely on a dead device; here a timeout means the ZMQ
#     REQ socket is desynced and the process must be restarted.
#   * Sockets are opened once and closed once. Reopening a REQ connection to
#     this device has been observed to leave it unresponsive on all ports
#     until a power cycle, so close() is final.

#### IMPORTS ####
import threading
import time
import warnings

import zmq

import logging
_log = logging.getLogger('XDAC')

import numpy as np

warnings.filterwarnings('ignore')

ATTEN_CHANNEL = [80, 60]
ATTEN_MAXVOLTAGE = 4.65
ATTEN_MAXCURRENT = 4.5                            # (mA) Compliance ceiling for the optical attenuator
XDAC_IP = "169.254.54.201"                      # change based on device network settings
XDAC_KEY = "nicslab"
XDACReqPort = 5555
XDACSubPort = 5556
XDACChannelMax = 120
XDACVoltageControlMax = 13.1
XDACVoltageControlMin = 0
XDACCurrentMax = 150
XDACCurrentMin = 0
XDACVoltageRangeIndex = 2                       # 0-20 V, covers XDACVoltageMax
XDACCurrentRangeIndex = 7                       # 0-300 mA, covers XDACCurrentMax
XDACVoltageDelay = 0.5                          # (sec) Delay after each XDAC voltage setting
XDACErrorDelay = 1                              # (sec) Delay between re-tries of sending XDAC commands if error encountered
XDACCommandDelay = 0.05
XDACMaxRetries = 2                              # Re-tries before giving up (XPOW retried forever)
XDACCommandTimeout = 5000                       # (ms) Reply timeout for a single command
XDACBroadcastTimeout = 3000                     # (ms) Wait for a broadcast message
XDACJoinDelay = 0.3                             # (sec) ZMQ slow-joiner window after subscribing
XDACMaxDrain = 1000                             # Most queued broadcasts to discard when finding the newest
XDACSockets = {}
XDACErrorNum = 0

_reqLock = threading.Lock()


class XDACError(RuntimeError):
    ''' Raised when the XDAC replies FALSE, i.e. it rejected the command. '''
    pass


#### XDAC CLASS ####

class XDAC:

    def __init__(self):
        ''' Upon initialization of class instance:
            1. ZMQ connection to XDAC breadboard.
            2. Unlocks the session.
            3. Configures hardware ranges on all channels.
            4. Resets all channels.

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
        ''' Opens connection to XDAC and resets all channels to 0 V.

            Also initializes set_voltages and set_currents, which track what
            was commanded on each channel. Measured values are not stored:
            the readout_voltages and readout_currents properties query the
            device on every access.

            Parameters
            ----------
            None

            Returns
            -------
            None
        '''
        XDAC._XDACOpen()
        XDAC._XDACUnlock()
        XDAC._XDACSetAllRanges()
        XDAC._XDACResetAllChannels()

        # Setpoints commanded on each channel, indexed from 0, so that
        # set_voltages[0] is channel 1. These record what was sent, which is
        # not what the hardware measures: this unit reads a few percent below
        # setpoint, by a different amount on each channel. They also go stale
        # if anything else touches the device, such as the vendor GUI.
        self.set_voltages = [0.0] * XDACChannelMax
        self.set_currents = [float(XDACCurrentMax)] * XDACChannelMax
        for channel in ATTEN_CHANNEL:
            self.set_currents[channel - 1] = float(ATTEN_MAXCURRENT)

        # Snapshot taken by turn_off / turn_off_all so that turn_on can put
        # things back. ZERO clears current setpoints as well as voltages, so
        # both have to be restored.
        self._saved_voltages = list(self.set_voltages)
        self._saved_currents = list(self.set_currents)

        _log.info('Connected to XDAC.')
        return

    #### XDAC LIVE READOUT PROPERTIES ####
    #
    # These query the device every time they are accessed, so each one costs a
    # broadcast round-trip of roughly 260 ms. Two consequences worth keeping
    # in mind:
    #
    #   * dac.readout_voltages[5] reads all 120 channels and throws away 119
    #     of them. Indexing in a loop is very slow -- bind the list once
    #     (v = dac.readout_voltages) and index that instead.
    #   * Reading voltages and currents separately is two round-trips, and
    #     they come from different broadcast messages. Use read_all() when you
    #     need both, or need them close together in time.

    @property
    def readout_voltages(self):
        ''' Measured voltage of every channel, read fresh from the device.

            Indexed from 0, so that readout_voltages[0] is channel 1. Each
            access costs roughly 260 ms.

            Returns
            -------
            voltages : list of float
                Measured voltage of each channel in V, channel 1 to 120
        '''
        return XDAC._XDACReadBroadcast(b'V')

    @property
    def readout_currents(self):
        ''' Measured current of every channel, read fresh from the device.

            Indexed from 0, so that readout_currents[0] is channel 1. Each
            access costs roughly 260 ms.

            Returns
            -------
            currents : list of float
                Measured current of each channel in mA, channel 1 to 120
        '''
        return XDAC._XDACReadBroadcast(b'C')

    @property
    def readout_powers(self):
        ''' Measured power of every channel, read fresh from the device.

            Indexed from 0, so that readout_powers[0] is channel 1. This is
            two round-trips, roughly 520 ms, since voltage and current arrive
            as separate broadcasts.

            Returns
            -------
            powers : list of float
                Measured power of each channel in mW, channel 1 to 120
        '''
        voltages = self.readout_voltages
        currents = self.readout_currents
        return [v * c for v, c in zip(voltages, currents)]

    ### SETTING FUNCTIONS ###
    def apply_voltage(self, channel, voltage):
        ''' Applies voltage to a channel.

            Parameters
            ----------
            channel : int
                Channel of interest, 1 to 120
            voltage : float
                Voltage to assign channel of interest

            Returns
            -------
            None
        '''
        channel = XDAC._XDACCheckChannel(channel)
        XDAC._XDACCheckVoltage(voltage, channel)
        applied = XDAC._XDACChannelAdjustVoltage(channel, voltage)
        self.set_voltages[channel - 1] = applied
        _log.debug(f"Channel {channel} set to {voltage} V.")
        return

    def apply_current(self, channel, current):
        ''' Applies a current setpoint to a channel.

            A channel will not source voltage unless its current setpoint is
            large enough to drive the load, so this is normally set once to
            XDACCurrentMax and left alone.

            Parameters
            ----------
            channel : int
                Channel of interest, 1 to 120
            current : float
                Current to assign channel of interest, in mA

            Returns
            -------
            None
        '''
        channel = XDAC._XDACCheckChannel(channel)
        XDAC._XDACCheckCurrent(current, channel)
        applied = XDAC._XDACChannelAdjustCurrent(channel, current)
        self.set_currents[channel - 1] = applied
        _log.debug(f"Channel {channel} set to {current} mA.")
        return

    def apply_voltage_all(self, voltages):
        ''' Applies voltages to every channel in a single command.

            Roughly 45 times faster than looping over apply_voltage, at the
            cost of not being able to report which channel failed.

            Parameters
            ----------
            voltages : list of float or float
                Voltage for each channel, ordered channel 1 to 120. A single
                number applies that voltage to every channel.

            Returns
            -------
            None
        '''
        voltages = XDAC._XDACExpand(voltages)

        for idx, voltage in enumerate(voltages):
            XDAC._XDACCheckVoltage(voltage, idx + 1)

        msg = "SETMULTIA;:" + ":".join(f"V{i + 1}:{v:.3f}" for i, v in enumerate(voltages)) + ";"
        XDAC._XDACRequest(msg)
        self.set_voltages = list(voltages)
        time.sleep(XDACVoltageDelay)
        _log.debug(f"All {XDACChannelMax} channels set, max {max(voltages)} V.")
        return

    def apply_current_all(self, currents):
        ''' Applies current setpoints to every channel in a single command.

            Parameters
            ----------
            currents : list of float or float
                Current for each channel in mA, ordered channel 1 to 120. A
                single number applies that current to every channel.

            Returns
            -------
            None
        '''
        currents = XDAC._XDACExpand(currents)

        for idx, current in enumerate(currents):
            XDAC._XDACCheckCurrent(current, idx + 1)

        # Note the ';;:' prefix and absent trailing ';': the current form of
        # this command differs from the voltage form above. Both are verbatim
        # from the vendor reference and neither tolerates other delimiters.
        msg = "SETMULTIA;;:" + ":".join(f"C{i + 1}:{c:.3f}" for i, c in enumerate(currents))
        XDAC._XDACRequest(msg)
        self.set_currents = list(currents)
        _log.debug(f"All {XDACChannelMax} channels set, max {max(currents)} mA.")
        return

    ### READOUT FUNCTIONS ###

    def read_channel(self, channel):
        ''' Readout voltage, current, and power of a channel.

            Parameters
            ----------
            channel : int
                Channel of interest, 1 to 120

            Returns
            -------
            actualvoltage : float
                Readout voltage of channel, in V
            current : float
                Readout current of channel, in mA
            power : float
                Readout power of channel, in mW
        '''
        actualvoltage, current, power = XDAC._XDACChannelGetData(channel)
        return actualvoltage, current, power

    def read_all(self):
        ''' Readout voltage, current, and power of every channel at once.

            Roughly 65 times faster per channel than looping over read_channel.
            Values come from the device's broadcast stream, so they carry 3
            decimal places for voltage and 2 for current, against 5 for
            read_channel. The voltage and current readings arrive as separate
            broadcasts and so are not from precisely the same instant.

            Parameters
            ----------
            None

            Returns
            -------
            voltages : list of float
                Readout voltage of each channel in V, ordered channel 1 to 120
            currents : list of float
                Readout current of each channel in mA, ordered channel 1 to 120
            powers : list of float
                Readout power of each channel in mW, ordered channel 1 to 120
        '''
        voltages = XDAC._XDACReadBroadcast(b'V')
        currents = XDAC._XDACReadBroadcast(b'C')
        powers = [v * c for v, c in zip(voltages, currents)]
        return np.vstack([voltages, currents, powers]).T
    
    def get_set_voltage(self, channel):
        ''' Returns the voltage setpoint most recently commanded on a channel.

            This is what was asked for, not what the hardware measures. Use
            get_readout_voltage or read_channel for a measurement.

            Parameters
            ----------
            channel : int
                Channel of interest, 1 to 120

            Returns
            -------
            voltage : float
                Commanded voltage of the channel, in V
        '''
        XDAC._XDACCheckChannel(channel)
        return self.set_voltages[channel - 1]

    def get_set_current(self, channel):
        ''' Returns the current setpoint most recently commanded on a channel.

            Parameters
            ----------
            channel : int
                Channel of interest, 1 to 120

            Returns
            -------
            current : float
                Commanded current of the channel, in mA
        '''
        XDAC._XDACCheckChannel(channel)
        return self.set_currents[channel - 1]

    def get_readout_voltage(self, channel):
        ''' Reads the measured voltage of a single channel from the device.

            Uses a single-channel command rather than the broadcast, so this
            costs roughly 140 ms and carries 5 decimal places, against 3 for
            readout_voltages.

            Parameters
            ----------
            channel : int
                Channel of interest, 1 to 120

            Returns
            -------
            voltage : float
                Measured voltage of the channel, in V
        '''
        XDAC._XDACCheckChannel(channel)
        return float(XDAC._XDACRequestRetry("MEASV:" + str(channel)))

    def get_readout_current(self, channel):
        ''' Reads the measured current of a single channel from the device.

            Uses a single-channel command rather than the broadcast, so this
            costs roughly 140 ms and carries 5 decimal places, against 2 for
            readout_currents.

            Parameters
            ----------
            channel : int
                Channel of interest, 1 to 120

            Returns
            -------
            current : float
                Measured current of the channel, in mA
        '''
        XDAC._XDACCheckChannel(channel)
        return float(XDAC._XDACRequestRetry("MEASC:" + str(channel)))

    ### CHANNEL ACTIVE/OFF FUNCTIONS ###

    def reset_all_channels(self):
            ''' Resets all channels to 0 V and returns current setpoints to maximum.
    
                ZERO:ALL clears current setpoints as well as voltages, so the
                currents are explicitly re-applied afterwards.
    
                Parameters
                ----------
                None
    
                Returns
                -------
                None
            '''
            XDAC._XDACResetAllChannels()
            self.set_voltages = [0.0] * XDACChannelMax
            self.set_currents = [float(XDACCurrentMax)] * XDACChannelMax
            for channel in ATTEN_CHANNEL:
                self.set_currents[channel - 1] = float(ATTEN_MAXCURRENT)
            return

    def turn_off(self, channel):
        ''' Turns off a single channel.

            ZERO clears the channel's current setpoint as well as its voltage,
            so the channel cannot source again until a current is re-applied.
            A voltage command on its own will be acknowledged and do nothing.
            Use turn_on to restore both.

            Parameters
            ----------
            channel : int
                Channel of interest, 1 to 120

            Returns
            -------
            None
        '''
        channel = XDAC._XDACCheckChannel(channel)

        self._saved_voltages[channel - 1] = self.set_voltages[channel - 1]
        self._saved_currents[channel - 1] = self.set_currents[channel - 1]

        XDAC._XDACRequestRetry("ZERO:" + str(channel))
        self.set_voltages[channel - 1] = 0.0
        self.set_currents[channel - 1] = 0.0
        time.sleep(XDACVoltageDelay)
        _log.debug(f"Channel {channel} turned off.")
        return

    def turn_off_all(self):
        ''' Turns off every channel.

            ZERO:ALL clears every current setpoint as well as every voltage,
            so no channel can source again until currents are re-applied. Use
            turn_on_all to restore what was set beforehand.

            Parameters
            ----------
            None

            Returns
            -------
            None
        '''
        self._saved_voltages = list(self.set_voltages)
        self._saved_currents = list(self.set_currents)

        XDAC._XDACRequestRetry("ZERO:ALL")
        self.set_voltages = [0.0] * XDACChannelMax
        self.set_currents = [0.0] * XDACChannelMax
        time.sleep(XDACVoltageDelay)
        _log.info('All XDAC channels turned off.')
        return

    def turn_on(self, channel):
        ''' Restores the setpoints a channel had before it was turned off.

            Current is restored before voltage, since a channel with no
            compliance will accept a voltage and source nothing.

            Parameters
            ----------
            channel : int
                Channel of interest, 1 to 120

            Returns
            -------
            None
        '''
        channel = XDAC._XDACCheckChannel(channel)

        current = self._saved_currents[channel - 1]
        voltage = self._saved_voltages[channel - 1]

        if current <= 0:
            _log.warning(
                f"Channel {channel} had no saved current setpoint, so it will "
                f"not source. Applying {XDACCurrentMax} mA instead."
            )
            current = float(ATTEN_MAXCURRENT) if np.isin(channel, ATTEN_CHANNEL) else float(XDACCurrentMax)

        self.apply_current(channel, current)
        self.apply_voltage(channel, voltage)
        _log.debug(f"Channel {channel} turned back on at {voltage} V, {current} mA.")
        return

    def turn_on_all(self):
        ''' Restores the setpoints every channel had before turn_off_all.

            Currents are restored before voltages, since a channel with no
            compliance will accept a voltage and source nothing.

            Parameters
            ----------
            None

            Returns
            -------
            None
        '''
        currents = list(self._saved_currents)

        # A channel saved with no compliance would silently fail to source, so
        # give it the default ceiling instead.
        for idx, current in enumerate(currents):
            if current <= 0:
                currents[idx] = float(ATTEN_MAXCURRENT) if np.isin((idx + 1), ATTEN_CHANNEL) else float(XDACCurrentMax)

        self.apply_current_all(currents)
        self.apply_voltage_all(list(self._saved_voltages))
        _log.info('All XDAC channels turned back on.')
        return

    ### DEVICE CONNECTION FUNCTIONS ###

    def lock(self):
        ''' Locks the session. The XDAC then rejects commands until unlock().

            Parameters
            ----------
            None

            Returns
            -------
            None
        '''
        XDAC._XDACRequest("LOCK")
        _log.info('XDAC session locked.')
        return

    def unlock(self):
        ''' Unlocks the session.

            Parameters
            ----------
            None

            Returns
            -------
            info : str
                Device info string, e.g. OK:120:20:U:1
        '''
        return XDAC._XDACUnlock()

    def close(self):
        ''' Resets all channels to 0 V and closes connection to XDAC.

            This is final. Reopening a connection in the same process has been
            observed to leave the device unresponsive until a power cycle, so
            restart the interpreter to reconnect.

            Parameters
            ----------
            None

            Returns
            -------
            None
        '''
        try:
            XDAC._XDACResetAllChannels()
            self.set_voltages = [0.0] * XDACChannelMax
            self.set_currents = [float(XDACCurrentMax)] * XDACChannelMax
            for channel in ATTEN_CHANNEL:
                self.set_currents[channel - 1] = float(ATTEN_MAXCURRENT)
            XDAC._XDACRequest("LOCK")
        except Exception as e:
            _log.warning(f"Could not reset/lock before closing: {type(e).__name__}: {e}")

        XDAC._XDACClearSockets()
        _log.info('Connection to XDAC closed.')
        return

    #### XDAC STATIC METHODS ####

    @staticmethod
    def _XDACExpand(values):
        """
        Function for turning a scalar or a sequence into a full channel list.

        Args:
            values: float or list  - one value, or one value per channel
        """

        if isinstance(values, (int, float)):
            return [float(values)] * XDACChannelMax

        values = [float(v) for v in values]

        if (len(values) != XDACChannelMax):
            raise ValueError("ERROR: Expected " + str(XDACChannelMax) + " values, got " + str(len(values)) + "!")

        return values

    @staticmethod
    def _XDACCheckChannel(channel):
        """
        Function for validating an XDAC channel number. Channels are 1-indexed.

        Rejects non-integers: a float or a bool would otherwise be formatted
        straight into the command, sending e.g. "SETV:3.5:2.000".

        Args:
            channel: int    - XDAC channel
        """

        if isinstance(channel, bool):
            raise ValueError("ERROR: XDAC channel must be an integer, got " + repr(channel) + "!")

        if not isinstance(channel, int):
            # numpy integers satisfy __index__ without being Python ints
            try:
                channel = channel.__index__()
            except (AttributeError, TypeError):
                raise ValueError("ERROR: XDAC channel must be an integer, got " + repr(channel) + "!")

        if (channel < 1) or (channel > XDACChannelMax):
            raise ValueError("ERROR: Invalid XDAC channel " + str(channel) + "!")

        return channel

    @staticmethod
    def _XDACCheckVoltage(voltage, channel):
        """
        Function for validating a voltage before it is sent.

        NaN is rejected explicitly: every comparison against NaN is False, so
        it would otherwise slip past the range guards and reach the device as
        "SETV:3:nan".

        The channel is required, not optional: the attenuator guard below
        depends on it, and a caller that omitted it would get a weaker check
        that still looked like validation.

        Args:
            voltage: float  - voltage to validate
            channel: int    - XDAC channel the voltage is destined for
        """

        if voltage != voltage:
            raise ValueError(f'Voltage cannot be NaN. Channel {channel}.')

        if voltage < XDACVoltageControlMin:
            raise ValueError(f'Voltage applied cannot be negative. Channel {channel} received: {voltage} V')

        if voltage > XDACVoltageControlMax:
            raise ValueError(f'Voltage applied cannot exceed {XDACVoltageControlMax} V. Channel {channel} received: {voltage} V')

        if (np.isin(channel, ATTEN_CHANNEL)) and (voltage > ATTEN_MAXVOLTAGE):
            raise ValueError(f'Voltage applied to optical attenuator cannot exceed {ATTEN_MAXVOLTAGE} V. Received: {voltage} V')

    @staticmethod
    def _XDACCheckCurrent(current, channel):
        """
        Function for validating a current before it is sent.

        The channel is required for the same reason as in _XDACCheckVoltage.

        Args:
            current: float  - current to validate, in mA
            channel: int    - XDAC channel the current is destined for
        """

        if current != current:
            raise ValueError(f'Current cannot be NaN. Channel {channel}.')

        if current < XDACCurrentMin:
            raise ValueError(f'Current applied cannot be negative. Channel {channel} received: {current} mA')

        if current > XDACCurrentMax:
            raise ValueError(f'Current applied cannot exceed {XDACCurrentMax} mA. Channel {channel} received: {current} mA')

        if (np.isin(channel, ATTEN_CHANNEL)) and (current > ATTEN_MAXCURRENT):
            raise ValueError(f'Current applied to optical attenuator cannot exceed {ATTEN_MAXCURRENT} mA. Received: {current} mA')

    @staticmethod
    def _XDACRequest(cmd, raise_on_false=True):
        """
        Function for sending a single command to the XDAC and returning its reply.

        Args:
            cmd: str             - command to send
            raise_on_false: bool - whether a FALSE reply raises XDACError
        """

        if 'req' not in XDACSockets:
            raise RuntimeError("ERROR: XDAC is not open!")

        _log.debug("Sent XDAC command: \"" + cmd + "\".")

        with _reqLock:
            try:
                XDACSockets['req'].send(cmd.encode('utf-8'))
                reply = XDACSockets['req'].recv().decode('utf-8')
            except zmq.Again:
                raise TimeoutError(
                    "XDAC did not reply to \"" + cmd + "\" within "
                    + str(XDACCommandTimeout) + " ms. The REQ socket is now "
                    "desynced, so restart the interpreter. If the device is "
                    "also silent on the broadcast port, it needs a power cycle."
                )

        _log.debug("XDAC answer = {" + reply + "}")

        if raise_on_false and (reply.strip().upper() == 'FALSE'):
            raise XDACError(
                "XDAC rejected \"" + cmd + "\" (replied FALSE). The session is "
                "most likely locked or not unlocked with a valid key."
            )

        return reply

    @staticmethod
    def _XDACRequestRetry(cmd):
        """
        Function for sending a command, re-unlocking and retrying if the XDAC
        rejects it. A rejection almost always means the session locked itself,
        which unlock() recovers. Unlike XPOW this gives up after
        XDACMaxRetries rather than looping forever.

        Args:
            cmd: str    - command to send
        """

        global XDACErrorNum

        for attempt in range(XDACMaxRetries + 1):
            try:
                return XDAC._XDACRequest(cmd)

            except XDACError:
                if (attempt == XDACMaxRetries):
                    raise

                XDACErrorNum += 1
                _log.error("\n-- ERROR " + str(XDACErrorNum) + " ENCOUNTERED, TRYING AGAIN... --")
                time.sleep(XDACErrorDelay)
                XDAC._XDACUnlock()

    @staticmethod
    def _XDACUnlock():
        """
        Function for unlocking the XDAC session.
        """

        _log.info('Unlocking connection to XDAC...')
        reply = XDAC._XDACRequest("GETINFO:" + XDAC_KEY)

        fields = reply.split(":")

        if (len(fields) > 1):
            try:
                reported = int(fields[1])
            except ValueError:
                pass
            else:
                if (reported != XDACChannelMax):
                    _log.warning("XDAC reports " + str(reported) + " channels but XDACChannelMax is " + str(XDACChannelMax) + "!")

        return reply

    @staticmethod
    def _XDACChannelGetData(channel):
        """
        Function for acquiring the voltage and current readings of a given XDAC channel.

        Args:
            channel: int    - XDAC channel
        """

        XDAC._XDACCheckChannel(channel)

        voltage = float(XDAC._XDACRequestRetry("MEASV:" + str(channel)))
        current = float(XDAC._XDACRequestRetry("MEASC:" + str(channel)))
        power = voltage * current

        return voltage, current, power

    @staticmethod
    def _XDACChannelAdjustVoltage(channel, voltage):
        """
        Function for setting a certain XDAC channel to a certain voltage.

        Args:
            channel: int    - XDAC channel
            voltage: float  - voltage to set XDAC channel to

        Returns the voltage actually sent, which may have been clamped.
        """

        XDAC._XDACCheckChannel(channel)

        if (voltage < XDACVoltageControlMin):
            voltage = XDACVoltageControlMin
        if (voltage > XDACVoltageControlMax):
            voltage = XDACVoltageControlMax

        XDAC._XDACRequestRetry("SETV:" + str(channel) + ":" + ("%.3f" % voltage))
        time.sleep(XDACVoltageDelay)

        return float(voltage)

    @staticmethod
    def _XDACChannelAdjustCurrent(channel, current):
        """
        Function for setting a certain XDAC channel to a certain current.

        Args:
            channel: int    - XDAC channel
            current: float  - current to set XDAC channel to

        Returns the current actually sent, which may have been clamped.
        """

        XDAC._XDACCheckChannel(channel)

        if (current < XDACCurrentMin):
            current = XDACCurrentMin
        if (current > XDACCurrentMax):
            current = XDACCurrentMax

        XDAC._XDACRequestRetry("SETC:" + str(channel) + ":" + ("%.3f" % current))

        return float(current)

    @staticmethod
    def _XDACChannelAdjustVoltageRange(channel, rangeIdx):
        """
        Function for setting the hardware voltage range of a given XDAC channel.
        Setpoints have no effect until this has been configured.

        Voltage Control Range Index
        Index 0 = 0 V - 5 V
        Index 1 = 0 V - 10 V
        Index 2 = 0 V - 20 V
        Index 3 = 0 V - 40 V

        Args:
            channel: int    - XDAC channel
            rangeIdx: int   - range index, 0 to 3
        """

        XDAC._XDACCheckChannel(channel)

        if rangeIdx not in range(0, 4):
            raise ValueError("ERROR: Invalid XDAC voltage range index " + str(rangeIdx) + "!")

        XDAC._XDACRequestRetry("SETRV:" + str(channel) + ":" + str(rangeIdx))

    @staticmethod
    def _XDACChannelAdjustCurrentRange(channel, rangeIdx):
        """
        Function for setting the hardware current range of a given XDAC channel.
        Setpoints have no effect until this has been configured.

        Current Control Range Index
        Index 0 = 0 mA - 3.125 mA
        Index 1 = 0 mA - 6.25 mA
        Index 2 = 0 mA - 12.5 mA
        Index 3 = 0 mA - 25 mA
        Index 4 = 0 mA - 50 mA
        Index 5 = 0 mA - 100 mA
        Index 6 = 0 mA - 200 mA
        Index 7 = 0 mA - 300 mA

        Args:
            channel: int    - XDAC channel
            rangeIdx: int   - range index, 0 to 7
        """

        XDAC._XDACCheckChannel(channel)

        if rangeIdx not in range(0, 8):
            raise ValueError("ERROR: Invalid XDAC current range index " + str(rangeIdx) + "!")

        XDAC._XDACRequestRetry("SETRC:" + str(channel) + ":" + str(rangeIdx))

    @staticmethod
    def _XDACSetAllRanges():
        """
        Function for configuring the hardware voltage and current range on every
        channel. Takes roughly 35 seconds, since it is 240 separate commands.

        Setpoints have no effect until ranges are configured: a channel with no
        range acknowledges SETV and sources nothing. Ranges appear to survive
        within a session, but this has not been checked across a power cycle,
        so open() reconfigures them every time rather than assuming.
        """

        _log.info("Setting XDAC channel ranges...")

        channel = 1

        while (channel <= XDACChannelMax):
            XDAC._XDACChannelAdjustVoltageRange(channel, XDACVoltageRangeIndex)
            XDAC._XDACChannelAdjustCurrentRange(channel, XDACCurrentRangeIndex)

            channel += 1

        _log.info("XDAC channel ranges set!")

    @staticmethod
    def _XDACReadBroadcast(topic, timeout_ms=XDACBroadcastTimeout):
        """
        Function for reading the newest broadcast of a given topic and parsing
        it into a list of floats, one per channel.

        The payload is '<topic>v1,0.000,v2,0.000,...;', so every second field
        is padding. The socket queues broadcasts continuously from the moment
        it subscribes, so everything already queued is drained and only the
        newest message kept, otherwise the reading would be stale.

        Args:
            topic: bytes        - b'V' or b'C'
            timeout_ms: int     - how long to wait for a message
        """

        sock = XDACSockets.get(topic)

        if sock is None:
            raise ValueError("ERROR: Unknown XDAC broadcast topic " + str(topic) + "!")

        raw = None
        drained = 0

        # The drain is bounded: the device publishes every 150-300 ms so a
        # handful of messages is normal, but an unbounded loop would hang if
        # the publisher ever outran it.
        while sock.poll(0) and (drained < XDACMaxDrain):
            raw = sock.recv(zmq.NOBLOCK)
            drained += 1

        if raw is None:
            if not sock.poll(timeout_ms):
                raise TimeoutError(
                    "No " + topic.decode() + " broadcast received from the XDAC within "
                    + str(timeout_ms) + " ms."
                )

            raw = sock.recv(zmq.NOBLOCK)
            drained = 0

            while sock.poll(0) and (drained < XDACMaxDrain):
                raw = sock.recv(zmq.NOBLOCK)
                drained += 1

        fields = raw.decode('utf-8')[1:-1].split(",")
        values = [float(x) for x in fields[0::2]]

        if (len(values) != XDACChannelMax):
            raise ValueError(
                "ERROR: Parsed " + str(len(values)) + " values from a "
                + str(len(fields)) + "-field " + topic.decode() + " broadcast, expected "
                + str(XDACChannelMax) + "! The payload format may have changed."
            )

        return values

    @staticmethod
    def _XDACMeasurementConfig(averagingSample, voltageConv, currentConv):
        """
        Function for setting the XDAC measurement averaging and conversion time.
        UNTESTED on this unit. Fewer samples and shorter conversion times
        should raise the broadcast rate at the cost of noise.

        Average Sample Number Index
        Index 0 = 1 Sample
        Index 1 = 4 Samples
        Index 2 = 16 Samples
        Index 3 = 64 Samples
        Index 4 = 128 Samples
        Index 5 = 256 Samples
        Index 6 = 512 Samples
        Index 7 = 1024 Samples

        V and I ADC Conversion Time Index
        Index 0 = 50 uS
        Index 1 = 84 uS
        Index 2 = 150 uS
        Index 3 = 280 uS
        Index 4 = 540 uS
        Index 5 = 1052 uS
        Index 6 = 2074 uS
        Index 7 = 4120 uS

        Args:
            averagingSample: int    - averaging index, 0 to 7
            voltageConv: int        - voltage conversion time index, 0 to 7
            currentConv: int        - current conversion time index, 0 to 7
        """

        for name, value in (("averaging sample", averagingSample),
                            ("voltage conversion", voltageConv),
                            ("current conversion", currentConv)):
            if value not in range(0, 8):
                raise ValueError("ERROR: Invalid XDAC " + name + " index " + str(value) + "!")

        return XDAC._XDACRequest("CONFIG:" + str(averagingSample) + ":" + str(voltageConv) + ":" + str(currentConv))

    @staticmethod
    def _XDACResetAllChannels():
        """
        Function for zeroing every XDAC channel and setting every current
        setpoint to maximum, so that channels are ready to source voltage.
        """

        XDAC._XDACRequest("ZERO:ALL")
        time.sleep(XDACVoltageDelay)

        # Every channel gets maximum compliance so it is ready to source,
        # except the optical attenuator, which is capped at ATTEN_MAXCURRENT.
        currents = [float(XDACCurrentMax)] * XDACChannelMax
        for channel in ATTEN_CHANNEL:
            currents[channel - 1] = float(ATTEN_MAXCURRENT)

        msg = "SETMULTIA;;:" + ":".join(f"C{i + 1}:{c:.3f}" for i, c in enumerate(currents))
        XDAC._XDACRequest(msg)

        _log.info("XDAC channels reset!")

    @staticmethod
    def _XDACCreateSockets():
        """
        Function for creating the XDAC request and broadcast sockets.

        One SUB socket per topic: a single socket carrying both SUBSCRIBE
        filters was tried and delivered only the 'C' topic on this firmware.
        """

        context = zmq.Context()
        XDACSockets['context'] = context

        req = context.socket(zmq.REQ)
        req.setsockopt(zmq.RCVTIMEO, XDACCommandTimeout)
        req.setsockopt(zmq.SNDTIMEO, XDACCommandTimeout)
        req.setsockopt(zmq.LINGER, 0)
        req.connect("tcp://%s:%d" % (XDAC_IP, XDACReqPort))
        XDACSockets['req'] = req

        for topic in (b'V', b'C'):
            sub = context.socket(zmq.SUB)
            sub.connect("tcp://%s:%d" % (XDAC_IP, XDACSubPort))
            sub.setsockopt(zmq.SUBSCRIBE, topic)
            XDACSockets[topic] = sub

        # ZMQ slow joiner: a subscription takes a moment to reach the
        # publisher, so the first recv after connecting can miss messages
        # already in flight.
        time.sleep(XDACJoinDelay)

        _log.info("XDAC sockets created!")

    @staticmethod
    def _XDACClearSockets():
        """
        Function for closing every XDAC socket and terminating the ZMQ context.
        """

        for key in (b'V', b'C', 'req'):
            sock = XDACSockets.pop(key, None)

            if sock is not None:
                try:
                    sock.close(linger=0)
                except Exception:
                    pass

                time.sleep(XDACCommandDelay)

        context = XDACSockets.pop('context', None)

        if context is not None:
            try:
                context.term()
            except Exception:
                pass

        _log.info("XDAC sockets cleared!")

    @staticmethod
    def _XDACOpen():
        """
        Function for opening the connection to the XDAC.
        """

        _log.debug("------------------------------------------\nChosen XDAC address: " + XDAC_IP + "\n------------------------------------------\n")

        if 'req' in XDACSockets:
            raise RuntimeError(
                "ERROR: XDAC is already open! Reopening a connection in the "
                "same process has been observed to leave the device "
                "unresponsive until a power cycle. Restart the interpreter "
                "instead."
            )

        XDAC._XDACCreateSockets()