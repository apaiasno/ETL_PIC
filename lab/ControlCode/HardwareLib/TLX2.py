# TLX2.py
# 
# Library for defining TLX class and routines for communicating with the TLX2
# 1.55 micron laser. Based on Yoo Jung's original class definition.
# Modified by: Anusha, 12/3/2025 
#
# Tuning notes, established by probing the instrument:
#
#   * The laser tunes by ITU channel, not by wavelength. SYStem:WAVElength only
#     accepts the value it is already at, and there is no frequency setter.
#     LASer:CHANnel: n is the tuning command, with n from 1 to 93.
#   * Channels are spaced 50 GHz apart, about 0.42 nm at 1590 nm. Channel 45 is
#     188550000 MHz, channel 1 is 1608.7602 nm and channel 93 is 1570.0050 nm.
#   * LASer:FINE: n applies a fine frequency offset in MHz, +/- 30000, filling
#     in between grid points. The offset is absolute, not cumulative.
#   * LASer:FREQ_NOMinal? reports the actual frequency in MHz including any
#     offset; LASer:FREQuency? reports the channel frequency in units of
#     0.01 GHz, excluding it. SYStem:WAVElength? rounds to whole nanometres and
#     is not used here.
#   * There is no error queue: SYSTem:ERRor? is undefined. A rejected command
#     is only visible as an 'Err:' string in its reply, and an accepted one
#     returns '1'.

import serial
import time

import logging
_log = logging.getLogger('TLX2')

COM_PORT = 'COM1'
dict_status = {'1\r': 'on', '0\r': 'off'}

VOA_MAX, VOA_MIN = 20, 1                        # dB
LASER_ON_DELAY = 10                             # seconds
LASER_OFF_DELAY = 2                             # seconds

C_MHZ_NM = 2.99792458e11                        # nm * MHz
CHANNEL_MIN, CHANNEL_MAX = 1, 93                # ITU channels this unit accepts
CHANNEL_REF, FREQ_REF_MHZ = 45, 188550000       # measured anchor
CHANNEL_STEP_MHZ = 50000                        # 50 GHz grid
FINETUNE_MIN, FINETUNE_MAX = -30000, 30000      # MHz, +/- 30 GHz
TUNE_DELAY = 10                                 # seconds
FINETUNE_DELAY = 2                              # seconds
SERIAL_TIMEOUT = 5.0                            # seconds to wait for a reply


WAV_SCALE = 1e-5                                # LASer:WAVE_NOMinal? units, nm


class TLX:

    def __init__(self, com_port=COM_PORT):
        ''' Upon initialization of class instance, opens serial communication with laser if open is set to True.

            Parameters
            ----------
            com_port : str
                Name of COM port for communication with laser. Default: COM_PORT

            Returns
            -------
            None
        '''
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
        self.ser = serial.Serial(
                    port=com_port,      # Change to your COM port
                    baudrate=115200,
                    bytesize=serial.EIGHTBITS,
                    parity=serial.PARITY_NONE,
                    stopbits=serial.STOPBITS_ONE,
                    timeout=3         # seconds
                )
        _log.info('Connected to TLX2 laser.')
        return

    def send_command(self, cmd, timeout=SERIAL_TIMEOUT):
        ''' Sends a SCPI-style serial command and reads the response.

            Reads until the instrument's terminator rather than waiting a fixed
            time. A fixed wait is not safe here: a slow reply gets picked up by
            the next command instead, putting every response one step out of
            sync with the command that produced it.

            Parameters
            ----------
            cmd : str
                Command to send laser.
            timeout : float
                Seconds to wait for a terminated reply. Default: SERIAL_TIMEOUT

            Returns
            -------
            resp : str
                Response from laser.        
        '''
        self.ser.reset_input_buffer()           # discard anything stale
        self.ser.write((cmd + '\r\n').encode('ascii'))

        old_timeout = self.ser.timeout
        self.ser.timeout = timeout
        try:
            resp = self.ser.read_until(b'\r').decode('ascii')
        finally:
            self.ser.timeout = old_timeout

        if not resp.endswith('\r'):
            _log.warning(f'No terminated reply to "{cmd}" within {timeout} s. Got: {resp!r}')

        return resp

    def send_command_checked(self, cmd, timeout=SERIAL_TIMEOUT):
        ''' Sends a command and raises if the laser rejects it.

            The instrument has no error queue, so this is the only way a failed
            command can be detected: rejections come back as an 'Err:' string
            in the reply itself.

            Parameters
            ----------
            cmd : str
                Command to send laser.
            timeout : float
                Seconds to wait for a terminated reply. Default: SERIAL_TIMEOUT

            Returns
            -------
            resp : str
                Response from laser.
        '''
        resp = self.send_command(cmd, timeout=timeout)

        if resp.strip().startswith('Err'):
            raise RuntimeError(f'Laser rejected "{cmd}": {resp.strip()}')

        return resp

    ### Laser on/off ###

    def laser_on(self):
        ''' Turns laser on.

            Parameters
            ----------
            None

            Returns
            -------
            None
        '''
        self.send_command_checked('LASer:POWer: 1')
        time.sleep(LASER_ON_DELAY)
        _log.debug(f"Laser set to {self.laser_status()}. VOA is {self.voa}.")
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
        self.send_command_checked('LASer:POWer: 0')
        time.sleep(LASER_OFF_DELAY)
        _log.debug(f"Laser set to {self.laser_status()}.")
        return

    def laser_status(self):
        ''' Returns on/off status of laser.

            Parameters
            ----------
            None

            Returns
            -------
            status : str
                'on' or 'off' to indicate laser status, or 'unknown' if the
                laser returned something unexpected.
        '''
        resp = self.send_command('LASer:POWer?')

        # .get rather than [] : an empty or malformed reply would otherwise
        # raise KeyError from inside a debug log line, taking down whatever
        # called it.
        status = dict_status.get(resp, 'unknown')

        if status == 'unknown':
            _log.warning(f'Unexpected laser status reply: {resp!r}')

        return status

    ### Wavelength tuning ###

    @staticmethod
    def channel_to_wavelength(channel):
        ''' Converts an ITU channel number to its wavelength.

            Parameters
            ----------
            channel : int
                ITU channel number

            Returns
            -------
            wav : float
                Wavelength in nm, excluding any fine tuning offset
        '''
        return C_MHZ_NM / (FREQ_REF_MHZ + (channel - CHANNEL_REF) * CHANNEL_STEP_MHZ)

    @staticmethod
    def wavelength_to_channel(wav):
        ''' Converts a wavelength to the nearest ITU channel number.

            Parameters
            ----------
            wav : float
                Wavelength in nm

            Returns
            -------
            channel : int
                Nearest ITU channel number, which may fall outside the tunable
                range
        '''
        return CHANNEL_REF + int(round((C_MHZ_NM / wav - FREQ_REF_MHZ) / CHANNEL_STEP_MHZ))

    @staticmethod
    def channel_to_frequency(channel):
        ''' Converts an ITU channel number to its frequency.

            Parameters
            ----------
            channel : int
                ITU channel number

            Returns
            -------
            freq : float
                Channel frequency in MHz, excluding any fine tuning offset
        '''
        return FREQ_REF_MHZ + (channel - CHANNEL_REF) * CHANNEL_STEP_MHZ

    def set_channel(self, channel):
        ''' Tunes the laser to an ITU channel.

            Parameters
            ----------
            channel : int
                ITU channel number, CHANNEL_MIN to CHANNEL_MAX

            Returns
            -------
            wav : float
                Wavelength in nm the laser actually tuned to
        '''
        if (channel < CHANNEL_MIN) or (channel > CHANNEL_MAX):
            raise ValueError(f"Channel must be between {CHANNEL_MIN} and {CHANNEL_MAX}. Received: {channel}")

        channel = int(channel)
        self.send_command_checked(f"LASer:CHANnel: {channel}")
        time.sleep(TUNE_DELAY)

        # Verified against frequency rather than LASer:CHANnel?, which has been
        # observed to keep reporting the previous channel after a successful
        # tune. FREQ_NOMinal includes any fine tuning offset, so compare against
        # the channel frequency with a tolerance of half a grid step.
        expected = self.channel_to_frequency(channel)
        actual = self.freq

        if abs(actual - expected) > CHANNEL_STEP_MHZ / 2:
            raise RuntimeError(
                f"Requested channel {channel} ({expected/1e6:.3f} THz) but laser "
                f"is at {actual/1e6:.3f} THz.")

        wav = self.wav
        _log.debug(f"Laser tuned to channel {channel}, {wav:.4f} nm.")
        return wav

    def fine_tune(self, offset):
        ''' Applies a fine tuning frequency offset from the current ITU channel.

            The offset is absolute rather than cumulative: calling this with 0
            returns the laser to the exact grid point.

            Parameters
            ----------
            offset : int
                Frequency offset in MHz, FINETUNE_MIN to FINETUNE_MAX

            Returns
            -------
            wav : float
                Wavelength in nm the laser actually tuned to
        '''
        if (offset < FINETUNE_MIN) or (offset > FINETUNE_MAX):
            raise ValueError(f"Fine tuning offset must be between {FINETUNE_MIN} and {FINETUNE_MAX} MHz. Received: {offset} MHz.")

        self.send_command_checked(f"LASer:FINE: {int(round(offset))}")
        time.sleep(FINETUNE_DELAY)

        wav = self.wav
        _log.debug(f"Fine tuning offset set to {offset} MHz, now at {wav:.4f} nm.")
        return wav

    def laser_tune(self, wav, fine=True):
        ''' Tunes the laser to a requested wavelength.

            With fine=True the laser first moves to the nearest ITU channel and
            then applies a fine tuning offset, which reaches any wavelength in
            range to within about 1e-5 nm. Half a grid step is 25 GHz and fine
            tuning covers +/- 30 GHz, so no wavelength falls between the two.
            Fine tuning also extends the range slightly past the edge channels.

            With fine=False the laser goes to the nearest grid point and the
            offset is cleared, so the wavelength reached can be up to half a
            grid step, about 0.21 nm, from the one requested.

            Parameters
            ----------
            wav : float
                Requested wavelength in nm
            fine : bool
                Whether to apply a fine tuning offset to reach wav exactly.
                Default: True

            Returns
            -------
            wav_actual : float
                Wavelength in nm the laser actually tuned to
        '''
        target_mhz = C_MHZ_NM / wav
        channel = self.wavelength_to_channel(wav)

        if not fine:
            if (channel < CHANNEL_MIN) or (channel > CHANNEL_MAX):
                wav_min = self.channel_to_wavelength(CHANNEL_MAX)
                wav_max = self.channel_to_wavelength(CHANNEL_MIN)
                raise ValueError(f"Wavelength must be between {wav_min:.4f} and {wav_max:.4f} nm without fine tuning. Received: {wav} nm.")

            self.set_channel(channel)
            return self.fine_tune(0)

        # Clamp to the tunable grid: a wavelength just outside the edge
        # channels is still reachable using the edge channel plus an offset.
        channel = max(CHANNEL_MIN, min(CHANNEL_MAX, channel))
        offset = int(round(target_mhz - self.channel_to_frequency(channel)))

        if (offset < FINETUNE_MIN) or (offset > FINETUNE_MAX):
            wav_min = C_MHZ_NM / (self.channel_to_frequency(CHANNEL_MAX) + FINETUNE_MAX)
            wav_max = C_MHZ_NM / (self.channel_to_frequency(CHANNEL_MIN) + FINETUNE_MIN)
            raise ValueError(f"Wavelength must be between {wav_min:.4f} and {wav_max:.4f} nm. Received: {wav} nm.")

        if self.channel != channel:
            self.set_channel(channel)

        wav_actual = self.fine_tune(offset)
        _log.debug(f"Requested {wav} nm, reached {wav_actual:.5f} nm on channel {channel} with {offset:+d} MHz offset.")
        return wav_actual


    ### Variable optical attenuator ###

    def voa_on(self):
        ''' Turns VOA on.

            Parameters
            ----------
            None

            Returns
            -------
            None
        '''
        self.send_command_checked("VOA:POWer: 1")
        time.sleep(1)
        _log.debug(f"VOA on, set to {self.voa} dB.")
        return

    def set_voa(self, atten):
        ''' Sets laser attenuation.

            Parameters
            ----------
            atten : float
                Attenuation in dB. Must be between VOA_MIN and VOA_MAX.

            Returns
            -------
            None
        '''
        if (atten < VOA_MIN) or (atten > VOA_MAX):
            raise ValueError(f"Attenuation must be between {VOA_MIN} and {VOA_MAX}. Received: {atten} dB.")

        self.send_command_checked(f"VOA:ATTen: {atten}")
        time.sleep(1)
        _log.debug(f"VOA set to {self.voa} dB.")
        return

    def close(self):
        ''' Resets laser frequency to middle of L-band. Closes serial communication with laser.
            
            Parameters
            ----------
            None

            Returns
            -------
            None
        '''
        self.set_channel(45)
        self.fine_tune(0)
        try:
            self.laser_off()
        except Exception as e:
            _log.warning(f'Could not turn laser off before closing: {type(e).__name__}: {e}')

        self.ser.close()
        _log.info('Connection to TLX2 laser closed.')
        return

    ### Properties ###

    @property
    def voa(self):
        ''' Property for laser's variable optical attenuation, in dB.'''
        return float(self.send_command('VOA:ATTen?'))

    @property
    def channel(self):
        ''' Property for laser's ITU channel number.'''
        return int(self.send_command('LASer:CHANnel?'))

    @property
    def freq(self):
        ''' Property for laser's actual frequency in MHz, including any fine
            tuning offset.'''
        return float(self.send_command('LASer:FREQ_NOMinal?'))

    @property
    def freq_channel(self):
        ''' Property for the frequency of the laser's ITU channel in MHz,
            excluding any fine tuning offset.

            LASer:FREQuency? reports in units of 0.01 GHz, hence the factor
            of 100.
        '''
        return float(self.send_command('LASer:FREQuency?')) * 100.0

    @property
    def finetune(self):
        ''' Property for the fine tuning offset in MHz.

            Derived from the actual frequency against the arithmetic channel
            frequency. LASer:FREQuency? is not used: like LASer:CHANnel?, it has
            been observed to keep reporting a previous value after a tune.
        '''
        ch = CHANNEL_REF + int(round((self.freq - FREQ_REF_MHZ) / CHANNEL_STEP_MHZ))
        return self.freq - self.channel_to_frequency(ch)

    @property
    def wav(self):
        ''' Property for laser's output wavelength in nm.

            Read directly from LASer:WAVE_NOMinal?, which reports in units of
            1e-5 nm and includes any fine tuning offset. SYStem:WAVElength? is
            not used: it rounds to whole nanometres, which hides the 0.42 nm
            channel spacing entirely, and made a failing laser_tune look
            successful.
        '''
        return float(self.send_command('LASer:WAVE_NOMinal?')) * WAV_SCALE