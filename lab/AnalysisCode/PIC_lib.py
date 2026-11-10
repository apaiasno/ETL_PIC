# PIC_lib.py
#
# Library of routines for PIC testbed.
# Modified by: Anusha, 06/09/2026 

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import imageio_ffmpeg
matplotlib.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
# import matplotlib.colors as colors
# from matplotlib.colors import ListedColormap

import sys, glob, re
sys.path.insert(0, r"D:\\Active_PIC\\ControlCode\\HardwareLib")

from Bench import BENCH

# A channel label is 'Ch69', 'KCh2', or a bare channel number from logs written
# before the Keysight was added. Anything else in the log filename is a phase or
# variant tag, such as the '_sweep' and '_track' that null_stability_track
# appends, and is not a channel.
CHANNEL_LABEL_RE = re.compile(r'^K?Ch\d+$|^\d+$')


def parse_channel_labels(filename_log, filename_root):
    ''' Extracts the channel labels a scan was run over from its log filename.

        Parts of the filename that do not look like a channel label are
        discarded, so a phase tag appended to the log name does not become a
        spurious channel.

        Parameters
        ----------
        filename_log : str
            Full path of the .log file produced by the scan
        filename_root : str
            Path and beginning identifier passed to the scan routine

        Returns
        -------
        labels : list of str
            Channel labels in scan order, e.g. ['Ch69', 'KCh2']
    '''
    channel_str = filename_log.split(filename_root + '_')[-1].split('.log')[0]
    labels = [part for part in channel_str.split('_') if CHANNEL_LABEL_RE.match(part)]

    if not labels:
        raise ValueError(
            f'No channel labels found in {filename_log}. Expected parts like '
            f"'Ch69' or 'KCh2' after the root {filename_root}.")

    return labels


def label_to_channel(label):
    ''' Converts a channel label back to a channel specification.

        Parameters
        ----------
        label : str
            Channel label, e.g. 'Ch69' or 'KCh2'

        Returns
        -------
        spec : int or tuple
            69 for an XCOM channel, ('keysight', 2) for a Keysight channel
    '''
    if label.startswith('KCh'):
        return ('keysight', int(label[3:]))
    if label.startswith('Ch'):
        return int(label[2:])
    return int(label)          # legacy bare-number label


def voltage_column(log, label):
    ''' Finds the input voltage column for a channel label.

        Accepts both the current naming ('inputVoltageCh69', 'inputVoltageKCh2')
        and the pre-Keysight naming ('inputVoltage69').

        Parameters
        ----------
        log : numpy record array
            Data log read from the scan's .log file
        label : str
            Channel label, e.g. 'Ch69' or 'KCh2'

        Returns
        -------
        column : str
            Name of the matching column in log
    '''
    candidates = [f'inputVoltage{label}']

    # Legacy: column named with the bare channel number
    digits = ''.join(c for c in label if c.isdigit())
    if digits:
        candidates.append(f'inputVoltage{digits}')

    for column in candidates:
        if column in log.dtype.names:
            return column

    raise KeyError(
        f'No input voltage column for channel {label}. Tried {candidates}. '
        f'Log has: {list(log.dtype.names)}'
    )


def read_scan_log(filename_root, v_decimals=3):
    ''' Reads a scan's data log and works out which channels it covered.

        Parameters
        ----------
        filename_root : str
            Path and beginning identifier passed to the scan routine
        v_decimals : int
            Decimal places the scan used in image filenames. Voltages are
            rounded to this before deduplicating, since that is the precision
            that determines which image file a log row corresponds to.
            Default: 3

        Returns
        -------
        log : numpy record array
            Data log contents
        labels : list of str
            Channel labels in scan order
        voltage_coords : 2D numpy array
            Unique combinations of commanded voltages actually present in the log
    '''
    filename_log = glob.glob(filename_root + "*.log")[0]
    labels = parse_channel_labels(filename_log, filename_root)

    log = np.loadtxt(filename_log, delimiter=',')
    with open(filename_log) as f:
        colnames = f.readline().strip().strip('# ').split(',')
    log = np.core.records.fromarrays(log.T, names=colnames)

    cols = [voltage_column(log, label) for label in labels]

    # Rounded before np.unique: the log holds full-precision voltages, so
    # repeated float addition in the adaptive scan produces values differing
    # in the 9th decimal that are distinct to np.unique but map to the same
    # image filename, giving duplicate rows that then reduce the same files.
    voltages = np.round(np.column_stack([log[col] for col in cols]), v_decimals)
    voltage_coords = np.unique(voltages, axis=0)

    return log, labels, voltage_coords


def build_box_masks(shape, coords, mask_range=5):
    ''' Builds square boolean masks around each output coordinate.

        Parameters
        ----------
        shape : tuple
            Shape of the image the masks apply to
        coords : 2D numpy array or None
            Coordinate pairs of output centers. If None, returns an empty list.
        mask_range : int
            Half-width in pixels of the box around each center. Default: 5

        Returns
        -------
        masks : list of 2D boolean numpy arrays
            One mask per coordinate, empty if coords is None
    '''
    masks = []

    if coords is None:
        return masks

    for coord in coords:
        mask = np.zeros(shape, bool)
        r, c = int(coord[0]), int(coord[1])
        R = mask_range
        mask[max(0, r - R):r + R + 1, max(0, c - R):c + R + 1] = 1
        masks.append(mask)

    return masks


# def reduce_voltage_scan(filename_root, num_outputs, filename_dark, filt_size=5,
#                         coords=None, find_centers=True, mask_range=5, r_max=15,
#                         r_lin=5, plot=False, v_decimals=3):
#     ''' Get output intensities for each image produced by a voltage scan.

#         Handles scans over any mix of XCOM and Keysight channels: the channels
#         are read from the log filename as labels, and the log columns are looked
#         up by label.

#         Parameters
#         ----------
#         filename_root : str
#             Path and beginning identifier of each file in voltage scan image collection
#         num_outputs : int
#             Number of outputs to identify in image
#         filename_dark : str
#             Filename of the dark frame to subtract
#         filt_size : int
#             Size of maximum filter (in pixels) for isolating local maxima to identify outputs. Default: 5
#         coords : 2D numpy array or None
#             Coordinate pairs of output centers; if None, finds output centers
#         find_centers : boolean
#             True if need to find output centers; else False. Cannot be False if coords is None.
#         mask_range : int
#             Mask radius (in pixels) around center of output for peak flux. Default: 5
#         r_max : int
#             Maximum pixel radius for photometry extraction method. Default: 15
#         r_lin : int
#             Lower limit of pixel radius at which photometry extraction method behaves linearly. Default: 5
#         plot : boolean
#             Flag for plotting masks overlayed on image and photometry extraction
#         v_decimals : int
#             Decimal places used in the voltage part of the image filenames. Default: 3

#         Returns
#         -------
#         voltage_coords : 2D numpy array
#             Voltage combination for each reduced point, columns in channel order
#         outputs : 2D numpy array
#             Output intensities at each voltage setting
#         attens : 2D numpy array
#             Attenuation in dB applied to each output at each voltage setting
#         output_fluxes : 2D numpy array
#             Peak dark-subtracted flux in each output mask, NaN if coords is None
#         labels : list of str
#             Channel labels in scan order, e.g. ['Ch69', 'KCh2']
#     '''
#     log, labels, voltage_coords = read_scan_log(filename_root)

#     dark = np.loadtxt(filename_dark, delimiter=',')
#     masks = build_box_masks(np.shape(dark), coords, mask_range=mask_range)

#     outputs = np.full((len(voltage_coords), num_outputs), np.nan)
#     attens = np.full((len(voltage_coords), num_outputs), np.nan)
#     output_fluxes = np.full((len(voltage_coords), num_outputs), np.nan)
#     output_names = [chr(65 + i) for i in range(num_outputs)]

#     for idx_i, voltage_coord in enumerate(voltage_coords):
#         v_str = ''
#         for voltage in voltage_coord:
#             v_i = f"{voltage:.{v_decimals}f}".replace('.', 'p')
#             v_str += f"*_V{v_i}"

#         for o, output_name in enumerate(output_names):
#             mode_ext = f'output{output_name}'
#             pattern = filename_root + f"{v_str}*{mode_ext}*"
#             matches = glob.glob(pattern)

#             if not matches:
#                 # A missing image is normal for an interrupted scan, so skip it
#                 # rather than failing the whole reduction.
#                 print(f'No image found for {pattern}')
#                 continue

#             filename = matches[0]
#             attens[idx_i, o] = float(
#                 filename.split(f'{mode_ext}_')[-1].split('.npy')[0].replace('p', '.')
#             )

#             data = np.load(filename)

#             if data.ndim == 2:
#                 frame = data - dark
#                 output = BENCH.extract_outputs(frame, num_outputs, r_max=r_max, r_lin=r_lin,
#                                                filt_size=filt_size, plot=plot, coords=coords,
#                                                find_centers=find_centers)
#             elif data.ndim == 3:
#                 output = []
#                 frames = []
#                 for data_i in data:
#                     frame_i = data_i - dark
#                     frames.append(frame_i)
#                     output.append(BENCH.extract_outputs(frame_i, num_outputs, r_max=r_max, r_lin=r_lin,
#                                                         filt_size=filt_size, plot=plot, coords=coords,
#                                                         find_centers=find_centers))
#                 output = np.nanmean(output, axis=0)
#                 # Peak flux from the mean of the frames, not whichever frame the
#                 # loop happened to end on.
#                 frame = np.nanmean(frames, axis=0)
#             else:
#                 raise ValueError(f'Image {filename} has {data.ndim} dimensions, expected 2 or 3')

#             outputs[idx_i, o] = output[o]

#             if masks:
#                 output_fluxes[idx_i, o] = np.nanmax(frame * masks[o])

#         print(voltage_coord, attens[idx_i])

#     return voltage_coords, outputs, attens, output_fluxes, labels

def parse_scan_filename(filename, filename_root, labels):
    ''' Extracts the settings a scan image was taken at from its filename.
 
        Parameters
        ----------
        filename : str
            Full path of one .npy image produced by a voltage scan
        filename_root : str
            Path and prefix passed to the scan routine
        labels : list of str
            Channel labels in scan order, e.g. ['Ch69', 'KCh2']
 
        Returns
        -------
        voltages : list of float
            Commanded voltage for each channel, in label order. NaN for any
            label not present in the filename.
        output_name : str or None
            Output port letter
        atten : float or None
            Attenuation in dB applied for this image
    '''
    # Only the part after the root is searched: filename_root can itself
    # contain a '_Ch..._V...' part for a channel held fixed outside the scan,
    # which would otherwise be picked up as one of the scanned channels.
    stem = filename[len(filename_root):]
 
    voltages = []
    for label in labels:
        match = re.search(rf'_{re.escape(label)}_V(\d+p\d+)', stem)
        voltages.append(float(match.group(1).replace('p', '.')) if match else np.nan)
 
    match = re.search(r'_output([A-Z])_', stem)
    output_name = match.group(1) if match else None
 
    match = re.search(r'_output[A-Z]_(\d+p\d+)\.npy$', stem)
    atten = float(match.group(1).replace('p', '.')) if match else None
 
    return voltages, output_name, atten
 
def _select_null_region(attens_null):
    ''' Picks the scan points making up the region around the null.

        The attenuation series forms plateaus as attenuation_adjust steps it in
        discrete increments. The null sits in the plateau of lowest attenuation,
        since a fading output needs less attenuation to stay in range. This
        returns that plateau plus the one immediately before and after it, which
        bracket the null with the points where the output was brighter.

        Parameters
        ----------
        attens_null : 1D numpy array
            Attenuation applied to the null port at each scan point, in scan order

        Returns
        -------
        selected : list of int
            Scan point indices in the null region, in scan order. Empty if every
            attenuation is NaN.
    '''
    a = np.asarray(attens_null, dtype=float)

    if np.all(np.isnan(a)):
        return []

    # Group consecutive equal values, treating NaN runs as their own plateau so
    # a gap of unreadable images does not merge the plateaus on either side.
    plateaus, start = [], 0
    for i in range(1, len(a) + 1):
        same = (i < len(a)) and (a[i] == a[i-1] or (np.isnan(a[i]) and np.isnan(a[i-1])))
        if not same:
            plateaus.append(list(range(start, i)))
            start = i

    i_null = int(np.nanargmin(a))
    k = next(k for k, idx in enumerate(plateaus) if i_null in idx)

    selected = []
    for kk in (k - 1, k, k + 1):
        if 0 <= kk < len(plateaus):
            selected.extend(plateaus[kk])

    return sorted(selected)


def reduce_voltage_scan(filename_root, num_outputs, filename_dark, filt_size=5,
                        coords=None, find_centers=True, mask_range=5, r_max=15,
                        r_lin=5, plot=False,
                        movie_filename=None, movie_fps=4, movie_stride=1,
                        movie_dpi=100, movie_clim=None, movie_bitrate=4000,
                        plot_log=False, movie_fit_panel=True,
                        movie_null_filename=None, ind_null=0,
                        _only_indices=None):
    ''' Get output intensities for each image produced by a voltage scan.

        Scan points are enumerated from the image files themselves rather than
        from the data log. The log holds full-precision voltages while filenames
        are written to three decimals, so reconstructing a filename from a log
        row fails whenever two points round together, whenever the channel order
        differs from the log's, or whenever a file was overwritten. Reading the
        filenames instead means every returned row corresponds to a file that
        exists, so there are no gaps.

        Handles scans over any mix of XCOM and Keysight channels: the channels
        are read from the log filename as labels.

        Optionally writes a movie of every frame for visual inspection. Frames
        are piped to ffmpeg as they are reduced rather than collected, so a long
        scan does not build up in memory. With movie_fit_panel each output gets
        its own image panel and, below it, the photometry fit that produced its
        intensity, so a bad fit can be spotted against the image it came from.

        Parameters
        ----------
        filename_root : str
            Path and beginning identifier of each file in voltage scan image collection
        num_outputs : int
            Number of outputs to identify in image
        filename_dark : str
            Filename of the dark frame to subtract
        filt_size : int
            Size of maximum filter (in pixels) for isolating local maxima to identify outputs. Default: 5
        coords : 2D numpy array or None
            Coordinate pairs of output centers; if None, finds output centers
        find_centers : boolean
            True if need to find output centers; else False. Cannot be False if coords is None.
        mask_range : int
            Mask radius (in pixels) around center of output for peak flux. Default: 5
        r_max : int
            Maximum pixel radius for photometry extraction method. Default: 15
        r_lin : int
            Lower limit of pixel radius at which photometry extraction method behaves linearly. Default: 5
        plot : boolean
            Flag for plotting masks overlayed on image and photometry extraction
        movie_filename : str or None
            If given, writes an .mp4 of every reduced frame here. Requires ffmpeg.
        movie_fps : float
            Movie playback rate. Default: 4
        movie_stride : int
            Keep every nth scan point in the movie. Default: 1
        movie_dpi : int
            Movie render resolution. Default: 100
        movie_clim : tuple of float or None
            (vmin, vmax) colour limits held fixed across every frame. If None,
            taken from the 1st and 99.5th percentiles of the first frame.
        movie_bitrate : int
            ffmpeg bitrate in kb/s. Sharp-edged image data needs more than the
            default. Default: 4000
        plot_log : boolean
            Plot images with log colorbar if True.
        movie_fit_panel : boolean
            Whether to draw the photometry fit beside each output's image in the
            movie. Requires extract_outputs to accept an ax argument. Default: True
        movie_null_filename : str or None
            If given, writes a second .mp4 covering only the region around the
            null: every point at the null port's lowest attenuation, plus the
            attenuation plateau immediately before and after it. Written in a
            second pass over just those files, so it costs a fraction of the
            full movie.
        ind_null : int
            Index of the null port, used to pick the attenuation series that
            defines the null region. Default: 0
        _only_indices : list of int or None
            Internal. Restricts the reduction to these scan points, used by the
            null-region pass to re-read only the files it needs.

        Returns
        -------
        voltage_coords : 2D numpy array
            Voltage combination for each reduced point, columns in channel order
        outputs : 2D numpy array
            Output intensities at each voltage setting
        attens : 2D numpy array
            Attenuation in dB applied to each output at each voltage setting
        output_fluxes : 2D numpy array
            Peak dark-subtracted flux in each output mask, NaN if coords is None
        labels : list of str
            Channel labels in scan order, e.g. ['Ch69', 'KCh2']
    '''
    filename_log = glob.glob(filename_root + "*.log")[0]
    labels = parse_channel_labels(filename_log, filename_root)

    dark = np.loadtxt(filename_dark, delimiter=',')
    masks = build_box_masks(np.shape(dark), coords, mask_range=mask_range)

    output_names = [chr(65 + i) for i in range(num_outputs)]

    # Enumerate the scan points from the output A images, which exist once per
    # voltage combination. Each point keeps its own filename, so the other
    # outputs are found by substituting the output letter rather than by
    # rebuilding the name from voltages.
    settings = []
    for filename in glob.glob(filename_root + f"*_output{output_names[0]}_*.npy"):
        voltages, _, _ = parse_scan_filename(filename, filename_root, labels)
        settings.append((voltages, filename))

    if not settings:
        raise FileNotFoundError(
            f'No images found matching {filename_root}*_output{output_names[0]}_*.npy')

    # Numeric sort, not the lexical order glob returns: 'V10p000' sorts before
    # 'V2p000' as a string, which would put the returned rows out of order.
    settings.sort(key=lambda s: list(s[0]))

    if _only_indices is not None:
        settings = [settings[i] for i in _only_indices]

    voltage_coords = np.array([s[0] for s in settings])

    outputs = np.full((len(settings), num_outputs), np.nan)
    attens = np.full((len(settings), num_outputs), np.nan)
    output_fluxes = np.full((len(settings), num_outputs), np.nan)

    # --- movie setup ---
    writer = None
    if movie_filename is not None:
        from matplotlib.animation import FFMpegWriter
        from matplotlib.colors import LogNorm

        if not FFMpegWriter.isAvailable():
            raise RuntimeError(
                'ffmpeg not found. pip install imageio-ffmpeg, then set\n'
                "  matplotlib.rcParams['animation.ffmpeg_path'] = "
                'imageio_ffmpeg.get_ffmpeg_exe()')

        # One row per output: its image on the left, the photometry fit that
        # produced its intensity on the right, so a bad fit can be read against
        # the image it came from.
        #
        # A bare Figure with its own Agg canvas, not plt.figure: under the inline
        # backend every pyplot figure is rendered into the notebook, so a figure
        # redrawn once per scan point would print a plot for every point. A
        # Figure that pyplot does not know about is never displayed, and
        # FFMpegWriter only needs the canvas.
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        n_cols = 2 if movie_fit_panel else 1
        fig_mov = Figure(figsize=(3.6*n_cols + 1.4, 3.2*num_outputs + 0.6),
                         layout='constrained')
        FigureCanvasAgg(fig_mov)
        gs = fig_mov.add_gridspec(num_outputs, n_cols)

        ax_img = [fig_mov.add_subplot(gs[i, 0]) for i in range(num_outputs)]
        for a in ax_img:
            a.set_axis_off()

        ims = [a.imshow(dark, vmin=0, vmax=1, origin='lower', interpolation='nearest')
               for a in ax_img]

        # Attached to the whole image column, so it does not steal width from
        # one panel and leave the others a different size.
        cbar_mov = fig_mov.colorbar(ims[0], ax=ax_img, fraction=0.035, pad=0.02)

        ax_fit = ([fig_mov.add_subplot(gs[i, 1]) for i in range(num_outputs)]
                  if movie_fit_panel else [None]*num_outputs)

        title_mov = fig_mov.suptitle('', fontsize=10)

        writer = FFMpegWriter(fps=movie_fps, bitrate=movie_bitrate,
                              metadata={'title': filename_root})
        writer.setup(fig_mov, movie_filename, dpi=movie_dpi)
        movie_clim_set = movie_clim is not None
        movie_vmin = None
        if movie_clim_set:
            movie_vmin = movie_clim[0]
            if plot_log:
                lo = movie_clim[0] if movie_clim[0] > 0 else 1e-3
                for im in ims:
                    im.set_norm(LogNorm(vmin=lo, vmax=max(movie_clim[1], lo*10)))
                movie_vmin = lo
            else:
                for im in ims:
                    im.set_clim(*movie_clim)
        n_movie_frames = 0

    try:
        for idx_i, (voltages, filename_A) in enumerate(settings):
            frames_this_point = [None]*num_outputs
            draw_this_point = (writer is not None) and (idx_i % movie_stride == 0)

            for o, output_name in enumerate(output_names):
                # Substitute the output letter in the filename that was found,
                # rather than rebuilding it, so a point can only be missing if
                # that one output's file is genuinely absent.
                pattern = re.sub(rf'_output{output_names[0]}_[\d p]+\.npy$',
                                 f'_output{output_name}_*.npy', filename_A)
                matches = glob.glob(pattern)

                if not matches:
                    print(f'No image found for {pattern}')
                    continue

                filename = matches[0]
                _, _, attens[idx_i, o] = parse_scan_filename(filename, filename_root, labels)

                try:
                    data = np.load(filename)
                except (ValueError, OSError, EOFError) as e:
                    # A truncated .npy: the header declares a shape the file
                    # does not contain, which happens if a scan was interrupted
                    # mid-write. Skip it rather than failing the reduction.
                    print(f'Unreadable image {filename}: {type(e).__name__}: {e}')
                    continue

                # The fit is drawn onto this output's panel as it is computed, so
                # the movie costs no extra photometry.
                ax_this = ax_fit[o] if (draw_this_point and movie_fit_panel) else None
                if ax_this is not None:
                    ax_this.clear()

                # The aperture circles and centre markers are drawn by
                # extract_outputs_frame from the centers it used, so they are
                # correct whether coords was supplied or found per frame. The
                # old ones are removed first, since the centers can move between
                # frames: circles are patches and the markers are lines, and
                # both would otherwise accumulate. The image itself is an
                # AxesImage, so clearing lines and patches leaves it alone.
                ax_img_this = ax_img[o] if draw_this_point else None
                if ax_img_this is not None:
                    for artist in list(ax_img_this.patches) + list(ax_img_this.lines):
                        artist.remove()

                if data.ndim == 2:
                    frame = data - dark
                    output = BENCH.extract_outputs(frame, num_outputs, r_max=r_max, r_lin=r_lin,
                                                   filt_size=filt_size, plot=plot, coords=coords,
                                                   find_centers=find_centers, ax=ax_this,
                                                   ax_img=ax_img_this)
                elif data.ndim == 3:
                    output, frames = [], []
                    for i_f, data_i in enumerate(data):
                        frame_i = data_i - dark
                        frames.append(frame_i)
                        # Only the middle frame draws, matching extract_outputs'
                        # own convention, so a 50 frame stack does not overplot.
                        middle = (i_f == int(len(data)/2))
                        output.append(BENCH.extract_outputs(
                            frame_i, num_outputs, r_max=r_max, r_lin=r_lin,
                            filt_size=filt_size, plot=plot, coords=coords,
                            find_centers=find_centers,
                            ax=ax_this if middle else None,
                            ax_img=ax_img_this if middle else None))
                    output = np.nanmean(output, axis=0)
                    # Peak flux from the mean of the frames, not whichever frame
                    # the loop happened to end on.
                    frame = np.nanmean(frames, axis=0)
                else:
                    raise ValueError(f'Image {filename} has {data.ndim} dimensions, expected 2 or 3')

                outputs[idx_i, o] = output[o]

                if masks:
                    output_fluxes[idx_i, o] = np.nanmax(frame * masks[o])

                frames_this_point[o] = frame

            # --- movie frame for this voltage point ---
            valid = [f for f in frames_this_point if f is not None]

            if draw_this_point and valid:
                if not movie_clim_set:
                    stacked = np.concatenate([f.ravel() for f in valid])
                    vmin, vmax = np.percentile(stacked, [1, 99.5])

                    if plot_log:
                        # LogNorm needs a positive vmin: after dark subtraction
                        # the background straddles zero, so fall back to a small
                        # fraction of the peak.
                        if vmin <= 0:
                            vmin = max(np.nanmax(stacked) * 1e-4, 1e-3)
                        for im in ims:
                            im.set_norm(LogNorm(vmin=vmin, vmax=max(vmax, vmin*10)))
                    else:
                        for im in ims:
                            im.set_clim(vmin, vmax)

                    movie_vmin = vmin
                    movie_clim_set = True

                for o in range(num_outputs):
                    panel = frames_this_point[o]
                    if panel is None:
                        panel = np.full_like(dark, np.nan)
                    elif plot_log:
                        # Clipped, not masked: LogNorm renders non-positive
                        # pixels with the colormap's 'bad' colour, which would
                        # leave holes across the background.
                        panel = np.clip(panel, movie_vmin, None)
                    ims[o].set_data(panel)
                    ax_img[o].set_title(
                        f'Output {output_names[o]}  {attens[idx_i, o]:.0f} dB', fontsize=9)

                title_mov.set_text(', '.join(f'{lab} {v:.3f} V'
                                             for lab, v in zip(labels, voltages)))
                writer.grab_frame(facecolor='white')
                n_movie_frames += 1

            print(voltages, attens[idx_i])

    finally:
        # finish() in a finally block so an interrupted reduction still leaves a
        # playable file rather than a truncated one.
        if writer is not None:
            writer.finish()
            print(f'Wrote {n_movie_frames} frames to {movie_filename}')

    # --- second movie, covering only the region around the null ---
    if movie_null_filename is not None:
        selected = _select_null_region(attens[:, ind_null])

        if not selected:
            print('No valid null-port attenuations, so no null movie was written.')
        else:
            a_sel = attens[selected, ind_null]
            print(f'Null region: {len(selected)} of {len(settings)} points, '
                  f'attenuations {sorted(set(a_sel[~np.isnan(a_sel)]))}')

            # A second pass over just these files, rather than buffering frames
            # during the first pass: the null plateau cannot be identified until
            # every attenuation is known, and the subset is small enough that
            # re-reading it is cheaper than holding the whole scan in memory.
            reduce_voltage_scan(
                filename_root, num_outputs, filename_dark, filt_size=filt_size,
                coords=coords, find_centers=find_centers, mask_range=mask_range,
                r_max=r_max, r_lin=r_lin, plot=False,
                movie_filename=movie_null_filename, movie_fps=movie_fps,
                movie_stride=1, movie_dpi=movie_dpi, movie_clim=movie_clim,
                movie_bitrate=movie_bitrate, plot_log=plot_log,
                movie_fit_panel=movie_fit_panel,
                movie_null_filename=None, ind_null=ind_null,
                _only_indices=selected)

    n_missing = int(np.sum(np.all(np.isnan(outputs), axis=1)))
    if n_missing:
        print(f'{n_missing} of {len(settings)} points had no readable image.')

    return voltage_coords, outputs, attens, output_fluxes, labels

def parse_track_filename(filename, filename_root):
    ''' Extracts the time index and attenuation from a tracking image filename.
 
        Parameters
        ----------
        filename : str
            Full path of one .npy image produced by the tracking phase
        filename_root : str
            Path and prefix passed to null_stability_track
 
        Returns
        -------
        index : int or None
            Time index from the '_tNNNNN' part of the filename
        output_name : str or None
            Output port letter
        atten : float or None
            Attenuation in dB applied for this image
    '''
    stem = filename[len(filename_root):]
 
    match = re.search(r'_t(\d+)_output', stem)
    index = int(match.group(1)) if match else None
 
    match = re.search(r'_output([A-Z])_', stem)
    output_name = match.group(1) if match else None
 
    match = re.search(r'_output[A-Z]_(\d+p\d+)\.npy$', stem)
    atten = float(match.group(1).replace('p', '.')) if match else None
 
    return index, output_name, atten
  
def reduce_null_track(filename_root, num_outputs, ind_null, filename_dark, filt_size=5,
                      coords=None, find_centers=True, mask_range=5, r_max=15,
                      r_lin=5, plot=False, flux_max=None, log_suffix='_track',
                      movie_filename=None, movie_fps=4, movie_stride=1,
                      movie_dpi=100, movie_clim=None, movie_bitrate=4000):
    ''' Get output intensities for each image taken during the tracking phase of
        null_stability_track.
 
        Points are ordered by the '_tNNNNN' index in the filename rather than by voltage. Elapsed time and
        temperature come from the tracking log, which holds one row per output
        per point in output order.
 
        Parameters
        ----------
        filename_root : str
            Path and prefix passed to null_stability_track
        num_outputs : int
            Number of outputs to identify in image
        ind_null : integer
            Index of null port
        filename_dark : str
            Filename of the dark frame to subtract
        filt_size : int
            Size of maximum filter (in pixels) for isolating local maxima to identify outputs. Default: 5
        coords : 2D numpy array or None
            Coordinate pairs of output centers; if None, finds output centers
        find_centers : boolean
            True if need to find output centers; else False. Cannot be False if coords is None.
        mask_range : int
            Mask radius (in pixels) around center of output for peak flux. Default: 5
        r_max : int
            Maximum pixel radius for photometry extraction method. Default: 15
        r_lin : int
            Lower limit of pixel radius at which photometry extraction method behaves linearly. Default: 5
        plot : boolean
            Flag for plotting masks overlayed on image and photometry extraction
        flux_max : float or None
            Attenuation-corrected maximum flux from the confirmation sweep. If
            given, nulls_dB is returned relative to it; otherwise nulls_dB is
            relative to the brightest point in the tracking series itself, which
            is only meaningful if the null drifted far enough to reach a peak.
        log_suffix : str
            Suffix distinguishing the tracking log from the sweep log.
            Default: '_track'
        movie_filename : str or None
            If given, writes an .mp4 of every tracking frame here. Requires ffmpeg.
        movie_fps : float
            Movie playback rate. Default: 4
        movie_stride : int
            Keep every nth point in the movie. Default: 1
        movie_dpi : int
            Movie render resolution. Default: 100
        movie_clim : tuple of float or None
            (vmin, vmax) colour limits held fixed across every frame. If None,
            taken from the 1st and 99.5th percentiles of the first frame.
        movie_bitrate : int
            ffmpeg bitrate in kb/s. Default: 4000
 
        Returns
        -------
        times : 1D numpy array
            Seconds since the first tracking measurement
        outputs : 2D numpy array
            Output intensities at each time
        attens : 2D numpy array
            Attenuation in dB applied to each output at each time
        output_fluxes : 2D numpy array
            Peak dark-subtracted flux in each output mask, NaN if coords is None
        nulls_dB : 1D numpy array
            Null port intensity in dB relative to flux_max, or to the brightest
            tracking point if flux_max is None
        temps : 1D numpy array
            Chip temperature at each time
        labels : list of str
            Channel labels held fixed during tracking
    '''
    outputs_ref = np.load(f'{filename_root}_outputs_sweep.npy')
    attens_ref = np.load(f'{filename_root}_attenuation_sweep.npy')
    reference = np.nanmax(outputs_ref * 10**(attens_ref/10))

    matches = glob.glob(filename_root + f"*{log_suffix}.log")
    if not matches:
        raise FileNotFoundError(f'No log matching {filename_root}*{log_suffix}.log')
    filename_log = matches[0]
    labels = parse_channel_labels(filename_log, filename_root)
 
    dark = np.loadtxt(filename_dark, delimiter=',')
    masks = build_box_masks(np.shape(dark), coords, mask_range=mask_range)
 
    output_names = [chr(65 + i) for i in range(num_outputs)]
 
    # Enumerate the tracking points from the output A images and order them by
    # time index, not by the lexical order glob returns.
    settings = []
    for filename in glob.glob(filename_root + f"*_t*_output{output_names[0]}_*.npy"):
        index, _, _ = parse_track_filename(filename, filename_root)
        if index is not None:
            settings.append((index, filename))
 
    if not settings:
        raise FileNotFoundError(
            f'No tracking images matching {filename_root}*_t*_output{output_names[0]}_*.npy')
 
    settings.sort(key=lambda s: s[0])
    indices = np.array([s[0] for s in settings])
 
    n = len(settings)
    outputs = np.full((n, num_outputs), np.nan)
    attens = np.full((n, num_outputs), np.nan)
    output_fluxes = np.full((n, num_outputs), np.nan)
 
    # --- elapsed time and temperature from the log ---
    # One row per output per point, in output order, so the per-point values are
    # every num_outputs-th row.
    times = np.full(n, np.nan)
    temps = np.full(n, np.nan)
 
    log = np.loadtxt(filename_log, delimiter=',')
    with open(filename_log) as f:
        colnames = f.readline().strip().strip('# ').split(',')
 
    if log.ndim == 1:
        log = log[np.newaxis, :]
 
    if 'timestamp' in colnames and 'temperature' in colnames:
        i_time = colnames.index('timestamp')
        i_temp = colnames.index('temperature')
        rows = log[::num_outputs]
 
        if len(rows) < n:
            print(f'Log has {len(rows)} points but {n} images were found; '
                  'times and temperatures will be incomplete.')
 
        m = min(len(rows), n)
        times[:m] = rows[:m, i_time] - rows[0, i_time]
        temps[:m] = rows[:m, i_temp]
    else:
        print(f'Log {filename_log} has no timestamp/temperature columns; '
              'times will be the frame index.')
        times = indices.astype(float)
 
    # --- movie setup ---
    writer = None
    if movie_filename is not None:
        from matplotlib.animation import FFMpegWriter
 
        if not FFMpegWriter.isAvailable():
            raise RuntimeError(
                'ffmpeg not found. pip install imageio-ffmpeg, then set\n'
                "  matplotlib.rcParams['animation.ffmpeg_path'] = "
                'imageio_ffmpeg.get_ffmpeg_exe()')
 
        fig_mov, ax_mov = plt.subplots(figsize=(3 * num_outputs, 3.5))
        ax_mov.set_axis_off()
        im_mov = ax_mov.imshow(np.hstack([dark] * num_outputs),
                               vmin=0, vmax=1, origin='lower',
                               interpolation='nearest')
        title_mov = ax_mov.set_title('', fontsize=9)
        fig_mov.tight_layout()
 
        writer = FFMpegWriter(fps=movie_fps, bitrate=movie_bitrate,
                              metadata={'title': filename_root})
        writer.setup(fig_mov, movie_filename, dpi=movie_dpi)
        movie_clim_set = movie_clim is not None
        if movie_clim_set:
            im_mov.set_clim(*movie_clim)
        n_movie_frames = 0
 
    try:
        for idx_i, (index, filename_A) in enumerate(settings):
            panels = []
 
            for o, output_name in enumerate(output_names):
                # Substitute the output letter in the filename that was found,
                # rather than rebuilding it, so a point can only be missing if
                # that one output's file is genuinely absent.
                pattern = re.sub(rf'_output{output_names[0]}_[\d p]+\.npy$',
                                 f'_output{output_name}_*.npy', filename_A)
                found = glob.glob(pattern)
 
                if not found:
                    print(f'No image found for {pattern}')
                    continue
 
                filename = found[0]
                _, _, attens[idx_i, o] = parse_track_filename(filename, filename_root)
 
                try:
                    data = np.load(filename)
                except (ValueError, OSError, EOFError) as e:
                    # A truncated .npy: the header declares a shape the file
                    # does not contain, which happens if a run was interrupted
                    # mid-write. Skip it rather than failing the reduction.
                    print(f'Unreadable image {filename}: {type(e).__name__}: {e}')
                    continue
 
                if data.ndim == 2:
                    frame = data - dark
                    output = BENCH.extract_outputs(frame, num_outputs, r_max=r_max, r_lin=r_lin,
                                                   filt_size=filt_size, plot=plot, coords=coords,
                                                   find_centers=find_centers)
                elif data.ndim == 3:
                    output, frames = [], []
                    for data_i in data:
                        frame_i = data_i - dark
                        frames.append(frame_i)
                        output.append(BENCH.extract_outputs(frame_i, num_outputs, r_max=r_max, r_lin=r_lin,
                                                            filt_size=filt_size, plot=plot, coords=coords,
                                                            find_centers=find_centers))
                    output = np.nanmean(output, axis=0)
                    frame = np.nanmean(frames, axis=0)
                else:
                    raise ValueError(f'Image {filename} has {data.ndim} dimensions, expected 2 or 3')
 
                outputs[idx_i, o] = output[o]
 
                if masks:
                    output_fluxes[idx_i, o] = np.nanmax(frame * masks[o])
 
                if writer is not None:
                    panels.append(frame)
 
            if writer is not None and panels and (idx_i % movie_stride == 0):
                composite = np.hstack(panels)
 
                if not movie_clim_set:
                    im_mov.set_clim(*np.percentile(composite, [1, 99.5]))
                    movie_clim_set = True
 
                im_mov.set_data(composite)
                title_mov.set_text(
                    f't = {times[idx_i]:.1f} s,  {temps[idx_i]:.3f} C   |   '
                    + '  '.join(f'{name} {attens[idx_i, o]:.0f} dB'
                                for o, name in enumerate(output_names)))
                writer.grab_frame(facecolor='white')
                n_movie_frames += 1
 
            print(f't {times[idx_i]:8.1f} s  {outputs[idx_i]}  {attens[idx_i]}')
 
    finally:
        if writer is not None:
            writer.finish()
            plt.close(fig_mov)
            print(f'Wrote {n_movie_frames} frames to {movie_filename}')
 
    # --- null depth ---
    flux = outputs[:, ind_null] * 10**(attens[:, ind_null] / 10)
 
    nulls_dB = 10 * np.log10(flux / reference)
 
    n_missing = int(np.sum(np.all(np.isnan(outputs), axis=1)))
    if n_missing:
        print(f'{n_missing} of {n} points had no readable image.')
 
    return times, outputs, attens, output_fluxes, nulls_dB, temps, labels


# def reduce_voltage_scan_fixedAtten(filename_root, mode_ext, num_outputs, filename_dark=None,
#                                    filt_size=5, coords=None, mask_range=5, r_max=15, r_lin=5,
#                                    plot=False, v_decimals=2):
#     ''' Get output intensities for each image produced by a voltage scan taken at
#         fixed attenuation.

#         Parameters
#         ----------
#         filename_root : str
#             Path and beginning identifier of each file in voltage scan image collection
#         mode_ext : str
#             Extension corresponding to attenuation mode of data to reduce
#         num_outputs : int
#             Number of outputs to identify in image
#         filename_dark : None or str
#             If None, uses a voltage-specific dark filename for each point; else uses
#             the provided string throughout
#         filt_size : int
#             Size of maximum filter (in pixels) for isolating local maxima to identify outputs. Default: 5
#         coords : 2D numpy array or None
#             Coordinate pairs of output centers; if None, finds output centers
#         mask_range : int
#             Mask radius (in pixels) around center of output. Default: 5
#         r_max : int
#             Maximum pixel radius for photometry extraction method. Default: 15
#         r_lin : int
#             Lower limit of pixel radius at which photometry extraction method behaves linearly. Default: 5
#         plot : boolean
#             Flag for plotting masks overlayed on image and photometry extraction
#         v_decimals : int
#             Decimal places used in the voltage part of the image filenames. Default: 2

#         Returns
#         -------
#         voltages : list of 1D numpy arrays
#             Voltages scanned by each channel
#         outputs : numpy array
#             Output intensities at each voltage setting, shaped (n1, ..., nk, num_outputs)
#         output_fluxes : numpy array
#             Peak dark-subtracted flux in each output mask, same shape as outputs
#         labels : list of str
#             Channel labels in scan order, e.g. ['Ch69', 'KCh2']
#     '''
#     log, labels, _ = read_scan_log(filename_root)

#     voltages = []
#     for label in labels:
#         column = voltage_column(log, label)
#         voltages.append(np.sort(np.unique(log[column])))
#     shape = tuple(len(v) for v in voltages)

#     # A fixed dark frame is loaded once; a per-point dark is loaded in the loop.
#     dark_fixed = None
#     if filename_dark is not None:
#         dark_fixed = BENCH.remove_stripes(np.loadtxt(filename_dark, delimiter=','))

#     outputs = np.full((*shape, num_outputs), np.nan)
#     output_fluxes = np.full((*shape, num_outputs), np.nan)
#     outputs_masks = None

#     for idx_i, idx in enumerate(np.ndindex(shape)):
#         voltage_coord = [voltages[d][idx[d]] for d in range(len(voltages))]
#         print(voltage_coord)

#         v_str = ''
#         for voltage in voltage_coord:
#             v_i = f"{voltage:.{v_decimals}f}".replace('.', 'p')
#             v_str += f"*_V{v_i}"

#         pattern = filename_root + f"{v_str}*{mode_ext}*"
#         matches = glob.glob(pattern)

#         if not matches:
#             print(f'No image found for {pattern}')
#             continue

#         filename = matches[0]

#         # Per-point dark, re-globbed for every voltage rather than latching onto
#         # the first one found.
#         if dark_fixed is not None:
#             dark = dark_fixed
#         else:
#             dark_matches = glob.glob(filename_root + v_str + '_dark*')
#             if not dark_matches:
#                 print(f'No dark found for {filename_root + v_str}_dark*')
#                 continue
#             dark = BENCH.remove_stripes(np.load(dark_matches[0]))

#         masks = build_box_masks(np.shape(dark), coords, mask_range=mask_range)

#         data = np.load(filename)

#         if (idx_i == 0) and (coords is not None):
#             reference = data if data.ndim == 2 else np.sum(data, axis=0)
#             outputs_masks = BENCH.build_output_mask(reference, num_outputs, coords=coords,
#                                                     mask_range=mask_range, filt_size=filt_size,
#                                                     plot=plot)

#         if data.ndim == 2:
#             if coords is None:
#                 outputs_masks = BENCH.build_output_mask(data - dark, num_outputs, coords=coords,
#                                                         mask_range=mask_range, filt_size=filt_size,
#                                                         plot=plot)
#             frame = BENCH.remove_stripes(data, outputs_masks) - dark
#             output = BENCH.extract_outputs(frame, num_outputs, r_max=r_max, r_lin=r_lin,
#                                            filt_size=filt_size, plot=plot, coords=coords)
#         elif data.ndim == 3:
#             output = []
#             frames = []
#             for data_i in data:
#                 if coords is None:
#                     outputs_masks = BENCH.build_output_mask(data_i - dark, num_outputs, coords=coords,
#                                                             mask_range=mask_range, filt_size=filt_size,
#                                                             plot=plot)
#                 frame_i = BENCH.remove_stripes(data_i, outputs_masks) - dark
#                 frames.append(frame_i)
#                 output.append(BENCH.extract_outputs(frame_i, num_outputs, r_max=r_max, r_lin=r_lin,
#                                                     filt_size=filt_size, plot=plot, coords=coords))
#             output = np.nanmean(output, axis=0)
#             frame = np.nanmean(frames, axis=0)
#         else:
#             raise ValueError(f'Image {filename} has {data.ndim} dimensions, expected 2 or 3')

#         outputs[tuple(idx)] = output

#         if masks:
#             for o in range(num_outputs):
#                 output_fluxes[tuple(idx)][o] = np.nanmax(frame * masks[o])

    return np.array(voltages, dtype=object), outputs, output_fluxes, labels


def parse_laser_scan_filename(filename, filename_root, labels):
    ''' Extracts the settings a laser scan image was taken at from its filename.

        Parameters
        ----------
        filename : str
            Full path of one .npy image produced by laser_scan
        filename_root : str
            Path and prefix passed to laser_scan
        labels : list of str
            Channel labels in scan order, e.g. ['Ch69', 'KCh2']

        Returns
        -------
        wav : float or None
            Requested wavelength in nm
        voltages : list of float
            Commanded voltage for each channel, in label order
        output_name : str or None
            Output port letter
        atten : float or None
            Attenuation in dB applied for this image
    '''
    stem = filename[len(filename_root):]

    match = re.search(r'_wav(\d+p\d+)', stem)
    wav = float(match.group(1).replace('p', '.')) if match else None

    voltages = []
    for label in labels:
        match = re.search(rf'_{re.escape(label)}_V(\d+p\d+)', stem)
        voltages.append(float(match.group(1).replace('p', '.')) if match else np.nan)

    match = re.search(r'_output([A-Z])_', stem)
    output_name = match.group(1) if match else None

    match = re.search(r'_output[A-Z]_(\d+p\d+)\.npy$', stem)
    atten = float(match.group(1).replace('p', '.')) if match else None

    return wav, voltages, output_name, atten

def reduce_laser_scan(filename_root, num_outputs, filename_dark, filt_size=5,
                      coords=None, find_centers=True, mask_range=5, r_max=15,
                      r_lin=5, plot=False,
                      movie_root=None, movie_fps=2, movie_stride=1,
                      movie_dpi=150, movie_clim=None, movie_bitrate=4000):
    ''' Get output intensities for each image produced by a laser scan.

        Settings are read from the image filenames rather than the data log,
        because laser_scan writes the requested wavelength into the filename
        and the achieved wavelength into the log, so the two do not match.

        Optionally writes one movie per wavelength for visual inspection, named
        movie_root + '_wav<wavelength>.mp4'. Frames are piped to ffmpeg as they
        are reduced rather than collected, so a long scan does not build up in
        memory.

        Parameters
        ----------
        filename_root : str
            Path and beginning identifier of each file in the scan image collection
        num_outputs : int
            Number of outputs to identify in image
        filename_dark : str
            Filename of the dark frame to subtract
        filt_size : int
            Size of maximum filter (in pixels) for isolating local maxima to identify outputs. Default: 5
        coords : 2D numpy array or None
            Coordinate pairs of output centers; if None, finds output centers
        find_centers : boolean
            True if need to find output centers; else False. Cannot be False if coords is None.
        mask_range : int
            Mask radius (in pixels) around center of output for peak flux. Default: 5
        r_max : int
            Maximum pixel radius for photometry extraction method. Default: 15
        r_lin : int
            Lower limit of pixel radius at which photometry extraction method behaves linearly. Default: 5
        plot : boolean
            Flag for plotting masks overlayed on image and photometry extraction
        movie_root : str or None
            If given, path and prefix for one .mp4 per wavelength. Requires ffmpeg.
        movie_fps : float
            Movie playback rate. Default: 4
        movie_stride : int
            Keep every nth point of each wavelength in its movie. Default: 1
        movie_dpi : int
            Movie render resolution. Default: 150
        movie_clim : tuple of float or None
            (vmin, vmax) colour limits held fixed across every frame of every
            wavelength. If None, taken from the first frame of the whole scan,
            so that wavelengths remain comparable to each other.
        movie_bitrate : int
            ffmpeg bitrate in kb/s. Sharp-edged image data needs more than the
            default. Default: 4000

        Returns
        -------
        wavs : 1D numpy array
            Requested wavelength in nm for each reduced point
        voltage_coords : 2D numpy array
            Voltage combination for each reduced point, columns in channel order
        outputs : 2D numpy array
            Output intensities at each setting
        attens : 2D numpy array
            Attenuation in dB applied to each output at each setting
        output_fluxes : 2D numpy array
            Peak dark-subtracted flux in each output mask, NaN if coords is None
        labels : list of str
            Channel labels in scan order, e.g. ['Ch69', 'KCh2']
    '''
    filename_log = glob.glob(filename_root + "*.log")[0]
    labels = parse_channel_labels(filename_log, filename_root)

    dark = np.loadtxt(filename_dark, delimiter=',')
    masks = build_box_masks(np.shape(dark), coords, mask_range=mask_range)

    output_names = [chr(65 + i) for i in range(num_outputs)]

    # Enumerate the scan points from the output A images, which exist once per
    # wavelength and voltage combination.
    settings = []
    for filename in glob.glob(filename_root + f"*_output{output_names[0]}_*.npy"):
        wav, voltages, _, _ = parse_laser_scan_filename(filename, filename_root, labels)
        settings.append((wav, voltages))

    if not settings:
        raise FileNotFoundError(f'No images found matching {filename_root}*_output{output_names[0]}_*.npy')

    # Numeric sort, not the lexical order glob returns: 'V10p000' sorts before
    # 'V2p000' as a string, which would put both the movie frames and the
    # returned rows out of order.
    settings.sort(key=lambda s: (s[0], list(s[1])))

    wavs = np.array([s[0] for s in settings])
    voltage_coords = np.array([s[1] for s in settings])

    outputs = np.full((len(settings), num_outputs), np.nan)
    attens = np.full((len(settings), num_outputs), np.nan)
    output_fluxes = np.full((len(settings), num_outputs), np.nan)

    # --- movie setup ---
    # One writer at a time, reusing a single figure: the writer is finished and
    # restarted whenever the wavelength changes.
    writer = None
    fig_mov = None
    if movie_root is not None:
        from matplotlib.animation import FFMpegWriter

        if not FFMpegWriter.isAvailable():
            raise RuntimeError(
                'ffmpeg not found. pip install imageio-ffmpeg, then set\n'
                "  matplotlib.rcParams['animation.ffmpeg_path'] = "
                'imageio_ffmpeg.get_ffmpeg_exe()'
            )

        fig_mov, ax_mov = plt.subplots(figsize=(4 * num_outputs, 5))
        ax_mov.set_axis_off()
        # [int(coords[0][0])-10:int(coords[0][0])+11, int(coords[0][1])-10:int(coords[0][1])+11]
        im_mov = ax_mov.imshow(np.hstack([dark] * num_outputs),
                               vmin=0, vmax=1, origin='lower',
                               interpolation='nearest')
        title_mov = ax_mov.set_title('', fontsize=9)
        fig_mov.tight_layout()

        movie_clim_set = movie_clim is not None
        if movie_clim_set:
            im_mov.set_clim(*movie_clim)

        current_wav = None
        movie_files = []
        n_frames_this_wav = 0
        wav_frame_index = 0

    try:
        for idx_i, (wav, voltages) in enumerate(settings):
            wav_str = f"{wav:.3f}".replace('.', 'p')

            # New wavelength: close the previous movie and open the next
            if writer is not None or (movie_root is not None and current_wav != wav):
                if movie_root is not None and current_wav != wav:
                    if writer is not None:
                        writer.finish()
                        print(f'  wrote {n_frames_this_wav} frames to {movie_files[-1]}')

                    movie_file = f'{movie_root}_wav{wav_str}.mp4'
                    writer = FFMpegWriter(fps=movie_fps, bitrate=movie_bitrate,
                                          metadata={'title': f'{filename_root} {wav} nm'})
                    writer.setup(fig_mov, movie_file, dpi=movie_dpi)
                    movie_files.append(movie_file)
                    current_wav = wav
                    n_frames_this_wav = 0
                    wav_frame_index = 0

            v_str = ''
            for label, voltage in zip(labels, voltages):
                v_i = f"{voltage:.3f}".replace('.', 'p')
                v_str += f"_{label}_V{v_i}"

            panels = []

            for o, output_name in enumerate(output_names):
                pattern = filename_root + f"_wav{wav_str}{v_str}_output{output_name}_*.npy"
                matches = glob.glob(pattern)

                if not matches:
                    print(f'No image found for {pattern}')
                    continue

                filename = matches[0]
                _, _, _, attens[idx_i, o] = parse_laser_scan_filename(filename, filename_root, labels)

                data = np.load(filename)

                if data.ndim == 2:
                    frame = data - dark
                    output = BENCH.extract_outputs(frame, num_outputs, r_max=r_max, r_lin=r_lin,
                                                   filt_size=filt_size, plot=plot, coords=coords,
                                                   find_centers=find_centers)
                elif data.ndim == 3:
                    output, frames = [], []
                    for data_i in data:
                        frame_i = data_i - dark
                        frames.append(frame_i)
                        output.append(BENCH.extract_outputs(frame_i, num_outputs, r_max=r_max, r_lin=r_lin,
                                                            filt_size=filt_size, plot=plot, coords=coords,
                                                            find_centers=find_centers))
                    output = np.nanmean(output, axis=0)
                    frame = np.nanmean(frames, axis=0)
                else:
                    raise ValueError(f'Image {filename} has {data.ndim} dimensions, expected 2 or 3')

                outputs[idx_i, o] = output[o]

                if masks:
                    output_fluxes[idx_i, o] = np.nanmax(frame * masks[o])

                if writer is not None:
                    panels.append(frame) #[int(coords[0][0])-10:int(coords[0][0])+11, int(coords[0][1])-10: int(coords[0][1])+11]

            # --- movie frame for this point ---
            if writer is not None and panels:
                if wav_frame_index % movie_stride == 0:
                    composite = np.hstack(panels)

                    if not movie_clim_set:
                        # Fixed across every frame of every wavelength, so
                        # brightness differences are real rather than an artifact
                        # of per-frame autoscaling. Percentiles rather than
                        # min/max, so one hot pixel does not set the scale.
                        im_mov.set_clim(*np.percentile(composite, [1, 99.5]))
                        movie_clim_set = True

                    im_mov.set_data(composite)
                    title_mov.set_text(
                        f'{wav:.3f} nm   '
                        + ', '.join(f'{lab} {v:.3f} V' for lab, v in zip(labels, voltages))
                        + '   |   '
                        + '  '.join(f'Output {name}: {attens[idx_i, o]:.0f} dB'
                                    for o, name in enumerate(output_names))
                    )
                    writer.grab_frame(facecolor='white')
                    n_frames_this_wav += 1

                wav_frame_index += 1

            print(f'{wav:9.3f} nm  {voltages}  {attens[idx_i]}')

    finally:
        # finish() in a finally block so an interrupted reduction still leaves a
        # playable file rather than a truncated one.
        if writer is not None:
            writer.finish()
            print(f'  wrote {n_frames_this_wav} frames to {movie_files[-1]}')
        if fig_mov is not None:
            plt.close(fig_mov)
            print(f'Wrote {len(movie_files)} movies, one per wavelength.')

    return wavs, voltage_coords, outputs, attens, output_fluxes, labels


# def reduce_laser_scan(filename_root, num_outputs, filename_dark, filt_size=5,
#                       coords=None, find_centers=True, mask_range=5, r_max=15,
#                       r_lin=5, plot=False):
#     ''' Get output intensities for each image produced by a laser scan.

#         Settings are read from the image filenames rather than the data log,
#         because laser_scan writes the requested wavelength into the filename
#         and the achieved wavelength into the log, so the two do not match.

#         Parameters
#         ----------
#         filename_root : str
#             Path and beginning identifier of each file in the scan image collection
#         num_outputs : int
#             Number of outputs to identify in image
#         filename_dark : str
#             Filename of the dark frame to subtract
#         filt_size : int
#             Size of maximum filter (in pixels) for isolating local maxima to identify outputs. Default: 5
#         coords : 2D numpy array or None
#             Coordinate pairs of output centers; if None, finds output centers
#         find_centers : boolean
#             True if need to find output centers; else False. Cannot be False if coords is None.
#         mask_range : int
#             Mask radius (in pixels) around center of output for peak flux. Default: 5
#         r_max : int
#             Maximum pixel radius for photometry extraction method. Default: 15
#         r_lin : int
#             Lower limit of pixel radius at which photometry extraction method behaves linearly. Default: 5
#         plot : boolean
#             Flag for plotting masks overlayed on image and photometry extraction

#         Returns
#         -------
#         wavs : 1D numpy array
#             Requested wavelength in nm for each reduced point
#         voltage_coords : 2D numpy array
#             Voltage combination for each reduced point, columns in channel order
#         outputs : 2D numpy array
#             Output intensities at each setting
#         attens : 2D numpy array
#             Attenuation in dB applied to each output at each setting
#         output_fluxes : 2D numpy array
#             Peak dark-subtracted flux in each output mask, NaN if coords is None
#         labels : list of str
#             Channel labels in scan order, e.g. ['Ch69', 'KCh2']
#     '''
#     filename_log = glob.glob(filename_root + "*.log")[0]
#     labels = parse_channel_labels(filename_log, filename_root)

#     dark = np.loadtxt(filename_dark, delimiter=',')
#     masks = build_box_masks(np.shape(dark), coords, mask_range=mask_range)

#     output_names = [chr(65 + i) for i in range(num_outputs)]

#     # Enumerate the scan points from the output A images, which exist once per
#     # wavelength and voltage combination.
#     settings = []
#     for filename in sorted(glob.glob(filename_root + f"*_output{output_names[0]}_*.npy")):
#         wav, voltages, _, _ = parse_laser_scan_filename(filename, filename_root, labels)
#         settings.append((wav, voltages))

#     if not settings:
#         raise FileNotFoundError(f'No images found matching {filename_root}*_output{output_names[0]}_*.npy')

#     wavs = np.array([s[0] for s in settings])
#     voltage_coords = np.array([s[1] for s in settings])

#     outputs = np.full((len(settings), num_outputs), np.nan)
#     attens = np.full((len(settings), num_outputs), np.nan)
#     output_fluxes = np.full((len(settings), num_outputs), np.nan)

#     for idx_i, (wav, voltages) in enumerate(settings):
#         wav_str = f"{wav:.3f}".replace('.', 'p')

#         v_str = ''
#         for label, voltage in zip(labels, voltages):
#             v_i = f"{voltage:.3f}".replace('.', 'p')
#             v_str += f"_{label}_V{v_i}"

#         for o, output_name in enumerate(output_names):
#             pattern = filename_root + f"_wav{wav_str}{v_str}_output{output_name}_*.npy"
#             matches = glob.glob(pattern)

#             if not matches:
#                 print(f'No image found for {pattern}')
#                 continue

#             filename = matches[0]
#             _, _, _, attens[idx_i, o] = parse_laser_scan_filename(filename, filename_root, labels)

#             data = np.load(filename)

#             if data.ndim == 2:
#                 frame = data - dark
#                 output = BENCH.extract_outputs(frame, num_outputs, r_max=r_max, r_lin=r_lin,
#                                                filt_size=filt_size, plot=plot, coords=coords,
#                                                find_centers=find_centers)
#             elif data.ndim == 3:
#                 output, frames = [], []
#                 for data_i in data:
#                     frame_i = data_i - dark
#                     frames.append(frame_i)
#                     output.append(BENCH.extract_outputs(frame_i, num_outputs, r_max=r_max, r_lin=r_lin,
#                                                         filt_size=filt_size, plot=plot, coords=coords,
#                                                         find_centers=find_centers))
#                 output = np.nanmean(output, axis=0)
#                 frame = np.nanmean(frames, axis=0)
#             else:
#                 raise ValueError(f'Image {filename} has {data.ndim} dimensions, expected 2 or 3')

#             outputs[idx_i, o] = output[o]

#             if masks:
#                 output_fluxes[idx_i, o] = np.nanmax(frame * masks[o])

#         print(f'{wav:9.3f} nm  {voltages}  {attens[idx_i]}')

#     return wavs, voltage_coords, outputs, attens, output_fluxes, labels



# import serial

# import logging
# _log = logging.getLogger('PIC_lib')

# import glob, time


# try:
#     from Zaber import UNIT, CAM_X, CAM_Y, CAM_FOCUS
#     # Use package_with_problem functionality here
# except ImportError:
#     print("Zaber not available, functionality will be disabled.")
#     UNIT, CAM_X, CAM_Y, CAM_FOCUS = None, None, None, None
# try:
#     from XenicsCam import NAVG
# except:
#     print("XenicsCam not available, functionality will be disabled.")
#     NAVG = None
# from XPOW import ATTEN_CHANNEL, XPOWVoltageControlMax

# MAXVOLTAGE = XPOWVoltageControlMax
# XPOW_DELAY = 1 # seconds
# ZABER_DELAY = 1 # seconds
# LASER_ON_DELAY = 10 # seconds
# LASER_OFF_DELAY = 2 # seconds
# VOA_MIN, VOA_MAX = 1, 20 # dB
# ARDUINO_PORT = 'COM7'
# EXTERNALVOA_POWER = 'D:\\Yoojung\\CalibrationData\\2026-05-15-Calibration\\ExternalVOA\\ExternalVOA_voltageScan_power.npy'
# EXTERNALVOA_VOLTAGE = 'D:\\Yoojung\\CalibrationData\\2026-05-15-Calibration\\ExternalVOA\\ExternalVOA_voltageScan_voltage.npy'

# ### Set-up hardware control routines ###

# def find_focus(mask_region, step, zaber, cam, tlx=None, dark=None, num_iter=20, step_unit=UNIT, navg=NAVG):
#     ''' Iteratively searches for camera focus position by maximizing flux in specified region.

#         Parameters
#         ----------
#         mask_region : 2x2 numpy array
#             x and y limits of region in camera image to add up flux for optimizing focus. Format: [[x_min, x_max], [y_min, y_max]]
#         step : float
#             Initial step size for moving camera focus position with Zaber to search for optimal focus
#         tlx : TLX object
#             Object for programmatically controlling TLX2 laser as defined in TLX2.py
#         zaber: ZABER object
#             Object for programmatically controlling Zaber actuators/stages as defined in Zaber.py
#         cam : XENICSCAM object
#             Object for programmatically controlling Xenics camera as defined in XenicsCam.py
#         num_iter : int
#             Number of iterations over which to optimize focus. Default: 20. 
#         step_unit : Units enum from zaber_motion library
#             Unit of step size for moving camera focus position with Zaber. Default: UNIT.
#         navg : int
#             Number of camera frames to average together when collecting data. Default: NAVG

#         Returns:
#         --------
#         zaber : ZABER object
#             Zaber control object after moving to optimize focus position        
#     '''

#     # Make mask
#     if tlx is not None:
#         tlx.laser_off()
#         time.sleep(LASER_OFF_DELAY)
#         _, _, dark, _ = cam.take_image(navg=navg)
#         tlx.laser_on()
#         time.sleep(LASER_ON_DELAY)
    
#     x, y, data, _ = cam.take_image(navg=navg)
#     x_dim, y_dim = len(x), len(y)

#     mask = np.zeros_like(data, dtype=bool)
#     if (mask_region[0][1] <= mask_region[0][0]) or (mask_region[1][1] <= mask_region[1][0]):
#         raise ValueError(f"Mask region array must define lower limits in first column and upper limits in second column. Received: \n {mask_region}")
#     ix = np.abs(x - np.mean(mask_region[0])) < (mask_region[0][1] - mask_region[0][0])/2
#     iy = np.abs(y - np.mean(mask_region[1])) < (mask_region[1][1] - mask_region[1][0])/2
#     mask[iy[:, None] & ix[None, :]] = True

#     # Plot mask
#     yellow_cmap = colors.ListedColormap(['yellow'])

#     fig, ax = plt.subplots(1, 1, figsize=(10,6))
#     cbar_norm = colors.Normalize(vmin=0, vmax=10000)  
#     im = ax.pcolormesh(x, y, data - dark, norm=cbar_norm, zorder=0)
#     masked_overlay = np.ma.masked_where(~mask, np.ones_like(mask))
#     ax.pcolormesh(x, y, masked_overlay, cmap=yellow_cmap, alpha=0.2, zorder=1)
#     fig.colorbar(im, ax=ax, fraction=0.05*y_dim/x_dim)
#     ax.set_xlabel('x (microns)', fontsize=18)
#     ax.set_ylabel('y (microns)', fontsize=18)
#     ax.set_aspect('equal')
#     plt.title('Masked region')
#     plt.show()

#     # Find focus
#     data_reduce = data[mask].flatten() - dark[mask].flatten()
#     metric = np.sum(data_reduce[np.argsort(data_reduce)[-5:]])
#     metric_prior, metric_first = np.copy(metric), np.copy(metric)
#     device_ind = CAM_FOCUS
#     _log.info('Starting metric:'+str(metric))

#     for i in range(20):
#         zaber.move_relative(device_ind, step)
#         time.sleep(0.5)
#         _, _, data, _ = cam.take_image(navg=navg)
#         data_reduce = data[mask].flatten() - dark[mask].flatten()
#         metric = np.sum(data_reduce[np.argsort(data_reduce)[-5:]])

#         if metric < metric_prior:
#             step = -0.8 * step
#         _log.info('Change in metric:'+str(float(metric - metric_prior)))
#         metric_prior = np.copy(metric)
#         _log.info('\n')

#     _log.info("Final metric: "+str(metric))
#     _log.info(f"Final change in metric from start: {metric/metric_first * 100:0.2f}%")
#     return zaber

# def center_frame(zaber, cam, num_outputs, tlx=None, x=None, y=None, data=None, dark=None):
#     ''' Centers camera on central output along x-axis. (Centering on y-axis isn't possible because the Zabers won't 
#         raise without a slip/stall error beyond a certain point.)

#         Parameters
#         ----------
#         tlx : TLX object
#             Object for programmatically controlling TLX2 laser as defined in TLX2.py
#         zaber: ZABER object
#             Object for programmatically controlling Zaber actuators/stages as defined in Zaber.py
#         cam : XENICSCAM object
#             Object for programmatically controlling Xenics camera as defined in XenicsCam.py 
#         num_outputs : int
#             Number of outputs to identify in reference image

#         Returns
#         -------
#         x, y : 1D numpy arrays
#             Image x- and y-axis in microns.
#         data : 2D numpy array
#             Image data with laser on
#         dark : 2D numpy array
#             Image data with laser off
#     '''
#     if tlx is not None:
#         # Dark
#         tlx.laser_off()
#         time.sleep(LASER_OFF_DELAY)
#         x, y, dark, _ = cam.take_image()

#         # Data
#         tlx.laser_on()
#         time.sleep(LASER_ON_DELAY)
#         x, y, data, _ = cam.take_image()
#     data_reduce = data - dark

#     # Find central output and shift
#     coords = find_output_centers(data_reduce, num_outputs)
#     coords_center = (np.max(coords[:, 1]) + np.min(coords[:, 1]))/2
#     shift = np.interp(coords_center, np.arange(len(x)), x) # in mm
#     print(shift)
#     zaber.move_relative(CAM_X, shift, camera=True)
#     time.sleep(ZABER_DELAY)

#     # Data
#     x, y, data, _ = cam.take_image()
#     data_reduce = data - dark

#     # Plot
#     fig, ax = plt.subplots(1, 1, figsize=(24,5))
#     norm = colors.Normalize(vmin=0, vmax=10000) 
#     im0 = ax.pcolormesh(x, y, data_reduce, norm=norm, cmap='RdBu_r')
#     fig.colorbar(im0, ax=ax, fraction=0.05*len(y)/len(x))
#     ax.set_xlabel('x (microns)', fontsize=18)
#     ax.set_ylabel('y (microns)', fontsize=18)
#     for i in range(3):
#         ax.set_aspect('equal')
#     plt.show()
#     return x, y, data, dark

# # def set_externalVOA(dB, xpow):
# #     ''' Applies attenuation with external VOA by setting current.

# #         Parameters
# #         ----------
# #         dB : float
# #             Decibels of attenuation applied by external VOA
# #         xpow : XPOW object
# #             Object for programmatically controlling XPOW breadboard using class definition in XPOW.py
        
# #         Returns
# #         -------
# #         None
# #     '''
# #     data = np.load('D:\\Yoojung\\CalibrationData\\2026-01-26-Calibration\\AttenuatorCalib\\AttenuatorCalib_currentScan_power.npy')
# #     ratio = np.abs(np.mean(data/data[0, :], axis=1))
# #     atten_max = np.max(10*np.log10(ratio))

# #     if dB > atten_max:
# #         raise ValueError(f'Cannot attenuate more than {atten_max:0.3f} dB. Received: {dB} dB')
    
# #     current = externalVOA_dB_to_current(dB)
# #     xpow.apply_current(ATTEN_CHANNEL, current)
# #     return

# def set_VOA(dB, xpow, tlx=None):
#     ''' Applies attenuation with external VOA by setting voltage (and laser if tlx is not None).

#         Parameters
#         ----------
#         dB : float
#             Decibels of attenuation to apply.
#         xpow : XPOW object
#             Object for programmatically controlling XPOW breadboard using class definition in XPOW.py
#         tlx : TLX object
#             None or Object for programmatically controlling TLX2 laser as defined in TLX2.py
        
#         Returns
#         -------
#         None
#     '''
#     data = np.load(EXTERNALVOA_POWER)
#     ratio = np.abs(np.mean(data[:,1:,0]/data[:,1, 0][:,None], axis=0))

#     if tlx is not None:
#         atten_max = np.max(np.abs(10*np.log10(ratio))) + 20
#         atten_min = 1.0
#         if (dB > atten_max) or (dB < atten_min):
#             raise ValueError(f'Attenuation limited between {atten_min:0.3f} dB to {atten_max:0.3f} dB. Received: {dB} dB')
#         elif (dB >= atten_min) and (dB <= 20.0):
#             tlx.set_voa(dB)
#             xpow.apply_voltage(ATTEN_CHANNEL, 0)
#         elif (dB > 20.0) and (dB <= atten_max):
#             tlx.set_voa(20.0)
#             voltage = externalVOA_dB_to_voltage(dB-20.0)
#             xpow.apply_voltage(ATTEN_CHANNEL, voltage)
#     else:
#         atten_max = np.max(np.abs(10*np.log10(ratio)))
#         atten_min = 0
#         if (dB > atten_max) or (dB < atten_min):
#             raise ValueError(f'Attenuation limited between {atten_min:0.3f} dB to {atten_max:0.3f} dB. Received: {dB} dB')
#         elif (dB >= atten_min) and (dB <= atten_max):
#             voltage = externalVOA_dB_to_voltage(dB)
#             xpow.apply_voltage(ATTEN_CHANNEL, voltage)
#     time.sleep(0.5)
#     return

# def get_VOA(xpow, tlx=None):
#     ''' Reads out attenuation with external VOA by setting voltage.

#         Parameters
#         ----------
#         xpow : XPOW object
#             Object for programmatically controlling XPOW breadboard using class definition in XPOW.py
#         tlx : TLX object
#             None or Object for programmatically controlling TLX2 laser as defined in TLX2.py
        
#         Returns
#         -------
#         dB : float
#             Decibels of attenuation applied by external VOA (and optionally laser)
#     '''
#     dB = externalVOA_voltage_to_dB(xpow.read_XPOW(ATTEN_CHANNEL)[0])
#     if tlx is not None:
#         dB += tlx.voa
#     return dB

# ### Hardware calibrated properties ###
# def externalVOA_voltage_to_dB(voltage):
#     ''' Converts voltage applied to external VOA to dB of attenuation.

#         Parameters
#         ----------
#         voltage : float
#             Voltage in V applied to external VOA
        
#         Returns
#         -------
#         dB : float
#             Decibels of attenuation applied by external VOA
#     '''
#     data = np.load(EXTERNALVOA_POWER)[:,1:,:]
#     external_voltage = np.load(EXTERNALVOA_VOLTAGE)[1:]
#     ratio = np.abs(np.mean(data[:,:,0]/data[:,0, 0][:,None], axis=0))
#     atten_measured = 10*np.log10(ratio)

#     dB = np.abs(np.interp(voltage, external_voltage, atten_measured))
#     return dB

# def externalVOA_dB_to_voltage(dB):
#     ''' Converts dB of attenuation to requisite voltage applied to external VOA in V.

#         Parameters
#         ----------
#         dB : float
#             Decibels of attenuation applied by external VOA
        
#         Returns
#         -------
#         voltage : float
#             Voltage in V applied to external VOA
#     '''
#     data = np.load(EXTERNALVOA_POWER)[:,1:,:]
#     external_voltage = np.load(EXTERNALVOA_VOLTAGE)[1:]
#     ratio = np.abs(np.mean(data[:,:,0]/data[:,0, 0][:,None], axis=0))
#     atten_measured = np.abs(10*np.log10(ratio))

#     voltage = np.interp(dB, atten_measured, external_voltage)
#     return voltage

# ### Data acquisition hardware control routines ###
# def attenuation_adjust(xpow, cam, coord, atten, dark, tlx=None, dB_step=3, R=5, navg=NAVG, stack=False, output_name=''):
#     ''' Dynamically adjust attenuation to ensure a given output is imaged within the linear regime of the camera.

#         Parameters
#         ----------
#         xpow : XPOW object
#             Object for programmatically controlling XPOW breadboard using class definition in XPOW.py
#         cam : XENICSCAM object
#             Object for programmatically controlling Xenics camera as defined in XenicsCam.py
#         coord : 1x2 numpy array 
#             Approximate coordinate of output port pixel
#         atten : float
#             Initial attenuation (in dB) setting
#         dark : 2D numpy array
#             Dark frame image
#         tlx : TLX object
#             None or object for programmatically controlling TLX2 laser as defined in TLX2.py. Default: None
#         dB_step : float
#             Increment (in dB) by which to adjust dB when output is outside linear regime of camera
#         R : integer
#             Pixel radius for mask around output coordinate to test for linearity
#         navg : int
#             Number of camera frames to average together when collecting final data image. Default: NAVG
#         stack : boolean
#             If True, stack all camera exposures together in final data image, else don't. Default: False
#         output_name : string
#             Name of output port associated with provided coordinate. Default: ''

#         Returns
#         -------
#         atten : float
#             Final attenuation (in dB) adopted for final data image with camera's linear regime
#         x, y : 1D numpy arrays
#             Image x- and y-axis in mm.
#         data : numpy array
#             Image data averaged over all frames if stack == True; else, collection of all frames.
#         timestamps : np.ndarray
#             Shape (navg,)
#             Camera or host timestamps (see notes)       
#     '''
#     set_VOA(atten, xpow, tlx)
#     time.sleep(3)
#     _, _, data, _ = cam.take_image(navg=1, stack=True) 
#     mask = np.zeros(np.shape(dark), bool)
#     r, c =  coord
#     mask[max(0,r-R):r+R+1, max(0,c-R):c+R+1] = 1

#     while np.max(data*mask) > 12000:
#         try:
#             atten = atten + dB_step
#             set_VOA(atten, xpow, tlx)
#             time.sleep(3)
#         except:
#             raise ValueError(f'Output {output_name} saturating at max attenuation limit of VOA.')
#         _, _, data, _ = cam.take_image(navg=1, stack=True)   
#     while np.max((data-dark)*mask) < 5*np.std(dark):
#         if atten >= 4.0:
#             atten = atten - dB_step
#             set_VOA(atten, xpow, tlx)
#             time.sleep(3)
#         else:
#             atten = 1.0
#             print(f'Output {output_name} underexposed at min attenuation limit of VOA.')
#         _, _, data, _ = cam.take_image(navg=1, stack=True)
#     x, y, data, timestamp = cam.take_image(navg=navg, stack=stack)
#     return atten, x, y, data, timestamp


# def voltage_scan(channels, voltages, filename_root, coords, attens, dark, cam, xpow, tec, tlx=None, navg=NAVG, remove_stripes=False):
#     ''' Performs a voltage scan over multiple active components with the power meter. Saves each image and the hardware settings to a table.

#         Parameters
#         ----------
#         channels : n x 1 numpy array of integers
#             List of channels for each active component to set
#         voltages : List of n sequences of floats
#             All sequences of voltages to scan through for each active component. This routine will go through all possible combinations of voltages for 
#             each active component based on their corresponding sequences of voltages to cycle through.
#         filename_root : string
#             Path and prefix for each file produced by this routine
#         coords : Nx2 numpy array or list of integers
#             Approximate coordinates of output port pixels
#         attens : 1xN numpy array
#             Initial attenuations for each of the outputs
#         dark : 2D numpy array
#             Dark frame image
#         cam : XENICSCAM object
#             Object for programmatically controlling Xenics camera as defined in XenicsCam.py
#         xpow : XPOW object
#             Object for programmatically controlling XPOW breadboard using class definition in XPOW.py
#         tec : TEC object
#             Object for programmatically managing Arduino-based temperature control as defined in TEC.py
#         tlx : None or TLX object
#             None or object for programmatically controlling TLX2 laser as defined in TLX2.py
#         navg : int
#             Number of camera frames to average together when collecting data. Default: NAVG
#         remove_stripes : boolean
#             Flag for turning on feature to remove stripes from images. Default: False

#         Returns
#         -------
#         dats : list
#             Table of data log containing XPOW and laser readout settings 
#     '''
#     # Construct all voltage combos
#     voltagecombos = np.meshgrid(*voltages)
#     voltagecombos = np.stack(voltagecombos, axis=-1).reshape(-1, len(voltagecombos))

#     assert np.min(voltagecombos) >= 0, "voltage should be non-negative"
#     assert np.max(voltagecombos) < MAXVOLTAGE, "exceeds max voltage allowed"

#     dats = []
    
#     masks = []
#     for coord in coords:
#         mask, R = np.zeros(np.shape(dark), bool), 5
#         r, c =  coord
#         mask[max(0,r-R):r+R+1, max(0,c-R):c+R+1] = 1
#         masks.append(mask)

#     for voltagecombo in voltagecombos:
#         table_row = []
#         channel_str = f''

#         # Set XPOW
#         for i, voltage in enumerate(voltagecombo):
#             # Set XPOW
#             xpow.apply_voltage(channels[i], voltage)
#             time.sleep(XPOW_DELAY)
#             # Readout settings for data log table
#             actualvoltage, current, component_power = xpow.read_XPOW(channels[i])
#             table_row = np.hstack([table_row, voltage, actualvoltage, current, component_power])
#             # Build filename extension for image data
#             v_str = "{:.3f}".format(round(voltage, 3)).replace('.', 'p')
#             channel_str = channel_str + f'_Ch{channels[i]}_V{v_str}'
        
#         # Take images for each output
#         for i, coord in enumerate(coords):
#             output_name = chr(i + 65)
#             # set_VOA(attens[i], xpow, tlx)
#             # x, y, data, timestamp = cam.take_image(navg=navg, stack=False)
#             # if remove_stripes:
#             #     data_noStripe, dark_noStripe = remove_stripes(np.mean(data, axis=0), masks), remove_stripes(dark)
#             # else:
#             #     data_noStripe, dark_noStripe = np.copy(data), np.copy(dark)

#             # from matplotlib.colors import ListedColormap
#             # yellow_cmap = ListedColormap(['yellow'])
#             # fig, ax = plt.subplots(1, 1, figsize=(10,6))
#             # cbar_norm = colors.Normalize(vmin=0, vmax=12000)  
#             # im = ax.pcolormesh(x, y, data_null[0]-dark, norm=cbar_norm, zorder=0)
#             # masked_overlay = np.ma.masked_where(~null_mask, np.ones_like(null_mask))
#             # ax.pcolormesh(x, y, masked_overlay, cmap=yellow_cmap, alpha=0.2, zorder=1)
#             # # ax.plot(x[coords[:, 1]], y[coords[:, 0]], 'r+')
#             # fig.colorbar(im, ax=ax, fraction=0.05*len(y)/len(x))
#             # ax.set_xlabel('x (microns)', fontsize=18)
#             # ax.set_ylabel('y (microns)', fontsize=18)
#             # ax.set_title('Data - Dark')
#             # ax.set_aspect('equal')
#             # plt.show()

#             # while np.max(data*mask) > 12000:
#             #     try:
#             #         attens[i] = attens[i] + 3
#             #         set_VOA(attens[i], xpow, tlx)
#             #         time.sleep(3)
#             #     except:
#             #         raise ValueError(f'Output {output_name} saturating at max attenuation limit of VOA.')
#             #     x, y, data, timestamp = cam.take_image(navg=navg, stack=False)  
#             # while np.max((data_noStripe-dark_noStripe)*mask) < 5*np.std(dark_noStripe):
#             #     if attens[i] >= 4.0:
#             #         attens[i] = attens[i] - 3
#             #         set_VOA(attens[i], xpow, tlx)
#             #         time.sleep(3)
#             #     else:
#             #         attens[i] = 1.0
#             #         print(f'Output {output_name} underexposed at min attenuation limit of VOA.')
#             #     x, y, data, timestamp = cam.take_image(navg=navg, stack=False)
#             #     if remove_stripes:
#             #         data_noStripe, dark_noStripe = remove_stripes(np.mean(data, axis=0), masks), remove_stripes(dark)
#             #     else:
#             #         data_noStripe, dark_noStripe = np.copy(data), np.copy(dark)
#             atten, x, y, data, timestamp = attenuation_adjust(xpow, cam, coord, attens[i], dark, tlx=tlx, navg=NAVG, output_name=output_name)
#             attens[i] = atten
#             atten_str = f"{attens[i]:.3f}".replace('.', 'p')
#             filename = filename_root + f"{channel_str}_output{output_name}_{atten_str}"
#             np.save(filename, data)
#             temp = tec.read_temperature()
#             if tlx is not None:
#                 dats.append(np.hstack([xpow.read_XPOW(ATTEN_CHANNEL), tlx.voa, table_row, np.mean(timestamp), temp]))
#             else:
#                 dats.append(np.hstack([xpow.read_XPOW(ATTEN_CHANNEL), table_row, np.mean(timestamp), temp]))
      
#     # Construct filename extension and table header for data log
#     channel_str = f''
#     header_str = f'externalAttenActualVoltage,externalAttenCurrent,externalAttenPower,'
#     if tlx is not None:
#         header_str = header_str + 'laserVOA,'
#     for ch in channels:
#         channel_str = channel_str + f'_Ch{ch}'
#         header_str = header_str + f'inputVoltage{ch},actualVoltage{ch},current{ch},componentPower{ch},'
#     header_str = header_str + 'timestamp,temperature'
#     output_filename = filename_root + channel_str + '.log'
#     x_filename = filename_root + '_x.csv'
#     y_filename = filename_root + '_y.csv'
#     np.savetxt(output_filename, dats, header=header_str, delimiter=",")
#     np.savetxt(x_filename, x, delimiter=",")
#     np.savetxt(y_filename, y, delimiter=",")
#     np.savetxt(filename_root+"_dark.csv", dark, delimiter=",")
#     return np.array(dats)

# def voltage_scan_adaptive(channels_fix, voltages_fix, channel_adapt, adapt_lim, adapt_step, filename_root, coords, 
#                           ind_null, attens, dark, cam, xpow, tec, tlx=None, navg=NAVG):
#     ''' Performs a voltage scan over multiple active components with the power meter. Saves each image and the hardware settings to a table.

#         Parameters
#         ----------
#         channels_fix : n x 1 numpy array of integers
#             List of channels for each active component to set according to a fixed grid
#         voltages_fix : List of n sequences of floats
#             All sequences of voltages to scan through for each active component set to a fixed grid. This routine will go through all possible combinations of voltages for 
#             each active component based on their corresponding sequences of voltages to cycle through.
#         channel_adapt : integer
#             Channel to modulate adaptively
#         adapt_lim : 1 x 2 numpy array or list
#             [min, max] of voltage range over which to modulate the adaptive channel
#         adapt_step : float
#             Initial step size for adaptive channel
#         filename_root : string
#             Path and prefix for each file produced by this routine
#         coords : Nx2 numpy array or list of integers
#             Approximate coordinates of output port pixels
#         ind_null : integer
#             Index of null port
#         attens : 1xN numpy array
#             Initial attenuations for each of the outputs
#         dark : 2D numpy array
#             Dark frame image
#         cam : XENICSCAM object
#             Object for programmatically controlling Xenics camera as defined in XenicsCam.py
#         xpow : XPOW object
#             Object for programmatically controlling XPOW breadboard using class definition in XPOW.py
#         tec : TEC object
#             Object for programmatically managing Arduino-based temperature control as defined in TEC.py
#         tlx : None or TLX object
#             None or object for programmatically controlling TLX2 laser as defined in TLX2.py
#         navg : int
#             Number of camera frames to average together when collecting data. Default: NAVG
#         remove_stripes : boolean
#             Flag for turning on feature to remove stripes from images. Default: False

#         Returns
#         -------
#         dats : list
#             Table of data log containing XPOW and laser readout settings 
#     '''
#     # Construct all voltage combos
#     voltagecombos = np.meshgrid(*voltages_fix)
#     voltagecombos = np.stack(voltagecombos, axis=-1).reshape(-1, len(voltagecombos))

#     assert np.min(voltagecombos) >= 0, "voltage should be non-negative"
#     assert np.max(voltagecombos) < MAXVOLTAGE, "exceeds max voltage allowed"

#     dats = []
    
#     masks = []
#     for coord in coords:
#         mask, R = np.zeros(np.shape(dark), bool), 5
#         r, c =  coord
#         mask[max(0,r-R):r+R+1, max(0,c-R):c+R+1] = 1
#         masks.append(mask)

#     for voltagecombo in voltagecombos:
#         table_row = []
#         channel_str = f''

#         # Set XPOW
#         for i, voltage in enumerate(voltagecombo):
#             # Set XPOW
#             xpow.apply_voltage(channels_fix[i], voltage)
#             time.sleep(XPOW_DELAY)
#             # Readout settings for data log table
#             actualvoltage, current, component_power = xpow.read_XPOW(channels_fix[i])
#             table_row = np.hstack([table_row, voltage, actualvoltage, current, component_power])
#             # Build filename extension for image data
#             v_str = "{:.3f}".format(round(voltage, 3)).replace('.', 'p')
#             channel_str = channel_str + f'_Ch{channels_fix[i]}_V{v_str}'

#         adapt_voltage = adapt_lim[0]
#         xpow.apply_voltage(channel_adapt, adapt_voltage)
#         time.sleep(XPOW_DELAY)
#         actualvoltage, current, component_power = xpow.read_XPOW(channel_adapt)
#         table_row_all = np.hstack([table_row, adapt_voltage, actualvoltage, current, component_power])
#         # Build filename extension for image data
#         v_str = "{:.3f}".format(round(adapt_voltage, 3)).replace('.', 'p')
#         channel_str_all = channel_str + f'_Ch{channel_adapt}_V{v_str}'
#         for i, coord in enumerate(coords):
#             output_name = chr(i + 65)
#             atten, x, y, data, timestamp = attenuation_adjust(xpow, cam, coord, attens[i], dark, tlx=tlx, navg=NAVG, stack=False, output_name=output_name)
#             attens[i] = atten
#             atten_str = f"{attens[i]:.3f}".replace('.', 'p')
#             filename = filename_root + f"{channel_str_all}_output{output_name}_{atten_str}"
#             np.save(filename, data)
#             temp = tec.read_temperature()
#             if tlx is not None:
#                 dats.append(np.hstack([xpow.read_XPOW(ATTEN_CHANNEL), tlx.voa, table_row_all, np.mean(timestamp), temp]))
#             else:
#                 dats.append(np.hstack([xpow.read_XPOW(ATTEN_CHANNEL), table_row_all, np.mean(timestamp), temp]))
#             if i == ind_null:
#                 flux_previous = np.sum(np.mean(data, axis=0)[masks[i]]) * atten

#         while adapt_voltage < adapt_lim[-1]:
#             adapt_voltage = adapt_voltage + adapt_step
#             if adapt_voltage > adapt_lim[-1]:
#                 adapt_voltage = adapt_lim[-1]
#             print(f'Phase: {adapt_voltage} V')
#             xpow.apply_voltage(channel_adapt, adapt_voltage)
#             time.sleep(XPOW_DELAY)
#             actualvoltage, current, component_power = xpow.read_XPOW(channel_adapt)
#             table_row_all = np.hstack([table_row, adapt_voltage, actualvoltage, current, component_power])
#             # Build filename extension for image data
#             v_str = "{:.3f}".format(round(adapt_voltage, 3)).replace('.', 'p')
#             channel_str_all = channel_str + f'_Ch{channel_adapt}_V{v_str}'
        
#             # Take images for each output
#             for i, coord in enumerate(coords):
#                 output_name = chr(i + 65)
#                 atten, x, y, data, timestamp = attenuation_adjust(xpow, cam, coord, attens[i], dark, tlx=tlx, navg=NAVG, stack=False, output_name=output_name)
#                 attens[i] = atten
#                 atten_str = f"{attens[i]:.3f}".replace('.', 'p')
#                 filename = filename_root + f"{channel_str_all}_output{output_name}_{atten_str}"
#                 np.save(filename, data)
#                 temp = tec.read_temperature()
#                 if tlx is not None:
#                     dats.append(np.hstack([xpow.read_XPOW(ATTEN_CHANNEL), tlx.voa, table_row_all, np.mean(timestamp), temp]))
#                 else:
#                     dats.append(np.hstack([xpow.read_XPOW(ATTEN_CHANNEL), table_row_all, np.mean(timestamp), temp]))
#                 if i == ind_null:
#                     flux_current = np.sum(np.mean(data, axis=0)[masks[i]]) * atten
#             adapt_step = ((flux_current/flux_previous)**2) * adapt_step
#             flux_previous = flux_current
#             if adapt_step < 0.001:
#                 adapt_step = 0.001
#             if adapt_step > 0.5:
#                 adapt_step = 0.5
      
#     # Construct filename extension and table header for data log
#     channel_str = f''
#     header_str = f'externalAttenActualVoltage,externalAttenCurrent,externalAttenPower,'
#     if tlx is not None:
#         header_str = header_str + 'laserVOA,'
#     for ch in channels_fix:
#         channel_str = channel_str + f'_Ch{ch}'
#         header_str = header_str + f'inputVoltage{ch},actualVoltage{ch},current{ch},componentPower{ch},'
#     channel_str = channel_str + f'_Ch{channel_adapt}'
#     header_str = header_str + f'inputVoltage{channel_adapt},actualVoltage{channel_adapt},current{channel_adapt},componentPower{channel_adapt},'
#     header_str = header_str + 'timestamp,temperature'
#     output_filename = filename_root + channel_str + '.log'
#     x_filename = filename_root + '_x.csv'
#     y_filename = filename_root + '_y.csv'
#     np.savetxt(output_filename, dats, header=header_str, delimiter=",")
#     np.savetxt(x_filename, x, delimiter=",")
#     np.savetxt(y_filename, y, delimiter=",")
#     np.savetxt(filename_root+"_dark.csv", dark, delimiter=",")
#     return np.array(dats)

# def voltage_scan_fixedAtten(channels, voltages, filename_root, cam, xpow, tec, navg=NAVG):
#     ''' Performs a voltage scan over multiple active components with the power meter. Saves each image and the hardware settings to a table.

#         Parameters
#         ----------
#         channels : n x 1 numpy array of integers
#             List of channels for each active component to set
#         voltages : List of n sequences of floats
#             All sequences of voltages to scan through for each active component. This routine will go through all possible combinations of voltages for 
#             each active component based on their corresponding sequences of voltages to cycle through.
#         outputs_masks : list of n numpy slice objects
#             List of slice objects to specify each output "aperture"
#         filename_root : string
#             Path and prefix for each file produced by this routine
#         cam : XENICSCAM object
#             Object for programmatically controlling Xenics camera as defined in XenicsCam.py
#         xpow : XPOW object
#             Object for programmatically controlling XPOW breadboard using class definition in XPOW.py
#         tec : TEC object
#             Object for programmatically managing Arduino-based temperature control as defined in TEC.py
#         navg : int
#             Number of camera frames to average together when collecting data. Default: NAVG

#         Returns
#         -------
#         dats : list
#             Table of data log containing XPOW and laser readout settings 
#     '''
#     # Construct all voltage combos
#     voltagecombos = np.meshgrid(*voltages)
#     voltagecombos = np.stack(voltagecombos, axis=-1).reshape(-1, len(voltagecombos))

#     assert np.min(voltagecombos) >= 0, "voltage should be non-negative"
#     assert np.max(voltagecombos) < MAXVOLTAGE, "exceeds max voltage allowed"

#     dats = []
#     for voltagecombo in voltagecombos:
#         table_row = []
#         channel_str = f''

#         # Set XPOW
#         for i, voltage in enumerate(voltagecombo):
#             # Set XPOW
#             xpow.apply_voltage(channels[i], voltage)
#             time.sleep(XPOW_DELAY)
#             # Readout settings for data log table
#             actualvoltage, current, component_power = xpow.read_XPOW(channels[i])
#             table_row = np.hstack([table_row, voltage, actualvoltage, current, component_power])
#             # Build filename extension for image data
#             v_str = "{:.2f}".format(round(voltage, 2)).replace('.', 'p')
#             channel_str = channel_str + f'_Ch{channels[i]}_V{v_str}'
        
#         # Take low attenuation image
#         ### Add temperature if we can stream that info ###
#         filename = filename_root + f"{channel_str}"
#         if navg == 1:
#             filename = filename + '.csv'
#         x, y, data, timestamp = cam.take_image(navg=navg, filename=filename, stack=False)
#         temp = tec.read_temperature()
#         dats.append(np.hstack([1, table_row, np.mean(timestamp), temp]))     
      
#     # Construct filename extension and table header for data log
#     channel_str = f''
#     header_str = f'laserStatus,'
#     for ch in channels:
#         channel_str = channel_str + f'_Ch{ch}'
#         header_str = header_str + f'inputVoltage{ch},actualVoltage{ch},current{ch},componentPower{ch},'
#     header_str = header_str + 'timestamp,temperature'
#     output_filename = filename_root + channel_str + '.log'
#     x_filename = filename_root + channel_str + '_x.csv'
#     y_filename = filename_root + channel_str + '_y.csv'
#     np.savetxt(output_filename, dats, header=header_str, delimiter=",")
#     np.savetxt(x_filename, x, delimiter=",")
#     np.savetxt(y_filename, y, delimiter=",")
#     return np.array(dats)

# def optimize_null(initial_voltages, channels, outputs_masks, ind_null, cam, xpow, dark_frame=None):
#     def fit_func(voltages):
        
#         # Set XPOW
#         for i, voltage in enumerate(voltages):
#             # Set XPOW
#             xpow.apply_voltage(channels[i], voltage)
#             time.sleep(XPOW_DELAY)
#             # # Readout settings for data log table
#             # actualvoltage, current, component_power = xpow.read_XPOW(channels[i])
#             # table_row = np.hstack([table_row, voltage, actualvoltage, current, component_power])
#             # # Build filename extension for image data
#             # v_str = "{:.2f}".format(round(voltage, 2)).replace('.', 'p')
#             # channel_str = channel_str + f'_Ch{channels[i]}_V{v_str}'
        
#         # Take bright image
#         # tlx.laser_on()
#         # laser_voa = tlx.voa
#         # time.sleep(LASER_ON_DELAY)
#         # ### Add temperature if we can stream that info ###
#         # filename = filename_root + f"{channel_str}.csv"
#         x, y, data, timestamp = cam.take_image(navg=NAVG)
#         if dark_frame is None:
#             data_reduce = np.copy(data)
#         else:
#             data_reduce = data - dark_frame

#         # Report intensities
#         B_intensity = np.sum(data_reduce[outputs_masks[1]])
#         null_intensity = np.sum(data_reduce[outputs_masks[2]])
#         D_intensity = np.sum(data_reduce[outputs_masks[3]])
#         _log.info("voltages:", voltages)
#         _log.info("intensities:", B_intensity, null_intensity, D_intensity)
#         _log.info("\n")
#         return null_intensity/(B_intensity + D_intensity)
    
#     from numpy.random import default_rng
#     rng = default_rng(seed=4524)
#     sig_v = 0.2 # [V]
#     n_v = len(initial_voltages)
#     initial_simplex = initial_voltages[np.newaxis,:] + sig_v*rng.normal(size=(n_v+1,n_v))

#     xatol = 0.05 # [V] should be bigger than resolution of xpow  FIXME  check this value
#     bounds = [(0., 12.)]*n_v

#     # _log.info(initial_simplex)
#     # _log.info(xatol)
#     # _log.info(bounds)
#     # _log.info(fit_func(initial_voltages))
#     from scipy.optimize import minimize
#     options = {'initial_simplex': initial_simplex, 
#                'xatol':xatol}
#     res = minimize(fit_func, None, method='Nelder-Mead', options=options, bounds=bounds)
#     null_depth = fit_func(res.x)
#     return res, null_depth


# def optimize_null_dynamicVOA(initial_voltages, channels, outputs_masks, ind_null, cam, xpow, tlx, dark_frame=None, dark_threshhold=3, bright_threshhold=10000, delta_VOA=3):
#     def fit_func(voltages):
#         # Set XPOW
#         for i, voltage in enumerate(voltages):
#             xpow.apply_voltage(channels[i], voltage)
#             time.sleep(XPOW_DELAY)
        
#         # Take bright image
#         x, y, data, timestamp = cam.take_image(navg=NAVG)
#         # data_reduce = np.copy(data)# * 10**((tlx.voa-20) / 10)
#         if dark_frame is None:
#             tlx.laser_off()
#             time.sleep(LASER_OFF_DELAY)
#             x, y, dark, timestamp = cam.take_image(navg=navg, filename=filename)
#             tlx.laser_off()
#             time.sleep(LASER_ON_DELAY)
#         else:
#             dark = np.copy(dark_frame)

#         # Subtract stripes
#         mask_stripes = outputs_masks.sum(axis=0)
#         inds = np.argwhere(mask_stripes==True)
#         inds_rows = list(set(inds[:,0]))
#         ind_min, ind_max = np.min(inds_rows), np.max(inds_rows)
#         inds_rows = np.arange(ind_min - 2*(ind_max-ind_min), ind_max + 2*(ind_max-ind_min))
#         for ind in inds_rows:
#             mask_stripes[ind,:] = True

#         data_stripes, dark_stripes = np.copy(data), np.copy(dark)
#         data_stripes[mask_stripes] = np.nan
#         dark_stripes[mask_stripes] = np.nan

#         data = data - np.nanmean(data_stripes, axis=0)
#         dark = dark - np.nanmean(dark_stripes, axis=0)
        
#         data_reduce = data - dark

#         # Report intensities
#         B_intensity = np.sum(data_reduce[outputs_masks[1]])
#         null_intensity = np.sum(data_reduce[outputs_masks[2]])
#         D_intensity = np.sum(data_reduce[outputs_masks[3]])
#         _log.info("voltages:", voltages)
#         _log.info("raw intensities:", np.sum(data[outputs_masks[1]]), np.sum(data[outputs_masks[2]]), np.sum(data[outputs_masks[3]]))
#         _log.info("intensities:", B_intensity, null_intensity, D_intensity)        

#         # Dark threshhold condition
#         null_bright = np.max(data[outputs_masks[ind_null]] )
#         null_dark = np.std(dark[outputs_masks[ind_null]])
#         _log.info("null_bright, null_dark:", null_bright, null_dark)
#         if null_bright/null_dark < dark_threshhold:
#             tlx.set_voa(tlx.voa - delta_VOA)
#         else:
#             # Bright threshhold condition
#             for mask in outputs_masks:
#                 if np.max(data[mask]) > bright_threshhold:
#                     tlx.set_voa(tlx.voa + delta_VOA)
#                     break 
#         _log.info("\n")
#         return null_intensity/(B_intensity + D_intensity)
    
#     from numpy.random import default_rng
#     rng = default_rng(seed=4524)
#     sig_v = 0.2 # [V]
#     n_v = len(initial_voltages)
#     initial_simplex = initial_voltages[np.newaxis,:] + sig_v*rng.normal(size=(n_v+1,n_v))

#     xatol = 0.05 # [V] should be bigger than resolution of xpow  FIXME  check this value
#     bounds = [(0., 12.)]*n_v

#     # _log.info(initial_simplex)
#     # _log.info(xatol)
#     # _log.info(bounds)
#     # _log.info(fit_func(initial_voltages))
#     from scipy.optimize import minimize
#     options = {'initial_simplex': initial_simplex, 
#                'xatol':xatol}
#     res = minimize(fit_func, None, method='Nelder-Mead', options=options, bounds=bounds)
#     null_depth = fit_func(res.x)
#     return res, null_depth


# def optimize_null(initial_voltages, channels, coords, ind_null, cam, xpow, tec,
#                   dark_frame, inds_norm=None, tlx=None, navg=NAVG, atten_null=40, atten_norm=40, mask_range=5, 
#                   filt_size=5, plot=False, r_max=20, r_lin=5, bounds=None, sig_v=0.5, xatol=0.5):
    
#     all_voltages = []
#     null_metrics = []
#     temps = [] 
#     dark = np.copy(dark_frame)
     

#     null_coords = coords[ind_null].astype(int)
#     if inds_norm is not None:
#         norm_coords = [coords[ind].astype(int) for ind in inds_norm]
#         attens_norm = atten_norm * np.ones(len(inds_norm))

#     def fit_func(voltages):
#         nonlocal atten_null, attens_norm
#         # Set XPOW
#         for i, voltage in enumerate(voltages):
#             xpow.apply_voltage(channels[i], voltage)
#             time.sleep(XPOW_DELAY)

#         # Take null mode image
#         atten, data_null = attenuation_adjust(xpow, cam, null_coords, dark_frame, tlx=tlx, R=mask_range, navg=NAVG, stack=False, output_name='null')
#         if atten is not None:
#             atten_null = np.copy(atten)
#         output_null = extract_outputs(data_null, 1, r_max=r_max, r_lin=r_lin, 
#                                     filt_size=filt_size, plot=plot, coords=np.array([null_coords]), find_centers=False) * 10**(atten_null/10)
#         # set_VOA(atten_null, xpow, tlx)
#         # _, _, data_null, _ = cam.take_image(navg=navg, stack=True)

#         # while np.max(data_null*null_mask) > 12000:
#         #     try:
#         #         atten_null = atten_null + 3
#         #         set_VOA(atten_null, xpow, tlx)
#         #         time.sleep(3)
#         #     except:
#         #         raise ValueError(f'Output {output_name} saturating at max attenuation limit of VOA.')
#         #     _, _, data_null, _ = cam.take_image(navg=navg, stack=False)   
#         # while np.max((data_null-dark)*null_mask) < 5*np.std(dark):
#         #     if atten_null > 0.1:
#         #         atten_null -= 0.1
#         #         xpow.apply_voltage(ATTEN_CHANNEL, atten_null)
#         #     else:
#         #         raise ValueError('Null mode underexposed at min attenuation limit of external VOA.')
#         #     x, y, data_null, timestamp = cam.take_image(navg=navg, stack=True)

#         # Take norm images
#         if inds_norm is not None:
#             outputs_norm = 0
#             for i, coord in enumerate(norm_coords):
#                 atten, data_norm = attenuation_adjust(xpow, cam, norm_coords[i], dark_frame, tlx=tlx, R=mask_range, navg=NAVG, stack=False, output_name=f'norm {i}')
#                 if atten is not None:
#                     attens_norm[i] = atten_norm
#                 outputs_norm += extract_outputs(data_norm, 1, r_max=r_max, r_lin=r_lin, 
#                                             filt_size=filt_size, plot=plot, coords=np.array([norm_coords[i]]), find_centers=False) * 10**(attens_norm[i]/10)
#         else:
#             outputs_norm = 1


#         # xpow.apply_voltage(ATTEN_CHANNEL, atten_side)
#         # x, y, data_side, timestamp = cam.take_image(navg=navg, stack=True)
#         # # print('Side saturation: ',np.max(data_side*side_mask) > 12000)
#         # while np.max(data_side*side_mask) > 12000:
#         #     if atten_side < 4.7:
#         #         atten_side += 0.1
#         #         xpow.apply_voltage(ATTEN_CHANNEL, atten_side)
#         #     else:
#         #         raise ValueError('Side mode saturating at max attenuation limit of external VOA.')
#         #     x, y, data_side, timestamp = cam.take_image(navg=navg, stack=True)
#         # # print('Side underexposure: ',np.max((data_side-dark)*side_mask) < 3*np.std(dark))
#         # while np.max((data_side-dark)*side_mask) < 3*np.std(dark):
#         #     if atten_side > 0.1:
#         #         atten_side -= 0.1
#         #         xpow.apply_voltage(ATTEN_CHANNEL, atten_side)
#         #     else:
#         #         raise ValueError('Side mode underexposed at min attenuation limit of external VOA.')
#         #     x, y, data_side, timestamp = cam.take_image(navg=navg, stack=True)

#         # outputs_masks_null = build_output_mask(data_null - dark, num_outputs, coords=coords, mask_range=mask_range,
#         #                                     filt_size=filt_size, plot=plot)
#         # outputs_masks_side = build_output_mask(data_side - dark, num_outputs, coords=coords, mask_range=mask_range,
#         #                                     filt_size=filt_size, plot=plot)
#         # data_null = remove_stripes(data_null, outputs_masks_null) - dark_noStripe
#         # data_side = remove_stripes(data_side, outputs_masks_side) - dark_noStripe
#         # outputs_null = extract_outputs(data_null, num_outputs, r_max=r_max, r_lin=r_lin, 
#         #                             filt_size=filt_size, plot=plot, coords=coords)
#         # outputs_side = extract_outputs(data_side, num_outputs, r_max=r_max, r_lin=r_lin, 
#         #                             filt_size=filt_size, plot=plot, coords=coords)

#         # # Report intensities
#         # atten_null_dB, atten_side_dB = externalVOA_voltage_to_dB(atten_null), externalVOA_voltage_to_dB(atten_side)
#         # scaling = 10**(abs(atten_null_dB-atten_side_dB)/10)
#         # null_value = outputs_null[ind_null]/(np.sum(outputs_side[inds_norm]) * scaling)
#         null_value = output_null/outputs_norm
#         _log.info("voltages:"+str(voltages))
#         # _log.info("raw intensities:", np.sum(data[outputs_masks[1]]), np.sum(data[outputs_masks[2]]), np.sum(data[outputs_masks[3]]))
#         _log.info("Null mode intensities: "+str(output_null))   
#         _log.info("Side mode intensities: "+str(outputs_norm))   
#         _log.info("Null: "+str(10*np.log10(null_value))+" dB")
#         _log.info("\n")
#         all_voltages.append(voltages)
#         null_metrics.append(null_value)
#         temps.append(tec.read_temperature())
#         return null_value
    
#     from numpy.random import default_rng
#     rng = default_rng(seed=4524)
#     n_v = len(initial_voltages)
#     initial_simplex = initial_voltages[np.newaxis,:] + sig_v*rng.normal(size=(n_v+1,n_v))

#     # xatol [V] should be bigger than resolution of xpow  FIXME  check this value
#     if bounds is None:
#         bounds = [(0., MAXVOLTAGE)]*n_v

#     # _log.info(initial_simplex)
#     # _log.info(xatol)
#     # _log.info(bounds)
#     # _log.info(fit_func(initial_voltages))
#     from scipy.optimize import minimize
#     options = {'initial_simplex': initial_simplex, 
#                'xatol':xatol,
#                'maxiter':10}
#     res = minimize(fit_func, None, method='Nelder-Mead', options=options, bounds=bounds)
#     null_depth = fit_func(res.x)
#     all_voltages = np.array(all_voltages)
#     null_metrics = np.array(null_metrics)
#     temps = np.array(temps)
#     return res, null_depth, all_voltages, null_metrics, temps


# def optimize_null_OSA(initial_voltages, channels, osa, xpow, tec, lambda_0=1550, bandwidth=12, navg=5, plot=False, bounds=None, 
#                       xatol=0.5):
    
#     all_voltages = []
#     min_metrics = []
#     temps = []
#     def fit_func(voltages):
#         # Set XPOW
#         for i, voltage in enumerate(voltages):
#             xpow.apply_voltage(channels[i], voltage)
#             time.sleep(XPOW_DELAY)
        
#         spectrum, sigma = osa.read_spectrum(navg=navg, plot=plot, filename=None)
#         crop_cond = abs(spectrum[0]-lambda_0) < bandwidth/2
#         min_metric = np.mean(spectrum[1][crop_cond])
#         _log.info("voltages:"+str(voltages))
#         _log.info("Minimum metric: "+str(min_metric))
#         _log.info("\n")
#         all_voltages.append(voltages)
#         min_metrics.append(min_metric)
#         temps.append(tec.read_temperature())
#         return min_metric
    
#     from numpy.random import default_rng
#     rng = default_rng(seed=4524)
#     sig_v = 0.5 # [V]
#     n_v = len(initial_voltages)
#     initial_simplex = initial_voltages[np.newaxis,:] + sig_v*rng.normal(size=(n_v+1,n_v))

#     # xatol [V] should be bigger than resolution of xpow  FIXME  check this value
#     if bounds is None:
#         bounds = [(0., MAXVOLTAGE)]*n_v

#     # _log.info(initial_simplex)
#     # _log.info(xatol)
#     # _log.info(bounds)
#     # _log.info(fit_func(initial_voltages))
#     from scipy.optimize import minimize
#     options = {'initial_simplex': initial_simplex, 
#                'xatol':xatol,
#                'maxiter':100}
#     res = minimize(fit_func, initial_voltages, method='Nelder-Mead', options=options, bounds=bounds)
#     min_metric = fit_func(res.x)
#     all_voltages = np.array(all_voltages)
#     min_metrics = np.array(min_metrics)
#     temps = np.array(temps)
#     return res, min_metric, all_voltages, min_metrics, temps

### Data reduction routines ###

# def find_output_centers(frame, num_outputs, init_coords=None, filt_size=5, plot=True):
#     ''' Finds the coordinates of each output using the pixel of maximum intensity for each output.
    
#         Parameters
#         ----------
#         frame : 2D numpy array
#             Image data
#         num_outputs : int
#             Number of outputs to identify in reference image
#         init_coords : 2D numpy array
#             Initial guess for coordinates of each output 
#         filt_size : int
#             Size of maximum filter (in pixels) for isolating local maxima to identify outputs. Default: 5
#         plot : boolean
#             Flag for plotting masks overlayed on image

#         Returns
#         -------
#         coords : 2D numpy array
#             Coordinates of each output    
#     '''

#     if init_coords is None:
#         # Find local maxima
#         filtered = scipy.ndimage.maximum_filter(frame, size=filt_size)
#         masks = (frame == filtered)
#         masks = np.stack([scipy.ndimage.binary_dilation(mask, iterations=2) for mask in masks])
#         # Label each local maximum region
#         labeled, num_features = scipy.ndimage.label(masks)
#         # Compute centroid of each labeled region
#         centroids = np.array(scipy.ndimage.center_of_mass(frame, labeled, range(1, num_features+1)))

#         # Compute total intensity in each region
#         region_sums = np.array([frame[labeled==i+1].sum() for i in range(num_features)])
#         top_idx = np.argsort(region_sums)[-num_outputs:]
#         coords = centroids[top_idx]

#         # Sort by x-coordinate to keep ABCDE order
#         coords = coords[np.argsort(coords[:, 1])]#.astype(int)    

#     # else:
#     #     num_features, labeled = len(init_coords), np.zeros_like(frame, dtype=int)
#     #     R = filt_size
#     #     for i, (r, c) in enumerate(init_coords, start=1):
#     #         r, c = int(r), int(c)
#     #         labeled[
#     #             max(0, r-R):r+R+1,
#     #             max(0, c-R):c+R+1
#     #         ] = i 
#     else:
#         coords = []

#         R = filt_size

#         for r, c in init_coords:

#             r, c = int(r), int(c)

#             r0 = max(0, r - R)
#             r1 = min(frame.shape[0], r + R + 1)

#             c0 = max(0, c - R)
#             c1 = min(frame.shape[1], c + R + 1)

#             # Crop local search region
#             subframe = frame[r0:r1, c0:c1]
#             subframe = np.clip(subframe, 0, None)

#             # Compute local center of mass only within this region
#             rr, cc = scipy.ndimage.center_of_mass(subframe)

#             coords.append([r0 + rr, c0 + cc])

#         coords = np.array(coords)       

    
#     if plot:
#         y_dim, x_dim = np.shape(frame)

#         fig, ax = plt.subplots(1, 1, figsize=(10,6))
#         im = ax.pcolormesh(frame, zorder=0)
#         for coord in coords:
#             ax.scatter(coord[1]+0.5, coord[0]+0.5, marker='+', color='r', alpha=0.5)
#         fig.colorbar(im, ax=ax, fraction=0.05*y_dim/x_dim)
#         ax.set_xlabel('x (pixels)', fontsize=18)
#         ax.set_ylabel('y (pixels)', fontsize=18)
#         ax.set_aspect('equal')
#         plt.show()
#     return np.asarray(coords)

# def build_output_mask(frame, num_outputs, coords=None, find_centers=True, mask_range=5, filt_size=5, plot=True):
#     ''' Builds a collection of boolean masks identifying each output in the reference image.
    
#         Parameters
#         ----------
#         frame : 2D numpy array
#             Image data
#         num_outputs : int
#             Number of outputs to identify in reference image
#         coords : 2D numpy array or None
#             Coordinate pairs of output centers; initial guess if find_centers=True, else used as fixed output centers.
#         find_centers : boolean
#             True if need to find output centers; else False. Cannot be False if coords is None.
#         mask_range : int
#             Mask radius (in pixels) around center of output. Default: 5
#         filt_size : int
#             Size of maximum filter (in pixels) for isolating local maxima to identify outputs. Default: 5
#         plot : boolean
#             Flag for plotting masks overlayed on image

#         Returns
#         -------
#         outputs_masks : 3D numpy array
#             Collection of 2D boolean masks for each output in reference image    
#     '''
#     if coords is None and find_centers == False:
#         raise ValueError("Cannot set find_centers to False if coords is None.")

#     # Determine coordinates of outputs
#     if find_centers:
#         coords = find_output_centers(frame, num_outputs, init_coords=coords, filt_size=filt_size, plot=False)

#     # Create 2D grid
#     height, width = np.shape(frame)
#     Y, X = np.ogrid[:height, :width]  # Y: (height,1), X: (1,width)
    
#     # Reshape coords for broadcasting: (N,1,1)
#     Xc = coords[:, 1][:, None, None]
#     Yc = coords[:, 0][:, None, None]
    
#     # Compute squared distance for each coordinate
#     dist2 = (X - Xc)**2 + (Y - Yc)**2  # shape: (N, height, width)
    
#     # Create masks
#     outputs_masks = dist2 <= mask_range**2

#     # # Make 2D masks of the outputs
#     # outputs_masks = []
#     # for coord in coords:
#     #     mask = np.zeros_like(frame, dtype=bool)
#     #     mask[coord[0]-mask_range:coord[0]+mask_range+1, coord[1]-mask_range:coord[1]+mask_range+1] = True 
#     #     outputs_masks.append(mask)

#     # Plot the reference data with the output mask
#     if plot:
#         yellow_cmap = ListedColormap(['red'])
#         y_dim, x_dim = np.shape(frame)

#         fig, ax = plt.subplots(1, 1, figsize=(10,6))
#         im = ax.pcolormesh(frame, zorder=0)
#         for mask in outputs_masks:
#             masked_overlay = np.ma.masked_where(~mask, np.ones_like(mask))
#             ax.pcolormesh(masked_overlay, cmap=yellow_cmap, alpha=0.3, zorder=1)
#         fig.colorbar(im, ax=ax, fraction=0.05*y_dim/x_dim)
#         ax.set_xlabel('x (pixels)', fontsize=18)
#         ax.set_ylabel('y (pixels)', fontsize=18)
#         ax.set_aspect('equal')
#         plt.show()

#     return np.array(outputs_masks)

# def plot_masked(frame, mask, vmax=12000):
#     masked_data = np.where(mask, frame, np.nan)

#     ys, xs = np.where(mask)

#     pad = 2  # extra pixels around mask

#     xmin = xs.min() - pad
#     xmax = xs.max() + pad
#     ymin = ys.min() - pad
#     ymax = ys.max() + pad

#     fig, ax = plt.subplots(1, 1, figsize=(6,6))

#     im = ax.pcolormesh(masked_data, zorder=0, vmin=0, vmax=vmax)

#     ax.set_xlim(xmin, xmax + 1)
#     ax.set_ylim(ymax + 1, ymin)   # invert y-axis to match image orientation

#     fig.colorbar(im, ax=ax)

#     ax.set_xlabel('x (pixels)', fontsize=18)
#     ax.set_ylabel('y (pixels)', fontsize=18)
#     ax.set_aspect('equal')

#     plt.show()
#     return

# def remove_stripes(frame, outputs_masks=None):
#     ''' Removes stripes from Xenics camera images.

#         Parameters
#         ----------
#         frame : 2D numpy array
#             Image data
#         outputs_masks : None or 3D numpy array
#             Collection of 2D boolean masks for each output in reference image

#         Returns
#         -------
#         frame_reduced : 2D numpy array
#             Image data with stripes removed
#     '''
#     if outputs_masks is not None:
#         mask_stripes = np.sum(outputs_masks, axis=0)
#         inds = np.argwhere(mask_stripes==True)
#         inds_rows = list(set(inds[:,0]))
#         ind_min, ind_max = np.min(inds_rows), np.max(inds_rows)
#         inds_rows = np.arange(ind_min - 5*(ind_max-ind_min), ind_max + 5*(ind_max-ind_min))
#         for ind in inds_rows:
#             mask_stripes[ind,:] = True

#         frame_stripes = np.copy(frame)
#         frame_stripes[mask_stripes] = np.nan
#     else:
#         frame_stripes = np.copy(frame)

#     frame_reduced = frame - np.nanmean(frame_stripes, axis=0)
#     return frame_reduced

# def extract_outputs_frame(frame, num_outputs, r_max=15, r_lin=5, filt_size=5, plot=False, coords=None, find_centers=True):
#     ''' Get output intensities from data frame using Nem's photometry extraction method.

#         Parameters
#         ----------
#         frame : 2D numpy array
#             Image data
#         num_outputs : int
#             Number of outputs to identify in reference image
#         r_max : int
#             Maximum pixel radius for photometry extraction method. Default: 15
#         r_lin : int
#             Lower limit of pixel radius at which photometry extraction method behaves linearly. Default: 5
#         filt_size : int
#             Size of maximum filter (in pixels) for isolating local maxima to identify outputs. Default: 5
#         plot : boolean
#             Flag for plotting masks overlayed on image and photometry extraction
#         coords : 2D numpy array or None
#             Coordinate pairs of output centers; if None, finds output centers
#         find_centers : boolean
#             True if need to find output centers; else False. Cannot be False if coords is None.

#         Returns
#         -------
#         outputs : list of numpy arrays
#     '''

#     radii = np.arange(1, r_max)
#     enclosed_fluxes = []

#     for r in radii:
#         if r == np.max(radii):

#             outputs_masks = build_output_mask(frame, num_outputs, mask_range=r, filt_size=filt_size, 
#                                               plot=plot, coords=coords, find_centers=find_centers)
#         else:
#             outputs_masks = build_output_mask(frame, num_outputs, mask_range=r, filt_size=filt_size, 
#                                               plot=False, coords=coords, find_centers=find_centers)
#         enclosed_fluxes.append(np.sum(outputs_masks * frame, axis=(1, 2)))
#     enclosed_fluxes = np.array(enclosed_fluxes).T

#     enclosed_pixels = np.pi*radii**2
#     fit = radii > r_lin

#     if plot:
#         import string
#         plt.figure()

#     outputs = []
#     for i in range(num_outputs):
#         enclosed_flux = enclosed_fluxes[i]
#         m, flux = np.polyfit(enclosed_pixels[fit], enclosed_fluxes[i][fit], 1)
#         flux = enclosed_flux[-1] ### REMOVE THIS

#         if plot:        
#             plt.plot(enclosed_pixels, enclosed_flux, 'o', color=f"C{i}", label=f"Output {string.ascii_uppercase[i]}")
#             plt.plot(enclosed_pixels, m*enclosed_pixels+flux, linestyle='dashed', color=f"C{i}")
#         outputs.append(flux)
#     outputs = np.array(outputs)
#     if plot:
#         plt.xlim(0, np.max(enclosed_pixels))
#         plt.xlabel('Enclosed pixels', fontsize=16)
#         plt.ylabel('Enclosed counts', fontsize=16)
#         plt.legend()
#         plt.show()
#     return outputs

# def extract_outputs(frames, num_outputs, r_max=15, r_lin=5, filt_size=5, plot=False, coords=None, find_centers=True, stack=True):
#     ''' Get output intensities from data frame using Nem's photometry extraction method.

#         Parameters
#         ----------
#         frame : 2D or 3D numpy array
#             Image data
#         num_outputs : int
#             Number of outputs to identify in reference image
#         r_max : int
#             Maximum pixel radius for photometry extraction method. Default: 15
#         r_lin : int
#             Lower limit of pixel radius at which photometry extraction method behaves linearly. Default: 5
#         filt_size : int
#             Size of maximum filter (in pixels) for isolating local maxima to identify outputs. Default: 5
#         plot : boolean
#             Flag for plotting masks overlayed on image and photometry extraction
#         coords : 2D numpy array or None
#             Coordinate pairs of output centers; if None, finds output centers
#         find_centers : boolean
#             True if need to find output centers; else False. Cannot be False if coords is None.
#         stack : boolean
#             If True, mean-stack all frames; else, don't combine frames.            

#         Returns
#         -------
#         outputs : list of numpy arrays
#     '''
#     if len(np.shape(frames)) == 2:
#         outputs = extract_outputs_frame(frames, num_outputs, coords=coords, find_centers=find_centers, r_max=r_max, r_lin=r_lin, 
#                                         filt_size=filt_size, plot=plot)
#     elif len(np.shape(frames)) == 3:
#         outputs = []
#         for i in range(len(frames)):
#             outputs.append(extract_outputs_frame(frames[i], num_outputs, coords=coords, find_centers=find_centers, r_max=r_max, 
#                                                  r_lin=r_lin, filt_size=filt_size, plot=plot))
#         outputs = np.array(outputs)
#         if stack:
#             outputs = np.mean(outputs, axis=0)
#     return outputs

# def reduce_voltage_scan(filename_root, num_outputs, filename_dark, filt_size=5,
#                         coords=None, find_centers=True, mask_range=5, r_max=15, r_lin=5, plot=False):
#     ''' Get output intensities for each image produced by a voltage scan.

#         Parameters
#         ----------
#         filename_root : str
#             Path and beginning identifier of each file in voltage scan image collection
#         num_outputs : int
#             Number of outputs to identify in image
#         filename_dark : None or str
#             If None, uses voltage-specific filename for reading dark image; else uses provided string as dark filename
#         filt_size : int
#             Size of maximum filter (in pixels) for isolating local maxima to identify outputs. Default: 5
#         coords : 2D numpy array or None
#             Coordinate pairs of output centers; if None, finds output centers
#         find_centers : boolean
#             True if need to find output centers; else False. Cannot be False if coords is None.
#         mask_range : int
#             Mask radius (in pixels) around center of output for stripe removal. Default: 5
#         r_max : int
#             Maximum pixel radius for photometry extraction method. Default: 15
#         r_lin : int
#             Lower limit of pixel radius at which photometry extraction method behaves linearly. Default: 5
#         plot : boolean
#             Flag for plotting masks overlayed on image and photometry extraction

#         Returns
#         -------
#         voltages : list of 1D numpy arrays
#             Voltages scanned by each channel
#         outputs : list of numpy arrays
#             Output intensities at each voltage setting
#     '''
#     # Get channels
#     filename_log = glob.glob(filename_root+"*.log")[0]
#     channel_str = filename_log.split(filename_root+'_')[-1].split('.log')[0]
#     channel_str = channel_str.split('_')
#     channels = [int(channel.split('Ch')[-1]) for channel in channel_str]

#     # Read voltages from data log
#     log = np.loadtxt(filename_log, delimiter=',')
#     with open(filename_log) as f:
#         colnames = f.readline().strip().strip('# ').split(',')
#     log = np.core.records.fromarrays(log.T, names=colnames)

#     voltages = []
#     for channel in channels:
#         voltages.append(np.sort(list(set(log['inputVoltage'+str(channel)]))))
#     shape = tuple(len(v) for v in voltages)

#     dark = np.loadtxt(filename_dark, delimiter=',')
#     # dark = remove_stripes(dark)

#     #####################################################
#     masks = []
#     if coords is not None:
#         for coord in coords:
#             mask, R = np.zeros(np.shape(dark), bool), 5
#             r, c =  coord
#             r, c = int(r), int(c)
#             mask[max(0,r-R):r+R+1, max(0,c-R):c+R+1] = 1
#             masks.append(mask)
#     ##########################################################
#     cols = [f'inputVoltage{channel}' for channel in channels]

#     # Actual measured voltage coordinates
#     voltage_coords = np.unique(
#         np.column_stack([log[col] for col in cols]),
#         axis=0
#     )

#     outputs = np.empty((len(voltage_coords), num_outputs))
#     attens = np.empty((len(voltage_coords), num_outputs))
#     output_fluxes = np.empty((len(voltage_coords), num_outputs))
#     output_names = [chr(65+i) for i in range(num_outputs)]

#     for idx_i, voltage_coord in enumerate(voltage_coords):
#         voltage_coord = [voltages[d][idx[d]] for d in range(len(voltages))]
#         v_str = ''
#         for i in range(len(voltage_coord)):
#             v_i = f"{voltage_coord[i]:.3f}".replace('.', 'p')
#             v_str += f"*_V{v_i}"

#         output_V = [np.nan]*num_outputs
#         output_flux = [np.nan]*num_outputs
#         atten_V = [np.nan]*num_outputs
#         for o, output_name in enumerate(output_names):
#             mode_ext = f'output{output_name}'
#             print(filename_root+f"{v_str}*{mode_ext}*")
#             filename = glob.glob(filename_root+f"{v_str}*{mode_ext}*")[0]
#             atten = float(filename.split(f'{mode_ext}_')[-1].split('.npy')[0].replace('p', '.'))
#             atten_V[o] = atten

#             data = np.load(filename)
#             # if (idx_i == 0) and (coords is not None):
#             #     if data.ndim == 2: # stacked frame
#             #         outputs_masks = build_output_mask(data, num_outputs, coords=coords, mask_range=mask_range,
#             #                                 filt_size=filt_size, plot=plot)
#             #     elif data.ndim == 3:
#             #         outputs_masks = build_output_mask(np.sum(data, axis=0), num_outputs, coords=coords, mask_range=mask_range,
#             #                                 filt_size=filt_size, plot=plot)

#             if data.ndim == 2:
#                 # if coords is None:
#                 #     outputs_masks = build_output_mask(data-dark, num_outputs, coords=coords, mask_range=mask_range,
#                                                     # filt_size=filt_size, plot=plot)
#                 # data = remove_stripes(data, outputs_masks)    
#                 output = BENCH.extract_outputs(data-dark, num_outputs, r_max=r_max, r_lin=r_lin, 
#                                         filt_size=filt_size, plot=plot, coords=coords, find_centers=find_centers)
#             elif data.ndim == 3:
#                 output = []
#                 for data_i in data:
#                 # data_i = data[21]
#                     # if coords is None:
#                     #     outputs_masks = build_output_mask(data_i-dark, num_outputs, coords=coords, mask_range=mask_range,
#                                                     # filt_size=filt_size, plot=plot)
#                     # data_i = remove_stripes(data_i, outputs_masks)    
#                     output.append(BENCH.extract_outputs(data_i-dark, num_outputs, r_max=r_max, r_lin=r_lin, 
#                                                 filt_size=filt_size, plot=plot, coords=coords, find_centers=find_centers))
#                 output = np.nanmean(output, axis=0)
#             output_V[o] = output[o]
#             if coords is not None:
#                 output_flux[o] = np.nanmax((data_i-dark)*masks[o])

#         # Determine VOA scaling
#         # cols = [f"inputVoltage{channel}" for channel in channels]
#         # row = np.all([log[name] == val for name, val in zip(cols, coord)], axis=0)
#         # voa = log[row]['laserVOA']

#         outputs[idx_i] = output_V
#         attens[idx_i] = atten_V
#         output_fluxes[idx_i] = output_flux
#         print(voltage_coord, atten_V)
        
#     return voltage_coords, outputs, attens, output_fluxes

# def reduce_voltage_scan_adaptive(filename_root, num_outputs, filename_dark, filt_size=5,
#                         coords=None, find_centers=True, mask_range=5, r_max=15, r_lin=5, plot=False):
#     ''' Get output intensities for each image produced by a voltage scan.

#         Parameters
#         ----------
#         filename_root : str
#             Path and beginning identifier of each file in voltage scan image collection
#         num_outputs : int
#             Number of outputs to identify in image
#         filename_dark : None or str
#             If None, uses voltage-specific filename for reading dark image; else uses provided string as dark filename
#         filt_size : int
#             Size of maximum filter (in pixels) for isolating local maxima to identify outputs. Default: 5
#         coords : 2D numpy array or None
#             Coordinate pairs of output centers; if None, finds output centers
#         find_centers : boolean
#             True if need to find output centers; else False. Cannot be False if coords is None.
#         mask_range : int
#             Mask radius (in pixels) around center of output for stripe removal. Default: 5
#         r_max : int
#             Maximum pixel radius for photometry extraction method. Default: 15
#         r_lin : int
#             Lower limit of pixel radius at which photometry extraction method behaves linearly. Default: 5
#         plot : boolean
#             Flag for plotting masks overlayed on image and photometry extraction

#         Returns
#         -------
#         voltages : list of 1D numpy arrays
#             Voltages scanned by each channel
#         outputs : list of numpy arrays
#             Output intensities at each voltage setting
#     '''
#     # Get channels
#     filename_log = glob.glob(filename_root+"*.log")[0]
#     channel_str = filename_log.split(filename_root+'_')[-1].split('.log')[0]
#     channel_str = channel_str.split('_')
#     channels = [int(channel.split('Ch')[-1]) for channel in channel_str]

#     # Read voltages from data log
#     log = np.loadtxt(filename_log, delimiter=',')
#     with open(filename_log) as f:
#         colnames = f.readline().strip().strip('# ').split(',')
#     log = np.core.records.fromarrays(log.T, names=colnames)

#     dark = np.loadtxt(filename_dark, delimiter=',')
#     # dark = remove_stripes(dark)
#     #####################################################
#     masks = []
#     if coords is not None:
#         for coord in coords:
#             mask, R = np.zeros(np.shape(dark), bool), 5
#             r, c =  coord
#             r, c = int(r), int(c)
#             mask[max(0,r-R):r+R+1, max(0,c-R):c+R+1] = 1
#             masks.append(mask)
#     ##########################################################
#     cols = [f'inputVoltage{channel}' for channel in channels]

#     # Actual measured voltage coordinates
#     voltage_coords = np.unique(
#         np.column_stack([log[col] for col in cols]),
#         axis=0
#     )

#     outputs = np.empty((len(voltage_coords), num_outputs))
#     attens = np.empty((len(voltage_coords), num_outputs))
#     output_fluxes = np.empty((len(voltage_coords), num_outputs))
#     output_names = [chr(65+i) for i in range(num_outputs)]

#     for idx_i, voltage_coord in enumerate(voltage_coords):
#         v_str = ''
#         for i in range(len(voltage_coord)):
#             v_i = f"{voltage_coord[i]:.3f}".replace('.', 'p')
#             v_str += f"*_V{v_i}"

#         output_V = [np.nan]*num_outputs
#         output_flux = [np.nan]*num_outputs
#         atten_V = [np.nan]*num_outputs
#         for o, output_name in enumerate(output_names):
#             mode_ext = f'output{output_name}'
#             print(filename_root+f"{v_str}*{mode_ext}*")
#             filename = glob.glob(filename_root+f"{v_str}*{mode_ext}*")[0]
#             atten = float(filename.split(f'{mode_ext}_')[-1].split('.npy')[0].replace('p', '.'))
#             atten_V[o] = atten

#             data = np.load(filename)
#             # if (idx_i == 0) and (coords is not None):
#             #     if data.ndim == 2: # stacked frame
#             #         outputs_masks = build_output_mask(data, num_outputs, coords=coords, mask_range=mask_range,
#             #                                 filt_size=filt_size, plot=plot)
#             #     elif data.ndim == 3:
#             #         outputs_masks = build_output_mask(np.sum(data, axis=0), num_outputs, coords=coords, mask_range=mask_range,
#             #                                 filt_size=filt_size, plot=plot)

#             if data.ndim == 2:
#                 # if coords is None:
#                 #     outputs_masks = build_output_mask(data-dark, num_outputs, coords=coords, mask_range=mask_range,
#                                                     # filt_size=filt_size, plot=plot)
#                 # data = remove_stripes(data, outputs_masks)    
#                 output = BENCH.extract_outputs(data-dark, num_outputs, r_max=r_max, r_lin=r_lin, 
#                                         filt_size=filt_size, plot=plot, coords=coords, find_centers=find_centers)
#             elif data.ndim == 3:
#                 output = []
#                 for data_i in data:
#                 # data_i = data[21]
#                     # if coords is None:
#                     #     outputs_masks = build_output_mask(data_i-dark, num_outputs, coords=coords, mask_range=mask_range,
#                                                     # filt_size=filt_size, plot=plot)
#                     # data_i = remove_stripes(data_i, outputs_masks)    
#                     output.append(BENCH.extract_outputs(data_i-dark, num_outputs, r_max=r_max, r_lin=r_lin, 
#                                                 filt_size=filt_size, plot=plot, coords=coords, find_centers=find_centers))
#                 output = np.nanmean(output, axis=0)
#             output_V[o] = output[o]
#             if coords is not None:
#                 output_flux[o] = np.nanmax((data_i-dark)*masks[o])

#         # Determine VOA scaling
#         # cols = [f"inputVoltage{channel}" for channel in channels]
#         # row = np.all([log[name] == val for name, val in zip(cols, coord)], axis=0)
#         # voa = log[row]['laserVOA']

#         outputs[idx_i] = output_V
#         attens[idx_i] = atten_V
#         output_fluxes[idx_i] = output_flux
#         print(voltage_coord, atten_V)
        
#     return voltage_coords, outputs, attens, output_fluxes

# def reduce_voltage_scan_fixedAtten(filename_root, mode_ext, num_outputs, filename_dark=None, filt_size=5,
#                         coords=None, mask_range=5, r_max=15, r_lin=5, plot=False):
#     ''' Get output intensities for each image produced by a voltage scan.

#         Parameters
#         ----------
#         filename_root : str
#             Path and beginning identifier of each file in voltage scan image collection
#         mode_ext : str
#             Extension corresponding to attenuation mode of data to reduce
#         num_outputs : int
#             Number of outputs to identify in image
#         filename_dark : None or str
#             If None, uses voltage-specific filename for reading dark image; else uses provided string as dark filename
#         filt_size : int
#             Size of maximum filter (in pixels) for isolating local maxima to identify outputs. Default: 5
#         coords : 2D numpy array or None
#             Coordinate pairs of output centers; if None, finds output centers
#         mask_range : int
#             Mask radius (in pixels) around center of output for stripe removal. Default: 5
#         r_max : int
#             Maximum pixel radius for photometry extraction method. Default: 15
#         r_lin : int
#             Lower limit of pixel radius at which photometry extraction method behaves linearly. Default: 5
#         plot : boolean
#             Flag for plotting masks overlayed on image and photometry extraction

#         Returns
#         -------
#         voltages : list of 1D numpy arrays
#             Voltages scanned by each channel
#         outputs : list of numpy arrays
#             Output intensities at each voltage setting
#     '''
#     # Get channels
#     filename_log = glob.glob(filename_root+"*.log")[0]
#     channel_str = filename_log.split(filename_root+'_')[-1].split('.log')[0]
#     channel_str = channel_str.split('_')
#     channels = [int(channel.split('Ch')[-1]) for channel in channel_str]

#     # Read voltages from data log
#     log = np.loadtxt(filename_log, delimiter=',')
#     with open(filename_log) as f:
#         colnames = f.readline().strip().strip('# ').split(',')
#     log = np.core.records.fromarrays(log.T, names=colnames)

#     voltages = []
#     for channel in channels:
#         voltages.append(np.sort(list(set(log['inputVoltage'+str(channel)]))))
#     shape = tuple(len(v) for v in voltages)

#     if filename_dark is not None:
#         dark = np.loadtxt(filename_dark, delimiter=',')
#         dark = remove_stripes(dark)

#     #####################################################
#     masks = []
#     if coords is not None:
#         for coord in coords:
#             mask, R = np.zeros(np.shape(dark), bool), 5
#             r, c =  coord
#             r, c = int(r), int(c)
#             mask[max(0,r-R):r+R+1, max(0,c-R):c+R+1] = 1
#             masks.append(mask)
#     ##########################################################
          
#     outputs = np.empty((*shape, num_outputs))
#     output_fluxes = np.empty((*shape, num_outputs))
#     for idx_i, idx in enumerate(np.ndindex(shape)):
#         voltage_coord = [voltages[d][idx[d]] for d in range(len(voltages))]
#         print(voltage_coord)
#         v_str = ''
#         for i in range(len(voltage_coord)):
#             v_i = f"{voltage_coord[i]:.2f}".replace('.', 'p')
#             v_str += f"*_V{v_i}"

#         filename = glob.glob(filename_root+f"{v_str}*{mode_ext}*")[0]
#         if filename_dark is None:
#             filename_dark = glob.glob(filename_root + v_str + '_dark*')[0]
#             dark = np.load(filename_dark)
#             dark = BENCH.remove_stripes(dark)
        
#         data = np.load(filename)
#         if (idx_i == 0) and (coords is not None):
#             if data.ndim == 2: # stacked frame
#                 outputs_masks = BENCH.build_output_mask(data, num_outputs, coords=coords, mask_range=mask_range,
#                                         filt_size=filt_size, plot=plot)
#             elif data.ndim == 3:
#                 outputs_masks = BENCH.build_output_mask(np.sum(data, axis=0), num_outputs, coords=coords, mask_range=mask_range,
#                                         filt_size=filt_size, plot=plot)

#         if data.ndim == 2:
#             if coords is None:
#                 outputs_masks = BENCH.build_output_mask(data-dark, num_outputs, coords=coords, mask_range=mask_range,
#                                                 filt_size=filt_size, plot=plot)
#             data = BENCH.remove_stripes(data, outputs_masks)    
#             output = BENCH.extract_outputs(data-dark, num_outputs, r_max=r_max, r_lin=r_lin, 
#                                      filt_size=filt_size, plot=plot, coords=coords)
#         elif data.ndim == 3:
#             output = []
#             for data_i in data[:1]:
#                 if coords is None:
#                     outputs_masks = BENCH.build_output_mask(data_i-dark, num_outputs, coords=coords, mask_range=mask_range,
#                                                 filt_size=filt_size, plot=plot)
#                 data_i = BENCH.remove_stripes(data_i, outputs_masks)    
#                 output.append(BENCH.extract_outputs(data_i-dark, num_outputs, r_max=r_max, r_lin=r_lin, 
#                                               filt_size=filt_size, plot=plot, coords=coords))
#             output = np.nanmean(output, axis=0)
#         output_flux = [np.nan]*num_outputs
#         for o in range(num_outputs):    
#             output_flux[o] = np.nanmax((data_i-dark)*masks[o])

#         # Determine VOA scaling
#         # cols = [f"inputVoltage{channel}" for channel in channels]
#         # row = np.all([log[name] == val for name, val in zip(cols, coord)], axis=0)
#         # voa = log[row]['laserVOA']

#         outputs[tuple(idx)] = output
#         output_fluxes[tuple(idx)] = output_flux
        
#     return np.array(voltages), outputs, output_fluxes

def scale_voltage_scan(outputs, attens):
    """Get output intensities for each image produced by a voltage scan.

    Parameters
    ----------
    outputs : array-like
        Output intensities at each voltage setting
    attens : array-like
        Attenuation applied for each output at each voltage setting

    Returns
    -------
    outputs_scaled : ndarray
        Output intensities scaled by attenuation
    """

    outputs = np.asarray(outputs, dtype=float)
    attens = np.asarray(attens, dtype=float)

    # # Create attenuation array in dB
    # attens_dB = np.full(attens.shape, np.nan)
    # # Convert unique attenuation voltages to dB
    # for atten in np.unique(attens[~np.isnan(attens)]):
    #     attens_dB[attens == atten] = externalVOA_voltage_to_dB(atten)

    # Convert dB attenuation to linear scaling
    scaling = 10 ** (attens / 10)

    # Apply scaling
    outputs_scaled = outputs * scaling

    return outputs_scaled

import re

def read_ThorSpectra(path, log=True, delimiter=','):
    with open(path) as f:
        text = f.read()

    data = text.split("[Data]\n",1)[1].split("[EndOfFile]",1)[0]

    arr = np.fromstring(data.replace(delimiter, " "), sep=" ")
    arr = arr.reshape(-1,2)
    wav, flux = arr[:,0], arr[:,1]
    if log:
        flux = 10**(flux/10)
    return wav, flux

def crop_spectrum(wav, flux, wav_range):
    ind = np.argwhere((wav>=wav_range[0]) * (wav<=wav_range[1]))
    return wav[ind], flux[ind]

def read_voltage_scan_spectra(data_dir, prefix, channel, sep=';', delimiter=','):
    ''' Outputs all spectra associated with input filename prefix and modulation channel.

        Parameters
        ----------
        data_dir: string
            Path to directory where data being read in are stored
        prefix: string
            Filename prefix (can include wildcard * to encompass multiple voltage scans)
        channel: integer
            Channel being modulated (in voltage) in data of interest
        sep: string
            Separator used in data files
        
        Returns
        -------
        all_spectra: 3D numpy array
            Collection of spectra (nx2 arrays with wavelength in the first column and spectral power in the second column)
        powers: 1D numpy array
            Powers associated with voltages applied to obtain spectra
        actualvoltages: 1D numpy array
            Measured voltages associated with voltages applied to obtain spectra   
    '''
    # List all files of interest with spectra to plot
    print(data_dir+'%s*_Ch%d_V*' % (prefix, channel))
    files = glob.glob(data_dir+'%s*_Ch%d_V*' % (prefix, channel))
    
    # Identify all voltages associated with data
    dats = np.genfromtxt(glob.glob(f'{data_dir}{prefix}*.log')[0], names=True, comments="#", delimiter=',')
    voltages = dats[f'inputVoltage{channel}']

    # Average and store all spectra associated with the same voltage applied to the same channel
    all_spectra = []
    for v in voltages:
        v_str = f'{v:0.3f}'.replace('.', 'p')
        files = glob.glob(f'{data_dir}{prefix}*_Ch{channel}_V{v_str}*') # All spectra of a given voltage
        spectra_v = [] 
        for filename in files:
            # Read and store spectrum
            wav, flux = read_ThorSpectra(filename, log=False, delimiter=delimiter)
            spectra_v.append(np.vstack([wav, flux]).T)      
        
        # Average all iterations      
        spectrum_v = np.mean(spectra_v, axis=0)
        all_spectra.append(spectrum_v)

    all_spectra = np.array(all_spectra)

    return voltages, all_spectra


# def extract_outputs(filename_root, outputs_masks, filename_dark=None):
#     ''' Gets output intensities for each image with specific file identifier.
    
#         Parameters
#         ----------
#         filename_root : str
#             Path and beginning identifier of each file in image collection for which the outputs masks are being constructed
#         outputs_masks : 3D numpy array
#             Collection of 2D boolean masks for each output in reference image   
#         filename_dark : None or str
#             If None, uses voltage-specific filename for reading dark image; else uses provided string as dark filename

#         Returns
#         -------
#         voltages : list of floats
#             Voltages scanned by each channel
#         outputs : list of numpy arrays

#     '''
#     # Get channels
#     filename_log = glob.glob(filename_root+"*.log")[0]
#     channel_str = filename_log.split(filename_root+'_')[-1].split('.log')[0]
#     channel_str = channel_str.split('_')
#     channels = [int(channel.split('Ch')[-1]) for channel in channel_str]

#     # Read voltages from data log
#     log = np.loadtxt(filename_log, delimiter=',')
#     with open(filename_log) as f:
#         colnames = f.readline().strip().strip('# ').split(',')
#     log = np.core.records.fromarrays(log.T, names=colnames)

#     voltages = []
#     for channel in channels:
#         voltages.append(np.sort(list(set(log['inputVoltage'+str(channel)]))))
    
#     # Read outputs for each voltage setting
#     voltages_mesh = np.meshgrid(*voltages, indexing="ij")
#     coords = np.column_stack([voltage.ravel() for voltage in voltages_mesh])

#     outputs_bright, outputs_null = [], []
#     shape = [len(voltage) for voltage in voltages]

#     if filename_dark is not None:
#         dark = np.loadtxt(filename_dark, delimiter=',')
#         dark = remove_stripes(dark)
        
#     for output_i, mask in enumerate(outputs_masks):
#         _log.info(f"Output {output_i}...")
#         output_bright, output_null = [], []
#         for coord in coords:
#             print(coord)
#             v_str = ''
#             for i in range(len(channels)):
#                 v_i = f"{coord[i]:.2f}".replace('.', 'p')
#                 v_str += f"*_V{v_i}"
            
#             filename = glob.glob(filename_root + v_str + "_bright.csv")[0]
#             data = np.loadtxt(filename, delimiter=',')
#             data_bright = remove_stripes(data)

#             filename = glob.glob(filename_root + v_str + "_null.csv")[0]
#             data = np.loadtxt(filename, delimiter=',')
#             data_null = remove_stripes(data)

#             if filename_dark is None:
#                 filename_dark = filename_root + v_str + '_dark.csv'
#                 dark = np.loadtxt(filename_dark, delimiter=',')
#                 dark = remove_stripes(dark)
            
#             # Determine VOA scaling
#             # cols = [f"inputVoltage{channel}" for channel in channels]
#             # row = np.all([log[name] == val for name, val in zip(cols, coord)], axis=0)
#             # voa = log[row]['laserVOA']

#             # Subtract darks from data
#             data_reduce_bright = (data_bright - dark) * 10**((20-20) / 10)
#             data_reduce_null = (data_null - dark) * 10**((1-20) / 10)
#             output_bright.append(np.sum(data_reduce_bright[mask]))
#             output_null.append(np.sum(data_reduce_null[mask]))

#         outputs_bright.append(np.array(output_bright).reshape(shape))
#         outputs_null.append(np.array(output_null).reshape(shape))
#     outputs_null = np.array(outputs_null).T 
#     outputs_bright= np.array(outputs_bright).T            
#     return voltages, outputs_null, outputs_bright
