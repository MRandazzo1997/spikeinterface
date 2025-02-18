"""Sorting components: peak detection."""

import numpy as np

from spikeinterface.core.job_tools import ChunkRecordingExecutor, _shared_job_kwargs_doc
from spikeinterface.toolkit import get_noise_levels, get_channel_distances

from ..toolkit import get_chunk_with_margin

from .peak_localization import (dtype_localize_by_method, init_kwargs_dict,
                                localize_peaks_center_of_mass, localize_peaks_monopolar_triangulation)

try:
    import numba

    HAVE_NUMBA = True
except ImportError:
    HAVE_NUMBA = False

base_peak_dtype = [('sample_ind', 'int64'), ('channel_ind', 'int64'),
                   ('amplitude', 'float64'), ('segment_ind', 'int64')]


def detect_peaks(recording, method='by_channel', peak_sign='neg', detect_threshold=5, n_shifts=2,
                 local_radius_um=50, noise_levels=None, random_chunk_kwargs={},
                 outputs='numpy_compact', localization_dict=None, **job_kwargs):
    """Peak detection based on threshold crossing in term of k x MAD.

    Parameters
    ----------
    recording: RecordingExtractor
        The recording extractor object.
    method: 'by_channel', 'locally_exclusive'
        Method to use. Options:
            * 'by_channel' : peak are detected in each channel independently
            * 'locally_exclusive' : a single best peak is taken from a set of neighboring channels
    peak_sign: 'neg', 'pos', 'both'
        Sign of the peak.
    detect_threshold: float
        Threshold, in median absolute deviations (MAD), to use to detect peaks.
    n_shifts: int
        Number of shifts to find peak.
        For example, if `n_shift` is 2, a peak is detected if a sample crosses the threshold,
        and the two samples before and after are above the sample.
    local_radius_um: float
        The radius to use for detection across local channels.
    noise_levels: array, optional
        Estimated noise levels to use, if already computed.
        If not provide then it is estimated from a random snippet of the data.
    random_chunk_kwargs: dict, optional
        A dict that contain option to randomize chunk for get_noise_levels().
        Only used if noise_levels is None.
    outputs: 'numpy_compact', 'numpy_split', 'sorting'
        The type of the output. By default, "numpy_compact" returns an array with complex dtype.
    localization_dict : dict, optional
        Can optionally do peak localization at the same time as detection.
        This avoids running `localize_peaks` separately and re-reading the entire dataset.
    {}

    Returns
    -------
    peaks: array
        Detected peaks.

    Notes
    -----
    This peak detection ported from tridesclous into spikeinterface.
    """

    assert method in ('by_channel', 'locally_exclusive')
    assert peak_sign in ('both', 'neg', 'pos')
    assert outputs in ('numpy_compact', 'numpy_split', 'sorting')

    if method == 'locally_exclusive' and not HAVE_NUMBA:
        raise ModuleNotFoundError('"locally_exclusive" need numba which is not installed')

    if noise_levels is None:
        noise_levels = get_noise_levels(recording, return_scaled=False, **random_chunk_kwargs)

    abs_threholds = noise_levels * detect_threshold

    if method == 'locally_exclusive':
        assert local_radius_um is not None
        channel_distance = get_channel_distances(recording)
        neighbours_mask = channel_distance < local_radius_um
    else:
        neighbours_mask = None

    # deal with margin
    if localization_dict is None:
        extra_margin = 0
    else:
        assert isinstance(localization_dict, dict)
        assert localization_dict['method'] in dtype_localize_by_method.keys()
        localization_dict = init_kwargs_dict(localization_dict['method'], localization_dict)

        nbefore = int(localization_dict['ms_before'] * recording.get_sampling_frequency() / 1000.)
        nafter = int(localization_dict['ms_after'] * recording.get_sampling_frequency() / 1000.)
        extra_margin = max(nbefore, nafter)

    # and run
    func = _detect_peaks_chunk
    init_func = _init_worker_detect_peaks
    init_args = (recording.to_dict(), method, peak_sign, abs_threholds, n_shifts,
                 neighbours_mask, extra_margin, localization_dict)
    processor = ChunkRecordingExecutor(recording, func, init_func, init_args,
                                       handle_returns=True, job_name='detect peaks', **job_kwargs)
    peaks = processor.run()
    peaks = np.concatenate(peaks)

    if outputs == 'numpy_compact':
        return peaks
    elif outputs == 'sorting':
        # @alessio : here we can do what you did in old API
        # the output is a sorting where unit_id is in fact one channel
        raise NotImplementedError


detect_peaks.__doc__ = detect_peaks.__doc__.format(_shared_job_kwargs_doc)


def _init_worker_detect_peaks(recording, method, peak_sign, abs_threholds, n_shifts,
                              neighbours_mask, extra_margin, localization_dict):
    """Initialize a worker for detecting peaks."""

    if isinstance(recording, dict):
        from spikeinterface.core import load_extractor
        recording = load_extractor(recording)

    # create a local dict per worker
    worker_ctx = {}
    worker_ctx['recording'] = recording
    worker_ctx['method'] = method
    worker_ctx['peak_sign'] = peak_sign
    worker_ctx['abs_threholds'] = abs_threholds
    worker_ctx['n_shifts'] = n_shifts
    worker_ctx['neighbours_mask'] = neighbours_mask
    worker_ctx['extra_margin'] = extra_margin
    worker_ctx['localization_dict'] = localization_dict

    if localization_dict is not None:
        worker_ctx['contact_locations'] = recording.get_channel_locations()
        channel_distance = get_channel_distances(recording)

        ms_before = worker_ctx['localization_dict']['ms_before']
        ms_after = worker_ctx['localization_dict']['ms_after']
        worker_ctx['localization_dict']['nbefore'] = \
            int(ms_before * recording.get_sampling_frequency() / 1000.)
        worker_ctx['localization_dict']['nafter'] = \
            int(ms_after * recording.get_sampling_frequency() / 1000.)

        # channel sparsity
        channel_distance = get_channel_distances(recording)
        neighbours_mask = channel_distance < localization_dict['local_radius_um']
        worker_ctx['localization_dict']['neighbours_mask'] = neighbours_mask

    return worker_ctx


def _detect_peaks_chunk(segment_index, start_frame, end_frame, worker_ctx):

    # recover variables of the worker
    recording = worker_ctx['recording']
    peak_sign = worker_ctx['peak_sign']
    abs_threholds = worker_ctx['abs_threholds']
    n_shifts = worker_ctx['n_shifts']
    method = worker_ctx['method']
    extra_margin = worker_ctx['extra_margin']
    localization_dict = worker_ctx['localization_dict']

    margin = n_shifts + extra_margin

    # load trace in memory
    recording_segment = recording._recording_segments[segment_index]
    traces, left_margin, right_margin = get_chunk_with_margin(recording_segment, start_frame, end_frame, 
                                                              None, margin, add_zeros=True)

    if extra_margin > 0:
        # remove extra margin for detection step
        trace_detection = traces[extra_margin:-extra_margin]
    else:
        trace_detection = traces

    if method == 'by_channel':
        peak_sample_ind, peak_chan_ind = detect_peaks_by_channel(trace_detection, peak_sign, abs_threholds, n_shifts)
    elif method == 'locally_exclusive':
        peak_sample_ind, peak_chan_ind = detect_peak_locally_exclusive(trace_detection, peak_sign, abs_threholds, 
                                                                       n_shifts, worker_ctx['neighbours_mask'])

    if extra_margin > 0:
        peak_sample_ind += extra_margin

    peak_dtype = base_peak_dtype
    if localization_dict is None:
        peak_dtype = base_peak_dtype
    else:
        method = localization_dict['method']
        peak_dtype = base_peak_dtype + dtype_localize_by_method[method]

    peak_amplitude = traces[peak_sample_ind, peak_chan_ind]

    peaks = np.zeros(peak_sample_ind.size, dtype=peak_dtype)
    peaks['sample_ind'] = peak_sample_ind
    peaks['channel_ind'] = peak_chan_ind
    peaks['amplitude'] = peak_amplitude
    peaks['segment_ind'] = segment_index

    if localization_dict is not None:
        contact_locations = worker_ctx['contact_locations']
        neighbours_mask_for_loc = worker_ctx['localization_dict']['neighbours_mask']
        nbefore = worker_ctx['localization_dict']['nbefore']
        nafter = worker_ctx['localization_dict']['nafter']

        # TO BE CONTINUED here
        if localization_dict['method'] == 'center_of_mass':
            peak_locations = localize_peaks_center_of_mass(traces, peaks, contact_locations,
                                                           neighbours_mask_for_loc, nbefore, nafter)

        elif localization_dict['method'] == 'monopolar_triangulation':
            max_distance_um = worker_ctx['localization_dict']['max_distance_um']
            peak_locations = localize_peaks_monopolar_triangulation(traces, peaks, contact_locations,
                                                                    neighbours_mask_for_loc, nbefore, nafter,
                                                                    max_distance_um)

        for k in peak_locations.dtype.fields:
            peaks[k] = peak_locations[k]

    # make absolute sample index
    peaks['sample_ind'] += (start_frame - left_margin)

    return peaks


def detect_peaks_by_channel(traces, peak_sign, abs_threholds, n_shifts):
    """Detect peaks using the 'by channel' method."""

    traces_center = traces[n_shifts:-n_shifts, :]
    length = traces_center.shape[0]

    if peak_sign in ('pos', 'both'):
        peak_mask = traces_center > abs_threholds[None, :]
        for i in range(n_shifts):
            peak_mask &= traces_center > traces[i:i + length, :]
            peak_mask &= traces_center >= traces[n_shifts + i + 1:n_shifts + i + 1 + length, :]

    if peak_sign in ('neg', 'both'):
        if peak_sign == 'both':
            peak_mask_pos = peak_mask.copy()

        peak_mask = traces_center < -abs_threholds[None, :]
        for i in range(n_shifts):
            peak_mask &= traces_center < traces[i:i + length, :]
            peak_mask &= traces_center <= traces[n_shifts + i + 1:n_shifts + i + 1 + length, :]

        if peak_sign == 'both':
            peak_mask = peak_mask | peak_mask_pos

    # find peaks
    peak_sample_ind, peak_chan_ind = np.nonzero(peak_mask)
    # correct for time shift
    peak_sample_ind += n_shifts

    return peak_sample_ind, peak_chan_ind


def detect_peak_locally_exclusive(traces, peak_sign, abs_threholds, n_shifts, neighbours_mask):
    """Detect peaks using the 'locally exclusive' method."""

    assert HAVE_NUMBA, 'You need to install numba'
    traces_center = traces[n_shifts:-n_shifts, :]

    if peak_sign in ('pos', 'both'):
        peak_mask = traces_center > abs_threholds[None, :]
        peak_mask = _numba_detect_peak_pos(traces, traces_center, peak_mask, n_shifts,
                                           abs_threholds, peak_sign, neighbours_mask)

    if peak_sign in ('neg', 'both'):
        if peak_sign == 'both':
            peak_mask_pos = peak_mask.copy()

        peak_mask = traces_center < -abs_threholds[None, :]
        peak_mask = _numba_detect_peak_neg(traces, traces_center, peak_mask, n_shifts,
                                           abs_threholds, peak_sign, neighbours_mask)

        if peak_sign == 'both':
            peak_mask = peak_mask | peak_mask_pos

    # Find peaks and correct for time shift
    peak_sample_ind, peak_chan_ind = np.nonzero(peak_mask)
    peak_sample_ind += n_shifts

    return peak_sample_ind, peak_chan_ind


if HAVE_NUMBA:
    @numba.jit(parallel=True)
    def _numba_detect_peak_pos(traces, traces_center, peak_mask, n_shifts,
                               abs_threholds, peak_sign, neighbours_mask):
        num_chans = traces_center.shape[1]
        for chan_ind in numba.prange(num_chans):
            for s in range(peak_mask.shape[0]):
                if not peak_mask[s, chan_ind]:
                    continue
                for neighbour in range(num_chans):
                    if not neighbours_mask[chan_ind, neighbour]:
                        continue
                    for i in range(n_shifts):
                        if chan_ind != neighbour:
                            peak_mask[s, chan_ind] &= traces_center[s, chan_ind] >= traces_center[s, neighbour]
                        peak_mask[s, chan_ind] &= traces_center[s, chan_ind] > traces[s + i, neighbour]
                        peak_mask[s, chan_ind] &= traces_center[s, chan_ind] >= traces[n_shifts + s + i + 1, neighbour]
                        if not peak_mask[s, chan_ind]:
                            break
                    if not peak_mask[s, chan_ind]:
                        break
        return peak_mask


    @numba.jit(parallel=True)
    def _numba_detect_peak_neg(traces, traces_center, peak_mask, n_shifts,
                               abs_threholds, peak_sign, neighbours_mask):
        num_chans = traces_center.shape[1]
        for chan_ind in numba.prange(num_chans):
            for s in range(peak_mask.shape[0]):
                if not peak_mask[s, chan_ind]:
                    continue
                for neighbour in range(num_chans):
                    if not neighbours_mask[chan_ind, neighbour]:
                        continue
                    for i in range(n_shifts):
                        if chan_ind != neighbour:
                            peak_mask[s, chan_ind] &= traces_center[s, chan_ind] <= traces_center[s, neighbour]
                        peak_mask[s, chan_ind] &= traces_center[s, chan_ind] < traces[s + i, neighbour]
                        peak_mask[s, chan_ind] &= traces_center[s, chan_ind] <= traces[n_shifts + s + i + 1, neighbour]
                        if not peak_mask[s, chan_ind]:
                            break
                    if not peak_mask[s, chan_ind]:
                        break
        return peak_mask
