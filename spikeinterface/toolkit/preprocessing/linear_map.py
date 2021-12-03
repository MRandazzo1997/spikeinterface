import numpy as np
from .basepreprocessor import BasePreprocessor, BasePreprocessorSegment


class LinearMapFilter(BasePreprocessor):
    name = 'Filter'

    def __init__(self, recording, matrix):
        """

        Parameters
        ----------
        recording: RecordingExtractor
            RecordingExtractor object
        matrix: np.ndarray
            Matrix of shape (n_features, n_channels)

        Returns
        ----------
        LinearMapFilter object. The filtered recording.
        """
        self.recording = recording
        if isinstance(matrix, list):
            self.M = np.asarray(matrix)
        else:
            self.M = matrix
        # if not recording.get_num_channels() == self.M.shape[0]:
        #     raise ArithmeticError(
        #         f"Matrix first dimension must be equal to number of channels: {recording.get_num_channels()}"
        #         f"It is: {self.M.shape[0]}")

        BasePreprocessor.__init__(self, recording)
        for i, parent_segment in enumerate(recording._recording_segments):
            segment = FilterRecordingSegment(parent_segment, self.M)
            self.add_recording_segment(segment)

        self._kwargs = dict(recording=recording.to_dict(), matrix=matrix)

    def get_num_channels(self):
        return self.M.shape[0]

    def get_channel_ids(self):
        return np.asarray([str(i + 1) for i in np.arange(self.M.shape[0])], dtype='<U64')


class FilterRecordingSegment(BasePreprocessorSegment):
    def __init__(self, parent_segment, M):
        BasePreprocessorSegment.__init__(self, parent_segment)
        self.M = M

    def get_traces(self, start_frame, end_frame, channel_indices):
        traces = self.parent_recording_segment.get_traces(start_frame, end_frame, slice(None)).T
        mean = traces.mean(axis=1)
        filtered_traces = np.matmul(self.M, traces - mean[:, np.newaxis])
        filtered_traces = filtered_traces[channel_indices, :]
        return filtered_traces.T


def lin_map(*args):
    return LinearMapFilter(*args)


lin_map.__doc__ = LinearMapFilter.__doc__
