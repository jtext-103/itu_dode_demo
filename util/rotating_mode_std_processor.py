from copy import deepcopy
from jddb.processor import Signal, BaseProcessor
import numpy as np


class RotatingModeStd(BaseProcessor):

    def __init__(self):
        super().__init__()

    def transform(self, *signal: Signal) -> Signal:
        """input signals within the same samplerate and lenth to calculate the std

        Args:
            The Mirnov array with the a whole circle.
            The last signal should be bt

        Returns: Signal: Rotating_std.

        """
        resampled_attributes = deepcopy(signal[-1].attributes)
        Mirnov_array = []
        for each_signal in signal:
            if signal.index(each_signal) is not len(signal) - 1:
                Mirnov_array.append(each_signal.data)
        std = np.std(np.array(Mirnov_array), axis=0)
        new_data = std / signal[-1].data
        return Signal(data=new_data, attributes=resampled_attributes)
