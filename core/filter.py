import numpy as np
from scipy.signal import butter, lfilter
from .base import Base


class Filter(Base):

    def __init__(self, config):
        self.config = config
        if self.config["low_pass"] != None:
            self.fs = self.config["low_pass"]["fs"]
            self.order = self.config["low_pass"]["order"]
            self.cutoff = self.config["low_pass"]["cutoff"]

    def lowpass(self, x: np.ndarray) -> np.ndarray:

        def butter_lowpass_filter(data, cutoff, fs, order=5):
            b, a = butter(order, cutoff, fs=fs, btype='low', analog=False)
            y = lfilter(b, a, data)
            return y

        y = butter_lowpass_filter(x, self.cutoff, self.fs, self.order)
        # y -= np.mean(y, axis=-1)
        return y

    def median_filter(self, x: np.ndarray, r: int) -> np.ndarray:
        assert (np.shape(x)[-1] %
                r) == 0, "Oopsie, array can't be evenly divided by {}".format(r)
        idxs = np.asarray(list(range(x.shape[-1])))
        idxs = [idxs[::r] + i for i in range(r)]
        idxs = np.stack(idxs)
        x = x[:, idxs]
        return np.mean(np.asarray(x), axis=1)
