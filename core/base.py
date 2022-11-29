import numpy as np


class Base:

    def min_max_norm(self, x):
        return (x - x.min(keepdims=True, axis=-1)) / (
            x.max(keepdims=True, axis=-1) - x.min(keepdims=True, axis=-1))

    def proj2normal(self, x):
        b, n = x.shape
        t = np.linspace(0, 1, n)
        t = np.tile(t[None, :], [x.shape[0], 1])
        mu, std = np.mean(x, keepdims=True, axis=-1), np.std(x,
                                                             keepdims=True,
                                                             axis=-1)
        factor = 1 / (std * (2 * np.pi)**0.5)
        return factor * np.exp(-0.5 * ((t - mu) / std)**2)

    def search_tri_cyc(self, i: int, data: np.ndarray, window: int) -> list:
        # True and False for wave going up and down, respectively
        trans_mask = np.array([[True, False]])
        tmp = []
        for k in range(i, i + 2):
            frame1 = data[k * window:(k + 1) * window]
            k += 1
            frame2 = data[k * window:(k + 1) * window]
            state = np.mean(frame2, axis=-1) - np.mean(frame1, axis=-1)
            tmp.append(state)
        tmp = np.asarray(tmp).T > 0
        is_peak = np.all(np.equal(tmp, trans_mask), axis=-1)
        is_peak = np.reshape(is_peak, (1))
        peak_idx = np.argmax(data[i * window:(i + 2) * window]) + i * window
        peak_idx = np.where(is_peak == True, peak_idx, -1)
        return is_peak, peak_idx
