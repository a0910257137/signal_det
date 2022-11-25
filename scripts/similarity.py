import numpy as np
import os
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt

import sys
from pathlib import Path
from scipy.signal import butter, lfilter
import time as ts
from scipy.signal import find_peaks, find_peaks_cwt

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.io import load_text, load_json, parse_txt
"""
command line:
    python3 scripts/similarity.py --config=./config/example.json --low_pass True  --is_draw True
"""


def search_tri_cyc(i: int, data: np.ndarray, window: int) -> list:
    # True and False for wave going up and down, respectively
    trans_mask = np.array([[True, False]])
    tmp = []
    for k in range(i, i + 2):
        frame1 = data[:, k * window:(k + 1) * window]
        k += 1
        frame2 = data[:, k * window:(k + 1) * window]
        state = np.mean(frame2, axis=-1) - np.mean(frame1, axis=-1)
        tmp.append(state)
    tmp = np.asarray(tmp).T > 0
    b = data.shape[0]
    is_peak = np.all(np.equal(tmp, trans_mask), axis=-1)
    is_peak = np.reshape(is_peak, (b, 1))
    peak_idx = np.argmax(data[:, i * window:(i + 2) * window]) + i * window
    peak_idx = np.where(is_peak == True, peak_idx, -1)
    return is_peak, peak_idx


def lowpass(x):

    def butter_lowpass_filter(data, cutoff, fs, order=5):
        b, a = butter(order, cutoff, fs=fs, btype='low', analog=False)
        y = lfilter(b, a, data)
        return y

    # create a butter worth filter
    order = 5
    fs = 1250.0
    cutoff = 30
    y = butter_lowpass_filter(x, cutoff, fs, order)
    y -= np.mean(y, axis=-1)
    return y


def min_max_norm(x):
    return (x - x.min(keepdims=True, axis=-1)) / (
        x.max(keepdims=True, axis=-1) - x.min(keepdims=True, axis=-1))


def median_filter(x: np.ndarray, r: int) -> np.ndarray:
    assert (np.shape(x)[-1] %
            r) == 0, "Oopsie, array can't be evenly divided by {}".format(r)
    idxs = np.asarray(list(range(x.shape[-1])))
    idxs = [idxs[::r] + i for i in range(r)]
    idxs = np.stack(idxs)
    x = x[:, idxs]
    return np.mean(np.asarray(x), axis=1)


def run_similarity(time,
                   src,
                   tri,
                   patterns,
                   root_dir,
                   is_lowpass=True,
                   is_draw=False):
    s_n = src.shape[0]
    src, tri = src.reshape([-1, s_n]), tri.reshape([-1, s_n])
    # sparse and smooth data by using median filter
    filted_tri = median_filter(tri, r=4)
    stride = 1
    sim_thres, interval_per = 10., 0.5
    eps = 1e-9
    tri_window = 2000
    kernel_size = patterns.shape[1]
    normed_src = min_max_norm(src)
    filted_src = lowpass(normed_src)
    filted_src = median_filter(filted_src, r=4)
    steps = (filted_src.shape[-1] - kernel_size + 1) / stride
    steps += 1  # remainder
    tmp, poss_intervals = [], []
    windowed_pattern = min_max_norm(patterns)

    tri_peak_idxs = []
    for i in tqdm(range(int(steps))):
        if ((i + 2) * tri_window) < filted_tri.shape[-1]:
            _, peak_idx = search_tri_cyc(i, filted_tri, tri_window)
            tri_peak_idxs.append(peak_idx)
        windowed_src = filted_src[:, i:kernel_size + i]
        # last step
        if i == steps - 1:
            windowed_pattern = windowed_pattern[:, :np.size(windowed_src)]
        windowed_src = min_max_norm(windowed_src)
        curr_dist = np.sqrt(
            np.sum(np.square(windowed_pattern - windowed_src), axis=-1) + eps)
        tmp.append(curr_dist)
        if i != 0 and len(tmp) % kernel_size == 0:
            tmp = np.asarray(tmp).T
            # hard-code threshold
            mask = tmp < sim_thres
            valid_sim = np.float32(mask)
            guessed_interval = np.sum(valid_sim, axis=-1) / kernel_size
            votes = np.sum((guessed_interval > 0.5).astype(np.float32) /
                           np.size(guessed_interval))
            if votes > interval_per:
                poss_intervals.append(i)
            tmp = []
    tri_peak_idxs = np.concatenate(tri_peak_idxs, axis=-1)
    tri_peak_idxs = tri_peak_idxs[tri_peak_idxs != -1]
    peak2peak_len = (tri_peak_idxs[1] - tri_peak_idxs[0]) // 2

    convert2time = time[poss_intervals]
    print('Finish evaluating')
    print('Totoal find %i' % len(poss_intervals))
    print('Window size %i' % kernel_size)
    # np.save(os.path.join(root_dir, "index.npy"), poss_intervals)
    if is_draw:
        # OR draw src
        min_max = [np.min(filted_src), np.max(filted_src)]
        for t in convert2time:
            plt.plot([t, t], min_max, 'r-')
        plt.plot(time, filted_src[0])
        plt.grid()
        plt.savefig('foo.png')


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--low_pass', default=True, type=bool)
    parser.add_argument('--is_draw', default=True, type=bool)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()
    print('Run signal detection with a known pattern')
    print(f"Use following config to produce results: {args.config}.")
    cfg = load_json(args.config)

    if os.path.isfile(cfg["pattern"]):
        patterns = np.stack([np.load(cfg["pattern"])])
    else:
        patterns = np.stack([
            np.load(os.path.join(cfg["pattern"], p))
            for p in os.listdir(cfg["pattern"])
        ])
    time, raw_data, triangle_waves = parse_txt(load_text(cfg["raw_data"]))
    root_dir = os.path.split(cfg["raw_data"])[0]
    run_similarity(time, raw_data, triangle_waves, patterns, root_dir,
                   args.low_pass, args.is_draw)
