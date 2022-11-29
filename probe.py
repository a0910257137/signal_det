import numpy as np
import os
import argparse
from tqdm import tqdm
from matplotlib import pyplot as plt

from core.base import Base
from core.filter import Filter

from utils.io import load_text, load_json, parse_txt
"""
command line:
    python3 probe.py --config=./config/example.json
"""


def run(cfg, src, patterns):
    filter = Filter(cfg["filter"])
    base = Base()
    root_dir = os.path.split(cfg["raw_data"])[0]
    crds_ds, tri_ds = src[0], src[1]
    N = crds_ds.shape[0]

    # sparse and smooth data by using median filter
    stride = 1
    sim_thres, interval_per = 10., 0.5
    eps = 1e-9
    tri_window = 2000
    kernel_size = patterns.shape[1]
    normed_crds_ds = base.min_max_norm(crds_ds)
    filted_crds_ds = filter.lowpass(normed_crds_ds)
    np.save("/aidata/anders/objects/company/hmm/t.npy", filted_crds_ds)
    xxx
    # filted_src = median_filter(filted_src, r=4)
    steps = (filted_crds_ds.shape[-1] - kernel_size + 1) / stride
    steps += 1  # remainder
    sim_tmp, poss_intervals = [], []
    windowed_pattern = base.min_max_norm(patterns)
    proj_pattern = base.proj2normal(windowed_pattern)

    std_pattern = (windowed_pattern -
                   np.mean(windowed_pattern, axis=-1, keepdims=True))

    tri_peak_idxs = []
    for i in tqdm(range(int(steps))):
        if ((i + 2) * tri_window) < tri_ds.shape[-1]:
            _, peak_idx = base.search_tri_cyc(i, tri_ds, tri_window)
            tri_peak_idxs.append(peak_idx)
        windowed_src = filted_crds_ds[i:kernel_size + i]
        # last step
        if i == steps - 1:
            windowed_pattern = windowed_pattern[:, :np.size(windowed_src)]
            proj_pattern = proj_pattern[:, :np.size(windowed_src)]
            std_pattern = (windowed_pattern -
                           np.mean(windowed_pattern, axis=-1, keepdims=True))

        windowed_src = base.min_max_norm(windowed_src)
        proj_src = base.proj2normal(windowed_src[None, :])
        curr_dist = np.sqrt(
            np.sum(np.square(proj_src - proj_pattern), axis=-1) + eps)
        # curr_dist = np.sqrt(
        #     np.sum(np.square(windowed_src - windowed_pattern), axis=-1) + eps)
        # std_src = windowed_src - np.mean(windowed_src, axis=-1, keepdims=True)
        # curr_corr = np.sum(std_pattern * std_src, axis=-1) / np.sqrt(
        #     np.sum(std_pattern**2, axis=-1) * np.sum(std_src**2, axis=-1))

        sim_tmp.append(curr_dist)
        if i != 0 and len(sim_tmp) % kernel_size == 0:
            sim_tmp = np.asarray(sim_tmp).T
            # hard-code threshold
            mask = sim_tmp < sim_thres
            valid_sim = np.float32(mask)
            guessed_interval = np.sum(valid_sim, axis=-1) / kernel_size
            votes = np.sum((guessed_interval > 0.5).astype(np.float32) /
                           np.size(guessed_interval))
            if votes > interval_per:
                sig_peak_idx = np.argmax(windowed_src, axis=-1) + i
                poss_intervals.append(sig_peak_idx)
            sim_tmp = []
    tri_peak_idxs = np.concatenate(tri_peak_idxs, axis=-1)
    poss_intervals = np.asarray(poss_intervals)
    tri_peak_idxs = tri_peak_idxs[tri_peak_idxs != -1]
    np.save(os.path.join(root_dir, "det_idxs.npy"), poss_intervals)

    print('Finish evaluating')
    print('Totoal find %i' % len(poss_intervals))
    print('Window size %i' % kernel_size)
    return tri_peak_idxs, np.asarray(poss_intervals)


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()
    print(f'Run signal detection with known patterns')
    print(f"Use the following config to produce results: {args.config}.")
    cfg = load_json(args.config)

    if os.path.isfile(cfg["pattern"]):
        patterns = np.stack([np.load(cfg["pattern"])])
    else:
        patterns = np.stack([
            np.load(os.path.join(cfg["pattern"], p))
            for p in os.listdir(cfg["pattern"])
        ])
    raw_data = np.load(cfg["raw_data"])
    run(cfg, raw_data, patterns)
