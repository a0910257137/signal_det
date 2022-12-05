import numpy as np
import os
import argparse
from tqdm import tqdm
from core.base import Base
from core.filter import Filter
from box import Box
from utils.io import load_text, load_json, parse_txt
"""
command line:
    python3 probe.py --config=./config/probe.json
"""


class Probe:

    def __init__(self, cfg):
        self.cfg = cfg
        self.filter = Filter(cfg["filter"])
        self.base = Base()
        self.stride = self.cfg.window_stride

        self.sim_thres = self.cfg.similarity_threshold
        self.vote_thres = self.cfg.vote_threshold
        self.tri_window = self.cfg.triangle_window

    def __call__(self, raw_data, patterns):
        time_ds, crds_ds, tri_ds = raw_data
        N = crds_ds.shape[0]
        eps = 1e-9
        kernel_size = patterns.shape[1]
        normed_crds_ds = self.base.min_max_norm(crds_ds)
        filted_crds_ds = self.filter.lowpass(normed_crds_ds)
        steps = (filted_crds_ds.shape[-1] - kernel_size + 1) / self.stride
        steps += 1  # remainder
        sim_tmp, sig_peak_idxs = [], []
        windowed_pattern = self.base.min_max_norm(patterns)
        proj_pattern = self.base.proj2normal(windowed_pattern)

        tri_peak_idxs = []
        for i in tqdm(range(int(steps))):
            if ((i + 2) * self.tri_window) < tri_ds.shape[-1]:
                _, peak_idx = self.base.search_tri_cyc(i, tri_ds,
                                                       self.tri_window)
                tri_peak_idxs.append(peak_idx)
            windowed_src = filted_crds_ds[i:kernel_size + i]
            if i == steps - 1:
                windowed_pattern = windowed_pattern[:, :np.size(windowed_src)]
                proj_pattern = proj_pattern[:, :np.size(windowed_src)]

            windowed_src = self.base.min_max_norm(windowed_src)
            # core method
            proj_src = self.base.proj2normal(windowed_src[None, :])
            curr_dist = np.sqrt(
                np.sum(np.square(proj_src - proj_pattern), axis=-1) + eps)
            # curr_dist = np.sqrt(
            #     np.sum(np.square(windowed_src - windowed_pattern), axis=-1) + eps)
            sim_tmp.append(curr_dist)
            if i != 0 and len(sim_tmp) % kernel_size == 0:
                sim_tmp = np.asarray(sim_tmp).T
                mask = sim_tmp < self.sim_thres
                valid_sim = np.float32(mask)
                guessed_interval = np.sum(valid_sim, axis=-1) / kernel_size
                votes = np.sum((guessed_interval > 0.5).astype(np.float32) /
                               np.size(guessed_interval))
                if votes > self.vote_thres:
                    sig_peak_idx = np.argmax(windowed_src, axis=-1) + i
                    sig_peak_idxs.append(sig_peak_idx)
                sim_tmp = []
        tri_peak_idxs = np.concatenate(tri_peak_idxs, axis=-1)
        sig_peak_idxs = np.asarray(sig_peak_idxs)
        tri_peak_idxs = tri_peak_idxs[tri_peak_idxs != -1]
        return tri_peak_idxs, sig_peak_idxs


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    return parser.parse_args()


if __name__ == '__main__':

    args = parse_config()
    print(f'Run signal detection with known patterns')
    print(f"Use the following config to produce results: {args.config}.")
    cfg = Box(load_json(args.config))
    if os.path.isfile(cfg.pattern):
        patterns = np.stack([np.load(cfg.pattern)])
    else:
        patterns = np.stack([
            np.load(os.path.join(cfg.pattern, p))
            for p in os.listdir(cfg.pattern)
        ])
    if 'txt' in cfg.raw_data.split('/')[-1]:
        raw_data = parse_txt(load_text(cfg.raw_data))
    elif 'npy' in cfg.raw_data.split('/')[-1]:
        raw_data = np.load(cfg.raw_data)
    probe = Probe(cfg)
    tri_peak_idxs, sig_peak_idxs = probe(raw_data, patterns)
    root_dir = os.path.split(cfg.raw_data)[0]

    save_path = os.path.join(root_dir, "det_idxs.npy")
    np.save(save_path, sig_peak_idxs)
    print('-' * 100)
    print('Finish evaluating')
    print('Totoal find %i' % len(sig_peak_idxs))
    print('Detect signal with Window %i size' % patterns.shape[-1])
    print('save detected index in {}'.format(save_path))
