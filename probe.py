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
        self.hyper_len = self.cfg.hyper_extend_len

    def __call__(self, raw_data, patterns):
        time_ds, crds_ds, tri_ds = raw_data
        N = crds_ds.shape[0]
        eps = 1e-9
        kernel_size = patterns.shape[1]
        filted_crds_ds = self.filter.lowpass(crds_ds)
        filted_crds_ds = self.base.min_max_norm(filted_crds_ds)
        # filted_crds_ds -= np.mean(filted_crds_ds, axis=-1)
        steps = (filted_crds_ds.shape[-1] - kernel_size + 1) / self.stride
        steps += 1  # remainder
        sig_peak_idxs, i_idxs = [], []
        tri_high_idxs, tri_low_idxs = [], []
        proj_t = np.linspace(0, 1, patterns.shape[-1])[None, :]
        proj_pattern_t = np.tile(proj_t, [patterns.shape[0], 1])
        windowed_pattern = self.base.min_max_norm(patterns)
        proj_pattern = self.base.proj2normal(proj_pattern_t, windowed_pattern)
        progress = tqdm(total=steps)
        i, j = 0, 0
        while i < steps:
            forward = None
            if ((j + 2) * self.tri_window) < N:
                highest_idx, lowest_idx = self.base.search_tri_cyc(
                    j, tri_ds, self.tri_window)
                if highest_idx != None:
                    tri_high_idxs.append(highest_idx[0])
                if lowest_idx != None:
                    tri_low_idxs.append(lowest_idx[0])

            windowed_src = filted_crds_ds[i:kernel_size + i]
            if i == steps - 1:
                windowed_pattern = windowed_pattern[:, :np.size(windowed_src)]
                proj_pattern = proj_pattern[:, :np.size(windowed_src)]
                proj_t = proj_t[:, :np.size(windowed_src)]

            # core method
            windowed_src = self.base.min_max_norm(windowed_src)
            proj_src = self.base.proj2normal(proj_t, windowed_src[None, :])
            curr_dist = np.sqrt(
                np.sum(np.square(proj_src - proj_pattern), axis=-1) + eps)
            statistics = np.float32(curr_dist < self.sim_thres)
            valid_sim = np.mean(statistics)
            if valid_sim > self.vote_thres:
                # extend searching signals
                hyperterm = self.hyper_len
                max_idx = np.argmax(filted_crds_ds[i:kernel_size + i +
                                                   hyperterm],
                                    axis=-1)
                sig_peak_idx = max_idx + i
                i_idxs.append(i)
                sig_peak_idxs.append(sig_peak_idx)
                forward = max_idx + hyperterm
            forward = forward if forward != None else 1
            i += forward
            j += 1
            progress.update(forward)
        if tri_high_idxs[0] > tri_low_idxs[0]:
            tri_low_idxs.pop(0)

        m = len(tri_low_idxs) if len(tri_high_idxs) > len(
            tri_low_idxs) else len(tri_high_idxs)
        tri_HL_indxs = np.asarray(list(zip(tri_high_idxs[:m],
                                           tri_low_idxs[:m])))
        sig_peak_idxs, i_idxs = np.asarray(sig_peak_idxs), np.asarray(i_idxs)
        return tri_HL_indxs, sig_peak_idxs, i_idxs


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
    tri_peak_idxs, sig_peak_idxs, i_idxs = probe(raw_data, patterns)
    root_dir = os.path.split(cfg.raw_data)[0]
    np.save(os.path.join(root_dir, "tri_idxs.npy"), tri_peak_idxs)
    np.save(os.path.join(root_dir, "det_idxs.npy"), sig_peak_idxs)
    np.save(os.path.join(root_dir, "i_idxs.npy"), i_idxs)
    print('-' * 100)
    print('Finish evaluating')
    print('Totoal find %i' % len(sig_peak_idxs))
    print('Detect signal with Window %i size' % patterns.shape[-1])
    print('save detected triangle and detection index in {}'.format(root_dir))
