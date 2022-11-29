import numpy as np
import os
import argparse
from utils.io import load_json


class Eval:

    def __init__(self, cfg):
        self.cfg = cfg

    def __call__(self):
        gt_idxs = np.load(self.cfg["gt_path"])
        det_idxs = np.load(self.cfg["det_path"])
        TP, FP, FN = [], [], []
        for gt_idx in gt_idxs:
            lower_mask = gt_idx - 375 < det_idxs
            masked_det_idxs = det_idxs[lower_mask]
            upper_mask = masked_det_idxs < gt_idx + 375
            masked_det_idxs = masked_det_idxs[upper_mask]

            if len(masked_det_idxs) != 0:
                index = np.where(det_idxs == masked_det_idxs)[0][0]
                det_idxs = det_idxs.tolist()
                det_idxs.pop(index)
                TP.extend(masked_det_idxs)
            else:
                print(gt_idx)

                FN.extend([gt_idx])
            # print(masked_det_idxs)
        # gt_idxs = gt_idxs.tolist()
        FN = np.size(gt_idxs) - len(TP)
        gt_num = gt_idxs.shape[0]
        det_num = det_idxs.shape[0]

        return


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_config()
    print('Evaluate detection performance')
    print(f"Use following config to produce results: {args.config}.")
    cfg = load_json(args.config)
    eval = Eval(cfg)
    eval()