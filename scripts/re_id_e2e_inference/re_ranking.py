# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.

# Original source taken from https://github.com/michuanhaohao/reid-strong-baseline

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Re-Identification Metrics."""

import numpy as np
import torch

from utils import re_ranking, eval_func


def euclidean_distance(qf, gf):
    """Return a similiarity matrix based on euclidian distance.

    Args:
        qf (Numpy): Matrix A.
        gf (Numpy): Matrix B

    Returns:
        dist_mat (Numpy): Distance Matrix.

    """
    m = qf.shape[0]
    n = gf.shape[0]
    dist_mat = torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n) + \
               torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mat.addmm_(qf, gf.t(), beta = 1, alpha = -2)
    return dist_mat.cpu().numpy()


def cosine_similarity(qf, gf):
    """Return a similiarity matrix.

    Args:
        qf (Numpy): Matrix A.
        gf (Numpy): Matrix B

    Returns:
        dist_mat (Numpy): Distance Matrix.

    """
    epsilon = 0.00001
    dist_mat = qf.mm(gf.t())
    qf_norm = torch.norm(qf, p=2, dim=1, keepdim=True)  # mx1
    gf_norm = torch.norm(gf, p=2, dim=1, keepdim=True)  # nx1
    qg_normdot = qf_norm.mm(gf_norm.t())

    dist_mat = dist_mat.mul(1 / qg_normdot).cpu().numpy()
    dist_mat = np.clip(dist_mat, -1 + epsilon, 1 - epsilon)
    dist_mat = np.arccos(dist_mat)
    return dist_mat


class R1_mAP_reranking():
    """R1_mAP Class for metrics with reranking."""

    def __init__(self, num_query, output_dir, feat_norm=True):
        """Initialize the R1_mAP class."""
        super(R1_mAP_reranking, self).__init__()
        self.num_query = num_query
        self.feat_norm = feat_norm
        self.feats = []
        self.pids = []
        self.camids = []
        self.img_paths = []
        self.output_dir = output_dir

    def reset(self):
        """Reset the data members."""
        self.feats = []
        self.pids = []
        self.camids = []
        self.img_paths = []

    def update(self, feat, pid, camid, img_path):
        """Append to the data members.

        Args:
            feat (Tensor): Feature embedding.
            pid (int): Person IDs.
            camid (int): Camera IDs.
            img_path (str): Image Paths.

        """
        self.feats.append(feat)
        self.pids.extend(np.asarray(pid))
        self.camids.extend(np.asarray(camid))
        self.img_paths.extend(img_path)

    def compute(self):
        """Compute the metrics.

        Returns:
            cmc (list): CMC Rank List.
            mAP (float): Mean average precision.

        """
        feats = torch.cat(self.feats, dim=0)
        if self.feat_norm:
            print("The test features are normalized.")
            feats = torch.nn.functional.normalize(feats, dim=1, p=2)

        # query
        qf = feats[:self.num_query]
        q_pids = np.asarray(self.pids[:self.num_query])
        q_camids = np.asarray(self.camids[:self.num_query])
        q_img_paths = self.img_paths[:self.num_query]
        # gallery
        gf = feats[self.num_query:]
        g_pids = np.asarray(self.pids[self.num_query:])
        g_camids = np.asarray(self.camids[self.num_query:])
        g_img_paths = self.img_paths[self.num_query:]

        print("The distance matrix is processed by re-ranking.")
        distmat = re_ranking(qf, gf, k1=20, k2=6, lambda_value=0.3)
        cmc, mAP = eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_img_paths, g_img_paths, self.output_dir)
        return cmc, mAP
