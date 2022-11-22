# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
# 
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
# 
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import random
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


def read_image(img_path):
    """Read an image given a file path.

    Args:
        img_path (str): Image path.

    Returns:
        img (Pillow): Image data.

    """
    if not os.path.exists(img_path):
        raise FileNotFoundError("{} does not exist".format(img_path))
    img = Image.open(img_path).convert('RGB')
    return img


def plot_evaluation_results(num_queries, query_maps, max_rank, output_dir):
    """Plot evaluation results from queries.

    Args:
        num_queries (int): Number of queries to plot.
        query_maps (dict): Dictonary of images to list of images.
        max_rank (int): Max rank to plot.
        output_dir (str): Output directory.

    """
    fig, ax = plt.subplots(num_queries, max_rank+1)
    fig.suptitle('Sampled Matches')
    random.shuffle(query_maps)
    query_maps = query_maps[:num_queries]
    for row, collections in enumerate(query_maps):
        for col, collection in enumerate(collections):
            if col != 0:
                img_path, keep = collection
                string = "Rank " + str(col)
                if keep:
                    outline = "green"
                else:
                    outline = "red"
            else:
                img_path, _ = collection
                outline = "blue"
                string = "Query"
            img = read_image(img_path)
            draw = ImageDraw.Draw(img)
            width, height = img.size
            draw.rectangle([(0,0), (width, height)], fill=None, outline=outline, width=10)
            ax[row, col].imshow(img)
            ax[row, col].tick_params(top=False, bottom=False, left=False, right=False,
                labelleft=False, labelbottom=False)
            if row == len(query_maps) -1:
                ax[row, col].set_xlabel(string, rotation = 80)
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.savefig(os.path.join(output_dir, "sampled_matches.png"))


def eval_func(distmat, q_pids, g_pids, q_camids, g_camids, q_img_paths, g_img_paths, output_dir):
    """Evaluation with market1501 metric. For each query identity, its gallery images from the same camera view are discarded.

    Args:
        distmat (Tensor): Distance Matrix.
        q_pids (Numpy): Query Person IDs.
        g_pids (Numpy): Gallery Person IDs.
        q_camids (Numpy): Query Camera IDs.
        g_camids (Numpy): Gallery Image Paths.
        q_img_paths (Numpy): Query Image Paths.
        g_img_paths (Numpy): Gallery Image Paths.
        output_dir (str): Output Directory.

    Returns:
        all_cmc (list): CMC Rank List.
        mAP (float): Mean Average Precision.
    """
    num_q, num_g = distmat.shape
    max_rank = 10
    if num_g < max_rank:
        max_rank = num_g
    indices = np.argsort(distmat, axis=1)
    matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)

    # compute cmc curve for each query
    all_cmc = []
    all_AP = []
    num_valid_q = 0.  # number of valid query
    query_maps = []
    for q_idx in range(num_q):
        query_map = []

        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]
        q_img_path = q_img_paths[q_idx]
        query_map.append([q_img_path, False])

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = (g_pids[order] == q_pid) & (g_camids[order] == q_camid)
        keep = np.invert(remove)
        res_list =list(map(g_img_paths.__getitem__, order))
        for g_img_path, value in zip(res_list[:max_rank], matches[q_idx][keep]):
            query_map.append([g_img_path, value])
        query_maps.append(query_map)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        orig_cmc = matches[q_idx][keep]
        if not np.any(orig_cmc):
            # this condition is true when query identity does not appear in gallery
            continue

        cmc = orig_cmc.cumsum()
        cmc[cmc > 1] = 1

        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision
        num_rel = orig_cmc.sum()
        tmp_cmc = orig_cmc.cumsum()
        tmp_cmc = [x / (i + 1.) for i, x in enumerate(tmp_cmc)]
        tmp_cmc = np.asarray(tmp_cmc) * orig_cmc
        AP = tmp_cmc.sum() / num_rel
        all_AP.append(AP)

    plot_evaluation_results(10, query_maps, max_rank, output_dir)

    assert num_valid_q > 0, "Error: all query identities do not appear in gallery."

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def re_ranking(probFea, galFea, k1, k2, lambda_value, local_distmat=None, only_local=False):
    """Return distance matrix based on re-ranking.

    Args:
        probFea (Numpy): Probability Features.
        galFea (Numpy): Gallery features.
        k1 (int): Constant value for reranking.
        k2 (int): Constant value for reranking.
        lambda_value (int): Constant value for reranking.

    Returns:
        final_dist (Numpy): Distance Matrix.

    """
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    if only_local:
        original_dist = local_distmat
    else:
        feat = torch.cat([probFea,galFea])
        distmat = torch.pow(feat,2).sum(dim=1, keepdim=True).expand(all_num,all_num) + \
                      torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(all_num, all_num).t().squeeze()
        distmat.addmm_(feat,feat.t(), beta = 1, alpha = -2)
        original_dist = distmat.cpu().numpy()
        del feat
        if not local_distmat is None:
            original_dist = original_dist + local_distmat
    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)

    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[candidate_forward_k_neigh_index,
                                               :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[fi_candidate]
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2 / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)

    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    return final_dist
