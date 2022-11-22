import torch
import re
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from re_ranking import R1_mAP_reranking


def main():
    if sys.argv[2]:
        json_metadata_path = sys.argv[1]
        output_dir = sys.argv[2]
        f = open(json_metadata_path)
        pattern = re.compile(r'([-\d]+)_c(\d)')
        data = json.load(f)
        
        pids = []
        camids = []
        img_paths = []
        embeddings = []
        num_query = 0

        for row in data:
            img_path = row["img_path"]
            if "query" in img_path:
                num_query += 1
            embedding = row["embedding"]
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are ignored
            camid -= 1  # index starts from 0
            embeddings.append(embedding)
            pids.append(pid)
            camids.append(camid)
            img_paths.append(img_path)
        metrics = R1_mAP_reranking(num_query, output_dir, feat_norm=True)
        metrics.reset()
        metrics.update(torch.tensor(embeddings), pids, camids, img_paths)
        cmc, _ = metrics.compute()
        f.close()

        plt.figure()
        cmc_percentages = [value * 100 for value in cmc]
        plt.xticks(np.arange(len(cmc_percentages)), np.arange(1, len(cmc_percentages)+1))
        plt.plot(cmc_percentages, marker="*")
        plt.title('Cumulative Matching Characteristics (CMC) curve')
        plt.grid()
        plt.ylabel('Matching Rate[%]')
        plt.xlabel('Rank')
        output_cmc_curve_plot_path = os.path.join(output_dir, 'cmc_curve.png')
        plt.savefig(output_cmc_curve_plot_path)

        print("Output CMC curve plot saved at %s" % output_cmc_curve_plot_path)

    else:
        print("Usage: %s json_metadata_path output_dir" % __file__)


if __name__ == '__main__':
    main()
