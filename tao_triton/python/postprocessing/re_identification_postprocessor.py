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

"""Simple class to run post processing of Triton Inference outputs."""

import os
import json

from tao_triton.python.postprocessing.postprocessor import Postprocessor


class ReIdentificationPostprocessor(Postprocessor):
    """Class to run post processing of Triton Tensors."""

    def __init__(self, batch_size, query_frames, test_frames, output_path, data_format):
        """Initialize a post processor class for a reidentification model.
        
        Args:
            batch_size (int): Number of sequences in the batch.
            query_frames (list): List of query images.
            test_frames (list): List of test images.
            output_path (str): Unix path to the output embeddings.
            data_format (str): The input dimension order. This can be "channels_first"
                or "channels_last". "channels_first" is in the CHW order,
                and "channels_last" is in HWC order.
        """
        super().__init__(batch_size, query_frames, output_path, data_format)
        self.batch_size = batch_size
        self.output_path = output_path
        self.output_name = "fc_pred"
        self.query_frames = query_frames
        self.test_frames = test_frames

    def apply(self, output_tensors, this_id, render=True, batching=True):
        """Apply the post processor to the outputs to the reidentification outputs."""
        output_array = output_tensors.as_numpy(self.output_name)
        if len(output_array) != self.batch_size:
            raise Exception("expected {} results, got {}".format(
                self.batch_size, len(output_array)))
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        output_file = os.path.join(self.output_path, "results.json")
        
        results = []
        for image_idx in range(self.batch_size):
            current_idx = (int(this_id) - 1) * self.batch_size + image_idx
            embedding = output_array[image_idx]

            if current_idx < len(self.query_frames):
                current_frame = self.query_frames[current_idx]
            elif len(self.query_frames) <= current_idx < len(self.query_frames) + len(self.test_frames):
                current_frame = self.test_frames[current_idx-len(self.query_frames)]
            else:
                break

            result = {"img_path": current_frame._image_path, "embedding": embedding.tolist()}
            results.append(result)

        if os.path.exists(output_file):
            with open(output_file) as fp:
                results_list = json.load(fp)
            results_list.extend(results)
            results = results_list

        with open(output_file, "w") as json_file:
            json.dump(results, json_file)
