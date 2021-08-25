# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
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
import numpy as np

from tao_triton.python.postprocessing.postprocessor import Postprocessor


class ClassificationPostprocessor(Postprocessor):
    """Class to run post processing of Triton Tensors."""

    def __init__(self, batch_size, frames, output_path, data_format):
        """Initialize a post processor class for a classification model.
        
        Args:
            batch_size (int): Number of images in the batch.
            frames (list): List of images.
            output_path (str): Unix path to the output rendered images and labels.
            data_format (str): Order of the input model dimensions.
                "channels_first": CHW order.
                "channels_last": HWC order.
        """
        super().__init__(batch_size, frames, output_path, data_format)
        self.output_name = "predictions/Softmax"

    def apply(self, output_tensors, this_id, render=True, batching=True):
        """Apply the post processor to the outputs to the classification outputs."""
        output_array = output_tensors.as_numpy(self.output_name)
        if len(output_array) != self.batch_size:
            raise Exception("expected {} results, got {}".format(
                batch_size, len(output_array)))

        # Include special handling for non-batching models
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        output_file = os.path.join(self.output_path, "results.txt")
        with open(output_file, "a") as wfile:
            for idx in range(self.batch_size):
                results = output_array[idx]
                current_idx = (int(this_id) - 1) * self.batch_size + idx
                if current_idx < len(self.frames):
                    wfile.write("{}".format(self.frames[current_idx]._image_path))
                    if not batching:
                        results = [results]
                    for result in results:
                        if output_array.dtype.type == np.object_:
                            cls = "".join(chr(x) for x in result).split(':')
                        else:
                            cls = result.split(':')
                        wfile.write(
                            ", {:0.4f}({})={}".format(
                                float(cls[0]), cls[1], cls[2]
                            )
                        )
                    wfile.write("\n")
                else:
                    break
