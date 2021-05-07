# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

"""Simple class to run post processing of Triton Inference outputs."""

import os
import numpy as np

from tlt_triton.python.postprocessing.postprocessor import Postprocessor


class ClassificationPostprocessor(Postprocessor):
    """Class to run post processing of Triton Tensors."""

    def __init__(self, batch_size, frames, output_path, data_format):
        """Initialize a post processor class."""
        super().__init__(batch_size, frames, output_path, data_format)
        self.output_name = "predictions/Softmax"

    def apply(self, output_tensors, this_id, render=True, batching=True):
        """Apply the post processor to the outputs."""
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
