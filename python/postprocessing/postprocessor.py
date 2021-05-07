# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

"""Simple class to run post processing of Triton Inference outputs."""

import os

class Postprocessor(object):
    """Class to run post processing of Triton Tensors."""

    def __init__(self, batch_size, frames, output_path, data_format):
        """Initialize a post processor class."""

        self.batch_size = batch_size
        self.frames = frames
        self.output_path = output_path
        self.data_format = data_format
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        self.initialized = True

    def apply(self, output_tensors, this_id, render=True):
        """Apply the post processor to the outputs."""
        raise NotImplementedError("Base class doesn't implement any post-processing")
