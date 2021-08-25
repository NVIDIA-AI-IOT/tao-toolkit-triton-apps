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

import os

from PIL import Image
import numpy as np

import tritonclient.grpc.model_config_pb2 as mc


class Frame(object):
    """Data structure to contain an image."""

    def __init__(self, image_path, data_format, dtype, target_shape):
        """Instantiate a frame object."""
        self._image_path = image_path
        if data_format not in [mc.ModelInput.FORMAT_NCHW, mc.ModelInput.FORMAT_NHWC]:
            raise NotImplementedError(
                "Data format not in the supported data format: {}".format(data_format)
            )
        self.data_format = data_format
        self.height = None
        self.width = None
        self.dtype = dtype
        assert len(target_shape) == 3, (
            "3 dimensions are required for input definitions. Got {}".format(len(target_shape))
        )
        if self.data_format == mc.ModelInput.FORMAT_NCHW:
            self.c, self.h, self.w = target_shape
        else:
            self.h, self.w, self.c = target_shape
        assert self.c in [1, 3], (
            "Number of channels should be 1 or 3. Got {}".format(self.c))
        self.target_shape = target_shape

    def load_image(self):
        """Load the image defined."""
        if not os.path.exists(self._image_path):
            raise NotFoundError("Cannot find image at {}".format(self._image_path))
        image = Image.open(self._image_path)
        self.width, self.height = image.size

        if self.c == 1:
            image = image.convert("L")
        else:
            image = image.convert("RGB")
        return image

    def as_numpy(self, image):
        """Return a numpy array."""
        image = image.resize((self.w, self.h), Image.ANTIALIAS)
        nparray = np.asarray(image).astype(self.dtype)
        if nparray.ndim == 2:
            nparray = nparray[:, :, np.newaxis]
        if self.data_format == mc.ModelInput.FORMAT_NCHW:
            nparray = np.transpose(nparray, (2, 0, 1))
        return nparray
