# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

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
