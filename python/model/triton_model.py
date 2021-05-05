# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.

"""Triton inference client for TLT model."""

from abc import abstractmethod
import os

import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype


class TritonModel(object):
    """Simple class to run model inference using Triton client."""

    def __init__(self, max_batch_size, input_names, output_names,
                 channels, height, width, data_format, triton_dtype):
        """Set up a triton model instance."""
        self.max_batch_size = max_batch_size
        self.input_names = input_names
        self.output_names = output_names
        self.c = channels
        self.h = height
        self.w = width
        self.data_format = data_format
        self.triton_dtype = triton_dtype

    @staticmethod
    def parse_model(model_metadata, model_config):
        """Simple class to parse model metadata and model config."""
        raise NotImplementedError("Base class doesn't implement this method.")

    @classmethod
    def from_metadata(cls, model_metadata, model_config):
        """Parse a model from the metadata config."""
        parsed_outputs = cls.parse_model(model_metadata, model_config)
        max_batch_size, input_names, output_names, channels, height, width, \
            data_format, triton_dtype = parsed_outputs
        return cls(
            max_batch_size, input_names, output_names,
            channels, height, width, data_format,
            triton_dtype
        )

    @abstractmethod
    def preprocess(self, image):
        """Preprocess an input image.

        Args:
            image (np.ndarray): Image object to be preprocessed.

        Returns:
            image (np.ndarray): Preprocessed input image.
        """
        raise NotImplementedError("Base class doesn't implement this method.")

    def get_config(self):
        """Get dictionary config."""
        config_dict = {
            "data_format": self.data_format,
            "max_batch_size": self.max_batch_size,
            "channels": self.c,
            "width": self.w,
            "height": self.h,
            "input_names": self.input_names,
            "output_names": self.output_names,
            "triton_dtype": self.triton_dtype
        }
        return config_dict
