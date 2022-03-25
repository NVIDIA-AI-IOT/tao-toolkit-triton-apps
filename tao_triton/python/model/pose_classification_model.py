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

"""Triton inference client for TAO Toolkit model."""

import os
import numpy as np

import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype

from tao_triton.python.model.triton_model import TritonModel


class PoseClassificationModel(TritonModel):
    """Simple class to run model inference using Triton client."""

    def __init__(self, max_batch_size, input_names, output_names,
                 channels, seq_length, num_joint, num_person,
                 triton_dtype):
        """Set up a pose_classification triton model instance.
        
        Args:
            max_batch_size(int): The maximum batch size of the TensorRT engine.
            input_names (str): List of the input node names
            output_names (str): List of the output node names
            channels (int): Number of chanels in the input dimensions
            seq_length (int): Length of the sequences
            num_joint (int): Number of joint points
            num_person (int): Number of persons
            triton_dtype (proto): Triton input data type
                
        Returns:
            An instance of the PoseClassificationModel.
        """
        pass

    @staticmethod
    def parse_model(model_metadata, model_config):
        """Parse model metadata and model config from the triton server."""
        if len(model_metadata.inputs) != 1:
            raise Exception("expecting 1 input, got {}".format(
                len(model_metadata.inputs)))
        if len(model_metadata.outputs) != 1:
            raise Exception("expecting 1 output, got {}".format(
                len(model_metadata.outputs)))

        if len(model_config.input) != 1:
            raise Exception(
                "expecting 1 input in model configuration, got {}".format(
                    len(model_config.input)))

        input_metadata = model_metadata.inputs[0]
        input_config = model_config.input[0]
        output_metadata = model_metadata.outputs[0]

        if output_metadata.datatype != "FP32":
            raise Exception("expecting output datatype to be FP32, model '" +
                            model_metadata.name + "' output type is " +
                            output_metadata.datatype)

        # Model input must have 4 dims CTVM (not counting the batch dimension)
        input_batch_dim = (model_config.max_batch_size > 0)
        expected_input_dims = 4 + (1 if input_batch_dim else 0)
        if len(input_metadata.shape) != expected_input_dims:
            raise Exception(
                "expecting input to have {} dimensions, model '{}' input has {}".
                format(expected_input_dims, model_metadata.name,
                    len(input_metadata.shape)))

            c = input_metadata.shape[1 if input_batch_dim else 0]
            t = input_metadata.shape[2 if input_batch_dim else 1]
            v = input_metadata.shape[3 if input_batch_dim else 2]
            m = input_metadata.shape[4 if input_batch_dim else 3]

        print(model_config.max_batch_size, input_metadata.name,
                output_metadata.name, c, t, v, m,
                input_metadata.datatype)

        return (model_config.max_batch_size, input_metadata.name,
                [output_metadata.name], c, t, v, m,
                input_metadata.datatype)
