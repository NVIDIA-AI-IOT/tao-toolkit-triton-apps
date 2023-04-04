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
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision import utils
import torch

from tao_triton.python.postprocessing.postprocessor import Postprocessor
from tao_triton.python.postprocessing.utils import pool_context


    
def make_numpy_grid(tensor_data, pad_value=0,padding=0,num_class=2,gt=None):
    vis = tensor_data.detach()
    vis = utils.make_grid(tensor_data, pad_value=0,padding=0)
    vis = np.array(vis.cpu()).transpose((1,2,0))

    if num_class>2:
        #multi-class visualisation
        #TODO: Supports 10 classes - make it more flexible and not hard coded
        vis_multi = vis[:,:,0]

        #Code for visualising FN/FP (gt!=pred)
        if isinstance(gt, torch.Tensor):
            gt = gt.detach()
            gt = utils.make_grid(gt, pad_value=pad_value,padding=padding)
            gt = np.array(gt.cpu()).transpose((1,2,0))[:,:,0]

    if vis.shape[2] == 1:
        vis = np.stack([vis, vis, vis], axis=-1)

    
    if num_class>2:
        color_coded = np.ones(np.shape(vis))
        for i in range(10):
            color_coded[vis_multi==i] = colour_mappings[str(i)]
        color_coded= color_coded/255
        color_coded  = color_coded.astype(float)
        
        #Code for visualising FN/FP (gt!=pred)
        if isinstance(gt, np.ndarray):
            color_coded_mismatch = np.copy(color_coded)
            color_coded_mismatch[vis_multi!=gt] = (0,0,0)
            color_coded_mismatch  = color_coded_mismatch.astype(float)
            return color_coded, color_coded_mismatch

        return color_coded
    else: 
        return vis

def _visualize_pred(pred):
    import torch
    pred = torch.Tensor(pred)
    pred = torch.argmax(pred, dim=1, keepdims=True)
    pred_vis = pred * 255
    return pred_vis

def de_norm(tensor_data):
    tensor_data = tensor_data * 0.5 + 0.5
    tensor_data = torch.Tensor(tensor_data)
    return tensor_data

class ChangeFormerPostprocessor(Postprocessor):
    """Class to run post processing of Triton Tensors."""

    def __init__(self, batch_size, frames, output_path, data_format):
        """Initialize a post processor class for a changeformer model.
        
        Args:
            batch_size (int): Number of images in the batch.
            frames (list): List of images.
            output_path (str): Unix path to the output rendered images and labels.
            data_format (str): Order of the input model dimensions.
                "channels_first": CHW order.
                "channels_last": HWC order.
        """
        super().__init__(batch_size, frames, output_path, data_format)
        self.output_names = ["output",
                             "6746",
                             "6833",
                             "6920",
                             "6938"]
        self.final_output = "6938"
        
    def apply(self, results, this_id, render=True, batching=True, img_name=None):
        """Apply the post processor to the outputs to the changeformer outputs."""

        output_cf = results.as_numpy(self.final_output)

        vis = make_numpy_grid(_visualize_pred(output_cf))
        vis = np.clip(vis, a_min=0.0, a_max=1.0)
        file_name = os.path.join(
            self.output_path, 'eval_' + str(img_name)+'.jpg')
        plt.imsave(file_name, vis)