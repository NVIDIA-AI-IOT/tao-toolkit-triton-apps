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

repo_root = os.getenv("TAO_TRITON_REPO_ROOT", "")
characters_list_file = os.path.join(repo_root, "model_repository/lprnet_tao/characters_list.txt")

def get_classes_id():
    with open(characters_list_file, "r") as f:
        temp_list = f.readlines()
    classes = [i.strip() for i in temp_list]
    blank_id = len(classes)

    return classes, blank_id


def decode_ctc_conf(pred,
                    classes,
                    blank_id):
    '''
    Decode ctc trained model's output.

    return decoded license plate and confidence.
    '''
    pred_id = pred['tf_op_layer_ArgMax']
    pred_conf = pred['tf_op_layer_Max']
    decoded_lp = []
    decoded_conf = []

    for idx_in_batch, seq in enumerate(pred_id):
        seq_conf = pred_conf[idx_in_batch]
        prev = seq[0]
        tmp_seq = [prev]
        tmp_conf = [seq_conf[0]]
        for idx in range(1, len(seq)):
            if seq[idx] != prev:
                tmp_seq.append(seq[idx])
                tmp_conf.append(seq_conf[idx])
                prev = seq[idx]
        lp = ""
        output_conf = []
        for index, i in enumerate(tmp_seq):
            if i != blank_id:
                lp += classes[i]
                output_conf.append(tmp_conf[index])
        decoded_lp.append(lp)
        decoded_conf.append(output_conf)

    return decoded_lp, decoded_conf


class LPRPostprocessor(Postprocessor):
    """Class to run post processing of Triton Tensors."""

    def __init__(self, batch_size, frames, output_path, data_format):
        """Initialize a post processor class for a lprnet model.
        
        Args:
            batch_size (int): Number of images in the batch.
            frames (list): List of images.
            output_path (str): Unix path to the output rendered images and labels.
            data_format (str): Order of the input model dimensions.
                "channels_first": CHW order.
                "channels_last": HWC order.
        """
        super().__init__(batch_size, frames, output_path, data_format)
        self.output_names = ["tf_op_layer_ArgMax",
                             "tf_op_layer_Max"]


    def apply(self, results, this_id, render=True, batching=True):
        """Apply the post processor to the outputs to the lprnet outputs."""

        output_array = {}

        for output_name in self.output_names:
            output_array[output_name] = results.as_numpy(output_name)

        classes, blank_id = get_classes_id()

        decoded_lp, _ = decode_ctc_conf(output_array,
                                        classes=classes,
                                        blank_id=blank_id)

        for idx in range(self.batch_size):
            current_idx = (int(this_id) - 1) * self.batch_size + idx
            image_id = self.frames[current_idx]._image_path
            print(image_id)
        
        print("inference result: {}\n".format(decoded_lp))


        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        output_file = os.path.join(self.output_path, "results.txt")
        with open(output_file, "a") as wfile:
            wfile.write("{} : {}\n".format(image_id,decoded_lp))
