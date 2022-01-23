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


class MultitaskClassificationPostprocessor(Postprocessor):
    """Class to run post processing of Triton Tensors."""

    def __init__(self, batch_size, frames, output_path, data_format):
        """Initialize a post processor class for a multitaskclassification model.
        
        Args:
            batch_size (int): Number of images in the batch.
            frames (list): List of images.
            output_path (str): Unix path to the output rendered images and labels.
            data_format (str): Order of the input model dimensions.
                "channels_first": CHW order.
                "channels_last": HWC order.
        """
        super().__init__(batch_size, frames, output_path, data_format)
        self.output_name_0 = "base_color/Softmax"
        self.output_name_1 = "category/Softmax"
        self.output_name_2 = "season/Softmax"
        self.task_name = ["base_color", "category", "season"]
        self.class_mapping = {"base_color": {"0": "Black", "1": "Blue", "2": "Brown", "3": "Green", \
                              "4": "Grey", "5": "Navy Blue", "6": "Pink", "7": "Purple", "8": "Red", \
                              "9": "Silver", "10": "White"}, 
                              "category": {"0": "Bags", "1": "Bottomwear", "2": "Eyewear", "3": "Fragrance", \
                              "4": "Innerwear", "5": "Jewellery", "6": "Sandal", "7": "Shoes", "8": "Topwear", \
                              "9": "Watches"}, 
                              "season": {"0": "Fall", "1": "Spring", "2": "Summer", "3": "Winter"}}

    def apply(self, output_tensors, this_id, render=True, batching=True):
        """Apply the post processor to the outputs to the classification outputs."""
        output_array_0 = output_tensors.as_numpy(self.output_name_0)
        output_array_1 = output_tensors.as_numpy(self.output_name_1)
        output_array_2 = output_tensors.as_numpy(self.output_name_2)
        output_array = [output_array_0 , output_array_1 , output_array_2]

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        output_file = os.path.join(self.output_path, "results.txt")

        for image_idx in range(self.batch_size):
            current_idx = (int(this_id) - 1) * self.batch_size + image_idx
            if current_idx >= len(self.frames):
                break
            current_frame = self.frames[current_idx]
            img_in_path = current_frame._image_path
            filename = os.path.basename(current_frame._image_path)
            print("filename is {}".format(filename))

            with open(output_file, "a") as j:
                j.write("\n{}:\n".format(filename))

            for idx, task in enumerate(self.task_name):
                pred = output_array[idx].reshape(-1)
                print("Task {}:".format(task))
                #print("Predictions: {}".format(pred))
                class_name = self.class_mapping[task][str(np.argmax(pred))]
                print("Class name = {}".format(class_name))
                print('********')

                with open(output_file, "a") as j:
                    j.write("{}: {}\n".format(task,class_name))
