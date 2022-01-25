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
import struct
import numpy as np
from PIL import Image, ImageDraw

from tao_triton.python.postprocessing.postprocessor import Postprocessor
from tao_triton.python.postprocessing.utils import pool_context


    
def trt_output_process_fn(self, y_encoded):
    "function to process TRT model output."
    det_out, keep_k = y_encoded
    result = []
    for idx, k in enumerate(keep_k.reshape(-1)):
        det = det_out[idx].reshape(-1, 7)[:k]
        xmin = det[:, 3] * self.model_input_width
        ymin = det[:, 4] * self.model_input_height
        xmax = det[:, 5] * self.model_input_width
        ymax = det[:, 6] * self.model_input_height
        cls_id = det[:, 1]
        conf = det[:, 2]
        result.append(np.stack((cls_id, conf, xmin, ymin, xmax, ymax), axis=-1))

    return result



class RetinanetPostprocessor(Postprocessor):
    """Class to run post processing of Triton Tensors."""

    def __init__(self, batch_size, frames, output_path, data_format):
        """Initialize a post processor class for a retinanet model.
        
        Args:
            batch_size (int): Number of images in the batch.
            frames (list): List of images.
            output_path (str): Unix path to the output rendered images and labels.
            data_format (str): Order of the input model dimensions.
                "channels_first": CHW order.
                "channels_last": HWC order.
        """
        super().__init__(batch_size, frames, output_path, data_format)
        self.output_names = ["NMS",
                             "NMS_1" ]
        self.threshold = 0.6
        self.keep_aspect_ratio = True
        self.class_mapping = {0: 'background', 1: 'bicycle', 2: 'car', 3: 'person', 4: 'road_sign'}
        self.model_input_width = 960
        self.model_input_height = 544

    def _get_bbox_and_kitti_label_single_img(
        self, img, img_ratio, y_decoded,
        is_draw_img, is_kitti_export
    ):
        """helper function to draw bbox on original img and get kitti label on single image.

        Note: img will be modified in-place.
        """
        kitti_txt = ""
        draw = ImageDraw.Draw(img)
        color_list = ['Black', 'Red', 'Blue', 'Gold', 'Purple']
        for i in y_decoded:
            if float(i[1]) < self.threshold:
                continue

            if self.keep_aspect_ratio:
                i[2:6] *= img_ratio
            else:
                orig_w, orig_h = img.size
                ratio_w = float(orig_w) / self.model_input_width
                ratio_h = float(orig_h) / self.model_input_height
                i[2] *= ratio_w
                i[3] *= ratio_h
                i[4] *= ratio_w
                i[5] *= ratio_h

            if is_kitti_export:
                kitti_txt += self.class_mapping[int(i[0])] + ' 0 0 0 ' +  ' '.join([str(x) for x in i[2:6]])+' 0 0 0 0 0 0 0 ' + str(i[1])+'\n'

            if is_draw_img:
                draw.rectangle(
                    ((i[2], i[3]), (i[4], i[5])),
                    outline=color_list[int(i[0]) % len(color_list)]
                )
                # txt pad
                draw.rectangle(((i[2], i[3]), (i[2] + 100, i[3]+10)),
                               fill=color_list[int(i[0]) % len(color_list)])

                draw.text((i[2], i[3]), "{0}: {1:.2f}".format(self.class_mapping[int(i[0])], i[1]))


        return img, kitti_txt


    def apply(self, results, this_id, render=True, batching=True):
        """Apply the post processor to the outputs to the retinanet outputs."""

        output_array = []      
  
        for output_name in self.output_names:
            if output_name == "NMS_1":
                nms_1_bytes = results.as_numpy(output_name)[0].tobytes()
                nms_1 = np.frombuffer(nms_1_bytes, dtype=np.int32)
                output_array.append(nms_1)
            else:
                output_array.append(results.as_numpy(output_name))

        #..and return results up to the actual batch size.
        #y_pred = [i.reshape(max_batch_size, -1)[:actual_batch_size] for i in output_array] 
        y_pred = [i.reshape(1, -1)[:1] for i in output_array]
        print("y_pred is {}".format(y_pred))

        y_pred_decoded = trt_output_process_fn(self, y_pred)


        for image_idx in range(self.batch_size):
            current_idx = (int(this_id) - 1) * self.batch_size + image_idx
            if current_idx >= len(self.frames):
                break
            current_frame = self.frames[current_idx]
            filename = os.path.basename(current_frame._image_path)

            img = Image.open(current_frame._image_path)
            orig_w, orig_h = img.size
            ratio = min(current_frame.w/float(orig_w), current_frame.h/float(orig_h))
            new_w = int(round(orig_w*ratio))
            ratio = float(orig_w)/new_w

            output_label_file = os.path.join(
                self.output_path, "infer_labels",
                "{}.txt".format(os.path.splitext(filename)[0])
            )
            output_image_file = os.path.join(
                self.output_path, "infer_images",
                "{}.jpg".format(os.path.splitext(filename)[0])
            )
            if not os.path.exists(os.path.dirname(output_label_file)):
                os.makedirs(os.path.dirname(output_label_file))
            if not os.path.exists(os.path.dirname(output_image_file)):
                os.makedirs(os.path.dirname(output_image_file))

            img, kitti_txt = self._get_bbox_and_kitti_label_single_img(img, ratio, y_pred_decoded[0], output_image_file, output_label_file)

            img.save(output_image_file)

            open(output_label_file, 'w').write(kitti_txt)

