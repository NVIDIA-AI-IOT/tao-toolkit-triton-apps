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
import PIL.ImageColor as ImageColor

from tao_triton.python.postprocessing.postprocessor import Postprocessor
from tao_triton.python.postprocessing.utils import pool_context

from skimage.transform import resize
from skimage import measure
import json

def postprocess_fn(y_pred, nms_size, mask_size, n_classes):
    """Proccess raw output from TRT engine."""
    y_detection = y_pred[0].reshape((-1, nms_size, 6))
    y_mask = y_pred[1].reshape((-1, nms_size, n_classes, mask_size, mask_size))
    y_mask[y_mask < 0] = 0
    return [y_detection, y_mask]

class PeoplesegnetPostprocessor(Postprocessor):
    """Class to run post processing of Triton Tensors."""

    def __init__(self, batch_size, frames, output_path, data_format):
        """Initialize a post processor class for a peoplesegnet model.
        
        Args:
            batch_size (int): Number of images in the batch.
            frames (list): List of images.
            output_path (str): Unix path to the output rendered images and labels.
            data_format (str): Order of the input model dimensions.
                "channels_first": CHW order.
                "channels_last": HWC order.
        """
        super().__init__(batch_size, frames, output_path, data_format)
        self.output_names = ["generate_detections",
                             "mask_fcn_logits/BiasAdd"]
        self.threshold = 0.8
        self.keep_aspect_ratio = True
        self.class_mapping = {1: 'people', 2: ''}
        self.mask_conf=0.4
        self.is_draw_mask = True
        self.dump_coco = True

    def draw_mask_on_image_array(self, pil_image, mask, color='red', alpha=0.4):
        """Draws mask on an image.

        Args:
            image: PIL image (img_height, img_height, 3)
            mask: a uint8 numpy array of shape (img_height, img_height) with
                values between either 0 or 1.
            color: color to draw the keypoints with. Default is red.
            alpha: transparency value between 0 and 1. (default: 0.4)

        Raises:
            ValueError: On incorrect data type for image or masks.
        """
        rgb = ImageColor.getrgb(color)
        solid_color = np.expand_dims(np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
        pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
        pil_mask = Image.fromarray(np.uint8(255.0 * alpha * mask)).convert('L')
        pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
        return pil_image

    
    def generate_annotation_single_img(self, img_in_path, img, img_ratio,
                                       y_decoded, y_mask, is_draw_bbox,
                                       is_label_export, is_draw_mask):
        """helper function to draw bbox on original img and get kitti label on single image.

        Note: img will be modified in-place.
        """

        kitti_txt = ''
        json_txt = []
        draw = ImageDraw.Draw(img)
        ww, hh = img.size
        color_list = ['Black', 'Red', 'Blue', 'Gold', 'Purple']
        for idx, i in enumerate(y_decoded):
            if float(i[-1]) < self.threshold:
                continue
            i[0:4] *= img_ratio
            ii = i[:4].astype(np.int)
            ii[0] = min(max(0, ii[0]), hh)
            ii[1] = min(max(0, ii[1]), ww)
            ii[2] = max(min(hh, ii[2]), 0)
            ii[3] = max(min(ww, ii[3]), 0)

            if (ii[2] - ii[0]) <= 0 or (ii[3] - ii[1]) <= 0:
                continue

            if is_draw_bbox:
                draw.rectangle(
                    ((ii[1], ii[0]), (ii[3], ii[2])),
                    outline=color_list[int(i[-2]) % len(color_list)]
                )
                # txt pad
                draw.rectangle(((ii[1], ii[0]), (ii[1] + 100, ii[0]+10)),
                               fill=color_list[int(i[-2]) % len(color_list)])

                if self.class_mapping:
                    draw.text(
                        (ii[1], ii[0]), "{0}: {1:.2f}".format(
                            self.class_mapping[int(i[-2])], i[-1]))
            # Compute masks
            mask_i = np.zeros((hh, ww))
            mask_i[ii[0]:ii[2], ii[1]:ii[3]] = resize(
                y_mask[idx, int(i[-2]), :, :], (ii[2] - ii[0], ii[3] - ii[1]))
            if np.max(mask_i) > 0:
                mask_i /= np.max(mask_i)
                # Apply mask smoothing
                mask_i = np.sqrt(mask_i)

            # Apply mask confidence threshold
            mask_i = (mask_i > self.mask_conf).astype(np.uint8)
            mask_i = np.asfortranarray(mask_i)
            if is_draw_mask:
                img = self.draw_mask_on_image_array(
                    img, mask_i, color=color_list[int(i[-2] % len(color_list))], alpha=0.6)
                draw = ImageDraw.Draw(img)
            if is_label_export:
                # KITTI export is for INTERNAL only
                if self.dump_coco:
                    json_obj = {}
                    hhh, www = ii[3] - ii[1], ii[2] - ii[0]
                    json_obj['area'] = int(www * hhh)
                    json_obj['is_crowd'] = 0
                    json_obj['image_id'] = os.path.basename(img_in_path)
                    json_obj['bbox'] = [int(ii[1]), int(ii[0]), int(hhh), int(www)]
                    json_obj['id'] = idx
                    json_obj['category_id'] = int(i[-2])
                    # convert mask to polygon
                    json_obj["segmentation"] = []
                    contours = measure.find_contours(mask_i, 0.5)
                    for contour in contours:
                        contour = np.flip(contour, axis=1)
                        segmentation = contour.ravel().tolist()
                        json_obj["segmentation"].append(segmentation)
                    json_txt.append(json_obj)
                else:
                    if i[-1] >= self.threshold:
                        kitti_txt += self.class_mapping[int(i[-2])] + ' 0 0 0 ' + \
                            ' '.join(str(x) for x in [ii[1], ii[0], ii[3], ii[2]]) + \
                            ' 0 0 0 0 0 0 0 ' + \
                            str(i[-1])+'\n'
        return img, json_txt if self.dump_coco else kitti_txt




    def apply(self, results, this_id, render=True, batching=True):
        """Apply the post processor to the outputs to the peoplesegnet outputs."""

        output_array = []      
        mask_size = 28
        nms_size = 100
        n_classes = 2
  
        for output_name in self.output_names:
            output_array.append(results.as_numpy(output_name))

        #y_pred = [i.reshape(max_batch_size, -1)[:actual_batch_size] for i in output_array]
        y_pred = [i.reshape(1, -1)[:1] for i in output_array]

        y_pred_decoded = postprocess_fn(y_pred, nms_size, mask_size, n_classes)


        for image_idx in range(self.batch_size):
            current_idx = (int(this_id) - 1) * self.batch_size + image_idx
            if current_idx >= len(self.frames):
                break
            current_frame = self.frames[current_idx]
            img_in_path = current_frame._image_path
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


            img, json_txt = self.generate_annotation_single_img(
                img_in_path,
                img, 
                ratio, 
                y_pred_decoded[0][0, ...],  
                y_pred_decoded[1][0, ...], 
                output_image_file, 
                output_label_file,
                self.is_draw_mask
            )


            img.save(output_image_file)

            with open(output_label_file, "w") as json_file:
                json.dump(json_txt, json_file, indent=4, sort_keys=True)
