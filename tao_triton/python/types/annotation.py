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

"""Definition for bbox annotations."""

import csv
import logging
import os

logger = logging.getLogger(__name__)


class BaseAnnotation(object):
    """Label annotation object."""

    def __init__(self):
        """Initialze an annotation object."""
        self.initialized = True

    def __str__(self):
        """String representation of the annotation object."""
        raise NotImplementedError("This method is not implemented in the base class.")


class KittiBbox(BaseAnnotation):
    """Label annotation for a kitti object."""

    def __init__(self, category, truncation,
                 occlusion, observation_angle,
                 box, height, width, length,
                 x, y, z, world_bbox_rot_y,
                 confidence_score=None):
        """Initialize a kitti annotation object."""
        self.category = category
        self.truncation = float(truncation)
        self.observation_angle = float(observation_angle)
        self.occlusion = int(occlusion)
        self.box = [float(x) for x in box]
        hwlxyz = [float(height), float(width), float(length),
                  float(x), float(y), float(z)]
        self.world_bbox = hwlxyz[3:6] + hwlxyz[0:3]
        self.world_bbox_rot_y = world_bbox_rot_y
        self.confidence = confidence_score
        super(KittiBbox, self).__init__()

    def __str__(self):
        """String representation of the label file."""
        assert self.initialized, ("Annotation should be initialized.")
        world_bbox_str = "{3:.2f} {4:.2f} {5:.2f} {0:.2f} {1:.2f} {2:.2f}".format(
            *self.world_bbox
        )
        bbox_str = "{:0.3f} {:0.3f} {:0.3f} {:0.3f}".format(*self.box)
        if self.confidence is not None:
            return "{0} {1:.2f} {2} {3:0.2f} {4} {5} {6:.2f} {7:0.2f}".format(
                self.category, self.truncation, self.occlusion, self.observation_angle,
                bbox_str, world_bbox_str, self.world_bbox_rot_y, self.confidence
            )
        return "{0} {1:.2f} {2} {3:0.2f} {4} {5} {6:.2f}".format(
            self.category, self.truncation, self.occlusion, self.observation_angle,
            bbox_str, world_bbox_str, self.world_bbox_rot_y
        )

