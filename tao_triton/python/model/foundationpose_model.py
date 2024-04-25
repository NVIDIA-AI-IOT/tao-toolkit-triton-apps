# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
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
import trimesh
import torch
import numpy as np
from transformations import euler_matrix
from pytorch3d.transforms import so3_exp_map

import tritonclient.grpc as grpcclient
import tritonclient.grpc.model_config_pb2 as mc
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException
from tritonclient.utils import triton_to_np_dtype

from tao_triton.python.model.triton_model import TritonModel
from tao_triton.python.postprocessing.foundationpose_postprocessor import *



class FoundationposeModel(TritonModel):
    """Simple class to run model inference using Triton client."""

    def __init__(self, max_batch_size, input_names, output_names,
                 channels, height, width, data_format,
                 triton_dtype):
        """Set up a foundationpose triton model instance.

        Args:
            max_batch_size(int): The maximum batch size of the TensorRT engine.
            input_names (str): List of the input node names
            output_names (str): List of the output node names
            channels (int): Number of chanels in the input dimensions
            height (int): Height of the input
            width (int): Width of the input
            data_format (str): The input dimension order. This can be "channels_first"
                or "channels_last". "channels_first" is in the CHW order,
                and "channels_last" is in HWC order.
            triton_dtype (proto): Triton input data type.
            channel_mode (str): String order of the C dimension of the input.
                "RGB" or "BGR"

        Returns:
            An instance of the DetectnetModel.
        """
        self.max_batch_size = max_batch_size
        self.input_names = input_names
        self.output_names = output_names
        self.c = channels
        assert channels in [1, 3, 6], (
            "TAO Toolkit models only support 1, 3 or 6 channel inputs."
        )
        self.h = height
        self.w = width
        self.data_format = data_format
        self.triton_dtype = triton_dtype
        self.scale = 1
        if channels == 3:
            self.mean = [0., 0., 0.]
        else:
            self.mean = [0]
        self.mean = np.asarray(self.mean).astype(np.float32)
        if self.data_format == mc.ModelInput.FORMAT_NCHW:
            self.mean = self.mean[:, np.newaxis, np.newaxis]

    @staticmethod
    def parse_model(model_metadata, model_config):
        """Parse model metadata and model config from the triton server."""

        if len(model_metadata.inputs) != 2:
            raise Exception("expecting 2 input, got {}".format(
                len(model_metadata.inputs)))
        if model_metadata['name'] == 'foundationpose_refiner_tao' and len(model_metadata.outputs) != 2:
            raise Exception("expecting 2 output, got {}".format(
                len(model_metadata.outputs)))
        if model_metadata['name'] == 'foundationpose_scorer_tao' and len(model_metadata.outputs) != 1:
            raise Exception("expecting 1 output, got {}".format(
                len(model_metadata.outputs)))

        if len(model_config.input) != 2:
            raise Exception(
                "expecting 2 input in model configuration, got {}".format(
                    len(model_config.input)))

        input_metadata = model_metadata.inputs
        input_config = model_config.input
        output_metadata = model_metadata.outputs

        for _, data in enumerate(output_metadata):
            if data.datatype != "FP32":
                raise Exception("expecting output datatype to be FP32, model '" +
                            data.name + "' output type is " +
                            data.datatype)

        # Model input must have 3 dims, either CHW or HWC (not counting
        # the batch dimension), either CHW or HWC
        input_batch_dim = (model_config.max_batch_size > 0)
        expected_input_dims = 3 + (1 if input_batch_dim else 0)
        if type(input_config) == tuple:
            for i in range(len(input_config)): 
                if len(input_metadata[i].shape) != expected_input_dims:
                    raise Exception(
                        "expecting input to have {} dimensions, model '{}' input has {}".
                        format(expected_input_dims, model_metadata.name,
                            len(input_metadata[i].shape)))

        if type(input_config) == tuple:
            for i in range(len(input_config)): 
                if type(input_config[i].format) == str:
                    FORMAT_ENUM_TO_INT = dict(mc.ModelInput.Format.items())
                    input_config[i].format = FORMAT_ENUM_TO_INT[input_config[i].format]

        if type(input_config) == tuple:
            for i in range(len(input_config)): 
                if ((input_config[i].format != mc.ModelInput.FORMAT_NCHW) and
                    (input_config[i].format != mc.ModelInput.FORMAT_NHWC)):
                    raise Exception("unexpected input format " +
                                    mc.ModelInput.Format.Name(input_config[i].format) +
                                    ", expecting " +
                                    mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NCHW) +
                                    " or " +
                                    mc.ModelInput.Format.Name(mc.ModelInput.FORMAT_NHWC))

        if type(input_config) == tuple: # Can take for only 1 of the inputs
            for i in range(len(input_config)): 
                if input_config[i].format == mc.ModelInput.FORMAT_NHWC:
                    h = input_metadata[i].shape[1 if input_batch_dim else 0]
                    w = input_metadata[i].shape[2 if input_batch_dim else 1]
                    c = input_metadata[i].shape[3 if input_batch_dim else 2]
                else:
                    c = input_metadata[i].shape[1 if input_batch_dim else 0]
                    h = input_metadata[i].shape[2 if input_batch_dim else 1]
                    w = input_metadata[i].shape[3 if input_batch_dim else 2]

        print(model_config.max_batch_size, [input_meta.name for input_meta in input_metadata],
                [data.name for data in output_metadata], c, h, w, input_config[0].format,
                input_metadata[0].datatype)

        return (model_config.max_batch_size, [input_meta.name for input_meta in input_metadata],
                [data.name for data in output_metadata], c, h, w, input_config[0].format,
                input_metadata[0].datatype)

    @classmethod
    def from_metadata(cls, modelA_metadata, modelA_config, modelB_metadata, modelB_config):
        """Parse a model from the metadata config."""
        parsed_outputs = cls.parse_model(modelA_metadata, modelA_config)

        max_batch_size, input_names, output_names, channels, height, width, \
            data_format, triton_dtype = parsed_outputs

        return cls(
            max_batch_size, input_names, output_names,
            channels, height, width, data_format,
            triton_dtype
        )

    def make_rotation_grid(self, min_n_views=40, inplane_step=60, device='cuda'):
        cam_in_obs = sample_views_icosphere(n_views=min_n_views)
        rot_grid = []
        for i in range(len(cam_in_obs)):
            for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
                cam_in_ob = cam_in_obs[i]
                R_inplane = euler_matrix(0, 0, inplane_rot)
                cam_in_ob = cam_in_ob @ R_inplane
                ob_in_cam = np.linalg.inv(cam_in_ob)
                rot_grid.append(ob_in_cam)

        rot_grid = np.asarray(rot_grid)
        self.rot_grid = torch.as_tensor(rot_grid, device=device, dtype=torch.float)

    def guess_translation(self, depth, mask, K):
        vs, us = np.where(mask > 0)
        if len(us) == 0:
            return np.zeros((3))
        uc = (us.min() + us.max()) / 2.0
        vc = (vs.min() + vs.max()) / 2.0
        valid = mask.astype(bool) & (depth >= 0.1)
        if not valid.any():
            return np.zeros((3))

        zc = np.median(depth[valid])
        center = (np.linalg.inv(K) @ np.asarray([uc, vc, 1]).reshape(3, 1)) * zc
        return center.reshape(3)

    def generate_random_pose_hypo(self, K, depth, mask, device):
        '''
        @scene_pts: torch tensor (N,3)
        '''
        ob_in_cams = self.rot_grid.clone()
        center = self.guess_translation(depth=depth, mask=mask, K=K)
        ob_in_cams[:, :3, 3] = torch.tensor(center, device=device, dtype=torch.float).reshape(1, 3)
        return ob_in_cams
    
    def preprocess(self, triton_client, mesh_file, intrinsic_file):
        self.triton_client = triton_client
        self.mesh = trimesh.load(mesh_file)
        self.to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.extent_bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
        self.mesh_reset, self.mesh_tensor, self.diameter, self.model_center = reset_object(model_pts=self.mesh.vertices, model_normals=self.mesh.vertex_normals, mesh=self.mesh)

        self.K = np.loadtxt(intrinsic_file).reshape(3, 3)
        self.make_rotation_grid(min_n_views=40, inplane_step=60)
        self.tracking_pose = None

    def make_rotation_grid(self, min_n_views=40, inplane_step=60, device='cuda'):
        cam_in_obs = sample_views_icosphere(n_views=min_n_views)
        rot_grid = []
        for i in range(len(cam_in_obs)):
            for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
                cam_in_ob = cam_in_obs[i]
                R_inplane = euler_matrix(0, 0, inplane_rot)
                cam_in_ob = cam_in_ob @ R_inplane
                ob_in_cam = np.linalg.inv(cam_in_ob)
                rot_grid.append(ob_in_cam)

        rot_grid = np.asarray(rot_grid)
        self.rot_grid = torch.as_tensor(rot_grid, device=device, dtype=torch.float)

    def refiner_predict(self, rgb, depth, K, ob_in_cams, xyz_map, normal_map=None, mesh=None, mesh_tensors=None, mesh_diameter=None, iteration=5, model_metadata=None):
        tf_to_center = np.eye(4)
        ob_centered_in_cams = ob_in_cams
        mesh_centered = mesh

        crop_ratio = 1.2
        B_in_cams = torch.as_tensor(ob_centered_in_cams, device='cuda', dtype=torch.float)
        rgb_tensor = torch.as_tensor(rgb, device='cuda', dtype=torch.float)
        depth_tensor = torch.as_tensor(depth, device='cuda', dtype=torch.float)
        xyz_map_tensor = torch.as_tensor(xyz_map, device='cuda', dtype=torch.float)

        trans_normalizer = [0.019999999552965164, 0.019999999552965164, 0.05000000074505806]
        trans_normalizer = torch.as_tensor(list(trans_normalizer), device='cuda', dtype=torch.float).reshape(1,3)
        render_size = (160, 160)

        input_names = [input_meta.name for input_meta in model_metadata.inputs]
        output_names = [output_meta.name for output_meta in model_metadata.outputs]
        triton_dtype = [input_meta.datatype for input_meta in model_metadata.inputs]
        model_name = model_metadata.name

        client = httpclient

        for _ in range(iteration):
            pose_data = make_crop_data_batch(render_size, B_in_cams, mesh_centered, rgb_tensor, depth_tensor, K, crop_ratio=crop_ratio, normal_map=normal_map, xyz_map=xyz_map_tensor, mesh_tensors=mesh_tensors, mesh_diameter=mesh_diameter)

            B_in_cams = []
            A = torch.cat([pose_data.rgbAs.cuda(), pose_data.xyz_mapAs.cuda()], dim=1).float()
            B = torch.cat([pose_data.rgbBs.cuda(), pose_data.xyz_mapBs.cuda()], dim=1).float()

            inputA = A.cpu().numpy()
            inputB = B.cpu().numpy()

            batched_image_data = list([inputA, inputB])

            input_array = []
            for i in range(len(input_names)):
                inputs = [client.InferInput(input_names[i], batched_image_data[i].shape, triton_dtype[i])]
                inputs[0].set_data_from_numpy(batched_image_data[i])
                input_array.append(inputs[0])
            inputs = input_array

            outputs = [
                client.InferRequestedOutput(
                    out_name
                ) for out_name in output_names
            ]

            response = self.triton_client.infer(model_name, inputs, request_id=str(1), outputs=outputs)
            
            trans = torch.tensor(response.as_numpy(output_names[0]))
            rot = torch.tensor(response.as_numpy(output_names[1]))

            trans_delta = trans

            rot_mat_delta = torch.tanh(rot) * 0.3490658503988659
            rot_mat_delta = so3_exp_map(rot_mat_delta).permute(0,2,1)


            trans_delta *= (mesh_diameter / 2)

            B_in_cam = egocentric_delta_pose_to_pose(pose_data.poseA, trans_delta=trans_delta, rot_mat_delta=rot_mat_delta)
            B_in_cams.append(B_in_cam)

            B_in_cams = torch.cat(B_in_cams, dim=0).reshape(len(ob_in_cams), 4, 4)

        B_in_cams_out = B_in_cams @ torch.tensor(tf_to_center[None], device='cuda', dtype=torch.float)
        torch.cuda.empty_cache()

        return B_in_cams_out

    @torch.inference_mode()
    def find_best_among_pairs(self, pose_data:BatchPoseData, model_metadata):
        ids = []
        scores = []

        input_names = [input_meta.name for input_meta in model_metadata.inputs]
        output_names = [output_meta.name for output_meta in model_metadata.outputs]
        triton_dtype = [input_meta.datatype for input_meta in model_metadata.inputs]
        
        client = httpclient

        A = torch.cat([pose_data.rgbAs.cuda(), pose_data.xyz_mapAs.cuda()], dim=1).float()
        B = torch.cat([pose_data.rgbBs.cuda(), pose_data.xyz_mapBs.cuda()], dim=1).float()
        
        if pose_data.normalAs is not None:
            A = torch.cat([A, pose_data.normalAs.cuda().float()], dim=1)
            B = torch.cat([B, pose_data.normalBs.cuda().float()], dim=1)


        inputA = A.cpu().numpy()
        inputB = B.cpu().numpy()

        batched_image_data = list([inputA, inputB])

        input_array = []
        for i in range(len(input_names)):
            inputs = [client.InferInput(input_names[i], batched_image_data[i].shape, triton_dtype[i])]
            inputs[0].set_data_from_numpy(batched_image_data[i])
            input_array.append(inputs[0])
        inputs = input_array

        outputs = [
            client.InferRequestedOutput(
                out_name
            ) for out_name in output_names
        ]

        response = self.triton_client.infer("foundationpose_scorer_tao", inputs, request_id=str(1), outputs=outputs)
        
        score_logit = torch.tensor(response.as_numpy("score_logit"))

        scores_cur = score_logit.float().reshape(-1)
        ids.append(scores_cur.argmax())
        scores.append(scores_cur)
        ids = torch.stack(ids, dim=0).reshape(-1)
        scores = torch.cat(scores, dim=0).reshape(-1)
        return ids, scores
    
    def scorer_predict(self, rgb, depth, K, ob_in_cams, mesh=None, mesh_tensors=None, mesh_diameter=None, model_metadata=None):
        '''
        @rgb: np array (H,W,3)
        '''
        ob_in_cams = torch.as_tensor(ob_in_cams, dtype=torch.float, device='cuda')

        rgb = torch.as_tensor(rgb, device='cuda', dtype=torch.float)
        depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)

        pose_data = make_crop_data_batch_score((160, 160), ob_in_cams, mesh, rgb, depth, K, crop_ratio=1.1, mesh_tensors=mesh_tensors, mesh_diameter=mesh_diameter)

        pose_data_iter = pose_data
        global_ids = torch.arange(len(ob_in_cams), device='cuda', dtype=torch.long)
        scores_global = torch.zeros((len(ob_in_cams)), dtype=torch.float, device='cuda')
        
        while 1:
            ids, scores = self.find_best_among_pairs(pose_data_iter, model_metadata)
            if len(ids)==1:
                scores_global[global_ids] = scores + 100
                break
            global_ids = global_ids[ids]
            pose_data_iter = pose_data.select_by_indices(global_ids)

        scores = scores_global

        torch.cuda.empty_cache()

        return scores
    
    def register(self, rgb, depth, ob_mask, mesh, mesh_tensor, mesh_diameter, mesh_model_center, modelA_metadata, modelB_metadata, iteration=2, device='cuda'):
        depth = bilateral_filter_depth(depth, radius=2, device=device)

        normal_map = None
        self.H, self.W = depth.shape[:2]
        self.ob_mask = ob_mask

        poses = self.generate_random_pose_hypo(K=self.K, depth=depth, mask=self.ob_mask, device=device)
        center = self.guess_translation(depth=depth, mask=self.ob_mask, K=self.K)
        poses = torch.as_tensor(poses, device=device, dtype=torch.float)
        poses[:, :3, 3] = torch.as_tensor(center.reshape(1, 3), device=device)

        xyz_map = depth2xyzmap(depth, self.K)

        poses = self.refiner_predict(mesh=mesh, mesh_tensors=mesh_tensor, rgb=rgb, 
                                     depth=depth, K=self.K, ob_in_cams=poses.data.cpu().numpy(), 
                                     normal_map=normal_map, xyz_map=xyz_map, mesh_diameter=mesh_diameter, 
                                     iteration=iteration, model_metadata=modelA_metadata)
        
        scores = self.scorer_predict(mesh=mesh, rgb=rgb, depth=depth, K=self.K, 
                                     ob_in_cams=poses.data.cpu().numpy(), 
                                     mesh_tensors=mesh_tensor, mesh_diameter=mesh_diameter, model_metadata=modelB_metadata)
        
        ids = torch.as_tensor(scores).argsort(descending=True)
        scores = scores[ids]
        poses = poses[ids]
        best_pose = poses[0] @ get_tf_to_centered_mesh(mesh_model_center)

        return best_pose.data.cpu().numpy(), poses[0]

    def track_one(self, rgb, depth, pose_last, mesh, mesh_tensor, mesh_diameter, mesh_model_center, model_metadata, iteration=2):

        depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)
        depth = bilateral_filter_depth(depth, radius=2, device='cuda')

        xyz_map = depth2xyzmap_batch(depth[None], torch.as_tensor(self.K, dtype=torch.float, device='cuda')[None], zfar=np.inf)[0]

        pose = self.refiner_predict(mesh=mesh, mesh_tensors=mesh_tensor, rgb=rgb, 
                                    depth=depth, K=self.K, ob_in_cams=pose_last.reshape(1, 4, 4).data.cpu().numpy(), 
                                    normal_map=None, xyz_map=xyz_map, mesh_diameter=mesh_diameter,
                                    iteration=iteration, model_metadata=model_metadata)

        return (pose @ get_tf_to_centered_mesh(mesh_model_center)).data.cpu().numpy().reshape(4, 4), pose
    
    def draw_image(self, color, pose, to_origin, extent_bbox):
        center_pose = np.array(pose) @ np.linalg.inv(np.array(to_origin))
        vis = draw_posed_3d_box(self.K, img=color, ob_in_cam=center_pose, bbox=extent_bbox)
        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=0.1, K=self.K, thickness=3, transparency=0, is_input_rgb=True)
        return vis

    def process(self, batched, sent_count, modelA_metadata, modelB_metadata, bbox):
        image, depth = batched
        H, W = depth.shape

        if sent_count == 0:
            bbox = bbox.split(',')
            umin, vmin, umax, vmax = bbox
            umin, vmin, umax, vmax = int(umin), int(vmin), int(umax), int(vmax)
            mask = np.zeros((H, W))
            mask[vmin:vmax, umin:umax] = 1

            pose, self.tracking_pose = self.register(image, depth, mask, self.mesh, self.mesh_tensor, self.diameter, self.model_center, modelA_metadata, modelB_metadata)
        
        else:
            pose, self.tracking_pose = self.track_one(image, depth, self.tracking_pose, self.mesh, self.mesh_tensor, self.diameter, self.model_center, modelA_metadata)

        viz = self.draw_image(image, pose, self.to_origin, self.extent_bbox)
        return viz