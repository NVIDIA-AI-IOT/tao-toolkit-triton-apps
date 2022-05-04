"""Convert pose data from deepstream-bodypose-3d to skeleton arrays."""

import json
import yaml
import numpy as np


def _create_data_numpy(data_numpy, pose_sequence, frame_start, frame_end, pose_type, num_joints, sequence_length_max):
    """Create a NumPy array for output.
    
    # Arguments
        data_numpy (np.array): Initial Numpy array.
        pose_sequence (list): List of pose sequence.
        frame_start (int): Starting frame index.
        frame_end (int): Ending frame index.
        pose_type (str): 3dbp, 25dbp or 2dbp.
        num_joints (int): Number of total joints in the skeleton graph layout.
        sequence_length_max (int): Maximum sequence length in frame.

    # Returns
        Processed Numpy array.
    """
    joint_dim = 3
    if pose_type == "2dbp":
        joint_dim = 2
    sequence = np.zeros((1, joint_dim, sequence_length_max, num_joints, 1), dtype="float32")
    f = 0
    for frame in range(frame_start, frame_end):
        for j in range(num_joints):
            for d in range(joint_dim):
                sequence[0, d, f, j, 0] = pose_sequence[frame][j][d]
        f += 1
    if data_numpy is None:
        data_numpy = sequence
    else:
        data_numpy = np.concatenate((data_numpy, sequence), axis=0)
    return data_numpy


def pose_cls_dataset_convert(pose_data_file, track_id, dataset_convert_config):
    """Extract sequences from pose data and apply normalization.
    
    # Arguments
        pose_data_file (str): Path to JSON pose data.
        track_id (int): Track ID to extract the pose sequences.
        dataset_convert_config (str): Path to the YAML config for dataset conversion.

    # Returns
        Numpy array of pose sequences.
    """
    with open(pose_data_file, 'r') as f:
        pose_data = json.load(f)

    with open(dataset_convert_config, 'r') as f:
        dc_config = yaml.load(f, Loader=yaml.FullLoader)

    # Extract pose data
    pose_type = dc_config["pose_type"]
    num_joints = dc_config["num_joints"]
    frame_width = float(dc_config["frame_width"])
    frame_height = float(dc_config["frame_height"])
    focal_length = dc_config["focal_length"]
    pose_sequence = []
    for batch in pose_data:
        assert batch["num_frames_in_batch"] == len(batch["batches"]), f"batch[\"num_frames_in_batch\"] "\
            f"{batch['num_frames_in_batch']} does not match len(batch[\"batches\"]) {len(batch['batches'])}."
        for frame in batch["batches"]:
            for person in frame["objects"]:
                object_id = person["object_id"]
                if object_id != track_id:
                    continue
                poses = []
                if pose_type == "3dbp":
                    if "pose3d" not in list(person.keys()):
                        raise KeyError("\"pose3d\" not found in input data. "\
                                       "Please run deepstream-bodypose-3d with \"--publish-pose pose3d\".")
                    assert num_joints == len(person["pose3d"]) // 4, f"The num_joints should be "\
                        f"{len(person['pose3d']) // 4}. Got {num_joints}."
                    for j in range(num_joints):
                        if person["pose3d"][j*4+3] == 0.0:
                            poses.append([0.0, 0.0, 0.0])
                            continue
                        x = (person["pose3d"][j*4+0] - person["pose3d"][0]) / focal_length
                        y = (person["pose3d"][j*4+1] - person["pose3d"][1]) / focal_length
                        z = (person["pose3d"][j*4+2] - person["pose3d"][2]) / focal_length
                        poses.append([x, y, z])
                elif pose_type in ("25dbp", "2dbp"):
                    if "pose25d" not in list(person.keys()):
                        raise KeyError("\"pose25d\" not found in input data. "\
                                       "Please run deepstream-bodypose-3d with \"--publish-pose pose25d\".")
                    assert num_joints == len(person["pose25d"]) // 4, f"The num_joints should be "\
                        f"{len(person['pose25d']) // 4}. Got {num_joints}."
                    for j in range(num_joints):
                        if person["pose25d"][j*4+3] == 0.0:
                            if pose_type == "25dbp":
                                poses.append([0.0, 0.0, 0.0])
                            else:
                                poses.append([0.0, 0.0])
                            continue
                        x = person["pose25d"][j*4+0] / frame_width - 0.5
                        y = person["pose25d"][j*4+1] / frame_height - 0.5
                        z = person["pose25d"][j*4+2]
                        if pose_type == "25dbp":
                            poses.append([x, y, z])
                        else:
                            poses.append([x, y])
                else:
                    raise NotImplementedError(f"Pose type {pose_type} is not supported.")
                pose_sequence.append(poses)

    # Create output data array
    sequence_length_max = dc_config["sequence_length_max"]
    sequence_length_min = dc_config["sequence_length_min"]
    sequence_length = dc_config["sequence_length"]
    sequence_overlap = dc_config["sequence_overlap"]
    step = int(sequence_length * sequence_overlap)
    data_numpy = None
    frame_start = 0
    sequence_count = 0
    while len(pose_sequence) - frame_start >= sequence_length_min:
        frame_end = frame_start + sequence_length
        if len(pose_sequence) - frame_start < sequence_length:
            frame_end = len(pose_sequence)
        data_numpy = _create_data_numpy(data_numpy, pose_sequence, frame_start, frame_end,
                                        pose_type, num_joints, sequence_length_max)
        frame_start += step
        sequence_count += 1
    if sequence_count > 0:
        return data_numpy
    else:
        raise ValueError(f"The given track ID {track_id} does not exist or does not have enough pose data.")
