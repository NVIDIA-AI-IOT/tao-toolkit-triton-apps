"""Convert pose data from deepstream-bodypose-3d to skeleton arrays."""

import json
import yaml
import numpy as np


def _create_data_array(data_array, pose_sequence, frame_start, frame_end, pose_type, num_joints, sequence_length_max):
    """Create a NumPy array for output.
    
    # Arguments
        data_array (np.array): Initial Numpy array.
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
    if data_array is None:
        data_array = sequence
    else:
        data_array = np.concatenate((data_array, sequence), axis=0)
    return data_array


def pose_cls_dataset_convert(pose_data_file, dataset_convert_config):
    """Extract sequences from pose data and apply normalization.
    
    # Arguments
        pose_data_file (str): Path to JSON pose data.
        dataset_convert_config (str): Path to the YAML config for dataset conversion.

    # Returns
        Numpy array of pose sequences and pose data with placeholder for action.
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
    pose_sequences = {}
    frame_data = {}
    for batch in pose_data:
        assert batch["num_frames_in_batch"] == len(batch["batches"]), f"batch[\"num_frames_in_batch\"] "\
            f"{batch['num_frames_in_batch']} does not match len(batch[\"batches\"]) {len(batch['batches'])}."
        for frame in batch["batches"]:
            for person in frame["objects"]:
                object_id = person["object_id"]
                if object_id not in pose_sequences:
                    pose_sequences[object_id] = []
                    frame_data[object_id] = []

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

                pose_sequences[object_id].append(poses)
                frame_data[object_id].append(frame["frame_num"])

    # Create output data array
    sequence_length_max = dc_config["sequence_length_max"]
    sequence_length_min = dc_config["sequence_length_min"]
    sequence_length = dc_config["sequence_length"]
    sequence_overlap = dc_config["sequence_overlap"]
    step = int(sequence_length * sequence_overlap)
    data_arrays = []
    segment_assignments = {}
    segment_id = 0

    for object_id in pose_sequences.keys():
        # Create segments of data arrays
        data_array = None
        frame_start = 0
        while len(pose_sequences[object_id]) - frame_start >= sequence_length_min:
            frame_end = frame_start + sequence_length
            if len(pose_sequences[object_id]) - frame_start < sequence_length:
                frame_end = len(pose_sequences[object_id])
            data_array = _create_data_array(data_array, pose_sequences[object_id], frame_start, frame_end,
                                            pose_type, num_joints, sequence_length_max)
            if frame_end - frame_start > step:
                for i in range(frame_start + step, frame_end):
                    segment_assignments[(object_id, frame_data[object_id][i])] = segment_id
            segment_id += 1
            frame_start += step

        # Accumulate data arrays of all objects
        data_arrays.append(data_array)

    # Update pose data for returning (removing pose metadata)
    for b in range(len(pose_data)):
        for f in range(len(pose_data[b]["batches"])):
            pose_data[b]["batches"][f].pop("num_obj_meta", None)
            frame_num = pose_data[b]["batches"][f]["frame_num"]
            for p in range(len(pose_data[b]["batches"][f]["objects"])):
                object_id = pose_data[b]["batches"][f]["objects"][p]["object_id"]
                pose_data[b]["batches"][f]["objects"][p]["segment_id"] = \
                    segment_assignments.get((object_id, frame_num), -1)
                pose_data[b]["batches"][f]["objects"][p]["action"] = ""

    return np.concatenate(data_arrays, axis=0), pose_data
