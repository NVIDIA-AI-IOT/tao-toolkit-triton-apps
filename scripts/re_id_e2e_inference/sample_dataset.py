import sys
import os
import glob
import re
import random


def sample_dataset(input_dir, output_dir, n_samples, use_ids = None):
    """Select a subset of images fom input_dir and move them to output_dir.
    
    Args:
        input_dir (str): Input Folder Path of the train images.
        output_dir (str): Output Folder Path of the test images.
        n_samples (int): Number of samples to use.
        use_ids(list int): List of IDs to grab from test and query folder.
        
    Returns:
        IDs used for sampling
    """
    img_paths = glob.glob(os.path.join(input_dir, '*.jpg'))
    pattern = re.compile(r'([-\d]+)_c(\d)')
    id_to_img = {}

    # Grab images with matching ids
    for img_path in img_paths:
        pid, _ = map(int, pattern.search(img_path).groups())
        if pid not in id_to_img:
            id_to_img[pid] = []
        id_to_img[pid].append(img_path)
    
    # Create directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        command = "rm -r " + output_dir
        os.system(command)
        os.makedirs(output_dir)

    assert id_to_img, "Dataset size cannot be 0."

    sampled_id_to_img = dict(random.sample(id_to_img.items(), n_samples))

    for key, img_paths in sampled_id_to_img.items():
        for img_path in img_paths:
            command = "cp " + img_path + " " + output_dir
            os.system(command)

    # Use same ids for test and query
    if use_ids:    
        
        # Create query dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        else:
            command = "rm -r " + output_dir
            os.system(command)
            os.makedirs(output_dir)

        # Find images in test with same id
        img_paths = glob.glob(os.path.join(input_dir, '*.jpg'))
        for id in use_ids:
            pattern = re.compile(r'([-\d]+)_c(\d)')
            for img_path in img_paths:
                pid, _ = map(int, pattern.search(img_path).groups())
                if id == pid:
                    print(img_path)
                    command = "cp " + img_path + " " + output_dir
                    os.system(command)

    return sampled_id_to_img.keys()


def main():
    if sys.argv[1]:
        data_dir = sys.argv[1]

        # Number of samples
        n_samples = 100

        # Create train dataset
        train_input_dir = os.path.join(data_dir, "bounding_box_train")
        train_output_dir = os.path.join(data_dir, "sample_train")
        sample_dataset(train_input_dir, train_output_dir, n_samples)

        # Create test dataset
        test_input_dir = os.path.join(data_dir, "bounding_box_test")
        test_output_dir = os.path.join(data_dir, "sample_test")
        ids = sample_dataset(test_input_dir, test_output_dir, n_samples)

        # Create query dataset
        query_input_dir = os.path.join(data_dir, "query")
        query_output_dir = os.path.join(data_dir, "sample_query")
        sample_dataset(query_input_dir, query_output_dir, n_samples, ids)

    else:
        print("Usage: %s data_dir" % __file__)


if __name__ == '__main__':
    main()
