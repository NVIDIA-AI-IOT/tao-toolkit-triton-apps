import sys
import os
import random
import json
import cv2


num_joints = 34
joint_links = {
    (255, 0, 0): [(0, 1), (1, 4), (4, 7), (7, 9), (9, 11), (7, 13), (1, 20), (6, 20), (20, 22), (22, 24), (24, 26), (24, 28), (24, 30), (24, 32)],
    (0, 255, 0): [(0, 3), (3, 6), (6, 15), (15, 16), (15, 17), (16, 18), (17, 19)],
    (0, 0, 255): [(0, 2), (2, 5), (5, 8), (8, 10), (10, 12), (8, 14), (2, 21), (6, 21), (21, 23), (23, 25), (25, 27), (25, 29), (25, 31), (25, 33)]
}


def main():
    if sys.argv[3]:
        json_metadata_path = sys.argv[1]
        input_video_path = sys.argv[2]
        output_video_path = sys.argv[3]
        with open(json_metadata_path, 'r') as f:
            person_metadata = json.load(f)

        objects = {}

        for batch in person_metadata:
            for frame in batch["batches"]:
                frame_num = frame["frame_num"]
                objects[frame_num] = []
                for person in frame["objects"]:
                    objects[frame_num].append(person)

        video_capture = cv2.VideoCapture(input_video_path)
        frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        video_writer = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'),
                                       fps, (frame_width, frame_height))
        success, image_frame = video_capture.read()
        frame_num = 0
        bbox_colors = {}
        while success:
            if frame_num in objects.keys():
                print("Plotting results for frame %06d" % frame_num)
                for person in objects[frame_num]:
                    # Use a random BGR color to represent the object ID
                    object_id = person["object_id"]
                    bbox_color = None
                    if object_id in bbox_colors.keys():
                        bbox_color = bbox_colors[object_id]
                    else:
                        bbox_color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                        bbox_colors[object_id] = bbox_color

                    # Plot 2D body pose
                    joints = []
                    for j in range(num_joints):
                        x = int(person["pose25d"][j*4+0])
                        y = int(person["pose25d"][j*4+1])
                        if person["pose25d"][j*4+3] == 0:
                            x = -1
                            y = -1
                        joints.append((x, y))
                    for link_color in joint_links.keys():
                        for joint_pair in joint_links[link_color]:
                            if joints[joint_pair[0]] == (-1, -1) or joints[joint_pair[1]] == (-1, -1):
                                continue
                            image_frame = cv2.line(image_frame, joints[joint_pair[0]], joints[joint_pair[1]], link_color, 2)
                    for j in range(num_joints):
                        image_frame = cv2.circle(image_frame, joints[j], 2, (255, 255, 255), 2)



                    #Fix bbox, using minmax joints point
                    x_list,y_list = zip(*joints)
                    x_min = min(x_list)
                    x_max = max(x_list)
                    y_min = min(y_list)
                    y_max = max(y_list)
                    bbox = [x_min,y_min,x_max,y_max]
                    
                    # Plot bounding box
                    bbox_top_left = [int(bbox[0]), int(bbox[1])]
                    bbox_bottom_right = [int(bbox[2]), int(bbox[3])]
                    if bbox_top_left[0] < 0:
                        bbox_top_left[0] = 0
                    if bbox_top_left[1] < 0:
                        bbox_top_left[1] = 0
                    if bbox_bottom_right[0] > frame_width - 1:
                        bbox_bottom_right[0] = frame_width - 1
                    if bbox_bottom_right[1] > frame_height - 1:
                        bbox_bottom_right[1] = frame_height - 1
                    image_frame = cv2.rectangle(image_frame, bbox_top_left, bbox_bottom_right, bbox_color, 2)

                    # Plot object ID and action
                    image_frame = cv2.putText(image_frame, f"{str(object_id)}: {person['action']}",
                                              (bbox_top_left[0], bbox_top_left[1] - 10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, bbox_color, 2, cv2.LINE_AA)

            # Plot frame number
            image_frame = cv2.putText(image_frame, "%06d" % frame_num, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                      0.8, (0, 0, 255), 2, cv2.LINE_AA)

            video_writer.write(image_frame)
            success, image_frame = video_capture.read()
            frame_num += 1

        video_capture.release()
        video_writer.release()

        print("Output video saved at %s" % output_video_path)

    else:
        print("Usage: %s json_metadata_path input_video_path output_video_path" % __file__)


if __name__ == '__main__':
    main()
