import cv2
import os
from tqdm import tqdm

import numpy as np
import torch

from xml.etree.ElementTree import parse
from collections import defaultdict

from tao_triton.python.model.nota_yolox import YOLOX

class RawMetrics:
    def __init__(self):
        self.data_dir = f'./selected'
        self.image_dir = f"{self.data_dir}/JPEGImages"
        self.annotation_dir = f"{self.data_dir}/Annotations"
        self.testset_txt_path = f"{self.data_dir}/ImageSets/Main/test.txt"
        
        self.npos = self._get_total_pos()
        self.yolox = YOLOX()

        self.cnts = {
            "bus": np.array([0, 0]), 
            "truck": np.array([0, 0]), 
            "car": np.array([0, 0]), 
            "motorcycle": np.array([0, 0]), 
            "pedestrian":np.array([0, 0]), 
            "bicycle": np.array([0, 0])
        }
        self.metrics = {}

    def _get_actual_bboxes(self, xml_path):
        tree = parse(xml_path)
        root = tree.getroot()

        objects = root.findall("object")
        bboxes = [[0,0,0,0] for _ in range(len(objects))]

        labels = []
        for i, obj in enumerate(objects):
            labels.append(obj.findtext("name"))
            bboxes[i][0] = int(obj.find("bndbox").findtext("xmin"))
            bboxes[i][1] = int(obj.find("bndbox").findtext("ymin"))
            bboxes[i][2] = int(obj.find("bndbox").findtext("xmax"))
            bboxes[i][3] = int(obj.find("bndbox").findtext("ymax"))
        
        return labels, bboxes
    
    
    def _get_total_pos(self,):
        anno_path = os.path.join(os.getcwd(), self.annotation_dir)
        with open(self.testset_txt_path, "r", encoding="utf-8") as f:
            files = f.readlines()
            
        total_post_class_dict = defaultdict(int)
        for file in files:
            file_path = os.path.join(anno_path, file[:-1]+".xml")
            tree = parse(file_path)
            root = tree.getroot()
            objects = root.findall("object")
            labels = [x.findtext("name") for x in objects]
            for label in labels:
                total_post_class_dict[label] += 1

        return total_post_class_dict


    def _count_res(self, pred_bboxes, pred_labels, actual_bboxes, actual_labels, iou_thr):
        if len(pred_labels) == 0 or len(actual_labels) == 0:
            return self.cnts

        for i, pred_bbox in enumerate(pred_bboxes):
            for j, actual_bbox in enumerate(actual_bboxes):
                iou = self._calc_iou(pred_bbox, actual_bbox)

                if iou > iou_thr:
                    if pred_labels[i] == actual_labels[j]:
                        self.cnts[pred_labels[i]][1] += 1
                    else:
                        self.cnts[pred_labels[i]][0] += 1
                else:
                    continue

        return self.cnts


    def _calc_iou(self, box1, box2):
        # box = (x1, y1, x2, y2)
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

        # obtain x1, y1, x2, y2 of the intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        # compute the width and height of the intersection
        w = max(0, x2 - x1 + 1)
        h = max(0, y2 - y1 + 1)

        inter = w * h
        iou = inter / (box1_area + box2_area - inter)
        return iou


    def get_raw_metrics(
        self, 
        file_name,
        pred_bboxes,
        pred_labels,
        nms_thr=0.45, 
        score_thr=0.4, 
        iou_thr=0.5
    ):
        actual_labels, actual_bboxes = self._get_actual_bboxes(xml_path=f"{self.annotation_dir}/{file_name}.xml")
            
        cnts = self._count_res(
            pred_bboxes=pred_bboxes, 
            pred_labels=pred_labels,
            actual_bboxes=actual_bboxes, 
            actual_labels=actual_labels,
            iou_thr=iou_thr,
        )

        for cls in self.cnts.keys():
            fp = self.cnts[cls][0]
            tp = self.cnts[cls][1]
            fn = self.npos[cls] - tp
            
            self.metrics[f"{cls}_FP"] = fp
            self.metrics[f"{cls}_TP"] = tp
            self.metrics[f"{cls}_FN"] = fn

        return self.metrics
    