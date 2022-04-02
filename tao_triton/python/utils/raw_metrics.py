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
        self.yolox_data_dir = f'./VOC2012'
        self.image_dir = f"{self.yolox_data_dir}/JPEGImages"
        self.annotation_dir = f"{self.yolox_data_dir}/Annotations"
        self.testset_txt_path = f"{self.yolox_data_dir}/ImageSets/Main/test.txt"
        
        self.npos = self._get_total_pos()
        self.yolox = YOLOX()

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
        cnt = {
            "car": np.array([0, 0]), 
            "bus": np.array([0, 0]), 
            "hdv": np.array([0, 0]), 
            "truck": np.array([0, 0]), 
            "motorcycle":np.array([0, 0]), 
            "bicycle": np.array([0, 0]), 
            "personal_mobility":np.array([0, 0]), 
            "firetruck":np.array([0, 0]), 
            "police":np.array([0, 0]), 
            "ambulance":np.array([0, 0]), 
            "pedestrian":np.array([0, 0])
        }

        if len(pred_labels) == 0 or len(actual_labels) == 0:
            return cnt

        for i, pred_bbox in enumerate(pred_bboxes):
            for j, actual_bbox in enumerate(actual_bboxes):
                iou = self._calc_iou(pred_bbox, actual_bbox)

                if iou > iou_thr:
                    if pred_labels[i] == actual_labels[j]:
                        cnt[pred_labels[i]][1] += 1
                    else:
                        cnt[pred_labels[i]][0] += 1
                else:
                    continue

        return cnt


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


    def predict(self, img_path, model, score_thr, nms_thr):
        origin_img = cv2.imread(img_path)
        # print(origin_img.shape)

        input_shape = "544,960"
        input_shape = tuple(map(int, input_shape.split(',')))
        img, ratio = self.yolox.preprocess(origin_img, input_shape)
    
        model = model.eval()        
        output = model(torch.tensor(img).view(1, 3, input_shape[0], input_shape[1]).cuda()).detach()
        output = torch.cat([output[..., :4], output[..., 4:].sigmoid()], dim=-1)
        output = output.cpu().numpy()

        predictions = self.yolox.demo_postprocess(output, input_shape)[0]

        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2]/2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3]/2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2]/2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3]/2.
        boxes_xyxy /= ratio

        dets = self.yolox.multiclass_nms(boxes_xyxy, scores, nms_thr=nms_thr, score_thr=score_thr)

        if dets is None:
            return None
        else:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]

        label_names = ["car", "bus", "hdv", "truck", "motorcycle", "bicycle", "personal_mobility", "firetruck", "police", "ambulance", "pedestrian"]

        results = []
        for i in range(dets.shape[0]):
            results.append({
                "label": label_names[int(final_cls_inds[i])],
                "points": list(final_boxes[i]),
            })

        return results


    def get_raw_metrics(
        self, 
        model,
        nms_thr=0.45, 
        score_thr=0.4, 
        iou_thr=0.5
    ):
        with open(self.testset_txt_path) as f:
            lines = f.read().splitlines()


        cnts = {
            "car": np.array([0, 0]), 
            "bus": np.array([0, 0]), 
            "hdv": np.array([0, 0]), 
            "truck": np.array([0, 0]), 
            "motorcycle":np.array([0, 0]), 
            "bicycle": np.array([0, 0]), 
            "personal_mobility":np.array([0, 0]), 
            "firetruck":np.array([0, 0]), 
            "police":np.array([0, 0]), 
            "ambulance":np.array([0, 0]), 
            "pedestrian":np.array([0, 0])
        }

        for line in tqdm(lines):
            results = self.predict(
                img_path=f"{self.image_dir}/{line}.jpg", 
                model=model, 
                score_thr=score_thr, 
                nms_thr=nms_thr
            )
            if results is None:
                continue

            result = {"point": [], "label": []}
            for r in results:
                result["point"].append(r["points"])
                result["label"].append(r["label"])
                
            actual_labels, actual_bboxes = self._get_actual_bboxes(xml_path=f"{self.annotation_dir}/{line}.xml")
            
            cnt = self._count_res(
                pred_bboxes=result["point"], 
                pred_labels=result["label"], 
                actual_bboxes=actual_bboxes, 
                actual_labels=actual_labels,
                iou_thr=iou_thr,
            )

            for cls in cnt.keys():
                cnts[cls] += cnt[cls]
        
        metrics = {}
        for cls in cnt.keys():
            fp = cnts[cls][0]
            tp = cnts[cls][1]
            fn = self.npos[cls] - tp
            
            metrics[f"{cls}_FP"] = fp
            metrics[f"{cls}_TP"] = tp
            metrics[f"{cls}_FN"] = fn
            
        return metrics
    