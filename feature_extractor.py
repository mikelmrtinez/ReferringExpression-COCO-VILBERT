""" This script stores as .npy files the extracted features by a FasteRCNN model of the COCO dataset images.
    You can decide to use the proposed bboxes by the FasteRCNN or your own bboxes such as the GT bboxes of COCOdataset.
    This features will the be visual input of ViLBERT.
"""


import argparse
import os

import cv2
import numpy as np
import torch
from PIL import Image
import copy
from types import SimpleNamespace
import yaml

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict


from tools.refer.refer import REFER
import sys
import os.path as osp
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm

class FeatureExtractor:
    MAX_SIZE = 1333
    MIN_SIZE = 800

    def __init__(self):
        self.args = self.get_parser().parse_args()
        self.detection_model = self._build_detection_model()

        os.makedirs(self.args.output_folder, exist_ok=True)

    def get_parser(self):

        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--model_file", 
            default="./models/detectron/detectron_model.pth", 
            type=str, 
            help="Detectron model file"
        )
        parser.add_argument(
            "--config_file", 
            default="./models/detectron/detectron_config.yaml", 
            type=str, 
            help="Detectron config file"
        )
        parser.add_argument(
            "--npy_gt_file",
            default="./data/features/refcoco.npy",
            type=str,
            help="npy file containing file path and bboxes.",
        )
        parser.add_argument(
            "--batch_size", 
             type=int, 
            default=1, 
            help="Batch size"
        )

        parser.add_argument(
            "--num_features", 
            type=int, 
            default=100, 
            help="Number of features to extract."
        )
        parser.add_argument(
            "--output_folder", 
            type=str, 
            default="./data/features/fastercnn_proposals",  #If proposed bboxes from fasterCNN desired "./data/features/fastercnn_proposals"
            help="Output folder. You should select coco_gt or fastercnn_proposals based on --coco_proposals "
        )
        parser.add_argument(
            "--coco_proposals", 
            type=bool, 
            default=False, 
            help="Partition to download."
        )
        parser.add_argument(
            "--feature_name",
            type=str,
            help="The name of the feature to extract",
            default="fc6",
        )
        parser.add_argument(
            "--confidence_threshold",
            type=float,
            default=0,
            help="Threshold of detection confidence above which boxes will be selected",
        )
        parser.add_argument(
            "--background",
            action="store_true",
            help="The model will output predictions for the background class when set",
        )
        parser.add_argument(
            "--partition", 
            type=int, 
            default=0, 
            help="Partition to download."
        )
        
        return parser

    def _build_detection_model(self):
        cfg.merge_from_file(self.args.config_file)
        cfg.freeze()

        model = build_detection_model(cfg)
        checkpoint = torch.load(self.args.model_file, map_location=torch.device("cpu"))

        load_state_dict(model, checkpoint.pop("model"))

        model.to("cuda")
        model.eval()
        return model

    def get_batch_proposals(self, images, im_scales, im_infos, proposals):
        proposals_batch = []
        
        for idx, img_info in enumerate(im_infos):
            boxes_tensor = torch.from_numpy(
                proposals[idx]["bbox"][: int(proposals[idx]["num_box"]), 0:]
            ).to("cuda")
            
            orig_image_size = (img_info["width"], img_info["height"])
            boxes = BoxList(boxes_tensor, orig_image_size)
            image_size = (images.image_sizes[idx][1], images.image_sizes[idx][0])
            boxes = boxes.resize(image_size)
            proposals_batch.append(boxes)
        return proposals_batch

    def _image_transform(self, path):
        img = Image.open(path)
        im = np.array(img).astype(np.float32)
        # IndexError: too many indices for array, grayscale images
        if len(im.shape) < 3:
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
        im = im[:, :, ::-1]
        im -= np.array([102.9801, 115.9465, 122.7717])
        im_shape = im.shape
        im_height = im_shape[0]
        im_width = im_shape[1]
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        # Scale based on minimum size
        im_scale = self.MIN_SIZE / im_size_min

        # Prevent the biggest axis from being more than max_size
        # If bigger, scale it down
        if np.round(im_scale * im_size_max) > self.MAX_SIZE:
            im_scale = self.MAX_SIZE / im_size_max

        im = cv2.resize(
            im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
        )
        img = torch.from_numpy(im).permute(2, 0, 1)

        im_info = {"width": im_width, "height": im_height}

        return img, im_scale, im_info

    def _process_feature_extraction(
        self, output, im_scales, im_infos, feature_name="fc6", conf_thresh=0
    ):
        batch_size = len(output[0]["proposals"])
        n_boxes_per_image = [len(boxes) for boxes in output[0]["proposals"]]
        score_list = output[0]["scores"].split(n_boxes_per_image)
        score_list = [torch.nn.functional.softmax(x, -1) for x in score_list]
        feats = output[0][feature_name].split(n_boxes_per_image)
        cur_device = score_list[0].device

        feat_list = []
        info_list = []

        for i in range(batch_size):
            dets = output[0]["proposals"][i].bbox / im_scales[i]
            scores = score_list[i]
            max_conf = torch.zeros((scores.shape[0])).to(cur_device)
            conf_thresh_tensor = torch.full_like(max_conf, conf_thresh)
            start_index = 1
            # Column 0 of the scores matrix is for the background class
            if self.args.background:
                start_index = 0
            for cls_ind in range(start_index, scores.shape[1]):
                cls_scores = scores[:, cls_ind]
                keep = nms(dets, cls_scores, 0.5)
                max_conf[keep] = torch.where(
                    # Better than max one till now and minimally greater than conf_thresh
                    (cls_scores[keep] > max_conf[keep])
                    & (cls_scores[keep] > conf_thresh_tensor[keep]),
                    cls_scores[keep],
                    max_conf[keep],
                )
            sorted_scores, sorted_indices = torch.sort(max_conf, descending=True)
            num_boxes = (sorted_scores[: self.args.num_features] != 0).sum()
            keep_boxes = sorted_indices[: self.args.num_features]
            feat_list.append(feats[i][keep_boxes])
            bbox = output[0]["proposals"][i][keep_boxes].resize(((im_infos[i]["width"], im_infos[i]["height"])))
            bbox = bbox.bbox
          
            # Predict the class label using the scores
            objects = torch.argmax(scores[keep_boxes][start_index:], dim=1)
            cls_prob = torch.max(scores[keep_boxes][start_index:], dim=1)[0]


            info_list.append(
                {
                    "bbox": bbox.cpu().numpy(),
                    "num_boxes": num_boxes,
                    "objects": objects.cpu().numpy(),
                    "image_width": im_infos[i]["width"],
                    "image_height": im_infos[i]["height"],
                    "cls_prob": cls_prob.cpu().numpy(),
                }
            )

        return feat_list, info_list

    def get_detectron_features(self, image_paths):
        img_tensor, im_scales, im_infos, im_bbox = [], [], [], []

        for image_path in image_paths:
            im, im_scale, im_info = self._image_transform(image_path["file_path"])
            img_tensor.append(im)
            im_scales.append(im_scale)
            im_infos.append(im_info)
            im_bbox.append(image_path)
        #print("   Images transformations done")
        # Image dimensions should be divisible by 32, to allow convolutions
        # in detector to work
        current_img_list = to_image_list(img_tensor, size_divisible=32)
        current_img_list = current_img_list.to("cuda")
        #import pdb;pdb.set_trace()

        if self.args.coco_proposals==True:

            proposals = self.get_batch_proposals(
                current_img_list, im_scales, im_infos, im_bbox
            )
            #print("   Getting batch proposals done")
            torch.cuda.empty_cache()
            with torch.no_grad():
                output = self.detection_model(current_img_list, proposals=proposals)
        else:
            torch.cuda.empty_cache()
            with torch.no_grad():
                output = self.detection_model(current_img_list)


        feat_list, info_list = self._process_feature_extraction(
            output,
            im_scales,
            im_infos,
            self.args.feature_name,
            self.args.confidence_threshold,
        )

        return feat_list, info_list

    def _chunks(self, array, chunk_size):
        for i in tqdm(range(0, len(array), chunk_size)):
            yield array[i : i + chunk_size]

    def _save_feature(self, file_name, feature, info):
        file_base_name = str(file_name).split(".")[0]
        info["image_id"] = file_base_name
        info["features"] = feature.cpu().numpy()
        file_base_name = str(file_base_name) + ".npy"

        np.save(os.path.join(self.args.output_folder, file_base_name), info)

    def extract_features(self):
        extraction = []
        files = np.load(self.args.npy_gt_file, allow_pickle=True)

        
        for chunk in self._chunks(files, self.args.batch_size):
            #print('Getting Batch features...')
            features, infos = self.get_detectron_features(chunk)
            extraction.append((features, infos))
            #print('Getting Batch features done!')
            for idx, c in enumerate(chunk):
                self._save_feature(c["file_name"], features[idx], infos[idx])
        print("\nSaved in: "+self.args.output_folder)

        return extraction


if __name__ == "__main__":


    # =============================
    # Feature Extractor GT with FasterCNN part
    # =============================

    feature_extractor = FeatureExtractor()
    extraction = feature_extractor.extract_features()
    print('\nDone!')


    