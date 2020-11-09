import sys
import os
import torch
import yaml

from easydict import EasyDict as edict
from pytorch_transformers.tokenization_bert import BertTokenizer
from vilbert.datasets import ConceptCapLoaderTrain, ConceptCapLoaderVal
from vilbert.vilbert import VILBertForVLTasks, BertConfig, BertForMultiModalPreTraining
from vilbert.task_utils import LoadDatasetEval

import numpy as np
import matplotlib.pyplot as plt
import PIL

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.layers import nms
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.utils.model_serialization import load_state_dict
from PIL import Image
import cv2
import argparse
import glob
from types import SimpleNamespace
import pdb
import _pickle as cPickle

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
            "--model_file", default=None, type=str, help="Detectron model file"
        )
        parser.add_argument(
            "--config_file", default=None, type=str, help="Detectron config file"
        )
        parser.add_argument(
            "--imdb_gt_file",
            default=None,
            type=str,
            help="Imdb file containing file path and bboxes.",
        )
        parser.add_argument("--batch_size", type=int, default=2, help="Batch size")
        parser.add_argument(
            "--num_features",
            type=int,
            default=100,
            help="Number of features to extract.",
        )
        parser.add_argument(
            "--output_folder", type=str, default="./output", help="Output folder"
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
            "--partition", type=int, default=0, help="Partition to download."
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

            feat_list.append(feats[i])
            num_boxes = len(feats[i])
            bbox = output[0]["proposals"][i]
            bbox = bbox.resize(((im_infos[i]["width"], im_infos[i]["height"])))
            bbox = bbox.bbox
            # Predict the class label using the scores
            objects = torch.argmax(scores[:, start_index:], dim=1)

            info_list.append(
                {
                    "bbox": bbox.cpu().numpy(),
                    "num_boxes": num_boxes,
                    "objects": objects.cpu().numpy(),
                    "image_width": im_infos[i]["width"],
                    "image_height": im_infos[i]["height"],
                    "cls_prob": scores.cpu().numpy(),
                }
            )

        return feat_list, info_list

    def get_detectron_features(self, image_paths):
        img_tensor, im_scales, im_infos, im_bbox = [], [], [], []

        for image_path in image_paths:
            #print('Image Path ' ,image_path)
            # print("image transformations...")
            im, im_scale, im_info = self._image_transform(image_path["file_path"])
            print("image transformations done")
            img_tensor.append(im)
            im_scales.append(im_scale)
            im_infos.append(im_info)
            im_bbox.append(image_path)

        # Image dimensions should be divisible by 32, to allow convolutions
        # in detector to work
        current_img_list = to_image_list(img_tensor, size_divisible=32)
        current_img_list = current_img_list.to("cuda")
        # print("Infos: curr image, im_scale, img_infos, image_path_bbox: \n",current_img_list, im_scales, im_infos, im_bbox )
        # print("Getting batch proposals...")
        # print("Infos: curr image, im_scale, img_infos, image_path_bbox: \n",current_img_list, im_scales, im_infos, image_paths['bbox'] )
        proposals = self.get_batch_proposals(
            current_img_list, im_scales, im_infos, im_bbox
        )
        print("Getting batch proposals done")
  
        
        with torch.no_grad():
            output = self.detection_model(current_img_list, proposals=proposals)

        feat_list = self._process_feature_extraction(
            output,
            im_scales,
            im_infos,
            self.args.feature_name,
            self.args.confidence_threshold,
        )
        print("Features extracted!")
        return feat_list

    def _chunks(self, array, chunk_size):
        for i in range(0, len(array), chunk_size):
            yield array[i : i + chunk_size]

    def _save_feature(self, file_name, feature, info):
        file_base_name = str(file_name).split(".")[0]
        info["image_id"] = file_base_name
        info["features"] = feature.cpu().numpy()
        file_base_name = str(file_base_name) + ".npy"

        np.save(os.path.join(self.args.output_folder, file_base_name), info)
        print("Saved in: "+os.path.join(self.args.output_folder, file_base_name))

    def extract_features(self):
        files = np.load(self.args.imdb_gt_file, allow_pickle=True)
        extracted_features = []
        # files = sorted(files)
        # files = [files[i: i+1000] for i in range(0, len(files), 1000)][self.args.partition]
        cnt = 1
        for chunk in self._chunks(files, self.args.batch_size):
            try:
                print('##############   CNT : ', cnt)
                cnt += 1
                print('Getting features...')
                # print(chunk)
                features, infos = self.get_detectron_features(chunk)
                extracted_features.append((features, infos))
                print('Getting batch features done!')
            except BaseException:
                continue
        np.save('gt_feat.npy', extracted_features)
        return extracted_features

def tokenize_batch(batch):
    return [tokenizer.convert_tokens_to_ids(sent) for sent in batch]

def untokenize_batch(batch):
    return [tokenizer.convert_ids_to_tokens(sent) for sent in batch]

def detokenize(sent):
    """ Roughly detokenizes (mainly undoes wordpiece) """
    new_sent = []
    for i, tok in enumerate(sent):
        if tok.startswith("##"):
            new_sent[len(new_sent) - 1] = new_sent[len(new_sent) - 1] + tok[2:]
        else:
            new_sent.append(tok)
    return new_sent

def printer(sent, should_detokenize=True):
    if should_detokenize:
        sent = detokenize(sent)[1:-1]
    print(" ".join(sent))

def show_boxes2(img_path, boxes, colors, texts=None, masks=None):
    # boxes [[xyxy]]
    plt.imshow(img)
    ax = plt.gca()
    print('boxes: ',boxes)
    for k in range(boxes.shape[0]):
        box = boxes[k]
        xmin, ymin, xmax, ymax = list(box)
        coords = (xmin, ymin), xmax - xmin + 1, ymax - ymin + 1
        color = colors[k]
        ax.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
        if texts is not None:
            ax.text(xmin, ymin, texts[k], bbox={'facecolor':'blue', 'alpha':0.5},fontsize=8, color='white')

            
# write arbitary string for given sentense. 
def plot_attention_maps(attn_maps, x_labels, y_labels, title, out_file, type):
    # create a 1920 x 1080 pixel image
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(19.2, 10.8))

    attn_head_idx = 0
    for row in range(0, 2):
        for col in range(0, 4):
            ax[row][col].imshow(attn_maps[attn_head_idx])

            ax[row][col].set_xticks(np.arange(len(x_labels)))
            ax[row][col].set_xticklabels(x_labels)
            if row == 0:
                ax[row][col].xaxis.tick_top()
                plt.setp(ax[row][col].get_xticklabels(), rotation=90, ha="left", rotation_mode="anchor")
            else:
                plt.setp(ax[row][col].get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor")

            # show y ticks only on left column
            if col == 0:
                ax[row][col].set_yticks(np.arange(len(y_labels)))
                ax[row][col].set_yticklabels(y_labels)
            else:
                ax[row][col].set_yticks([])
                ax[row][col].set_yticklabels([])

            attn_head_idx += 1

    fig.tight_layout()
    plt.text(24.25, 0, title, size=18, verticalalignment='center', rotation=270)

    # move vision on text attention maps more to the top and text on vision attention maps to the bottom such that
    # larger words fit into the visualization
    if type == 'vis':
        plt.subplots_adjust(left=0.1, right=0.98, top=1.0)
    else:
        plt.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.0)

    plt.savefig(out_file)
