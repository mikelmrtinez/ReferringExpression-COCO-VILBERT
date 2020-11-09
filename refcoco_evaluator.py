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

from easydict import EasyDict as edict
from pytorch_transformers.tokenization_bert import BertTokenizer
from vilbert.datasets import ConceptCapLoaderTrain, ConceptCapLoaderVal
from vilbert.vilbert import VILBertForVLTasks, BertConfig, BertForMultiModalPreTraining
from vilbert.task_utils import LoadDatasetEval

from tools.refer.refer import REFER
import sys
import os.path as osp
import json
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

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
            img_tensor.append(im)
            im_scales.append(im_scale)
            im_infos.append(im_info)
            im_bbox.append(image_path)
        print("   Images transformations done")
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
        print("   Getting batch proposals done")
  
        
        with torch.no_grad():
            output = self.detection_model(current_img_list, proposals=proposals)

        feat_list = self._process_feature_extraction(
            output,
            im_scales,
            im_infos,
            self.args.feature_name,
            self.args.confidence_threshold,
        )
        print("   Features batch extracted!")
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
        extraction = []
        files = np.load(self.args.imdb_gt_file, allow_pickle=True)
        cnt = 1
        for chunk in self._chunks(files, self.args.batch_size):
            print('##############   CNT : ', cnt)
            cnt += 1
            print('Getting Batch features...')
            # print(chunk)
            features, infos = self.get_detectron_features(chunk)
            extraction.append((features, infos))
            print('Getting Batch features done!')
            for idx, c in enumerate(chunk):
                self._save_feature(c["file_name"], features[idx], infos[idx])
        return extraction
#Metric
def computeIoU(box1, box2):
    # each box is of [x1, y1, w, h]
    inter_x1 = max(box1[0], box2[0])
    inter_y1 = max(box1[1], box2[1])
    inter_x2 = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
    inter_y2 = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

    if inter_x1 < inter_x2 and inter_y1 < inter_y2:
        inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
    else:
        inter = 0
    union = box1[2]*box1[3] + box2[2]*box2[3] - inter
    return float(inter)/union
def score(IoU, threshold):
    pre_scores = np.where(IoU>threshold, 1, 0)
    return pre_scores.mean()


def prediction_refering_expression(question, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, task_tokens, model, infos ):

    vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, attn_data_list = model(
        question, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, task_tokens, output_all_attention_masks=True
    )
    
    width, height = float(infos[0]['image_width']), float(infos[0]['image_height'])

    # grounding: 
    logits_vision = torch.max(vision_logit, 1)[1].data
    grounding_val, grounding_idx = torch.sort(vision_logit.view(-1), 0, True)    
        
    print("Spatials ---> ", spatials)
    # for whole batch to do!
    top_idx = grounding_idx[0]
    print('top_idx: ',top_idx)
    top_box = spatials[0][top_idx][:4].tolist() 
    y1 = int(top_box[1] * height)
    y2 = int(top_box[3] * height)
    x1 = int(top_box[0] * width)
    x2 = int(top_box[2] * width)
    
    predicted_bboxes = [x1, y1, x2, y2]
    return predicted_bboxes

def untokenize_batch(batch):
    return [tokenizer.convert_ids_to_tokens(sent) for sent in batch]


def custom_prediction(query, task, features, infos, tokenizer, model):
    print(query)
    tokens = tokenizer.encode(query)
    print(tokens)
    tokens = tokenizer.add_special_tokens_single_sentence(tokens)
    print(tokens)
    print(untokenize_batch(tokens))
    segment_ids = [0] * len(tokens)
    input_mask = [1] * len(tokens)

    max_length = 37
    if len(tokens) < max_length:
        # Note here we pad in front of the sentence
        padding = [0] * (max_length - len(tokens))
        tokens = tokens + padding
        input_mask += padding
        segment_ids += padding

    text = torch.from_numpy(np.array(tokens)).cuda().unsqueeze(0)
    input_mask = torch.from_numpy(np.array(input_mask)).cuda().unsqueeze(0)
    segment_ids = torch.from_numpy(np.array(segment_ids)).cuda().unsqueeze(0)
    task = torch.from_numpy(np.array(task)).cuda().unsqueeze(0)

    num_image = len(infos)

    feature_list = []
    image_location_list = []
    image_mask_list = []
    for i in range(num_image):
        image_w = infos[i]['image_width']
        image_h = infos[i]['image_height']
        feature = features[i]
        num_boxes = feature.shape[0]

        g_feat = torch.sum(feature, dim=0) / num_boxes
        num_boxes = num_boxes + 1
        feature = torch.cat([g_feat.view(1,-1), feature], dim=0)
        boxes = infos[i]['bbox']
        image_location = np.zeros((boxes.shape[0], 5), dtype=np.float32)
        image_location[:,:4] = boxes
        image_location[:,4] = (image_location[:,3] - image_location[:,1]) * (image_location[:,2] - image_location[:,0]) / (float(image_w) * float(image_h))
        image_location[:,0] = image_location[:,0] / float(image_w)
        image_location[:,1] = image_location[:,1] / float(image_h)
        image_location[:,2] = image_location[:,2] / float(image_w)
        image_location[:,3] = image_location[:,3] / float(image_h)
        g_location = np.array([0,0,1,1,1])
        image_location = np.concatenate([np.expand_dims(g_location, axis=0), image_location], axis=0)
        image_mask = [1] * (int(num_boxes))

        feature_list.append(feature)
        image_location_list.append(torch.tensor(image_location))
        image_mask_list.append(torch.tensor(image_mask))

    features = torch.stack(feature_list, dim=0).float().cuda()
    spatials = torch.stack(image_location_list, dim=0).float().cuda()
    image_mask = torch.stack(image_mask_list, dim=0).byte().cuda()
    co_attention_mask = torch.zeros((num_image, num_boxes, max_length)).cuda()

    pred_bboxes = prediction_refering_expression(text, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, task, model, infos)
    return pred_bboxes

if __name__ == "__main__":


    # =============================
    # Feature Extractor GT with FasterCNN part
    # =============================

    # feature_extractor = FeatureExtractor()
    # extraction = feature_extractor.extract_features()


    # =============================
    # ViLBERT part
    # =============================

    args = SimpleNamespace(from_pretrained= 'save/multi_task_model.bin',#"save/refcoco_bert_base_6layer_6conect-finetune_from_multi_task_model_refcoco/pytorch_model_19.bin",
                        bert_model="bert-base-uncased",
                        config_file="config/bert_base_6layer_6conect.json",
                        max_seq_length=101,
                        train_batch_size=1,
                        do_lower_case=True,
                        predict_feature=False,
                        seed=42,
                        num_workers=0,
                        baseline=False,
                        img_weight=1,
                        distributed=False,
                        objective=1,
                        visual_target=0,
                        dynamic_attention=False,
                        task_specific_tokens=True,
                        tasks='1',
                        save_name='',
                        in_memory=False,
                        batch_size=1,
                        local_rank=-1,
                        split='mteval',
                        clean_train_sets=True
                        )

    config = BertConfig.from_json_file(args.config_file)
    with open('./vilbert_tasks.yml', 'r') as f:
        task_cfg = edict(yaml.safe_load(f))

    task_names = []
    for i, task_id in enumerate(args.tasks.split('-')):
        task = 'TASK' + task_id
        name = task_cfg[task]['name']
        task_names.append(name)

    timeStamp = args.from_pretrained.split('/')[-1] + '-' + args.save_name
    config = BertConfig.from_json_file(args.config_file)
    default_gpu=True

    if args.predict_feature:
        config.v_target_size = 2048
        config.predict_feature = True
    else:
        config.v_target_size = 1601
        config.predict_feature = False

    if args.task_specific_tokens:
        config.task_specific_tokens = True    

    if args.dynamic_attention:
        config.dynamic_attention = True

    config.visualization = True
    num_labels = 3129

    if args.baseline:
        model = BaseBertForVLTasks.from_pretrained(
            args.from_pretrained, config=config, num_labels=num_labels, default_gpu=default_gpu
            )
    else:
        model = VILBertForVLTasks.from_pretrained(
            args.from_pretrained, config=config, num_labels=num_labels, default_gpu=default_gpu
            )
        
    model.eval()
    cuda = torch.cuda.is_available()
    if cuda: model = model.cuda(0)
    tokenizer = BertTokenizer.from_pretrained(
        args.bert_model, do_lower_case=args.do_lower_case
    )

    
    data_root = './data'  # contains refclef, refcoco, refcoco+, refcocog and images
    dataset = 'refcoco'
    splitBy = 'unc'
    feat_root = "./data/feat_gt"
    refer = REFER(data_root, dataset, splitBy)
    ref_ids = refer.getRefIds(split='test')

    cnt=0
    scores = []

    for ref_id in ref_ids:
        if cnt >= 10:
            break
        cnt += 1
        #Load img id
        img_id = refer.getImgIds(ref_id)
        #Load info of image COCO
        img = refer.loadImgs(img_id)[0]
        #Load feat gt bbox of image
        feat_gt = np.load(os.path.join(feat_root, str(img['file_name']).split(".")[0]+ '.npy'), allow_pickle=True).reshape(-1,1)[0][0]
        features = [torch.from_numpy(feat_gt['features'])]
        infos = copy.deepcopy(feat_gt)
        infos.pop('features')
        infos.pop('image_id')
        infos = [infos]
        #get the refCOCO references of image
        ref = refer.Refs[ref_id]
        ref_bbox = refer.getRefBox(ref['ref_id'])
        task = [9]
    
        for indx, sentence in enumerate (ref['sentences']):
            query = sentence['sent']
            pred_bboxes = custom_prediction(query, task, features, infos, tokenizer, model)
            print("Predicted BBOX: ", pred_bboxes)
            print("refCOCO BBOX: ", ref_bbox)
            plt.figure()
            refer.showRef(ref, seg_box='box')
            ax = plt.gca()
            box_plot = Rectangle((pred_bboxes[0], pred_bboxes[1]), pred_bboxes[2], pred_bboxes[3], fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(box_plot)
            plt.show()
            scores.append(computeIoU(pred_bboxes,ref_bbox))
    print("Score = ", score(np.array(scores), 0.5))
        





 
        
        
        
    

