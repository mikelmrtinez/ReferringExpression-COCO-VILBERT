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
from tqdm import tqdm

#Metric
def computeIoU(box1_list, box2_list):
    # each box is of [x1, y1, w, h]
    iou = []
    for i in range(len(box1_list)):
        box1, box2 = box1_list[i], box2_list[i]
        inter_x1 = max(box1[0], box2[0])
        inter_y1 = max(box1[1], box2[1])
        inter_x2 = min(box1[0]+box1[2]-1, box2[0]+box2[2]-1)
        inter_y2 = min(box1[1]+box1[3]-1, box2[1]+box2[3]-1)

        if inter_x1 < inter_x2 and inter_y1 < inter_y2:
            inter = (inter_x2-inter_x1+1)*(inter_y2-inter_y1+1)
        else:
            inter = 0
        union = box1[2]*box1[3] + box2[2]*box2[3] - inter
        iou_score = float(inter)/union
        iou.append(iou_score)
    
    return iou


def score(IoU, threshold):
    pre_scores = np.where(IoU>threshold, 1, 0)
    return pre_scores.mean()


def prediction_refering_expression(question, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, task_tokens, model, infos ):

    vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, attn_data_list = model(
        question, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, task_tokens, output_all_attention_masks=True
    )
    
    width, height = int(infos[0]['image_width']), int(infos[0]['image_height'])

    # grounding: 
    logits_vision = torch.max(vision_logit, 1)[1].data
    grounding_val, grounding_idx = torch.sort(vision_logit.squeeze(2), 1, True)    
    
     # for whole batch
    top_idx = grounding_idx[:,0].tolist()
    predicted_bboxes = []
    for indx, value in enumerate(top_idx):
        y1 = spatials[indx, value,1]*height
        # y1 = top_box[:,1] * height
        y2 = spatials[indx, value,3]*height
        #y2 = top_box[:,3] * height
        x1 = spatials[indx, value,0]*width
        #x1 = top_box[:,0] * width
        x2 = spatials[indx, value,2]*width
        #x2 = top_box[:,2] * width
        predicted_bboxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
    
    
    return predicted_bboxes

def untokenize_batch(batch):
    return [tokenizer.convert_ids_to_tokens(sent) for sent in batch]


def custom_prediction(query, task, features, infos, tokenizer, model):

    num_image = len(infos)

    feature_list = []
    image_location_list = []
    image_mask_list = []
    text_list = []
    input_mask_list = []
    segment_ids_list = []
    task_list = []

    for i in range(num_image):
        tokens = tokenizer.encode(query[i])
        tokens = tokenizer.add_special_tokens_single_sentence(tokens)
        segment_ids = [0] * len(tokens)
        input_mask = [1] * len(tokens)

        max_length = 37
        if len(tokens) < max_length:
            # Note here we pad in front of the sentence
            padding = [0] * (max_length - len(tokens))
            tokens = tokens + padding
            input_mask += padding
            segment_ids += padding

        text_list.append(torch.from_numpy(np.array(tokens)).cuda().unsqueeze(0))
        input_mask_list.append(torch.from_numpy(np.array(input_mask)).cuda().unsqueeze(0))
        segment_ids_list.append(torch.from_numpy(np.array(segment_ids)).cuda().unsqueeze(0))
        task_list.append(torch.from_numpy(np.array(task)).cuda().unsqueeze(0))

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

    text = torch.stack(text_list, dim=0).long().cuda().squeeze(1)
    input_mask = torch.stack(input_mask_list, dim=0).long().cuda().squeeze(1)
    segment_ids = torch.stack(segment_ids_list, dim=0).long().cuda().squeeze(1)
    task = torch.stack(task_list, dim=0).long().cuda().squeeze(1)

    pred_bboxes = prediction_refering_expression(text, features, spatials, segment_ids, input_mask, image_mask, co_attention_mask, task, model, infos)
    
    return pred_bboxes

def callVILBert():

    args = SimpleNamespace(from_pretrained= 'models/vilbert/multi_task_model.bin',#"save/refcoco_bert_base_6layer_6conect-finetune_from_multi_task_model_refcoco/pytorch_model_19.bin",
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

    return tokenizer, model

def xy_to_wh(bboxes):
    bbox_reshaped = []
    for bbox in bboxes:
        x1, y1, w, h = bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1]
        bbox_reshaped.append([x1, y1, w, h])
    
    return bbox_reshaped
def batch_loader(i, batch_size, refer, ref_ids):
    features_list, query_list, infos_list,  ref_bbox_list = [], [], [], []
    for j in range(i, (i+1)*batch_size):
        ref_id = ref_ids[j]
        img_id = refer.getImgIds(ref_id)
        #Load info of image COCO
        img = refer.loadImgs(img_id)[0]
        #Load features gt bbox of image and infos
        feat_gt = np.load(os.path.join(feat_root, str(img['file_name']).split(".")[0]+ '.npy'), allow_pickle=True).reshape(-1,1)[0][0]
        features = torch.from_numpy(feat_gt['features'])
        #Prepare infos removing the last two keys of dictionary loaded form gt_bboxes
        infos = copy.deepcopy(feat_gt)
        infos.pop('features')
        infos.pop('image_id')
        #get the refCOCO bboxes of image
        ref = refer.Refs[ref_id]
        ref_bbox = refer.getRefBox(ref['ref_id'])

        for indx, sentence in enumerate (ref['sentences']):
            query_list.append(sentence['sent'])
            features_list.append(features)
            infos_list.append(infos)
            ref_bbox_list.append(ref_bbox)

    return features_list, query_list, infos_list,  ref_bbox_list




if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_root",
        default="./data",
        type=str,
        help="Data set root.",
    )
    parser.add_argument(
        "--dataset",
        default="refcoco",
        type=str,
        help="dataset.",
    )
    parser.add_argument(
        "--splitBy",
        default="unc",
        type=str,
        help="Type of split, RefCOCO, COCO+, ...",
    )
    parser.add_argument(
        "--desired_split",
        default="test",
        type=str,
        help="The desired split to store, test, train or val.",
    )
    parser.add_argument(
        "--feat_root",
        default="./data/features/fastercnn_proposals/",
        type=str,
        help="Directory of the features .npy file",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--visualize", type=bool, default=False, help="Batch size")

    args_data = parser.parse_args()

    data_root = args_data.data_root  # contains refclef, refcoco, refcoco+, refcocog and images
    dataset = args_data.dataset
    splitBy = args_data.splitBy
    feat_root = args_data.feat_root

    refer = REFER(data_root, dataset, splitBy)
    ref_ids = refer.getRefIds(split=args_data.desired_split)

    tokenizer, model = callVILBert()
    ious = []

    for i in tqdm(range(0, len(ref_ids), args_data.batch_size)):
        features, query, infos,  ref_bboxes = batch_loader(i, args_data.batch_size, refer, ref_ids)
        task = [9]
        pred_bboxes = custom_prediction(query, task, features, infos, tokenizer, model)
        pred_bboxes = xy_to_wh(pred_bboxes)
        iou_batch = computeIoU(pred_bboxes, ref_bboxes)
        ious.append(iou_batch)

        if args_data.visualize==True:

            for j, pred_bbox in enumerate(pred_bboxes):
                plt.figure()
                gt_bboxes = infos[j]['bbox'].tolist()
                gt_bboxes = xy_to_wh(gt_bboxes)
                
                for k, bbox in enumerate(gt_bboxes):
                    if k==0:
                        ax = plt.gca()
                        box_plot = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, label='COCO GT bbox', edgecolor='blue', linewidth=0.5)
                        ax.add_patch(box_plot)
                    else:
                        ax = plt.gca()
                        box_plot = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='blue', linewidth=0.5)
                        ax.add_patch(box_plot)

                # draw box of the ann using 'red'
                ax = plt.gca()
                box_plot = Rectangle((pred_bbox[0], pred_bbox[1]), pred_bbox[2], pred_bbox[3], fill=False,ls='--', label='ViLBERT bbox', edgecolor='red', linewidth=2)
                ax.add_patch(box_plot)

                ref = refer.Refs[ref_ids[i]]
                refer.showRef(ref, seg_box='box')
                plt.legend()
                plt.show()

    print("\nFinal Score of Evaluation: {} % ".format(np.round(score(np.array(ious).flatten()), 2)*100))
        





 
        
        
        
    

