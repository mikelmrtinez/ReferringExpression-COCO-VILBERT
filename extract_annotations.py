
""" This script extracts into a .npy file the annotations of the COCO dataset for the desired
    refCOCO, refCOCO+, refCOCOg or refCLEF clean split's for Train, Test or Validation.
    You should execute this script to run the feature_extractor script to extract from a fasteRCNN
    model the fetures which are the visual inputs of ViLBERT.
    
        Arguments:
            + data_root --> path to data folder structured as:  data |---images | mscoco | images | train2014
                                                                     |
                                                                     |---refcoco | annotations (Coco dataset)
                                                                                 | instances.json
                                                                                 | refs(google).p
                                                                                 | refs(unc).p
            + output_folder --> Path to store the .npy file with annotations
            + dataset --> refcoco, refcoco+ ...
            + splitBy --> type of split, unc, google,...
            + desired_split --> Test, Train, Validation
            + visualize --> True to visualize how are the bboxes stored on an example image
                                                                  
"""
import os
import argparse
from tools.refer.refer import REFER
import numpy as np
import matplotlib.pyplot as plt
import PIL
import torch
from matplotlib.patches import Rectangle
from tqdm import tqdm



def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_root",
        default="./data",
        type=str,
        help="Data set root.",
    )
    parser.add_argument(
        "--output_folder",
        default="./data/features/",
        type=str,
        help="Path to folder where you want to store the data annotations of COCO",
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
        help="The desired split to store, Test, Train or Val.",
    )
    parser.add_argument(
        "--visualize",
        default=False,
        type=bool,
        help="True to visualize how the bboxes have been stored in output folder.",
    )
    
    args = parser.parse_args()

    data_root = args.data_root
    dataset = args.dataset
    splitBy = args.splitBy

    refer = REFER(data_root, dataset, splitBy)
    print('dataset [%s_%s] contains: ' % (dataset, splitBy))
    ref_ids = refer.getRefIds()
    image_ids = refer.getImgIds()
    print('\nAmong them:')
    if dataset == 'refclef':
        if splitBy == 'unc':
            splits = ['train', 'val', 'testA', 'testB', 'testC']
        else:
            splits = ['train', 'val', 'test']
    elif dataset == 'refcoco':
        splits = ['train', 'val', 'test']
    elif dataset == 'refcoco+':
        splits = ['train', 'val', 'test']
    elif dataset == 'refcocog':
        splits = ['train', 'val']  # we don't have test split for refcocog right now.

    print('%s expressions for %s refs in %s images.' % (len(refer.Sents), len(ref_ids), len(image_ids)))
    for split in splits:
        ref_ids = refer.getRefIds(split=split)
        print('%s refs are in split [%s].' % (len(ref_ids), split))
    
    print("\nApplied split: ", args.desired_split)
    print()

    # Store all ref ids of desired split in variable ref_ids
    ref_ids = refer.getRefIds(split=args.desired_split)
    # Loop through all references and create data
    stock = []
    proccesed_images = []
    for i in tqdm(range(len(ref_ids))):
        ref_id = ref_ids[i]
        #Load img id
        img_id = refer.getImgIds(ref_id)
        #Load info of image COCO
        img = refer.loadImgs(img_id)[0]
        if img['file_name'] not in proccesed_images:
            #print(img)
            proccesed_images.append(img['file_name'])
            #get the coco annotations' id of image
            ann_ids = refer.getAnnIds(img['id'])
            #loop through all annotations per image to get the GT bbox
            bboxes = []
            for i in range(len(ann_ids)):
                ann = refer.loadAnns(ann_ids[i])[0]
                # For a proper forward giving the proposals (FasteRCNN), the bboxes need 
                # to have the shape [x1, y1, x2, y2]. The COCO annotations provide the
                # the bboxes as [x1, y1, width, heigth]
                bboxes.append([ann['bbox'][0], ann['bbox'][1], ann['bbox'][0]+ann['bbox'][2], ann['bbox'][1]+ann['bbox'][3]])
            info = {}
            info['file_name'] = img['file_name']
            info['file_path'] = refer.IMAGE_DIR+'/'+img['file_name']
            info['bbox'] = np.array(bboxes)
            info['num_box'] = len(ann_ids)
            stock.append(info)
            

    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    np.save(args.output_folder+args.dataset, stock, allow_pickle=True)

    print('\nSAVED INFOS in '+os.path.join(args.output_folder+args.dataset))

    if args.visualize==True:
        data = np.load(args.output_folder+args.dataset+'.npy',  allow_pickle=True).reshape(-1,1)[0][0]
        image_path = data["file_path"]
        img = PIL.Image.open(image_path).convert('RGB')
        img = torch.tensor(np.array(img))
        plt.axis('off')
        plt.imshow(img)
        for i in range(data['bbox'].shape[0]):
            bbox = data['bbox'][i]
            ax = plt.gca()
            box_plot = Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(box_plot)
        plt.show()


if __name__ == "__main__":

    main()
    
    

