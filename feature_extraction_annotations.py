import argparse
from tools.refer.refer import REFER
import numpy as np
import matplotlib.pyplot as plt
import PIL
import torch
from matplotlib.patches import Rectangle

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
        "--dataset",
        default="refcoco+",
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

    # Store all ref ids of desired split in variable ref_ids
    ref_ids = refer.getRefIds(split=args.desired_split)
    # Loop through all references and create data
    stock = []
    proccesed_images = []
    
    for ref_id in ref_ids:
      
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
                bboxes.append(ann['bbox'])
            info = {}
            info['file_name'] = img['file_name']
            info['file_path'] = refer.IMAGE_DIR+'/'+img['file_name']
            info['bbox'] = np.array(bboxes)
            info['num_box'] = len(ann_ids)
            stock.append(info)

    np.save('data/gt_data_'+args.dataset, stock, allow_pickle=True)

    print('SAVED INFOS!')

if __name__ == "__main__":
    
    main()
    # UNCOMENT TO CHECK IF IS CORRECTLY STORED THE BBOXEZS
    # data = np.load('gt_data.npy',  allow_pickle=True).reshape(-1,1)[3][0]
    # print(data)
    # image_path = data["file_path"]
    # img = PIL.Image.open(image_path).convert('RGB')
    # img = torch.tensor(np.array(img))
    # plt.axis('off')
    # plt.imshow(img)

    # for i in range(data['bbox'].shape[0]):
    #     bbox = data['bbox'][i]
    #     ax = plt.gca()
    #     box_plot = Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], fill=False, edgecolor='red', linewidth=2)
    #     ax.add_patch(box_plot)
    # plt.show()

