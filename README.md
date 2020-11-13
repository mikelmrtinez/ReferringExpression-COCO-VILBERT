# Referring Expression Tutorial with ViLBERT for COCO Data Set

This tutorial shows how to evaluate the different Referring Expressions splits for the Coco data set. This repository is forked from the [12-in-1: Multi-Task Vision and Language Representation Learning](https://github.com/facebookresearch/vilbert-multi-task) repository developed by Facebook research. 

Modifications have been applied for a better comprehension of our task. 

## Quicke-Start

With the ```demo.ipybn ``` you can test by yourself the results regarding referring expression of ViLBERT approach.

## Evaluate refCOCO, refCOCO+, refCOCOg, and refclef
To evaluate the whole dataset you simply have to run the following code lines from the root of the repository.
### Extract annotations from COCO dataset (refCOCO example)
```
python extract_annotations.py --dataset refcoco --splitBy unc --desired_split test --visualize True
```
### Extract Features from fasteRCNN 

If you want to use the bounding boxes provided by FasteRCNN run:
```
python feature_extractor.py 
```
If you want to use the boudning boxes provided by COCO dataset you should run:
```
python feature_extractor.py --coco_gt True --output_folder ./data/features/coco_gt
```
### Evaluate refCOCO

This script will print the accuracy acquired for the desired Referring Expression using ViLBERT and FasteRCNN as feature extractor.

If you want to evaluate with the bounding boxes provided by FasteRCNN run:
```
python evaluator.py --dataset refcoco --splitBy unc --desired_split test
```

If you want to evaluate with the bounding boxes provided by COCO dataset run:
```
python evaluator.py --dataset refcoco --splitBy unc --desired_split test --feat_root ./data/features/coco_gt/
```

## Repository Setup
The setup might be a bit tedious. Don't skip any of the steps and pay attention to the structures mentioned. Once the Repository setup is finished you will be able to run the tutorial in ```demo.ipybn ```.
### Setup Environmnets

1. Create a fresh conda environment and set variable INSTALL_DIR

```text
conda create -n vilbert-mt python=3.6
conda activate vilbert-mt
git clone --recursive https://github.com/facebookresearch/vilbert-multi-task.git

export INSTALL_DIR=$PWD
```

2. Install some required libraries
```
conda install ipython pip 

pip install ninja yacs cython matplotlib tqdm opencv-python
```
3. Install PyTorch
```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch
```
4. Install apex
```
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python setup.py install --cuda_ext --cpp_ext
```
5. Intall vqa-maskrcnn-benchmark
```
cd $INSTALL_DIR
git clone https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark.git
cd maskrcnn-benchmark
python setup.py build develop

```

6. Install ViLBERT
```
cd $INSTALL_DIR
cd vilbert-multi-task
pip install -r requirements.txt
python setup.py develop
```
7. Finish the installation of ViLBERT by running the ```Make file``` in the ```tools/refer/```.
```
cd tools/refer/
make
```

### Data Setup

It is very important that the data folder has the following structure
```
data |-- images |-- mscoco |-- images |-- train2014
     |
     |-- refcoco |-- annotations (Coco dataset)
                 |-- instances.json
                 |-- refs(google).p
                 |-- refs(unc).p
                                                                            
```

Download refCOCO:

  * i. http://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip

  * ii. http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip

  * iii. http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip

  * iv. http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip

Download COCO images .zip, unzip it and allocate it as follow in the scheme above:

* i. http://images.cocodataset.org/zips/train2014.zip

Download COCO annotations .zip, unzip it and allocate it as follow in the scheme above:

* i. http://images.cocodataset.org/annotations/annotations_trainval2014.zip

### Set Up the models for VilBERT and Feature Extractor

Structure of the models folder:
```
models |-- detectron |-- detectron_model.pth
       |             |-- detectron_config.yaml
       |
       |-- vilbert   |-- multi_task_model.bin
                                                                            
```
#### Feature Extractor
```
cd models/detectron
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_model.pth
wget https://dl.fbaipublicfiles.com/vilbert-multi-task/detectron_config.yaml
```

#### Visiolinguistic Pre-training and Multi Task Training


Download the ```multi_task_model.bin``` on the following [link](https://dl.fbaipublicfiles.com/vilbert-multi-task/multi_task_model.bin)


## ViLBERT 12-in-1: Multi-Task Vision and Language Representation Learning

Please cite the following if you use this code. Code and pre-trained models for [12-in-1: Multi-Task Vision and Language Representation Learning](http://openaccess.thecvf.com/content_CVPR_2020/html/Lu_12-in-1_Multi-Task_Vision_and_Language_Representation_Learning_CVPR_2020_paper.html):

```
@InProceedings{Lu_2020_CVPR,
author = {Lu, Jiasen and Goswami, Vedanuj and Rohrbach, Marcus and Parikh, Devi and Lee, Stefan},
title = {12-in-1: Multi-Task Vision and Language Representation Learning},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```

