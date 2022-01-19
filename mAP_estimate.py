import torch
from torch.utils.data import DataLoader

import cv2
import numpy as np
from torchvision import transforms
from YOLO_V1_DataSet import YoloV1DataSet
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt

from nms import generate_q_sigmoid, sigmoid_lut, post_process, NMS_max, torch_post_process, torch_NMS_max
from sigmoid import generate_q_sigmoid, sigmoid_lut, q17p14, q_mul, q_div

import importlib
mod = importlib.import_module("yolov1_bn_model_noaffine")

import sys, os
sys.path.append("../../../") # go to the directory of ai8x
import ai8x

from map import calculate_map_main, NMS, gt_std
from YOLO_V1_DataSet import YoloV1DataSet

ai8x.set_device(85, simulate=False, round_avg=False, verbose=True)


dataSet = YoloV1DataSet(imgs_dir="../YOLO_V1_GPU/VOC2007/Test/JPEGImages",
                            annotations_dir="../YOLO_V1_GPU/VOC2007/Test/Annotations",
                            ClassesFile="../YOLO_V1_GPU/VOC2007/Test/VOC_remain_class_V2.data",
                            train_root="../YOLO_V1_GPU/VOC2007/Test/ImageSets/Main/",
                            img_per_class=None)

dataLoader = DataLoader(dataSet, batch_size=256, shuffle=True, num_workers=4)

Yolo = mod.Yolov1_net(num_classes=dataSet.Classes, bias=True)
qat_policy = {'start_epoch':150,
              'weight_bits':8,
              'bias_bits':8,
              'shift_quantile': 0.99}

ai8x.fuse_bn_layers(Yolo)
ai8x.initiate_qat(Yolo, qat_policy)
checkpoint_fname = './log/QAT-20210924-175040/scaled224_noaffine_shift0.99_maxim_yolo_qat_ep400.pth'   # batch_size 16 full training set
Yolo.load_state_dict(torch.load(checkpoint_fname, map_location=lambda storage, loc: storage))


pred_results = []
gt_results = []

with torch.no_grad():

    for batch_index, batch_test in enumerate(dataLoader):

        data = batch_test[0].float()
        label_data = batch_test[1].float()
        
        # print(data.shape)
        bb_pred, _ = Yolo(data)
        # bb_pred[:, :, :, 0:4] = bb_pred[:, :, :, 0:4]*224
        # bb_pred[:, :, :, 5:9] = bb_pred[:, :, :, 5:9]*224
        pred_results.append(bb_pred)
        gt_results.append(label_data)
        break

gt_results = torch.cat(gt_results)
gt_results = gt_results.squeeze(dim=3)
gt_results_std = gt_std(gt_results, S=7, B=2, img_size=224)


pred_results = torch.cat(pred_results) # N x 7x7 x 20
bounding_box_pred = NMS(pred_results, img_size=224, confidence_threshold=0.1) # , iou_threshold=0.)
bounding_box_pred = [[[y[0], y[1], y[2], y[3], y[5], y[4]] for y in z] for z in bounding_box_pred]

print(len(bounding_box_pred))
print(len(gt_results_std))

# calculate_map_main(gt_results, bounding_box_pred, iou_gt_thr=0.01, class_num=5)


