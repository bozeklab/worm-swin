# Created by: Deserno, M. et al., 2022
# Calculates the DSB mAP for Instance Segmentation results from MMDetection

import itertools
import pickle
import scipy.optimize as opt

import numpy as np
import pycocotools.mask as mask
import cv2
from pycocotools.coco import COCO
import argparse
from alive_progress import alive_bar


# src: https://www.kaggle.com/competitions/data-science-bowl-2018/overview/evaluation
def dsb_precision(result_matrix, gt_len, pred_len, thresholds):
    precisions = np.zeros(len(thresholds))

    for thresh_id, thresh in enumerate(thresholds):
        # match objects for theshold
        gt_assignment, pred_assignment = opt.linear_sum_assignment(1 - result_matrix)

        tp_count = 0
        fp_count = 0
        fn_count = 0

        # calc TPs
        assert len(gt_assignment) == len(pred_assignment)
        for ass in range(len(gt_assignment)):
            fp_fn_found = False

            gt_ass = gt_assignment[ass]
            pred_ass = pred_assignment[ass]

            if gt_ass >= gt_len: # check if assignment with fake gt
                fp_count += 1
                fp_fn_found = True
            if pred_ass >= pred_len:  # check if assignment with fake gt
                fn_count += 1
                fp_fn_found = True

            if not fp_fn_found:
                res_iou = result_matrix[gt_ass, pred_ass]
                if res_iou > thresh:
                    tp_count += 1
                else:
                    fp_count += 1
                    fn_count += 1
            
        # calc precision for threshold
        precisions[thresh_id] = tp_count/(tp_count + fp_count + fn_count)
    
    return ((1/len(thresholds)) * precisions.sum(), precisions) # mAP, APs

# src: https://github.com/matterport/Mask_RCNN/issues/2440#issuecomment-788629008
def mask_iou(pred_mask, gt_mask):
    intersection = np.sum((pred_mask + gt_mask) > 1)
    union = np.sum((pred_mask + gt_mask) > 0)
    iou_score = intersection / float(union)

    return iou_score

def pack_array(array):
    packed_array = []
    for i in range(0, len(array), 2):
        packed_array.append(np.array([int(array[i]), int(array[i + 1])]))
    return np.array(packed_array)

# extract predictions from .pkl files generated by MMDetection
def extract_predictions(score_thres, data_path):
    # open result file
    data_pkl = open(data_path, 'rb')
    data = pickle.load(data_pkl)

    det_scores_per_image = {}
    det_masks_per_image = []
    det_boxes_per_image = []

    # iterate over detections for all images
    for img_i, img_data in enumerate(data):
    # index 1 of tuple -> t[0]: BBox + Score, t[1]: Size + RLE; 0 to because of surrounding list
        detection_scores = img_data[0][0]
        detection_masks = img_data[1][0]

    # store masks and score per image
        det_scores = []
        det_masks = []
        det_boxes = []
        for det_idx in range(len(detection_masks)):
            det_score = detection_scores[det_idx]

            if det_score[4] <= score_thres:
                continue

            det_scores.append(det_score[4])
            det_boxes.append(det_score[:4])

            roi_dict = detection_masks[det_idx]
            dec_mask = mask.decode(roi_dict)
            det_masks.append(dec_mask)

        det_scores_per_image[img_i] = det_scores
        det_masks_per_image.append(det_masks)
        det_boxes_per_image.append(det_boxes)
    data_pkl.close()
    
    return det_scores_per_image, det_masks_per_image, det_boxes_per_image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("results", help = "Path to results .pkl file")
    parser.add_argument("gts", help = "Path to COCO gt .json file")

    args = parser.parse_args()

    results_path = args.results
    gt_path = args.gts

    score_thres = 0.5  # confidence threshold
    iou_threshs = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]  # iou thresholds

    coco = COCO(gt_path)
    img_ids = coco.getImgIds()

    det_scores_per_image, det_masks_per_image, det_boxes_per_image = extract_predictions(score_thres, results_path)

    # iterate over images
    dataset_dsb_map_sum = 0
    dataset_dsb_ap_sums = np.zeros(len(iou_threshs))
    with alive_bar(len(det_masks_per_image)) as bar:
        for img_count in range(len(det_masks_per_image)):
            img_id = img_ids[img_count]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)

            img_info = coco.imgs[img_id]
            img_name = img_info["file_name"]

            img_height = img_info["height"]
            img_width = img_info["width"]

            img_scores = det_scores_per_image[img_count]
            pred_masks = det_masks_per_image[img_count]
            img_boxes = det_boxes_per_image[img_count]

            img_sum_iou = 0
            alt_img_sum_iou = 0

            if len(pred_masks) == 0:
                print("Not detections found! Skipping...")
                continue

            compl_gt_masks = []
            for ann in anns:
                gt_masks = ann["segmentation"]

                packed_masks = np.zeros((img_height, img_width), np.uint8)
                packed_masks_arr = []
                for gt_mask in gt_masks: # one anno mask can be splitted in sub-masks, we merge them here
                    packed_masks_arr.append(np.array(pack_array(gt_mask)))
                cv2.fillPoly(packed_masks, packed_masks_arr, 1) # draw worm foreground
                compl_gt_masks.append(packed_masks)

            # calc IoU GT - PRED
            prods = itertools.product(range(len(anns)), range(len(pred_masks)))

            iou_matrix_size = max(len(anns), len(pred_masks))
            iou_matrix = np.zeros((iou_matrix_size, iou_matrix_size))
            for prod in prods:
                i, j = prod  # GT, Pred

                a = compl_gt_masks[i].astype('int')  # GT
                b = pred_masks[j]  # Pred

                iou_matrix[i][j] = mask_iou(b, a)

            # calc DSB AP per image
            dsb_map, dsb_precisions = dsb_precision(iou_matrix, len(anns), len(pred_masks), iou_threshs)

            for pre_ind, dsb_pre in enumerate(dsb_precisions):
                dataset_dsb_ap_sums[pre_ind] += dsb_pre

            dataset_dsb_map_sum += dsb_map
            bar()
    m_ap = dataset_dsb_map_sum/len(det_masks_per_image)
    print(f"\nDataset DSB mAP: {m_ap:.4f}.\n")

    for ap_sum_id, ap_sum in enumerate(dataset_dsb_ap_sums):
        ap = ap_sum/len(det_masks_per_image)
        print(f"Dataset DSB AP {iou_threshs[ap_sum_id]:.2f}: {ap:.4f}.")
