# Evaluate segmentation results

import colorsys
import itertools
import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask
import scipy.ndimage as ndimage
from PIL import Image

RESULT_PKL = "/Users/maurice/sciebo/results/htc++/htc_24/results.pkl"  # path to results .pkl file
IMGS_PATH = "/Users/maurice/Data/Datasets/worms_from_matthias/extracted_frames/"  # Path to images
GT_JSON = "/Users/maurice/Data/Datasets/worms_from_matthias/coco_annotations/new_test_annotations.json"  # Path to GT annotation file

SCORE_THRES = 0.5  # Detection confidence threshold
# Threshold to reject overlapping predictions (set >1.0 to deactivate)
PRED_IOU_THRES = 1.1
DISPLAY_SEG_PER_IMG = True  # Display segmentations per image
PRINT_SCORES_PER_IMG = False  # Print all scores per image


# calc mask IOU
# from https://github.com/matterport/Mask_RCNN/issues/2440#issuecomment-788629008
def mask_iou(pred_mask, gt_mask):
    intersection = np.sum((pred_mask + gt_mask) > 1)
    union = np.sum((pred_mask + gt_mask) > 0)
    iou_score = intersection / float(union)
    return iou_score


# get random n colors
# from https://stackoverflow.com/a/9701141
# with modifications
def get_colors(num_colors, ret_float=False):
    colors = []
    for i in np.arange(0.0, 360.0, 360.0 / num_colors):
        hue = i / 360.0
        lightness = (50 + np.random.rand() * 10) / 100.0
        saturation = (90 + np.random.rand() * 10) / 100.0

        rgb = np.array(colorsys.hls_to_rgb(hue, lightness, saturation))

        # return float or 8-bit
        if not ret_float:
            colors.append(
                np.around(255 * rgb)
            )  # convert 0 - 1 to 0 - 255 RGB values
        else:
            colors.append(rgb)
    return colors


# load gt data
gt_file = open(GT_JSON)
gt_data = json.load(gt_file)
gt_file.close()

# match image/result id to file
gt_id_to_image = []
for img_entry in gt_data["images"]:
    gt_id_to_image.append((img_entry["id"], img_entry["file_name"]))

gt_masks_to_image = {}
for anno in gt_data["annotations"]:
    image_id = anno["image_id"]

    if image_id not in gt_masks_to_image:
        gt_masks_to_image[image_id] = []
    gt_masks_to_image[image_id].append(anno["segmentation"][0])

# load result data
data_pkl = open(RESULT_PKL, "rb")
data = pickle.load(data_pkl)

# iterate over detections for all images
det_scores_per_image = {}
det_masks_per_image = []
for img_i, img_data in enumerate(data):
    # index 1 of tuple -> t[0]: BBox + Score, t[1]: Size + RLE; 0 to because of surrounding list
    detection_scores = img_data[0][0]
    detection_masks = img_data[1][0]

    # store masks and score per image
    det_scores = []
    det_masks = []
    for det_idx in range(len(detection_masks)):
        det_score = detection_scores[det_idx]

        if det_score[4] < SCORE_THRES:
            continue

        det_scores.append(det_score[4])

        roi_dict = detection_masks[det_idx]
        dec_mask = mask.decode(roi_dict)
        det_masks.append(dec_mask)

    det_scores_per_image[img_i] = det_scores
    det_masks_per_image.append(det_masks)
data_pkl.close()

if PRINT_SCORES_PER_IMG:
    sorted_scores = sorted(
        det_scores_per_image.items(),
        key=lambda item: (np.sum(item[1]) / len(item[1])),
        reverse=True,
    )

    for w_idx in reversed(range(len(sorted_scores))):
        worst_img = sorted_scores[w_idx]
        print(
            "Img: "
            + str(worst_img[0])
            + " Score: "
            + str(np.sum(worst_img[1]) / len(worst_img[1]))
        )

# display results
if DISPLAY_SEG_PER_IMG:
    for img_count in range(len(det_masks_per_image)):
        img_id = gt_id_to_image[img_count][0]
        img_scores = det_scores_per_image[img_count]
        img_masks = det_masks_per_image[img_count]
        gt_masks = gt_masks_to_image[img_id]

        if len(img_masks) == 0:
            print("Not detections found! Skipping...")
            continue

        # Create an empty image to store the masked array
        r_masks = []
        for c_id, contour in enumerate(gt_masks):
            r_mask = np.zeros_like(img_masks[0], dtype="int")
            # Create a contour image by using the contour coordinates rounded to their nearest integer value
            r_mask[
                np.round(contour[1::2]).astype("int"),
                np.round(contour[0::2]).astype("int"),
            ] = 1

            # Fill in the hole created by the contour boundary
            r_mask = ndimage.binary_fill_holes(r_mask)
            r_masks.append(r_mask)

        # IoU GT - PRED
        prods = itertools.product(range(len(gt_masks)), range(len(img_masks)))
        iou_gt_pred_scores = {}
        for prod in prods:
            i, j = prod

            a = r_masks[i].astype("int")  # GT
            b = img_masks[j]  # Pred

            if i not in iou_gt_pred_scores:
                iou_gt_pred_scores[i] = {}
            iou_gt_pred_scores[i][j] = mask_iou(b, a)

        # IoU PRED - PRED
        combis = itertools.combinations(range(len(img_masks)), 2)
        iou_pred_pred_score = {}
        iou_rejected = []
        for combi in combis:
            i, j = combi

            a = img_masks[i]
            b = img_masks[j]

            if i not in iou_pred_pred_score:
                iou_pred_pred_score[i] = {}
            iou_score = mask_iou(b, a)
            iou_pred_pred_score[i][j] = iou_score

            # check if mask needs to be rejected
            if (
                iou_score > PRED_IOU_THRES
            ):  # too large overlap with other pred.
                conf_scores = (img_scores[i], img_scores[j])
                rejected_mask = (
                    j if np.argmin(conf_scores) else i
                )  # get mask with the lowest confidence score
                iou_rejected.append(rejected_mask)

        file_name = gt_id_to_image[img_count][1]
        img = Image.open(os.path.join(IMGS_PATH, file_name)).convert("RGBA")

        fig, ax = plt.subplots(1, 2)
        fig.set_size_inches(18.5, 10.5)

        # get GT data
        gt_img = img.copy()
        ax[0].set_title("GT")
        ax[0].imshow(gt_img)

        gt_colors = get_colors(
            len(gt_masks), ret_float=True
        )  # generate a different color per gt

        for gt_mask_id in range(len(gt_masks)):
            gt_mask = gt_masks[gt_mask_id]

            x_coords = gt_mask[0::2]
            y_coords = gt_mask[1::2]

            gt_color = gt_colors[gt_mask_id].tolist()
            gt_color = tuple(gt_color)

            ax[0].fill(x_coords, y_coords, gt_color)  # create polygon

        # plot masks on image
        det_count = 0
        det_colors = get_colors(
            len(img_masks)
        )  # generate a different color per detection
        for img_mask_id in range(len(img_masks)):
            # check if mask was rejected because of IoU
            if img_mask_id in iou_rejected:
                print(f"Rejected IoU: {img_mask_id}")
                continue

            color = det_colors[img_mask_id].tolist()
            color.append(127.0)  # 50 percent transparency for mask

            mask = img_masks[img_mask_id]
            mask_size = np.count_nonzero(mask == 1)
            print(f"Mask: {img_mask_id}, Size: {mask_size}.")

            height = mask.shape[0]
            width = mask.shape[1]
            colored_mask = np.zeros((height, width, 4), np.uint8)

            colored_mask[mask == 0] = [
                0,
                0,
                0,
                0,
            ]  # no mask -> white pixel, 100 percent transparency
            colored_mask[mask == 1] = color

            mask_img = Image.fromarray(colored_mask)
            img = Image.alpha_composite(img, mask_img)  # add mask to image

            det_count += 1

        fig.suptitle(
            "Image: "
            + file_name
            + " -- GTs: "
            + str(len(gt_masks))
            + " -- Preds: "
            + str(det_count)
        )
        print("Image: " + file_name + " - Preds.: " + str(det_count))

        ax[1].set_title("Pred.")
        ax[1].imshow(img)

        plt.waitforbuttonpress(-1)
        plt.close()  # clear figure to reuse window for next image
