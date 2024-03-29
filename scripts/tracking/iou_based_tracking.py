# Created by: Deserno, M. et. al., 2023 for https://www.nature.com/articles/s41598-023-38213-7
# Experimental code to match detections globally over video frames

import copy
import itertools
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pycocotools.mask as mask
import scipy.optimize as opt
import seaborn as sns
from alive_progress import alive_bar
from pycocotools.coco import COCO


def vis_iou_matrix(iou_matrix: np.ndarray) -> None:
    # visualize IoU Matrix as Heatmap
    sns.set_theme()

    ax = sns.heatmap(
        iou_matrix,
        annot=True,
        vmax=1.0,
        vmin=0.0,
        center=0.0,
        cbar=False,
        fmt=".2f",
        square=True,
    )
    ax.set(xlabel="Support", ylabel="Target")
    plt.show()


def mask_iou(pred_mask, gt_mask) -> float:
    # src: https://github.com/matterport/Mask_RCNN/issues/2440#issuecomment-788629008
    intersection = np.sum((pred_mask + gt_mask) > 1)
    union = np.sum((pred_mask + gt_mask) > 0)
    iou_score = intersection / float(union)

    return iou_score


def extract_predictions(score_thres: float, data_path: str):
    # extract predictions from .pkl files generated by MMDetection
    # open result file
    data_pkl = open(data_path, "rb")
    data = pickle.load(data_pkl)

    det_scores_per_image: dict[int, list[np.float32]] = {}
    det_masks_per_image: list[list[np.ndarray]] = []
    det_boxes_per_image: list[list[np.ndarray]] = []

    # iterate over detections for all images
    for img_i, img_data in enumerate(data):
        # index 1 of tuple -> t[0]: BBox + Score, t[1]: Size + RLE; 0 to because of surrounding list
        detection_scores = img_data[0][0]
        detection_masks = img_data[1][0]

        # store masks and score per image
        det_scores: list[np.float32] = []
        det_masks: list[np.ndarray] = []
        det_boxes: list[np.ndarray] = []
        for det_idx in range(len(detection_masks)):
            det_score: np.ndarray = detection_scores[det_idx]

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


def pack_array(array):
    packed_array = []
    for i in range(0, len(array), 2):
        packed_array.append(np.array([int(array[i]), int(array[i + 1])]))
    return np.array(packed_array)


def extract_gt(gt_path: str, img_height: int, img_width: int):
    # extract and format GT the same way as the pred.
    coco = COCO(gt_path)
    img_ids = coco.getImgIds()

    gt_scores_per_image = {}
    gt_masks_per_image = []

    for img_id in img_ids:
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        gt_scores_per_image[img_id] = []

        compl_gt_masks = []  # all masks of an image
        for ann in anns:
            gt_scores_per_image[img_id].append(1.0)
            gt_masks = ann["segmentation"]

            packed_masks = np.zeros((img_height, img_width), np.uint8)
            packed_masks_arr = []
            for (
                gt_mask
            ) in (
                gt_masks
            ):  # one anno mask can be splitted in sub-masks, we merge them here
                packed_masks_arr.append(np.array(pack_array(gt_mask)))
            # draw worm foreground
            cv2.fillPoly(packed_masks, packed_masks_arr, 1)
            compl_gt_masks.append(packed_masks)

        gt_masks_per_image.append(compl_gt_masks)

    return gt_scores_per_image, gt_masks_per_image


def reconnect_paths(
    global_tracking_dict: dict[int, list[int]],
    lost_global_ids: dict[int, int],
    appeard_worm_id: int,
    current_frame_id: int,
    detected_masks_per_image: list[list[np.ndarray]],
) -> int:
    # tries to reconnect previous lost global ids
    iou_ids: list[int] = []  # global ids, matching the iou score
    # iou scores between previous lost worms and the new appeared worm
    iou_scores: list[float] = []

    lookback: int = 10  # define lookback distance
    iou_threshold: float = 0.5  # threshold for reconnect
    # youngest frame to consider when searching beginning of disconnected part
    youngest_frame_id: int = max(0, current_frame_id - lookback)
    appeared_worm_mask: np.ndarray = detected_masks_per_image[
        current_frame_id
    ][appeard_worm_id]

    # get all neccessary data for lost worm
    for lost_global_id in lost_global_ids.items():
        # lost global id was last seen here
        last_seen_frame_id = lost_global_id[1] - 1
        if last_seen_frame_id >= youngest_frame_id:
            # event is not too old -> we still consider it
            # get worm id which was assigned to the global_worm_id for this frame
            lost_local_worm_id = global_tracking_dict[lost_global_id[0]][
                last_seen_frame_id
            ]
            lost_worm_mask = detected_masks_per_image[last_seen_frame_id][
                lost_local_worm_id
            ]

            # calc iou between a lost worm and the found one
            iou_score = mask_iou(lost_worm_mask, appeared_worm_mask)
            iou_scores.append(iou_score)
            iou_ids.append(lost_global_id[0])

    # no possible match found
    if len(iou_scores) == 0:
        return -1

    # get best match for reappeard worm
    best_matching_worm = np.argmax(iou_scores)
    re_found_global_id = iou_ids[best_matching_worm]
    best_iou = iou_scores[best_matching_worm]

    # check if iou is greater than threshold
    if iou_threshold <= best_iou:
        print(
            f"Re-found global track {re_found_global_id} with IoU {best_iou}."
        )
    else:
        re_found_global_id = -1

    return re_found_global_id


def build_global_tracking(
    tracking_dict: dict[int, list[tuple[int, int]]],
    detected_masks_per_image: list[list[np.ndarray]],
) -> dict[int, list[int]]:
    # check local frame tracking and construct global tracking dict
    global_tracking_dict: dict[int, list[int]] = {}
    # storing last local worm ids to their assigned global ids
    latest_worm_assignment: dict[int, int] = {}
    lost_global_ids: dict[
        int, int
    ] = {}  # which global ID got lost in what frame

    frame_ids = sorted(list(tracking_dict.keys()))
    for frame_id in frame_ids:  # iterate over frames
        # this will replace previous one after iterating over the current frame
        tmp_latest_worm_assignment: dict[int, int] = {}
        # saves global worm ids seen in the current frame
        tracked_worms: list[int] = []
        frame_trackings: list[tuple[int, int]] = tracking_dict[frame_id]

        for track in frame_trackings:  # iterate over tracked worms
            prev_worm_id: int = track[0]
            curr_worm_id: int = track[1]
            global_worm_id: int = -1

            if curr_worm_id != -1 and prev_worm_id == -1:
                # found new worm...
                # attempting reconnection
                re_found_global_id = reconnect_paths(
                    global_tracking_dict,
                    lost_global_ids,
                    curr_worm_id,
                    frame_id,
                    detected_masks_per_image,
                )

                # re-found a global track
                if re_found_global_id > -1:
                    global_worm_id = re_found_global_id
                    del lost_global_ids[
                        global_worm_id
                    ]  # delete it from lost dict
            elif prev_worm_id in latest_worm_assignment.keys():
                # worm was seen before, get it's global ID
                global_worm_id = latest_worm_assignment[prev_worm_id]

            # not a refound track nor a previous seen worm
            if global_worm_id == -1:
                # new worm track, find and assign new ID
                if not global_tracking_dict:  # empty dicts == False
                    # dict is empty, this is the first track
                    global_worm_id = 0
                else:
                    # find a new global id
                    global_worm_keys = list(global_tracking_dict.keys())
                    global_worm_id = max(global_worm_keys) + 1

                # add new worm to tracking list
                # add -1 for all previous untracked frames
                global_tracking_dict[global_worm_id] = [-1] * (frame_id - 1)
                # add previous worm id, because that is for sure associated with the global id
                global_tracking_dict[global_worm_id].append(prev_worm_id)
                print(
                    f"New worm track found in frame ID {frame_id}. Assigning ID: {global_worm_id}."
                )

            # update tracking dict
            global_tracking_dict[global_worm_id].append(curr_worm_id)
            tracked_worms.append(global_worm_id)

            # update temporary last worm assignment
            tmp_latest_worm_assignment[curr_worm_id] = global_worm_id

        # mark all non seen global tracks as lost
        for global_worm_id in global_tracking_dict.keys():
            if global_worm_id not in tracked_worms:
                global_tracking_dict[global_worm_id].append(-1)
                if global_worm_id not in lost_global_ids.keys():
                    print(
                        f"Global worm ID {global_worm_id} was lost in frame ID {frame_id}."
                    )
                    lost_global_ids[global_worm_id] = frame_id

        # make temporary last worm assignment permanent
        latest_worm_assignment = tmp_latest_worm_assignment

    return global_tracking_dict


if __name__ == "__main__":
    iou_threshold: float = 0.5

    results_path: str = "pred_res.pkl"
    tracking_pickle_out: str = "tracking.pkl"

    score_thres: float = 0.5  # confidence threshold

    (
        det_scores_per_image,
        det_masks_per_image,
        det_boxes_per_image,
    ) = extract_predictions(score_thres, results_path)

    # iterate over images
    tracking_dict: dict[int, list[tuple[int, int]]] = {}
    with alive_bar(len(det_masks_per_image) - 1) as bar:
        for target_img_id in range(1, len(det_masks_per_image)):
            tracking_dict[target_img_id] = []

            # target frame data
            target_img_scores = det_scores_per_image[target_img_id]
            target_pred_masks = det_masks_per_image[target_img_id]

            # support frame data
            support_image_id: int = target_img_id - 1
            support_img_scores: list[np.float32] = det_scores_per_image[
                support_image_id
            ]
            support_pred_masks: list[np.ndarray] = det_masks_per_image[
                support_image_id
            ]

            if len(target_pred_masks) == 0 or len(support_pred_masks) == 0:
                print("Not detections found! Skipping...")
                continue

            # calc IoU
            len_target_masks: int = len(target_pred_masks)
            len_support_masks: int = len(support_pred_masks)
            prods = itertools.product(
                range(len(target_pred_masks)), range(len(support_pred_masks))
            )

            iou_matrix: np.ndarray = np.zeros(
                (len(target_pred_masks), len(support_pred_masks))
            )

            for prod in prods:
                target_id, support_id = prod

                target_mask = target_pred_masks[target_id]
                support_mask = support_pred_masks[support_id]

                iou_matrix[target_id][support_id] = mask_iou(
                    support_mask, target_mask
                )

            # apply threshold for matching
            matching_iou_matrix: np.ndarray = copy.deepcopy(iou_matrix)

            target_assignment: np.ndarray
            support_assignment: np.ndarray
            target_assignment, support_assignment = opt.linear_sum_assignment(
                1 - matching_iou_matrix
            )

            final_assigned_targets: list[
                int
            ] = []  # list of all targets which got matched
            final_assigned_supports: list[
                int
            ] = []  # list of all supports which got matched

            assert len(target_assignment) == len(support_assignment)
            for assignment_id in range(len(target_assignment)):
                assigned_target: int = target_assignment[assignment_id]
                assigned_support: int = support_assignment[assignment_id]
                assignment_iou: float = iou_matrix[assigned_target][
                    assigned_support
                ]
                multi_hit: bool = False
                tr_iou_matrix = iou_matrix.T

                if assignment_iou < iou_threshold:
                    continue
                tracking_dict[target_img_id].append(
                    (assigned_support, assigned_target)
                )
                final_assigned_supports.append(assigned_support)
                final_assigned_targets.append(assigned_target)

            for target_id in range(len_target_masks):
                if target_id not in final_assigned_targets:
                    tracking_dict[target_img_id].append((-1, target_id))
                    # no support worm found for detected worm in target
                    print(
                        f"Worm {target_id} was (re-)found in frame {target_img_id}."
                    )

            for support_id in range(len_support_masks):
                if support_id not in final_assigned_supports:
                    # worm was lost in target frame
                    print(
                        f"Support worm {support_id} was lost in frame {target_img_id}."
                    )

            bar()

    global_tracking_dict = build_global_tracking(
        tracking_dict, det_masks_per_image
    )

    print(f"Saving tracking dict to: {tracking_pickle_out}")
    with open(tracking_pickle_out, "wb") as handle:
        pickle.dump(
            global_tracking_dict, handle, protocol=pickle.HIGHEST_PROTOCOL
        )

    print("Done.")
