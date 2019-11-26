"""
This module provide
Authors: jiaohanzhe(jiaohanzhe@baidu.com)
Date: 2018/12/06
"""

import os
import time

from PIL import Image
from PIL import ImageFilter

from conf.basic_config import CLASS_NAME_DICT_WATER, CLASS_NAME_DICT_BOTTLE, BOX_IOU_THRESHOLD, \
    WATER_SCORE_THRESHOLD, CAIPI_SCORE_THRESHOLD, BOTTLE_SCORE_THRESHOLD, CHUYU_SCORE_THRESHOLD, MEDIUM_WATER_SCORE_THRESHOLD, BIG_WATER_SCORE_THRESHOLD, SMALL_WATER_SCORE_THRESHOLD, BOTTLE_BOTTLE_SCORE_THRESHOLD, TRASHBAG_BOTTLE_SCORE_THRESHOLD, PLASTIC_BOTTLE_SCORE_THRESHOLD, BOTTLE_CHEZAI_SCORE_THRESHOLD, TRASHBAG_CHEZAI_SCORE_THRESHOLD, PLASTIC_CHEZAI_SCORE_THRESHOLD, BOTTLE_CANCHU_SCORE_THRESHOLD, TRASHBAG_CANCHU_SCORE_THRESHOLD, PLASTIC_CANCHU_SCORE_THRESHOLD  

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np


def NMSBbox(all_box):
    vis = np.zeros(len(all_box))
    # print(len(all_box))
    rmpos = []
    for p in range(len(all_box)):
        if vis[p]:
            continue
        vis[p] = 1

        for q in range(len(all_box) - p - 1):
            if vis[q + p + 1]:
                continue
            bbox1 = all_box[p]
            bbox2 = all_box[q + p + 1]
            if compute_iou(bbox1, bbox2) > BOX_IOU_THRESHOLD:
                if all_box[p + q + 1][4] > all_box[p][4]:
                    rmpos.append(p)
                    break
                else:
                    rmpos.append(q + p + 1)
                    vis[q + p + 1] = 1
                # break
    # print('rmpos')
    rmpos.sort(reverse=True)
    # print(rmpos)

    for p in rmpos:
        all_box.pop(p)
    return all_box


def compute_iou(bbox1, bbox2):
    bbox1ymin = bbox1[1]
    bbox1xmin = bbox1[0]
    bbox1ymax = bbox1[3]
    bbox1xmax = bbox1[2]
    bbox2ymin = bbox2[1]
    bbox2xmin = bbox2[0]
    bbox2ymax = bbox2[3]
    bbox2xmax = bbox2[2]
    area1 = (bbox1ymax - bbox1ymin) * (bbox1xmax - bbox1xmin)
    area2 = (bbox2ymax - bbox2ymin) * (bbox2xmax - bbox2xmin)
    bboxxmin = max(bbox1xmin, bbox2xmin)
    bboxxmax = min(bbox1xmax, bbox2xmax)
    bboxymin = max(bbox1ymin, bbox2ymin)
    bboxymax = min(bbox1ymax, bbox2ymax)
    if bboxxmin >= bboxxmax:
        return 0
    if bboxymin >= bboxymax:
        return 0
    area = (bboxymax - bboxymin) * (bboxxmax - bboxxmin)
    iou = area / (area1 + area2 - area)
    return iou


def check_single_image(sess, image, tensor_dict, image_tensor, type):
    """

    :param sess:
    :param image:
    :param tensor_dict:
    :param image_tensor:
    :param type:
    :return:
    """
    bef_bur_time = time.time()
    image = Image.open(image)
    #image = image.filter(ImageFilter.MedianFilter(5))

    start_time = time.time()
    #print("bur_time is {}".format(start_time - bef_bur_time))
    # Run inference
    output_dict = sess.run(tensor_dict,
                           feed_dict={image_tensor: np.expand_dims(np.array(image), 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    # postprocess
    # MNSBOX
    total_boxes = []
    for k in range(output_dict['num_detections']):
        ymin = output_dict['detection_boxes'][k][0]
        xmin = output_dict['detection_boxes'][k][1]
        ymax = output_dict['detection_boxes'][k][2]
        xmax = output_dict['detection_boxes'][k][3]
        temp = [ymin, xmin, ymax, xmax, output_dict['detection_scores'][k],
                output_dict['detection_classes'][k]]
        total_boxes.append(temp)
    #new_total_boxes = NMSBbox(total_boxes)
    new_total_boxes = total_boxes

    # post_process
    new_detection_boxes = []
    new_detection_classes = []
    new_detection_scores = []
    for box in new_total_boxes:
        new_detection_boxes.append(box[0:4])
        new_detection_scores.append(box[4])
        new_detection_classes.append(box[5])

    new_detection_scores = np.array(new_detection_scores)
    new_detection_classes = np.array(new_detection_classes)
    new_detection_boxes = np.array(new_detection_boxes)

    inference_end = time.time()

    trash = []
    count = 0
    score_threshold = 0.1
    if type == 'water':
        class_name_dict = CLASS_NAME_DICT_WATER
        score_threshold = WATER_SCORE_THRESHOLD
    elif type == 'caipi':
        class_name_dict = CLASS_NAME_DICT_BOTTLE
        score_threshold = CAIPI_SCORE_THRESHOLD
    elif type == 'bottle':
        class_name_dict = CLASS_NAME_DICT_BOTTLE
        score_threshold = BOTTLE_SCORE_THRESHOLD
    elif type == 'chuyu':
        class_name_dict = CLASS_NAME_DICT_BOTTLE
        score_threshold = CHUYU_SCORE_THRESHOLD
    for i in range(len(new_detection_scores)):
        # TODO
        #print("score_threshold is ", score_threshold)
        #print("new_detection_scores[i] is ", new_detection_scores[i])
        #if new_detection_scores[i] <= score_threshold or int(new_detection_classes[i]) > 3:
         #   continue
        if type == 'water':
            if int(new_detection_classes[i]) == 1 and new_detection_scores[i] <= SMALL_WATER_SCORE_THRESHOLD:
                continue
            elif int(new_detection_classes[i]) == 2 and new_detection_scores[i] <= MEDIUM_WATER_SCORE_THRESHOLD:
                continue
            elif int(new_detection_classes[i]) == 3 and new_detection_scores[i] <= BIG_WATER_SCORE_THRESHOLD:
                continue
        if type == 'bottle':
            if int(new_detection_classes[i]) == 1 and new_detection_scores[i] <= BOTTLE_BOTTLE_SCORE_THRESHOLD:
                continue
            elif int(new_detection_classes[i]) == 2 and new_detection_scores[i] <= TRASHBAG_BOTTLE_SCORE_THRESHOLD:
                continue
            elif int(new_detection_classes[i]) == 3 and new_detection_scores[i] <= PLASTIC_BOTTLE_SCORE_THRESHOLD:
                continue
        if type == 'chuyu':
            if int(new_detection_classes[i]) == 1 and new_detection_scores[i] <= BOTTLE_CHEZAI_SCORE_THRESHOLD:
                continue
            elif int(new_detection_classes[i]) == 2 and new_detection_scores[i] <= TRASHBAG_CHEZAI_SCORE_THRESHOLD:
                continue
            elif int(new_detection_classes[i]) == 3 and new_detection_scores[i] <= PLASTIC_CHEZAI_SCORE_THRESHOLD:
                continue
        if type == 'caipi':
            if int(new_detection_classes[i]) == 1 and new_detection_scores[i] <= BOTTLE_CANCHU_SCORE_THRESHOLD:
                continue
            elif int(new_detection_classes[i]) == 2 and new_detection_scores[i] <= TRASHBAG_CANCHU_SCORE_THRESHOLD:
                continue
            elif int(new_detection_classes[i]) == 3 and new_detection_scores[i] <= PLASTIC_CANCHU_SCORE_THRESHOLD:
                continue

        current_detection = {'name': class_name_dict[int(new_detection_classes[i])],
                             'score': float(new_detection_scores[i]),
                             'position': new_detection_boxes[i].tolist()
                             }
        count += 1
        trash.append(current_detection)
    response = {
        'trash_num': count,
        'trash': trash
    }
    response_end = time.time()
    print('TIME: {}, {} '.format(response_end - inference_end, inference_end - start_time))
    print(response)
    return response
