"""
This module provide
Authors: Liyujun(qq751220449@126.com)
Date: 2019/11/25
"""


import numpy as np
import conf.yolov3_config as cfg
import cv2
import os
import tensorflow as tf
from util import tools

_train_input_sizes = cfg.TRAIN_INPUT_SIZES
_test_input_size = cfg.TEST_INPUT_SIZE
_classes = cfg.CLASSES
_num_classes = len(_classes)
_class_to_ind = dict(zip(_classes, range(_num_classes)))
_score_threshold = cfg.SCORE_THRESHOLD
_iou_threshold = cfg.IOU_THRESHOLD
# _dataset_path = cfg.DATASET_PATH
#_project_path = cfg.PROJECT_PATH
# moving_ave_decay = cfg.MOVING_AVE_DECAY
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # 默认使用GPU0进行预测
# test_weight = "./weights/yolo.ckpt-98-0.7907"


def __predict(image, test_input_size, valid_scale, sess , input_data, training, pred_sbbox, pred_mbbox, pred_lbbox):
    org_image = np.copy(image)
    org_h, org_w, _ = org_image.shape

    yolo_input = tools.img_preprocess2(image, None, (test_input_size, test_input_size), False)
    yolo_input = yolo_input[np.newaxis, ...]  # 扩充Batch_Size维度
    pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
        [pred_sbbox, pred_mbbox, pred_lbbox],
        feed_dict={
            input_data: yolo_input,
            training: False
        }
    )
    pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + _num_classes)),
                                np.reshape(pred_mbbox, (-1, 5 + _num_classes)),
                                np.reshape(pred_lbbox, (-1, 5 + _num_classes))], axis=0)
    bboxes = __convert_pred(pred_bbox, test_input_size, (org_h, org_w), valid_scale)
    return bboxes


def __convert_pred(pred_bbox, test_input_size, org_img_shape, valid_scale):
    """
    将yolo输出的bbox信息(xmin, ymin, xmax, ymax, confidence, probability)进行转换，
    其中(xmin, ymin, xmax, ymax)是预测bbox的左上角和右下角坐标
    confidence是预测bbox属于物体的概率，probability是条件概率分布
    (xmin, ymin, xmax, ymax) --> (xmin_org, ymin_org, xmax_org, ymax_org)
    --> 将预测的bbox中超出原图的部分裁掉 --> 将分数低于score_threshold的bbox去掉
    :param pred_bbox: yolo输出的bbox信息，shape为(output_size * output_size * gt_per_grid, 5 + num_classes)
    :param test_input_size: 测试尺寸
    :param org_img_shape: 存储格式必须为(h, w)，输入原图的shape
    :return: bboxes
    假设有N个bbox的score大于score_threshold，那么bboxes的shape为(N, 6)，存储格式为(xmin, ymin, xmax, ymax, score, class)
    其中(xmin, ymin, xmax, ymax)的大小都是相对于输入原图的，score = conf * prob，class是bbox所属类别的索引号
    """
    pred_bbox = np.array(pred_bbox)

    pred_coor = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # (1)
    # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
    # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
    # 需要注意的是，无论我们在训练的时候使用什么数据增强方式，都不影响此处的转换方式
    # 假设我们对输入测试图片使用了转换方式A，那么此处对bbox的转换方式就是方式A的逆向过程
    org_h, org_w = org_img_shape
    resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
    dw = (test_input_size - resize_ratio * org_w) / 2
    dh = (test_input_size - resize_ratio * org_h) / 2
    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # (2)将预测的bbox中超出原图的部分裁掉
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    # (3)将无效bbox的coor置为0
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # (4)去掉不在有效范围内的bbox
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # (5)将score低于score_threshold的bbox去掉
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > _score_threshold

    mask = np.logical_and(scale_mask, score_mask)

    coors = pred_coor[mask]
    scores = scores[mask]
    classes = classes[mask]

    bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

    return bboxes


def get_bbox(image, sess , input_data, training, pred_sbbox, pred_mbbox, pred_lbbox, multi_test=False, flip_test=False):
    """
    :param image: 要预测的图片
    :return: 返回NMS后的bboxes，存储格式为(xmin, ymin, xmax, ymax, score, class)
    """
    h, w, c = image.shape
    if multi_test:
        test_input_sizes = _train_input_sizes[::3]  # 这里应该是选择[320,416,512,608]
        bboxes_list = []
        for test_input_size in test_input_sizes:
            valid_scale = (0, np.inf)
            bboxes_list.append(__predict(image, test_input_size, valid_scale, sess, input_data,
                                         training, pred_sbbox, pred_mbbox, pred_lbbox))
            if flip_test:
                bboxes_flip = __predict(image[:, ::-1, :], test_input_size, valid_scale, sess,
                                        input_data, training, pred_sbbox, pred_mbbox, pred_lbbox)
                bboxes_flip[:, [0, 2]] = image.shape[1] - bboxes_flip[:, [2, 0]]
                bboxes_list.append(bboxes_flip)
        bboxes = np.row_stack(bboxes_list)
    else:
        bboxes = __predict(image, _test_input_size, (0, np.inf), sess , input_data, training,
                           pred_sbbox, pred_mbbox, pred_lbbox)
    bboxes = tools.nms(bboxes, _score_threshold, _iou_threshold, method='nms')
    trash = []
    count = np.array(bboxes).shape[0]
    for i in range(count):
        current_detection = {"name": _classes[int(bboxes[i][5])],
                             "score": float(bboxes[i][4]),
                             # "position":bboxes[i][0:4].tolist()
                             "position": [bboxes[i][1]/h, bboxes[i][0]/w, bboxes[i][3]/h, bboxes[i][2]/w]
                            }
        trash.append(current_detection)
    response ={
        "trash_num": count,
        "trash": trash
    }
    print(response)
    return response
