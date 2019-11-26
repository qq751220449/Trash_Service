"""
This module provide
Authors: Liyujun(qq751220449@126.com)
Date: 2019/11/25
"""

import json

import flask
import time
from flask import Flask
import tensorflow as tf
from PIL import Image
import cv2
import numpy as np

import conf.yolov3_config as cfg    #yolov3配置文件
from util.image_util import convert_base64_to_image
from model_server.yolov3_model_service import get_bbox


PATH_TO_CKPT_BOTTLE = cfg.weiht_file_pb
PATH_TO_LABELS_BOTTLE = cfg.CLASSES
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)   #设置每个进程使用GPU内存所占的比例


# 加载湿垃圾模型
bottle_detection_graph = tf.Graph()
with bottle_detection_graph.as_default():
    output_graph_def_bottle = tf.GraphDef()
    with open(PATH_TO_CKPT_BOTTLE, "rb") as f:
        output_graph_def_bottle.ParseFromString(f.read())
        for node in output_graph_def_bottle.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                node.op = 'Sub'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'AssignAdd':
                node.op = 'Add'
                if 'use_locking' in node.attr: del node.attr['use_locking']
            elif node.op == 'Assign':
                node.op = 'Identity'
                if 'use_locking' in node.attr: del node.attr['use_locking']
                if 'validate_shape' in node.attr: del node.attr['validate_shape']
                if len(node.input) == 2:
                    # input0: ref: Should be from a Variable node. May be uninitialized.
                    # input1: value: The value to be assigned to the variable.
                    node.input[0] = node.input[1]
                    del node.input[1]
        tf.import_graph_def(output_graph_def_bottle, name="")
bottle_sess = tf.Session(graph=bottle_detection_graph,
                            config=tf.ConfigProto(gpu_options=gpu_options))
# bottle_sess.run(tf.global_variables_initializer())
input_image_tensor = bottle_sess.graph.get_tensor_by_name("input/input_data:0")
input_is_training_tensor = bottle_sess.graph.get_tensor_by_name("input/training:0")
output_node_sbbox_tensor_name = bottle_sess.graph.get_tensor_by_name("yolov3/pred_sbbox/concat_2:0")
output_node_mbbox_tensor_name = bottle_sess.graph.get_tensor_by_name("yolov3/pred_mbbox/concat_2:0")
output_node_lbbox_tensor_name = bottle_sess.graph.get_tensor_by_name("yolov3/pred_lbbox/concat_2:0")


app = Flask(__name__)


@app.route('/lajijiance', methods=['POST', 'GET'])
def result_feed():

    get_data_time = time.time()
    response = None
    if flask.request.method == 'POST':
        request_body = flask.request.get_json()
        if not request_body['image']:
            return 'Image is a required field', 400
        base64_image = request_body['image']
        if base64_image:
            image = convert_base64_to_image(base64_image)
        else:
            return 'Json Unmarshal Error', 500
        bef_pre_time = time.time()
        print("before predict time is {}".format(bef_pre_time - get_data_time))
        image = Image.open(image)
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if request_body['type'] == 'bottle':
            response = get_bbox(
                                image,
                                bottle_sess,
                                input_image_tensor,
                                input_is_training_tensor,
                                output_node_sbbox_tensor_name,
                                output_node_mbbox_tensor_name,
                                output_node_lbbox_tensor_name,
                                False,
                                False
                                )
        else:
            return 'Unknown type', 400
        bef_return_time = time.time()
        print("all_predict time is {}".format(bef_return_time - bef_pre_time))
    return json.dumps(response), 200


def start():
    app.run(host='0.0.0.0', debug=False, port=8008, threaded=True)
