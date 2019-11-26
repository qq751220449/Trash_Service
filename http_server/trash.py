"""
This module provide
Authors: jiaohanzhe(jiaohanzhe@baidu.com)
Date: 2018/12/06
"""

import json

import flask
import time
from flask import Flask

from conf.basic_config import PATH_TO_CKPT_WATER, PATH_TO_LABELS_WATER, WATER_NUM_CLASSES, \
    PATH_TO_CKPT_BOTTLE, PATH_TO_LABELS_BOTTLE, BOTTLE_NUM_CLASSES, PATH_TO_CKPT_CHUYU, \
    PATH_TO_CKPT_CAIPI, PATH_TO_LABELS_CHUYU, CHUYU_NUM_CLASSES
from http_server.inference_graph import InferenceGraph
from model_server.model_service import check_single_image
from util.image_util import convert_base64_to_image

# 加载水流模型
#water_inference_graph = InferenceGraph(PATH_TO_CKPT_WATER, PATH_TO_LABELS_WATER, WATER_NUM_CLASSES, gpu_memory=0.2)
#water_sess, water_tensor_dict, water_image_tensor = water_inference_graph.start_session()

# 加载瓶子模型
bottle_inference_graph = InferenceGraph(PATH_TO_CKPT_BOTTLE, PATH_TO_LABELS_BOTTLE,
                                        BOTTLE_NUM_CLASSES, gpu_memory=0.8)
bottle_sess, bottle_tensor_dict, bottle_image_tensor = bottle_inference_graph.start_session()

# 加载厨余垃圾模型
#chuyu_inference_graph = InferenceGraph(PATH_TO_CKPT_CHUYU, PATH_TO_LABELS_CHUYU, CHUYU_NUM_CLASSES, gpu_memory=0.2)
#chuyu_sess, chuyu_tensor_dict, chuyu_image_tensor = chuyu_inference_graph.start_session()

# 加载菜皮模型
#caipi_inference_graph = InferenceGraph(PATH_TO_CKPT_CAIPI, PATH_TO_LABELS_CHUYU, CHUYU_NUM_CLASSES, gpu_memory=0.2)
#caipi_sess, caipi_tensor_dict, caipi_image_tensor = caipi_inference_graph.start_session()

app = Flask(__name__)


@app.route('/lajijiance', methods=['POST', 'GET'])
def result_feed():
    """
    结果流feed
    :return:
    """
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
        #print("before predict time is {}".format(bef_pre_time - get_data_time))
        if request_body['type'] == 'water':
            response = check_single_image(water_sess,
                                          image,
                                          water_tensor_dict,
                                          water_image_tensor,
                                          'water'
                                          )
        elif request_body['type'] == 'bottle':
            response = check_single_image(bottle_sess,
                                          image,
                                          bottle_tensor_dict,
                                          bottle_image_tensor,
                                          'bottle'
                                          )
        elif request_body['type'] == 'chuyu':
            response = check_single_image(chuyu_sess,
                                          image,
                                          chuyu_tensor_dict,
                                          chuyu_image_tensor,
                                          'chuyu'
                                          )
        elif request_body['type'] == 'caipi':
            response = check_single_image(caipi_sess,
                                          image,
                                          caipi_tensor_dict,
                                          caipi_image_tensor,
                                          'caipi'
                                          )
        else:
            return 'Unknown type', 400
        bef_return_time = time.time()
        print("all_predict time is {}".format(bef_return_time - bef_pre_time))
    return json.dumps(response), 200


def start():
    """
    启动
    :return:
    """
    app.run(host='0.0.0.0', debug=False, port=8008, threaded=True)
