# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
"""
# This module provide
# Authors: jiaohanzhe(jiaohanzhe@baidu.com)
# Date: 2018/12/18
"""
import tensorflow as tf

from tf_server import label_map_util


class InferenceGraph(object):
    """
    推理图
    """

    def __init__(self, path_to_ckpt, path_to_labels, num_classes, gpu_memory=0.2):
        print(path_to_ckpt, " ", gpu_memory)
        self.path_to_ckpt = path_to_ckpt
        self.path_to_labels = path_to_labels
        self.num_classes = num_classes
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_memory)
        self.detection_graph = self._load_detection_graph()
        self.label_map, self.category_index = self._load_label_map()

    def _load_detection_graph(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def_water = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph_water = fid.read()
                od_graph_def_water.ParseFromString(serialized_graph_water)
                tf.import_graph_def(od_graph_def_water, name='')
        return detection_graph

    def _load_label_map(self):
        label_map = label_map_util.load_labelmap(self.path_to_labels)
        categories = label_map_util.convert_label_map_to_categories(
            label_map,
            max_num_classes=self.num_classes,
            use_display_name=False)
        category_index = label_map_util.create_category_index(categories)
        return label_map, category_index

    def start_session(self):
        sess = tf.Session(graph=self.detection_graph,
                          config=tf.ConfigProto(gpu_options=self.gpu_options))
        ops = self.detection_graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = self.detection_graph.get_tensor_by_name(tensor_name)

        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        return sess, tensor_dict, image_tensor
