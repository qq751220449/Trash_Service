"""
This module provide
Authors: jiaohanzhe(jiaohanzhe@baidu.com)
Date: 2018/12/06
"""

# 输入输出路径配置
#PATH_TO_CKPT_WATER = '/data/trash_service/model_store/water_frozen_inference_graph_v6.pb'
PATH_TO_CKPT_WATER = '/data/trash_service/model_store/frozen_inference_graph_water_0430.pb'
#PATH_TO_CKPT_BOTTLE = '/data/trash_service/model_store/frozen_inference_graph_bottle_nas_0508.pb'
PATH_TO_CKPT_BOTTLE = '/data/trash_service/model_store/frozen_inference_graph_bottle_0423.pb'

#PATH_TO_CKPT_CHUYU = '/data/trash_service/model_store/chuyu_frozen_inference_graph_v3.pb'
#PATH_TO_CKPT_CAIPI = '/data/trash_service/model_store/caipi_frozen_inference_graph_v1.pb'
PATH_TO_CKPT_CAIPI = '/data/trash_service/model_store/frozen_inference_graph_chezaishilaji_0416.pb'
PATH_TO_CKPT_CHUYU = '/data/trash_service/model_store/frozen_inference_graph_canchu_0417.pb'
PATH_TO_LABELS_WATER = '/data/trash_service/conf/label_map_water.pbtxt'
PATH_TO_LABELS_BOTTLE = '/data/trash_service/conf/label_map_bottle.pbtxt'
PATH_TO_LABELS_CHUYU = '/data/trash_service/conf/label_map_chuyu.pbtxt'

# 模型图片配置
WATER_NUM_CLASSES = 3
CLASS_NAME_DICT_WATER = {
    1: "water-small",
    2: "water-medium",
    3: "water-big",
}

BOTTLE_NUM_CLASSES = 3
CLASS_NAME_DICT_BOTTLE = {
    1: "bottle",
    2: "trash_bag",
    3: "plastic_case",
}

CHUYU_NUM_CLASSES = 3

# 后处理配置
IOU_THRESHOLD = 0.3
BOX_IOU_THRESHOLD = 0.25

WATER_SCORE_THRESHOLD = 0.5
MEDIUM_WATER_SCORE_THRESHOLD = 0.9
BIG_WATER_SCORE_THRESHOLD = 0.9
SMALL_WATER_SCORE_THRESHOLD = 0.5
CAIPI_SCORE_THRESHOLD = 0.9
CHUYU_SCORE_THRESHOLD = 0.9
BOTTLE_SCORE_THRESHOLD = 0.5
BOTTLE_BOTTLE_SCORE_THRESHOLD = 0.8
TRASHBAG_BOTTLE_SCORE_THRESHOLD = 0.8
PLASTIC_BOTTLE_SCORE_THRESHOLD = 0.8
BOTTLE_CHEZAI_SCORE_THRESHOLD = 0.5
TRASHBAG_CHEZAI_SCORE_THRESHOLD = 0.9
PLASTIC_CHEZAI_SCORE_THRESHOLD = 0.5
BOTTLE_CANCHU_SCORE_THRESHOLD = 0.5
TRASHBAG_CANCHU_SCORE_THRESHOLD = 0.5
PLASTIC_CANCHU_SCORE_THRESHOLD = 0.5
