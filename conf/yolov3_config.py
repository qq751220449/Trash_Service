# coding:utf-8
import os

ROOTDIR_Project = os.path.abspath(os.path.join(__file__, '../..'))
print("ROOTDIR is ", ROOTDIR_Project)


# yolo
TRAIN_INPUT_SIZES = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
TEST_INPUT_SIZE = 544
STRIDES = [8, 16, 32]
IOU_LOSS_THRESH = 0.5

# train
BATCH_SIZE = 4
LEARN_RATE_INIT = 1e-4
LEARN_RATE_END = 1e-6
WARMUP_PERIODS = 2
MAX_PERIODS = 200 

GT_PER_GRID = 3
MOVING_AVE_DECAY = 0.9995

# test
SCORE_THRESHOLD = 0.6    # The threshold of the probability of the classes
IOU_THRESHOLD = 0.45     # The threshold of the IOU when implement NMS

CLASSES = ['bottle', 'plastic_case', 'trash_bag']

weiht_file_pb = os.path.join(ROOTDIR_Project, 'model_store', 'frozen_model_yolov3_20191125.pb')
print("weights_file is in ", weiht_file_pb)


