"""
# This module provide
# Authors: jiaohanzhe(jiaohanzhe@baidu.com)
# Date: 2018/12/18
"""

import sys
import os

#sys.path.append("/root/trash_service")
ROOTDIR = os.path.abspath(os.path.join(__file__, '../..'))
print("ROOTDIR is ", ROOTDIR)
sys.path.append(ROOTDIR)
from http_server import yolov3_trash

if __name__ == "__main__":
    yolov3_trash.start()
