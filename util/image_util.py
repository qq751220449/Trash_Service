"""
This module provide
Authors: jiaohanzhe(jiaohanzhe@baidu.com)
Date: 2018/12/06
"""
import io

import numpy as np
import base64


def load_image_into_numpy_array(origin_image):
    """

    :param origin_image:
    :return:
    """
    (im_width, im_height) = origin_image.size
    return np.array(origin_image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def convert_base64_to_image(base64_image):
    """

    :param base64_image:
    :return:
    """
    image = io.BytesIO(base64.b64decode(base64_image))
    return image
