"""
Python class representing RGB colour value.
"""

import numpy as np
from PIL import Image, ImageDraw


class Error(Exception):
    """Base class for exceptions in this module.
    """


class InputError(Error):
    """Exception raised for errors in the input.

    Args:
        expr (obj): Input expression in which the error occurred.
        msg  (str): Explanation of the error.
    """
    def __init__(self, expr, msg):
        self.expr = expr
        self.msg = msg


class RGB():
    """An RGB colour value.

    Args:
        r_value (int): Input R value should range from 0 to 255.
        g_value (int): Input G value should range from 0 to 255.
        b_value (int): Input B value should range from 0 to 255.
    """
    def __init__(self, r_value, g_value, b_value):
        for value in [r_value, g_value, b_value]:
            if (value < 0) or (value > 255):
                raise InputError(value, 'Not an RGB value.')
        self.r_value = r_value
        self.g_value = g_value
        self.b_value = b_value
        self.RGB = (r_value, g_value, b_value)
        self.hex = f'#{self.r_value:02X}{self.g_value:02X}{self.b_value:02X}'
        self.img = None


    def generate_img(self, font_col):
        """Generates an image of an RGB-coloured box with text of the given font colour.

        Args:
            font_col (str): A hex value.
        """
        self.img = Image.new(mode='RGB', size=(100, 100), color=self.RGB)
        img_draw = ImageDraw.Draw(self.img)
        img_draw.text((36, 45), 'Text', fill=font_col)


def generate_rgb_data(size, extreme=False, extreme_magnitude=200):
    """Generates a list filled with 'size' number of RGB class values.

    Can optionally generate cols that are very dark + very light for training.

    Args:
        size (int): Number of desired RGB instances.
        extreme (bool): Optional. Generates very dark + very light cols when True.
        extreme_magnitude (int): Number between 1 and 254.
    """
    if extreme is True:
        cols = []
        for count in range(size):
            minimum = extreme_magnitude*(count%2)
            maximum = 255-(extreme_magnitude*(not count%2))
            rgb = RGB(np.random.randint(low=minimum, high=maximum),
                      np.random.randint(low=minimum, high=maximum),
                      np.random.randint(low=minimum, high=maximum))
            cols.append(rgb)
        return cols
    return [RGB(np.random.randint(0, 255),
                np.random.randint(0, 255),
                np.random.randint(0, 255))
            for i in range(size)]
