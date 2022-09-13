import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw


class Error(Exception):
    """Base class for exceptions in this module.
    """
    pass


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
        R (int): Input R value should range from 0 to 255.
        G (int): Input G value should range from 0 to 255.
        B (int): Input B value should range from 0 to 255.
    """
    def __init__(self, R, G, B):
        for X in [R, G, B]:
            if (X < 0) or (X > 255):
                raise InputError(X, 'Not an RGB value.')
        self.R = R
        self.G = G
        self.B = B
        self.RGB = (R, G, B)
        # Automatically converts RGB to hex values
        self.hex = '#{:02X}{:02X}{:02X}'.format(self.R,self.G,self.B)
    

    def display(self, font_col):
        """Displays RGB value as a coloured box with text of the given font colour.

        Args:
            font_col (str): A hex value.
        """
        img = Image.new(mode='RGB', size=(100, 100), color=self.RGB)
        img_draw = ImageDraw.Draw(img)
        img_draw.text((36, 45), 'Text', fill=font_col)
        plt.imshow(img)
        plt.show()


def generate_RGB_data(X, extreme=False, extreme_magnitude=200):
    """Generates a list filled with X number of RGB class values.

    Can optionally generate cols that are very dark + very light for training.
    
    Args:
        X (int): Number of desired RGB instances.
        extreme (bool): Optional. Generates very dark + very light cols when True.
        extreme_magnitude (int): Number between 1 and 254.
    """
    if extreme == True:
        cols = []
        for x in range(X):
            minimum = extreme_magnitude*(x%2)
            maximum = 255-(extreme_magnitude*(not x%2))
            rgb = RGB(np.random.randint(low=minimum, high=maximum),
                      np.random.randint(low=minimum, high=maximum),
                      np.random.randint(low=minimum, high=maximum))
            cols.append(rgb)
        return cols     
    else:
        return [RGB(np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255))
                for i in range(X)]