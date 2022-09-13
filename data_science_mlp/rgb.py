class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class InputError(Error):
    """Exception raised for errors in the input.

    Attributes:
        expr -- Input expression in which the error occurred.
        msg  -- Explanation of the error."""
    def __init__(self, expr, msg):
        self.expr = expr
        self.msg = msg


class RGB():
    """Defined with values for RGB as input.
    
    Attributes:
        RGB -- Input RGB values should range from 0 to 255.
        hex -- Automatically converts RGB to hex values."""
    def __init__(self, R, G, B):
        for X in [R, G, B]:
            if (X < 0) or (X > 255):
                raise InputError(X, 'Not an RGB value.')
        self.R = R
        self.G = G
        self.B = B
        self.RGB = (R, G, B)
        self.hex = '#{:02X}{:02X}{:02X}'.format(self.R,self.G,self.B)


def generate_RGB_data(X, extreme=False, extreme_magnitude=200):
    """Generates a list filled with X number of RGB class values.
    Optional: generate cols that are v. dark + v. light for training.
    
    Attributes:
        X -- Number of desired RGB instances.
        extreme -- Boolean to generate v. dark + v. light cols.
        extreme_magnitude -- Int between 1 and 254."""
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


def display_RGB_colour(colour, font_col='#000'):
    """Will draw a box of given colour;
    and fill with text of given font colour.
    
    Attributes:
        colour -- String containing a RGB or hex value.
        font_col -- String containing a RGB or hex value."""
    img = Image.new(mode='RGB', size=(100, 100), color=colour)
    img_draw = ImageDraw.Draw(img)
    img_draw.text((36, 45), 'Text', fill=font_col)
    plt.imshow(img)
    plt.show();
