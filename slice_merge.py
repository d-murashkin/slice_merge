"""
A tool to slice large images into a set of small ones.
And then to merge them back into the image of the original size.
"""
import numpy as np


class TiledImage(object):
    """ Class contains the image, information about the split and a list of tiles.
        Either tile size or number_of_tiles should be specified.
        Both options can be integers or tuples.
        keep_rest option defines weather tiles of a smaller size should be kept.
        *channels* can be 'last' if channes of the image is the last index or 'first' otherwise.
    """
    def __init__(self, image, tile_size=0, number_of_tiles=0, keep_rest=True, channels='last'):
        self.image = image
        self.keep_rest = keep_rest
        self.channels = channels

        """ Dimentions of the input image """
        if len(image.shape) == 2:
            X, Y = image.shape
        elif len(image.shape) == 3:
            if channels.lower() == 'last':
                X, Y, _ = image.shape
            elif channels.lower() == 'first':
                _, X, Y = image.shape
            else:
                print('Unknown option {1} for *channels*.'.format(channels))
                return False
        else:
            print('Image has less than 2 dimentions or more than 3.')
            print('Currently such images are not supported.')
            return False

        """ Set tile width and hight """
        if tile_size:
            try:
                X_sub, Y_sub = tile_size
            except:
                X_sub = Y_sub = tile_size
        elif number_of_tiles == 0:
            print('Either tile_size or number_of_tiles should be specified.')
            return False
        else:
            try:
                X_num, Y_num = number_of_tiles
            except:
                X_num = Y_num = number_of_tiles
            X_sub = X // X_num
            Y_sub = Y // Y_num

        """ Define grid """
        if (not keep_rest) or (X % X_sub == 0):
            xs = np.arange(0, X, X_sub)
        else:
            xs = np.arange(0, X + X_sub, X_sub)
        
        if (not keep_rest) or (Y % Y_sub == 0):
            ys = np.arange(0, Y, Y_sub)
        else:
            ys = np.arange(0, Y + Y_sub, Y_sub)
        
        self.coords = [{'x_min': x, 'x_max': x + X_sub, 'y_min': y, 'y_max': y + Y_sub, 'subimage_location': (i, j)} for i, x in enumerate(xs) for j, y in enumerate(ys)]
    
    def split(self):
        self.tiles = [Tile(self.image, item['x_min'], item['x_max'], item['y_min'], item['y_max']) for item in self.coords]
    

class Tile(object):
    def __init__(self, image, x_min, x_max, y_min, y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.image = image[x_min:x_max, y_min:y_max]
                

if __name__ == '__main__':
    pass
    image_path = ''
    from PIL import Image
    image = Image.open(image_path)
    img = np.array(image)
    img_split = TiledImage(img, 256)
