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
    """
    def __init__(self, image, tile_size=0, number_of_tiles=0, keep_rest=True):
        self.keep_rest = keep_rest
        self.tiles = []

        """ Dimentions of the input image """
        if len(image.shape) == 2:
            self.image = image[:, :, np.newaxis]
        elif len(image.shape) == 3:
            self.image = image
        else:
            print('Image has less than 2 dimentions or more than 3.')
            print('Currently such images are not supported.')
            return False
        self.X, self.Y, _ = self.image.shape

        """ Set tile width and hight """
        if tile_size:
            try:
                self.X_sub, self.Y_sub = tile_size
            except:
                self.X_sub = self.Y_sub = tile_size
        elif number_of_tiles == 0:
            print('Either tile_size or number_of_tiles should be specified.')
            return False
        else:
            try:
                self.X_num, self.Y_num = number_of_tiles
            except:
                self.X_num = self.Y_num = number_of_tiles
            self.X_sub = self.X // self.X_num
            self.Y_sub = self.Y // self.Y_num

        """ Define grid """
        if (not keep_rest) or (self.X % self.X_sub == 0):
            xs = np.arange(0, self.X, self.X_sub)
        else:
            xs = np.arange(0, self.X + self.X_sub, self.X_sub)
        
        if (not keep_rest) or (self.Y % self.Y_sub == 0):
            ys = np.arange(0, self.Y, self.Y_sub)
        else:
            ys = np.arange(0, self.Y + self.Y_sub, self.Y_sub)
        
        self.coords = [{'x_min': x, 'x_max': x + self.X_sub, 'y_min': y, 'y_max': y + self.Y_sub, 'subimage_location': (i, j)} for i, x in enumerate(xs) for j, y in enumerate(ys)]
    
    def split(self):
        """ Split the image into tiles.
            The list of tiles is available at self.tiles
        """
        self.tiles = [Tile(self.image, item['x_min'], item['x_max'], item['y_min'], item['y_max']) for item in self.coords]

    def merge(self):
        """ Merge data from tiles into the large image.
        """
        self.output = np.zeros_like(self.image)
        for tile in self.tiles:
            self.output[tile.x_min:tile.x_max, tile.y_min:tile.y_max] = tile.image
    

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
