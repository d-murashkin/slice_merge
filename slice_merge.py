"""
A tool to slice large images into a set of small ones.
And then to merge them back into the image of the original size.
"""
from __future__ import division
import numpy as np


class TiledImage(object):
    """ Class contains the image, information about the split and a list of tiles.
        Either tile size or number_of_tiles should be specified.
        Both options can be integers or tuples.
        keep_rest option defines weather tiles of a smaller size should be kept.
    """
    def __init__(self, image, tile_size=0, number_of_tiles=0, keep_rest=True, offset=0):
        self.keep_rest = keep_rest
        self.tiles = []
        self.offset = offset
        try:
            self.offset_x, self.offset_y = offset
        except:
            self.offset_x = self.offset_y = offset

        """ Dimentions of the input image """
        if len(image.shape) == 2:
            image_copy = image[:, :, np.newaxis]
        elif len(image.shape) == 3:
            image_copy = image
        else:
            print('Image has less than 2 dimentions or more than 3.')
            print('Currently such images are not supported.')
            return False
        self.X, self.Y, self.Z = image_copy.shape

        """ If offset is not 0, extend the input image. """
        if self.offset:
            image_with_offset = np.zeros((self.X + self.offset_x, self.Y + self.offset_y, self.Z), dtype=image_copy.dtype)
            image_with_offset[self.offset_x:, self.offset_y:] = image_copy
            image_copy = image_with_offset
            self.X, self.Y, self.Z = image_copy.shape

        """ Set tile width and hight """
        if tile_size:
            try:
                self.X_sub, self.Y_sub = tile_size
            except:
                self.X_sub = self.Y_sub = tile_size

            if self.keep_rest:
                self.X_num = int(np.ceil(self.X / self.X_sub))
                self.Y_num = int(np.ceil(self.Y / self.Y_sub))
            else:
                self.X_num = self.X // self.X_sub
                self.Y_num = self.Y // self.Y_sub

        elif number_of_tiles == 0:
            print('Either tile_size or number_of_tiles should be specified.')
            return False
        else:
            try:
                self.X_num, self.Y_num = number_of_tiles
            except:
                self.X_num = self.Y_num = number_of_tiles
            if self.keep_rest:
                self.X_sub = int(np.ceil(self.X / self.X_num))
                self.Y_sub = int(np.ceil(self.Y / self.Y_num))
            else:
                self.X_sub = self.X // self.X_num
                self.Y_sub = self.Y // self.Y_num

        """ Represent the image as an 5d array """
        image_eq_div = np.zeros((self.X_sub * self.X_num, self.Y_sub * self.Y_num, self.Z), dtype=image.dtype)
        if self.keep_rest:
            image_eq_div[:self.X, :self.Y, :] = image_copy
        else:
            image_eq_div = image_copy[:image_eq_div.shape[0], :image_eq_div.shape[1]]
        self.data = np.array([np.hsplit(item, self.Y_num) for item in np.vsplit(image_eq_div, self.X_num)])
    
    def list_tiles(self, tile_2d=False):
        i, j, X, Y, Z = self.data.shape
        reshaped = np.reshape(self.data, (i * j, X, Y, Z))
        if tile_2d:
            return reshaped[:, :, :, 0]
        return reshaped
    
    def list_tile_indices(self):
        return [(i, j) for i in range(self.X_num) for j in range(self.Y_num)]
    
    def merge(self, data):
        """ Merge the *data* 5D array with results into a 3D array image size of the original image.
        """
        """ If data is a list of tiles reshape it into a 2D array of tiles. """
        if type(data) == list:
            shape = data[0].shape
            data = np.array(data).reshape(self.X_num, self.Y_num, *shape)

        X, Y = data[0].shape[1:3]
        scale_x = X / self.X_sub
        scale_y = Y / self.Y_sub
            
        merged = np.hstack(np.hstack(data))[int(self.offset_x * scale_x):int(self.X * scale_x), int(self.offset_y * scale_y):int(self.Y * scale_y)]
        return merged

    def image(self):
        """ Return the original 3D array.
        """
        return self.merge(self.data)
    
    def apply(self, fucntion, parallel=False, tile_dim=3):
        """ Apply the specified function to each tile.
            Note, that lambda functions do not work when parallel=True.
            Also note, this is not an in-place operation.
        """
        if tile_dim == 2:
            list_tiles = self.list_tiles(tile_2d=True)
        else:
            list_tiles = self.list_tiles()
        if parallel:
            from multiprocessing import Pool
            pool = Pool()
            result = pool.map(fucntion, list_tiles)
        else:
            result = list(map(fucntion, list_tiles))
        return result
    
    def get_tile(self, i, j):
        return self.data[i, j]

    def set_tile(self, i, j, data):
        if len(data.shape) == 2:
            X, Y = data.shape
            Z = 1
        elif len(data.shape) == 3:
            X, Y, Z = data.shape
        else:
            print('Data should be 2d or 3d array, got data shape {0}'.format(data.shape))
            return False
        if (X, Y, Z) != (self.X_sub, self.Y_sub, self.Z):
            print("data dimentions {0} do not correspond the tile size {1}".forman(data.shape, (self.X_sub, self.Y_sub, self.Z)))
            return False
        try:
            self.data[i, j] = data
        except:
            print('Something went wrong...')
            return False


if __name__ == '__main__':
    pass
