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
    def __init__(self, image, tile_size=0, number_of_tiles=0, keep_rest=True, offset=0, overlay=0):
        self.keep_rest = keep_rest
        self.tiles = []
        self.offset = offset
        self.overlay = overlay
        try:
            self.offset_x, self.offset_y = offset
        except TypeError:
            self.offset_x = self.offset_y = offset

        """ Dimensions of the input image """
        if len(image.shape) == 2:
            image_copy = image[:, :, np.newaxis]
        elif len(image.shape) == 3:
            image_copy = image
        else:
            print('Image has less than 2 dimensions or more than 3.')
            print('Currently such images are not supported.')
            return False
        self.X, self.Y, self.Z = image_copy.shape

        """ If offset or overlay is not 0, extend the input image. """
        if self.offset_x > 0 or self.offset_y > 0 or overlay > 0:
            image_copy_ext = np.zeros((image_copy.shape[0] + self.offset_x * 2 + overlay * 2, image_copy.shape[1] + self.offset_y * 2 + overlay * 2, image_copy.shape[2]), dtype=image_copy.dtype)
            x = -self.offset_x - overlay if self.offset_x + overlay > 0 else None
            y = -self.offset_y - overlay if self.offset_y + overlay > 0 else None
            image_copy_ext[self.offset_x + overlay:x, self.offset_y + overlay:y, :] = image_copy
            image_copy_ext[:self.offset_x + overlay, :, :] = image_copy_ext[self.offset_x + overlay: (self.offset_x + overlay) * 2, :, :][::-1, :, :]
            image_copy_ext[-self.offset_x - overlay:, :, :] = image_copy_ext[-2 * (self.offset_x + overlay):-self.offset_x - overlay, :, :][::-1, :, :]
            image_copy_ext[:, :self.offset_y + overlay, :] = image_copy_ext[:, self.offset_y + overlay: (self.offset_y + overlay) * 2, :][:, ::-1, :]
            image_copy_ext[:, -self.offset_y - overlay:, :] = image_copy_ext[:, -2 * (self.offset_y + overlay):-self.offset_y - overlay, :][:, ::-1, :]
            image_copy = image_copy_ext

        """ Set tile width and hight """
        if tile_size:
            try:
                self.X_sub, self.Y_sub = tile_size
            except TypeError:
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
            except TypeError:
                self.X_num = self.Y_num = number_of_tiles
            if self.keep_rest:
                self.X_sub = int(np.ceil(self.X / self.X_num))
                self.Y_sub = int(np.ceil(self.Y / self.Y_num))
            else:
                self.X_sub = self.X // self.X_num
                self.Y_sub = self.Y // self.Y_num

        """ Represent the image as an 5d array """
        image_eq_div = np.zeros(((self.X_sub + overlay * 2) * self.X_num,
                                 (self.Y_sub + overlay * 2) * self.Y_num,
                                 self.Z), dtype=image.dtype)
        for x in range(self.X_num):
            for y in range(self.Y_num):
                patch = image_copy[self.X_sub * x: self.X_sub * (x + 1) + 2 * overlay,
                                   self.Y_sub * y: self.Y_sub * (y + 1) + 2 * overlay, :]
                image_eq_div[(self.X_sub + 2 * overlay) * x:min((self.X_sub + overlay * 2) * (x + 1), (self.X_sub + 2 * overlay) * x + patch.shape[0]),
                             (self.Y_sub + 2 * overlay) * y:min((self.Y_sub + overlay * 2) * (y + 1), (self.Y_sub + 2 * overlay) * y + patch.shape[1]),
                             :] = patch
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
        if type(data) == list or data.ndim == 4:
            shape = data[0].shape
            data = np.array(data).reshape(self.X_num, self.Y_num, *shape)

        X, Y = data[0].shape[1:3]
        scale_x = X / (self.X_sub + 2 * self.overlay)
        scale_y = Y / (self.Y_sub + 2 * self.overlay)

        merged = np.hstack(np.hstack(data))[int(self.offset_x * scale_x):int(self.X * scale_x),
                                            int(self.offset_y * scale_y):int(self.Y * scale_y)]
        return merged

    def image(self):
        """ Return the original 3D array.
        """
        return self.merge(self.data)

    def apply(self, function, parallel=False, tile_dim=3):
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
            result = pool.map(function, list_tiles)
        else:
            result = list(map(function, list_tiles))
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
            print("data dimensions {0} do not correspond the tile size {1}".format(data.shape, (self.X_sub, self.Y_sub, self.Z)))
            return False
        try:
            self.data[i, j] = data
        except Exception:
            print('Something went wrong...')
            return False


if __name__ == '__main__':
    pass
