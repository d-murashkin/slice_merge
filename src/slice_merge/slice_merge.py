"""
A tool to slice large images into a set of small ones.
And then to merge them back into the image of the original size.
"""
from operator import index

import numpy as np


def _pair(value):
    if np.isscalar(value):
        return index(value), index(value)
    x, y = value
    return index(x), index(y)


class TiledImage(object):
    """ Class contains the image, information about the split and a list of tiles.
        Either tile size or number_of_tiles should be specified.
        Both options can be integers or tuples.
        keep_rest option defines weather tiles of a smaller size should be kept.
    """
    def __init__(
        self, image, tile_size: int | tuple[int, int] = 0,
        number_of_tiles: int | tuple[int, int] = 0, keep_rest=True,
        offset: int | tuple[int, int] = 0, overlay=0,
    ):
        self.keep_rest = keep_rest
        self.tiles = []
        self.offset = offset
        self.offset_x, self.offset_y = _pair(offset)
        self.overlay = overlay = index(overlay)
        if min(self.offset_x, self.offset_y, overlay) < 0:
            raise ValueError("offset and overlay must be nonnegative")

        image = np.asarray(image)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        if image.ndim != 3:
            raise ValueError("image must have two or three dimensions")
        self.X, self.Y, self.Z = image.shape
        extent_x = self.offset_x + self.X
        extent_y = self.offset_y + self.Y

        def divide(extent, divisor):
            if divisor <= 0:
                raise ValueError("tile sizes and counts must be positive")
            return (extent + divisor - 1) // divisor if keep_rest else extent // divisor

        if tile_size:
            self.X_sub, self.Y_sub = _pair(tile_size)
            self.X_num = divide(extent_x, self.X_sub)
            self.Y_num = divide(extent_y, self.Y_sub)
        elif number_of_tiles:
            self.X_num, self.Y_num = _pair(number_of_tiles)
            self.X_sub = divide(extent_x, self.X_num)
            self.Y_sub = divide(extent_y, self.Y_num)
        else:
            raise ValueError("Either tile_size or number_of_tiles must be specified")
        if min(self.X_num, self.Y_num, self.X_sub, self.Y_sub) == 0:
            raise ValueError("the requested grid has no complete, nonempty tiles")

        # Only offsets and halos request symmetric extension. Ordinary partial
        # tiles retain their historical zero fill outside the source image.
        if self.offset_x or self.offset_y or overlay:
            pad_x = (self.offset_x + overlay, max(0, self.X_num * self.X_sub + overlay - extent_x))
            pad_y = (self.offset_y + overlay, max(0, self.Y_num * self.Y_sub + overlay - extent_y))
            image = np.pad(image, (pad_x, pad_y, (0, 0)), mode="symmetric")

        height = self.X_sub + 2 * overlay
        width = self.Y_sub + 2 * overlay
        shape = (self.X_num, self.Y_num, height, width, self.Z)
        self.data = np.empty(shape, dtype=image.dtype)
        for i in range(self.X_num):
            x = i * self.X_sub
            for j in range(self.Y_num):
                y = j * self.Y_sub
                patch = image[x:x + height, y:y + width]
                tile = self.data[i, j]
                if patch.shape != tile.shape:
                    tile.fill(0)
                tile[:patch.shape[0], :patch.shape[1]] = patch

    def list_tiles(self, tile_2d=False):
        i, j, X, Y, Z = self.data.shape
        reshaped = np.reshape(self.data, (i * j, X, Y, Z))
        if tile_2d:
            return reshaped[:, :, :, 0]
        return reshaped

    def list_tile_indices(self):
        return [(i, j) for i in range(self.X_num) for j in range(self.Y_num)]

    def merge(self, data, *, tile_layout="auto"):
        """Join row-major tiles and crop to the retained image extent.

        With overlap, auto accepts either original full tiles or original
        cores. Use ``tile_layout="core"`` for resized, already-trimmed cores.
        """
        if tile_layout not in ("auto", "core"):
            raise ValueError("tile_layout must be 'auto' or 'core'")
        grid = np.asarray(data)
        counts = (self.X_num, self.Y_num)
        if grid.ndim in (3, 4):
            if grid.shape[0] != self.X_num * self.Y_num:
                raise ValueError("tile count does not match the grid")
            grid = grid.reshape(*counts, *grid.shape[1:])
        elif grid.ndim != 5 or grid.shape[:2] != counts:
            raise ValueError("expected a flat sequence of tiles or a matching 5-D tile grid")

        halo = self.overlay
        if halo and tile_layout == "auto":
            if grid.shape[2:4] == self.data.shape[2:4]:
                grid = grid[:, :, halo:-halo, halo:-halo]
            elif grid.shape[2:4] != (self.X_sub, self.Y_sub):
                raise ValueError("trim resized tiles to their cores and use tile_layout='core'")

        height, width = grid.shape[2:4]
        channels = grid.shape[4:]  # Empty for 2-D results.
        mosaic = np.empty((self.X_num * height, self.Y_num * width, *channels), dtype=grid.dtype)
        mosaic.reshape(self.X_num, height, self.Y_num, width, *channels).swapaxes(1, 2)[:] = grid
        return mosaic[
            self.offset_x * height // self.X_sub:(self.offset_x + self.X) * height // self.X_sub,
            self.offset_y * width // self.Y_sub:(self.offset_y + self.Y) * width // self.Y_sub,
        ]

    def image(self):
        """ Return the original 3D array.
        """
        return self.merge(self.data)

    def apply(self, function, parallel=False, tile_dim=3):
        """ Apply the specified function to each tile.
            Note, that lambda functions do not work when parallel=True.
            Also note, this is not an in-place operation.
        """
        tiles = self.list_tiles(tile_2d=(tile_dim == 2))
        if parallel:
            from multiprocessing import Pool
            from os import cpu_count
            with Pool(processes=min(len(tiles), cpu_count() or 1)) as pool:
                return pool.map(function, tiles)
        return [function(tile.copy()) for tile in tiles]

    def get_tile(self, i, j):
        return self.data[i, j]

    def set_tile(self, i, j, data):
        """Replace a full tile or its core without updating neighboring halos."""
        data = np.asarray(data)
        if data.ndim == 2:
            data = data[:, :, np.newaxis]
        try:
            target = self.data[i, j]
        except IndexError:
            return False
        if data.shape != target.shape:
            if data.shape != (self.X_sub, self.Y_sub, self.Z):
                return False
            h = self.overlay
            target = target[h:h + self.X_sub, h:h + self.Y_sub]
        try:
            target[:] = data
        except (ValueError, TypeError, OverflowError):
            return False


if __name__ == '__main__':
    pass
