import unittest

import numpy as np

from slice_merge import TiledImage


class TestTiledImage(unittest.TestCase):
    def test_slicing(self):
        data = np.zeros((5, 5, 2))

        sliced = TiledImage(data, tile_size=2, keep_rest=True)
        self.assertEqual(sliced.data.shape, (3, 3, 2, 2, 2))
        self.assertEqual(sliced.image().shape, data.shape)

        sliced = TiledImage(data, tile_size=2, keep_rest=False)
        self.assertEqual(sliced.data.shape, (2, 2, 2, 2, 2))

        sliced = TiledImage(data, number_of_tiles=2, keep_rest=True)
        self.assertEqual(sliced.data.shape, (2, 2, 3, 3, 2))
        self.assertEqual(sliced.image().shape, data.shape)

        sliced = TiledImage(data, number_of_tiles=2, keep_rest=False)
        self.assertEqual(sliced.data.shape, (2, 2, 2, 2, 2))
    
    def test_set_tile(self):
        data = np.zeros((2, 2, 1))
        sliced = TiledImage(data, tile_size=1, keep_rest=True)
        new_tile = np.ones((1, 1, 1))
        data[1, 0, 0] = 1
        sliced.set_tile(1, 0, new_tile)
        self.assertTrue(np.array_equal(sliced.image(), data))
        self.assertTrue(np.array_equal(new_tile, sliced.get_tile(1, 0)))


if __name__ == '__main__':
    unittest.main()
