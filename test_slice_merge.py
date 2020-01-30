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
    
    def test_apply(self):
        data = np.arange(25).reshape((5, 5, 1))
        true_result = data**2
        sliced = TiledImage(data, tile_size=2, keep_rest=True)
        result = sliced.merge(sliced.apply(lambda x: x**2))
        self.assertTrue(np.array_equal(result, true_result))
        result = sliced.merge(sliced.apply(np.square, parallel=True))
    
    def test_list_tiles_2d(self):
        data = np.arange(25).reshape((5, 5, 1))
        true_result = np.arange(4).reshape((2, 2))
        sliced = TiledImage(data, 2)
        self.assertEqual(sliced.list_tiles(tile_2d=True)[0].shape, true_result.shape)


if __name__ == '__main__':
    unittest.main()
