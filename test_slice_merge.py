import unittest

import numpy as np

from slice_merge import TiledImage


class TestTiledImage(unittest.TestCase):
    def test_slicing(self):
        data = np.zeros((5, 5, 2))

        sliced = TiledImage(data, tile_size=2, keep_rest=True)
        self.assertEqual(sliced.data.shape, (3, 3, 2, 2, 2))

        sliced = TiledImage(data, tile_size=2, keep_rest=False)
        self.assertEqual(sliced.data.shape, (2, 2, 2, 2, 2))

        sliced = TiledImage(data, number_of_tiles=2, keep_rest=True)
        self.assertEqual(sliced.data.shape, (2, 2, 3, 3, 2))

        sliced = TiledImage(data, number_of_tiles=2, keep_rest=False)
        self.assertEqual(sliced.data.shape, (2, 2, 2, 2, 2))


if __name__ == '__main__':
    unittest.main()
