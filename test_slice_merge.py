import unittest

import numpy as np

from slice_merge import TiledImage


def coordinate_ramp(rows, columns):
    return (
        1
        + 100 * np.arange(rows)[:, None, None]
        + np.arange(columns)[None, :, None]
        + np.array([0, 1000, 2000])
    )


def increment_in_place(tile):
    tile += 1
    return tile


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

    def test_size_based_offset_coverage(self):
        data = coordinate_ramp(5, 8)
        # Size (3, 4): expected grid counts and retained source extents.
        cases = (
            ((1, 0), (2, 2), (2, 2), (5, 8)),
            ((0, 1), (2, 3), (1, 2), (3, 7)),
            ((1, 1), (2, 3), (2, 2), (5, 7)),
        )
        for offset, all_counts, full_counts, (rows, columns) in cases:
            with self.subTest(offset=offset, keep_rest=True):
                sliced = TiledImage(data, tile_size=(3, 4), offset=offset)
                self.assertEqual(sliced.data.shape, (*all_counts, 3, 4, 3))
                np.testing.assert_array_equal(sliced.image(), data)
            with self.subTest(offset=offset, keep_rest=False):
                sliced = TiledImage(
                    data, tile_size=(3, 4), offset=offset, keep_rest=False
                )
                self.assertEqual(sliced.data.shape, (*full_counts, 3, 4, 3))
                np.testing.assert_array_equal(sliced.image(), data[:rows, :columns])

    def test_count_based_offset_coverage(self):
        data = coordinate_ramp(5, 8)
        # Grid (2, 4): expected core sizes and retained source extents.
        cases = (
            ((1, 0), (3, 2), (3, 2), (5, 8)),
            ((0, 1), (3, 3), (2, 2), (4, 7)),
            ((1, 1), (3, 3), (3, 2), (5, 7)),
        )
        for offset, all_size, full_size, (rows, columns) in cases:
            with self.subTest(offset=offset, keep_rest=True):
                sliced = TiledImage(data, number_of_tiles=(2, 4), offset=offset)
                self.assertEqual(sliced.data.shape, (2, 4, *all_size, 3))
                np.testing.assert_array_equal(sliced.image(), data)
            with self.subTest(offset=offset, keep_rest=False):
                sliced = TiledImage(
                    data, number_of_tiles=(2, 4), offset=offset, keep_rest=False
                )
                self.assertEqual(sliced.data.shape, (2, 4, *full_size, 3))
                np.testing.assert_array_equal(sliced.image(), data[:rows, :columns])

    def test_partial_tile_without_offset_or_overlay_is_zero_filled(self):
        data = coordinate_ramp(3, 5)
        sliced = TiledImage(data, tile_size=(2, 3))
        expected = np.zeros((2, 3, 3), dtype=data.dtype)
        expected[0, :2] = data[2, 3:]
        np.testing.assert_array_equal(sliced.get_tile(1, 1), expected)
        np.testing.assert_array_equal(sliced.image(), data)

    def test_set_tile_accepts_a_2d_grayscale_core(self):
        data = np.zeros((4, 4, 1), dtype=int)
        sliced = TiledImage(data, tile_size=2)
        new_tile = np.array([[1, 2], [3, 4]])
        sliced.set_tile(1, 0, new_tile)

        expected = np.zeros_like(data)
        expected[2:4, :2, 0] = new_tile
        np.testing.assert_array_equal(sliced.image(), expected)
        np.testing.assert_array_equal(sliced.get_tile(1, 0), new_tile[:, :, None])

    def test_set_tile_replaces_overlay_or_only_the_core(self):
        data = coordinate_ramp(4, 4)
        sliced = TiledImage(data, tile_size=2, overlay=1)
        neighbor = sliced.get_tile(0, 0).copy()

        full_tile = coordinate_ramp(4, 4) + 5000
        sliced.set_tile(0, 1, full_tile)
        np.testing.assert_array_equal(sliced.get_tile(0, 1), full_tile)

        core = coordinate_ramp(2, 2) + 9000
        sliced.set_tile(0, 1, core)
        expected_tile = full_tile.copy()
        expected_tile[1:-1, 1:-1] = core
        np.testing.assert_array_equal(sliced.get_tile(0, 1), expected_tile)
        np.testing.assert_array_equal(sliced.get_tile(0, 0), neighbor)
        expected_image = data.copy()
        expected_image[:2, 2:4] = core
        np.testing.assert_array_equal(sliced.image(), expected_image)

        before = sliced.data.copy()
        for shape in ((2, 2, 1), (1, 2, 3)):
            with self.subTest(shape=shape):
                self.assertIs(sliced.set_tile(0, 1, np.zeros(shape)), False)
                np.testing.assert_array_equal(sliced.data, before)

    def test_apply(self):
        data = np.arange(25).reshape((5, 5, 1))
        true_result = data**2
        sliced = TiledImage(data, tile_size=2, keep_rest=True)
        result = sliced.merge(sliced.apply(lambda x: x**2))
        np.testing.assert_array_equal(result, true_result)
        result = sliced.merge(sliced.apply(np.square, parallel=True))
        np.testing.assert_array_equal(result, true_result)

    def test_apply_does_not_mutate_stored_tiles(self):
        original = coordinate_ramp(4, 6)
        for parallel in (False, True):
            with self.subTest(parallel=parallel):
                sliced = TiledImage(original.copy(), tile_size=(2, 3), overlay=1)
                processed = sliced.apply(increment_in_place, parallel=parallel)
                np.testing.assert_array_equal(sliced.merge(processed), original + 1)
                np.testing.assert_array_equal(sliced.image(), original)

    def test_list_tiles_2d(self):
        data = np.arange(25).reshape((5, 5, 1))
        true_result = np.arange(4).reshape((2, 2))
        sliced = TiledImage(data, 2)
        self.assertEqual(sliced.list_tiles(tile_2d=True)[0].shape, true_result.shape)

    def test_merge_flat_and_list_2d_tiles(self):
        data = np.arange(24).reshape((4, 6, 1))
        sliced = TiledImage(data, tile_size=(2, 3))
        tiles = sliced.list_tiles(tile_2d=True)

        np.testing.assert_array_equal(sliced.merge(tiles), data[:, :, 0])
        np.testing.assert_array_equal(sliced.merge(list(tiles)), data[:, :, 0])

    def test_list_indices(self):
        data = np.arange(25).reshape((5, 5, 1))
        sliced = TiledImage(data, tile_size=2)
        tile_indices = sliced.list_tile_indices()
        tile_list = sliced.list_tiles()
        tile_by_index = sliced.get_tile(*tile_indices[1])
        tile_from_list = tile_list[1]
        self.assertTrue(np.array_equal(tile_from_list, tile_by_index))
    
    def test_overlay(self):
        data = np.arange(16).reshape((4, 4, 1))
        sliced = TiledImage(data, tile_size=2, overlay=1, keep_rest=True)
        result = np.array([
                          [[ 0,  0,  1,  2],
                           [ 0,  0,  1,  2],
                           [ 4,  4,  5,  6],
                           [ 8,  8,  9, 10]],
    
                          [[ 1,  2,  3,  3],
                           [ 1,  2,  3,  3],
                           [ 5,  6,  7,  7],
                           [ 9, 10, 11, 11]],
    
                          [[ 4,  4,  5,  6],
                           [ 8,  8,  9, 10],
                           [12, 12, 13, 14],
                           [12, 12, 13, 14]],
    
                          [[ 5,  6,  7,  7],
                           [ 9, 10, 11, 11],
                           [13, 14, 15, 15],
                           [13, 14, 15, 15]]])
        self.assertEqual(sliced.list_tiles(tile_2d=True).shape, (4, 4, 4))
        self.assertEqual(sliced.data.shape, (2, 2, 4, 4, 1))
        self.assertTrue(np.array_equal(sliced.list_tiles(tile_2d=True), result))
    
    def test_overlay_on_a_single_pixel(self):
        data = np.array([[[7, 11, 23]]])
        sliced = TiledImage(data, tile_size=1, overlay=2)
        np.testing.assert_array_equal(
            sliced.get_tile(0, 0), np.broadcast_to(data, (5, 5, 3))
        )

    def test_symmetric_padding_on_one_axis(self):
        data = np.array([[[3], [7]]])
        sliced = TiledImage(data, tile_size=(1, 8), offset=(0, 3))
        expected = np.array([7, 7, 3, 3, 7, 7, 3, 3]).reshape((1, 8, 1))
        np.testing.assert_array_equal(sliced.get_tile(0, 0), expected)

    def test_merge_overlayed(self):
        cases = (
            (np.arange(16).reshape((4, 4, 1)), 2, (0, 0)),
            (coordinate_ramp(4, 6), (3, 4), (1, 1)),
        )
        for data, tile_size, offset in cases:
            with self.subTest(tile_size=tile_size, offset=offset):
                sliced = TiledImage(data, tile_size=tile_size, offset=offset, overlay=1)
                full_tiles = sliced.list_tiles()
                trimmed_tiles = [tile[1:-1, 1:-1] for tile in full_tiles]
                np.testing.assert_array_equal(sliced.image(), data)
                np.testing.assert_array_equal(sliced.merge(full_tiles), data)
                np.testing.assert_array_equal(sliced.merge(trimmed_tiles), data)

    def test_merge_decimated_overlay_cores_requires_core_layout(self):
        data = coordinate_ramp(8, 8)
        sliced = TiledImage(data, tile_size=4, overlay=1)
        cores = [tile[1:-1:2, 1:-1:2] for tile in sliced.list_tiles()]

        with self.assertRaises(ValueError):
            sliced.merge(cores)
        np.testing.assert_array_equal(
            sliced.merge(cores, tile_layout="core"), data[::2, ::2]
        )

    def test_merge_preserves_all_29_pixels_of_a_resized_core(self):
        sliced = TiledImage(np.zeros((100, 100, 1)), tile_size=100)
        processed = np.arange(29 * 29).reshape((29, 29, 1))
        np.testing.assert_array_equal(sliced.merge([processed]), processed)

    def test_merge_core_layout_takes_precedence_over_full_tile_shape(self):
        data = coordinate_ramp(4, 4)
        sliced = TiledImage(data, tile_size=2, overlay=1)
        cores = [
            tile[1:-1, 1:-1].repeat(2, axis=0).repeat(2, axis=1)
            for tile in sliced.list_tiles()
        ]
        expected = data.repeat(2, axis=0).repeat(2, axis=1)
        np.testing.assert_array_equal(sliced.merge(cores, tile_layout="core"), expected)

    def test_merge_rectangular_downscaled_cores_with_offset(self):
        sliced = TiledImage(
            coordinate_ramp(7, 8), tile_size=(4, 5), offset=(3, 3), overlay=1
        )
        processed_grid = coordinate_ramp(6, 9)
        cores = [
            processed_grid[row:row + 2, column:column + 3]
            for row in (0, 2, 4)
            for column in (0, 3, 6)
        ]
        # Input-grid bounds [3:10, 3:11] map to [1:5, 1:6].
        np.testing.assert_array_equal(
            sliced.merge(cores, tile_layout="core"), processed_grid[1:5, 1:6]
        )

    def test_constructor_requires_tile_geometry(self):
        with self.assertRaisesRegex(ValueError, r"tile_size.*number_of_tiles"):
            TiledImage(np.zeros((2, 3, 1)))

    def test_readme_red_channel_workflow(self):
        image = coordinate_ramp(3, 5)
        sliced = TiledImage(image, tile_size=2)

        def get_red_channel(tile):
            return tile[:, :, 0]

        listed_red_tiles = [get_red_channel(tile) for tile in sliced.list_tiles()]
        applied_red_tiles = sliced.apply(get_red_channel, parallel=False)
        np.testing.assert_array_equal(sliced.merge(listed_red_tiles), image[:, :, 0])
        np.testing.assert_array_equal(sliced.merge(applied_red_tiles), image[:, :, 0])


if __name__ == '__main__':
    unittest.main()
