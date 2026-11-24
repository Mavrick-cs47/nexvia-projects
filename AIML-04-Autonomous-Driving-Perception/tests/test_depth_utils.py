import numpy as np

from src.depth_estimation.midas_depth import box_depth


def test_box_depth_median():
    depth = np.zeros((100, 100), dtype=np.float32)
    depth[20:40, 20:40] = 0.8
    assert abs(box_depth(depth, (20, 20, 40, 40)) - 0.8) < 1e-6


def test_box_depth_out_of_bounds_is_nan():
    depth = np.zeros((50, 50), dtype=np.float32)
    assert np.isnan(box_depth(depth, (60, 60, 80, 80)))
