import numpy as np

from mise.pipeline import structure


def test_a_mirror_is_see_through_all_over():
    """A mirror labelled only along its frame and in patches (the rest reads as the wall it
    reflects) is see-through across its whole hull."""
    classes = ["wall", "floor", "mirror", "bed"]
    m = np.zeros((60, 100), np.uint8)  # wall everywhere
    m[50:, :] = 1  # floor
    m[10:40, 20:24] = 2  # the mirror's left edge
    m[10:40, 66:70] = 2  # right edge
    m[10:13, 20:70] = 2  # top edge
    m[37:40, 20:70] = 2  # bottom edge
    out = structure.fill_see_through({"f": m}, classes, min_px=10)["f"]
    assert (out[15:35, 30:60] == 2).all()  # the reflection inside counts as mirror
    assert (out[50:, :] == 1).all() and (out[:5, :] == 0).all()  # floor and wall outside untouched
