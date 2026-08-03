"""even_crop() guards against a bug that does not look like a bug.

The source is chroma-subsampled, so ffmpeg silently snaps an odd crop onto the
chroma grid - ask for width 347 and you get 346. Reading w*h*3 bytes per frame
out of a stream of (w-1)*h*3 byte frames shears every frame progressively.
The pixel histogram still looks normal, so detection quietly degrades instead
of failing. These cases pin the geometry that keeps ffmpeg from rounding.
"""
import pytest

from shot_clipper.video_source import even_crop


@pytest.mark.parametrize("roi,frame,expected", [
    # the real case that exposed the bug: odd x, odd width
    ((517, 0, 864, 576), (2688, 1512), (516, 0, 348, 576, 1, 0)),
    # already even - must be left exactly alone
    ((4, 2, 10, 8), (100, 100), (4, 2, 6, 6, 0, 0)),
    # odd on every side
    ((1, 1, 9, 9), (100, 100), (0, 0, 10, 10, 1, 1)),
    # growing outward must not run past the frame
    ((1, 1, 99, 99), (100, 100), (0, 0, 100, 100, 1, 1)),
    # full frame, nothing to do
    ((0, 0, 100, 100), (100, 100), (0, 0, 100, 100, 0, 0)),
])
def test_even_crop_geometry(roi, frame, expected):
    assert even_crop(roi, *frame) == expected


@pytest.mark.parametrize("roi", [
    (517, 0, 864, 576), (1, 1, 9, 9), (3, 7, 101, 203), (0, 0, 1, 1), (99, 99, 100, 100),
])
def test_even_crop_always_even_and_covers_roi(roi):
    x, y, w, h, off_x, off_y = even_crop(roi, 2688, 1512)
    assert x % 2 == 0 and y % 2 == 0, "ffmpeg rounds odd offsets onto the chroma grid"
    assert w % 2 == 0 and h % 2 == 0, "ffmpeg rounds odd sizes onto the chroma grid"
    rx1, ry1, rx2, ry2 = roi
    # the slice the caller takes back out has to land inside what we asked for
    assert x + off_x == rx1 and y + off_y == ry1
    assert off_x + (rx2 - rx1) <= w and off_y + (ry2 - ry1) <= h
