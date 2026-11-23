import sys
from pathlib import Path

from shapely.geometry import Polygon, box

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import osm_detail  # noqa: E402


def test_an_arcade_without_a_mapped_ground_floor_gets_its_back_wall():
    # a 30 x 20 m block, the south 4 m an arcade under the upper floors; OSM maps no ground-floor part
    fp = box(0, 0, 30, 20)
    arcade = box(0, 0, 30, 4)
    segs = osm_detail.back_wall_segments(arcade, fp, Polygon(), [arcade])
    assert len(segs) == 1 and abs(sum(s.length for s in segs) - 30) < 1.0     # less 0.4 m at each facade
    assert all(abs(c[1] - 4) < 1e-6 for s in segs for c in s.coords)       # along y = 4, not on the facade


def test_a_passage_through_the_building_gets_both_walls():
    fp = box(0, 0, 30, 20)
    passage = box(12, 0, 18, 20)
    segs = osm_detail.back_wall_segments(passage, fp, Polygon(), [passage])
    assert abs(sum(s.length for s in segs) - 40) < 2.0                       # x = 12 and x = 18, 20 m each


def test_a_fully_open_ground_floor_has_no_back_wall():
    fp = box(0, 0, 30, 20)
    segs = osm_detail.back_wall_segments(fp, fp, Polygon(), [fp])
    assert segs == []
