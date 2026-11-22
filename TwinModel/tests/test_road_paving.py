import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import paving_textures  # noqa: E402
import road_paving  # noqa: E402


def test_herringbone_covers_the_plane_once():
    t = np.linspace(0.01, 39.99, 400)
    x, y = np.meshgrid(t, t)
    a, b, vert, u, v, w, h = paving_textures.herringbone(x, y)
    assert ((u >= 0) & (u < w) & (v >= 0) & (v < h)).all()
    # bricks are 2 x 1: horizontal ones 2 wide, vertical ones 2 high
    assert set(np.unique(w[vert])) == {1.0} and set(np.unique(h[vert])) == {2.0}


def test_textures_tile_seamlessly():
    d, n, o = paving_textures.render("clinker", px=256)
    assert d.shape == (256, 256, 3)
    # opposite edges continue each other: the edge rows differ about as much as neighbouring rows inside
    inner = np.abs(d[128].astype(int) - d[129].astype(int)).mean()
    edge = np.abs(d[0].astype(int) - d[-1].astype(int)).mean()
    assert edge < 3 * inner + 5


def test_style_of():
    assert road_paving.style_of({"surface": "paving_stones"}, "NL") == "clinker"
    assert road_paving.style_of({"surface": "paving_stones"}, "DE") == "pavers"
    assert road_paving.style_of({"surface": "paving_stones", "colour": "grey"}, "NL") == "pavers"
    assert road_paving.style_of({"surface": "sett"}, "NL") == "sett"
    assert road_paving.style_of({"surface": "asphalt"}, "NL") is None


def test_mark_xodr(tmp_path):
    src = tmp_path / "a.xodr"
    src.write_text('<OpenDRIVE>\n  <header name="T">\n    <geoReference>x</geoReference>\n  </header>\n</OpenDRIVE>\n')
    dst = tmp_path / "b.xodr"
    road_paving.mark_xodr(src, dst, ["sett", "clinker"])
    road_paving.mark_xodr(dst, dst, ["clinker"])        # idempotent: one marker
    text = dst.read_text()
    assert text.count("twinPaving") == 1 and 'value="clinker"' in text
    src.write_text('<OpenDRIVE><header name="T"/></OpenDRIVE>')
    road_paving.mark_xodr(src, dst, ["clinker"])
    assert '<header name="T">' in dst.read_text() and "</header>" in dst.read_text()
