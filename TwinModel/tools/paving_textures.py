"""Procedural road paving textures (base colour, normal, ORM) for tools/road_paving.py: seamless 2 x 2 m tiles.

* ``clinker``: Dutch fired clay road bricks (gebakken klinkers, waalformaat about 20 x 10 cm on the road) laid in
  herringbone (keperverband, the usual bond for roads with traffic), red-brown with the usual spread of darker
  purple-brown and lighter orange bricks, sand-filled 3 mm joints, slightly worn arrises.
* ``pavers``: grey concrete paving blocks (betonstraatstenen elsewhere: paving_stones outside NL), same bond.
* ``sett``: granite setts about 15 x 10 cm in courses across the road (random lengths, staggered joints), grey with
  blue-grey and brown-grey stones, wider 8 mm joints, rounded tops.

Nothing is derived from imagery: the stones are drawn from these numbers. Deterministic (seeded by the name).

    python tools/paving_textures.py OUT_DIR [--px 2048]
"""
from __future__ import annotations

import argparse
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

TILE_M = 2.0      # one texture tile covers 2 x 2 m (road_paving.py writes UVs as metres / TILE_M)

STYLES = {
    # base RGB (0-255), per-stone spread (value, hue), joint RGB, joint width (m), roughness, bevel (m)
    "clinker": {"rgb": (138, 62, 46), "spread": (0.09, 0.03), "joint": (122, 112, 96), "joint_m": 0.003,
                "rough": 0.82, "bevel_m": 0.006, "unit_m": 0.1, "bond": "herringbone",
                "mix": [((116, 52, 44), 0.14), ((154, 74, 50), 0.12)]},
    "pavers": {"rgb": (128, 126, 120), "spread": (0.10, 0.01), "joint": (104, 100, 92), "joint_m": 0.003,
               "rough": 0.86, "bevel_m": 0.005, "unit_m": 0.1, "bond": "herringbone",
               "mix": [((116, 114, 110), 0.2), ((138, 134, 126), 0.15)]},
    "sett": {"rgb": (118, 116, 112), "spread": (0.14, 0.02), "joint": (78, 74, 66), "joint_m": 0.008,
             "rough": 0.7, "bevel_m": 0.018, "unit_m": 0.1, "bond": "courses",
             "mix": [((108, 110, 115), 0.25), ((124, 117, 108), 0.2)]},
}


def _hash01(*keys, seed=0):
    """Per-stone uniform numbers in [0, 1) from integer keys (vectorised)."""
    h = np.uint64(seed * 0x9E3779B1 + 0x85EBCA77)
    for k in keys:
        h = (h ^ (np.asarray(k).astype(np.int64).astype(np.uint64) + np.uint64(0x9E3779B97F4A7C15))) * \
            np.uint64(0xBF58476D1CE4E5B9)
        h ^= h >> np.uint64(31)
    return (h % np.uint64(1 << 24)).astype(np.float64) / float(1 << 24)


def herringbone(x, y):
    """90-degree herringbone of 2 x 1 bricks (x, y in brick widths). Lattice (1, 1), (2, -2); horizontal brick
    [0, 2) x [0, 1), vertical [0, 1) x [1, 3). Returns (stone id a, stone id b, vertical?, u, v, w, h): local
    coordinates in the stone and its size."""
    fx, fy = np.floor(x), np.floor(y)
    out_a, out_b = np.zeros_like(x), np.zeros_like(x)
    vert = np.zeros(x.shape, bool)
    u = np.zeros_like(x)
    v = np.zeros_like(x)
    w = np.full_like(x, 2.0)
    h = np.ones_like(x)
    done = np.zeros(x.shape, bool)
    for s in (fx, fx - 1):          # horizontal: s in (x-2, x], d = floor(y), s = d (mod 4)
        m = ~done & (np.mod(s - fy, 4) == 0)
        out_a[m], out_b[m], u[m], v[m] = s[m], fy[m], (x - s)[m], (y - fy)[m]
        done |= m
    for d in (fy - 1, fy - 2):      # vertical: s = floor(x), d in (y-3, y-1]
        m = ~done & (np.mod(fx - d, 4) == 0)
        out_a[m], out_b[m], u[m], v[m] = fx[m], d[m], (x - fx)[m], (y - d - 1)[m]
        vert[m] = True
        w[m], h[m] = 1.0, 2.0
        done |= m
    assert done.all(), "herringbone left a gap"
    return out_a, out_b, vert, u, v, w, h


def courses(x, y, n_units, seed):
    """Setts in courses across the road: rows 1 unit high, stones 1.2-1.8 units long, staggered, periodic over
    n_units. Returns like herringbone."""
    row = np.floor(y)
    v = y - row
    rows = int(n_units)
    edges = {}
    rng = np.random.default_rng(seed)
    for r in range(rows):           # stone boundaries of each row, summing to n_units exactly
        cuts, pos = [], rng.uniform(0, 1.5)
        while pos < n_units - 1.0:
            cuts.append(pos)
            pos += rng.uniform(1.2, 1.8)
        edges[r] = np.array(cuts)
    a = np.zeros_like(x)
    u = np.zeros_like(x)
    w = np.ones_like(x)
    for r in range(rows):
        m = np.mod(row, rows) == r
        if not m.any():
            continue
        c = edges[r]
        xx = np.mod(x[m], n_units)
        k = np.searchsorted(c, xx, side="right") - 1          # -1: the stone that wraps around the tile edge
        left = np.where(k >= 0, c[np.clip(k, 0, None)], c[-1] - n_units)
        right = np.where(k + 1 < len(c), c[np.clip(k + 1, 0, len(c) - 1)], c[0] + n_units)
        a[m] = np.mod(k, len(c))
        u[m] = xx - left
        w[m] = right - left
    return a, row, np.zeros(x.shape, bool), u, v, w, np.ones_like(x)


def render(style: str, px: int = 2048, seed: int | None = None):
    """(base colour uint8 HxWx3, normal uint8, orm uint8) of one 2 x 2 m tile."""
    st = STYLES[style]
    seed = zlib.crc32(style.encode()) if seed is None else seed
    n_units = TILE_M / st["unit_m"]
    t = (np.arange(px) + 0.5) / px * n_units
    x, y = np.meshgrid(t, t)
    if st["bond"] == "herringbone":
        a, b, vert, u, v, w, h = herringbone(x, y)
    else:
        a, b, vert, u, v, w, h = courses(x, y, n_units, seed)
    a_w, b_w = np.mod(a, n_units), np.mod(b, n_units)       # the same stone on both sides of a tile edge
    edge = np.minimum.reduce([u, w - u, v, h - v]) * st["unit_m"]      # metres to the stone's edge
    j, bev = st["joint_m"] / 2, st["bevel_m"]
    top = np.clip((edge - j) / bev, 0, 1)
    top = top * top * (3 - 2 * top)                         # smoothstep: the worn arris
    joint = edge < j
    # colour per stone: the base or one of the mix colours, then value and hue jitter
    r1, r2, r3 = (_hash01(a_w, b_w, vert, k, seed=seed) for k in (1, 2, 3))
    base = np.array(st["rgb"], float)
    col = np.broadcast_to(base, x.shape + (3,)).copy()
    acc = 0.0
    for rgb, share in st["mix"]:
        m = (r1 >= acc) & (r1 < acc + share)
        col[m] = rgb
        acc += share
    val = 1.0 + (r2 - 0.5) * 2 * st["spread"][0]
    hue = (r3 - 0.5) * 2 * st["spread"][1]
    col *= val[..., None]
    col[..., 0] *= 1 + hue
    col[..., 2] *= 1 - hue
    rng = np.random.default_rng(seed)
    grain = rng.normal(0, 1, x.shape)
    grain = (grain + np.roll(grain, 1, 0) + np.roll(grain, 1, 1) + np.roll(grain, -1, 0)) / 4    # fine, seamless
    col *= (1 + 0.05 * grain)[..., None]
    # weathering: soft, seamless patches of dirt and wear (whole periods over the tile)
    ph = rng.uniform(0, 2 * np.pi, 6)
    k = 2 * np.pi / n_units
    soft = sum(np.sin(k * f * x + ph[i]) * np.sin(k * g * y + ph[i + 3]) / (i + 1)
               for i, (f, g) in enumerate(((1, 2), (3, 1), (2, 3))))
    col *= (1 + 0.04 * soft)[..., None]
    col *= (0.88 + 0.12 * top)[..., None]                   # darker, dusty arrises
    jc = np.array(st["joint"], float) * (1 + 0.06 * grain)[..., None]
    col = np.where(joint[..., None], jc, col)
    base_color = np.clip(col, 0, 255).astype(np.uint8)
    # height: stone tops (with a slight per-stone tilt and dome for setts), joints low
    dome = 0.0 if st["bond"] == "herringbone" else 0.4
    height = top * (1.0 + dome * np.sin(np.pi * np.clip(u / w, 0, 1)) * np.sin(np.pi * np.clip(v / h, 0, 1)))
    height += 0.15 * (r2 - 0.5) + 0.02 * grain
    height = np.where(joint, -0.3, height)
    m_per_px = TILE_M / px
    hscale = st["bevel_m"] * 0.6
    gx = (np.roll(height, -1, 1) - np.roll(height, 1, 1)) * hscale / (2 * m_per_px)
    gy = (np.roll(height, -1, 0) - np.roll(height, 1, 0)) * hscale / (2 * m_per_px)
    nrm = np.dstack([-gx, gy, np.ones_like(gx)])            # image rows run down: +v is -y (OpenGL-style green up)
    nrm /= np.linalg.norm(nrm, axis=2, keepdims=True)
    normal = ((nrm * 0.5 + 0.5) * 255).round().astype(np.uint8)
    ao = np.where(joint, 0.55, 0.8 + 0.2 * top)
    rough = np.clip(st["rough"] + 0.08 * (r3 - 0.5) + np.where(joint, 0.1, 0.0), 0, 1)
    orm = np.dstack([ao, rough, np.zeros_like(ao)])
    orm = (np.clip(orm, 0, 1) * 255).round().astype(np.uint8)
    return base_color, normal, orm


def write(style: str, out_dir: Path, prefix: str, px: int = 2048) -> dict:
    """Write T_<prefix>_paving_<style>_{d,n,orm}.png; returns the manifest texture entry (paths relative to ue/)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    d, n, o = render(style, px)
    names = {}
    for slot, img, suf in (("base_color", d, "d"), ("normal", n, "n"), ("orm", o, "orm")):
        p = out_dir / f"T_{prefix}_paving_{style}_{suf}.png"
        Image.fromarray(img).save(p, optimize=True)
        names[slot] = f"textures/{p.name}"
    names["source"] = "procedural (tools/paving_textures.py)"
    return names


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out")
    ap.add_argument("--px", type=int, default=2048)
    ap.add_argument("--prefix", default="Preview")
    args = ap.parse_args(argv)
    for s in STYLES:
        print(s, write(s, Path(args.out), args.prefix, args.px))


if __name__ == "__main__":
    main()
