#!/usr/bin/env python3
"""Check that the generated case fits the phone the same way the reference STL does.

Usage: python3 validate_fit.py path/to/reference.stl   (run generate_case.py first)

1. Phone-clash test: an iPhone 17 Pro Max proxy (78.0 x 163.4 x 8.75 mm, ~1 mm
   rounded front edge) is grown step by step; the volume it overlaps each case
   is printed. Matching columns = the phone is held identically.
2. Pocket / lip / camera-opening measurements on both models.
Requires: numpy, trimesh, manifold3d, shapely, scipy, networkx
"""
import sys
import numpy as np
import trimesh
from manifold3d import Manifold, Mesh
from shapely.geometry import LineString
from shapely.ops import unary_union
import generate_case as g

PARTS = ["skeleton", "corner_top_right", "corner_top_left", "corner_bottom_right", "corner_bottom_left"]
PHONE_W, PHONE_L, PHONE_H, EDGE_R = 78.0, 163.4, 8.75, 1.0


def load_manifold(paths):
    out = None
    for p in paths:
        t = trimesh.load(p)
        m = Manifold(Mesh(vert_properties=t.vertices.astype(np.float32), tri_verts=t.faces.astype(np.uint32)))
        out = m if out is None else out + m
    return out


def phone(grow, plan_r):
    slices = []
    for z in np.r_[0, np.linspace(PHONE_H - EDGE_R, PHONE_H, 7)]:
        t = z - (PHONE_H - EDGE_R)
        inset = EDGE_R - np.sqrt(max(EDGE_R ** 2 - t ** 2, 0)) if t > 0 else 0
        zz = z + (grow if z > 0 else 0)
        cs = g.rrect(PHONE_W + 2 * (grow - inset), PHONE_L + 2 * (grow - inset), plan_r + grow - inset)
        slices.append(g.prism(cs, zz, zz + 0.001))
    return Manifold.batch_hull(slices)


def section(mesh, z):
    s = mesh.section(plane_origin=[0, 0, z], plane_normal=[0, 0, 1])
    return unary_union(s.to_2D(to_2D=np.eye(4))[0].polygons_full)


def wall_intrusion(mesh, z, p, n):
    """How far solid reaches inward past point p (pocket wall) along -n at height z."""
    L = section(mesh, z).intersection(LineString([p - n * 6, p + n * 3]))
    ds = [np.dot(np.array(q) - p, n) for gg in getattr(L, "geoms", [L]) for q in getattr(gg, "coords", [])]
    return -min(ds) if ds else float("nan")


def camera_hole(mesh, z=-2.0):
    big = max(getattr(section(mesh, z), "geoms", [section(mesh, z)]), key=lambda q: q.area)
    h = max((h for h in big.interiors if h.bounds[1] > 20), key=lambda h: h.bounds[2] - h.bounds[0])
    return np.round(h.bounds, 2)


def main():
    ref_path = sys.argv[1]
    ref_m, mine_m = load_manifold([ref_path]), load_manifold([f"stl/{p}.stl" for p in PARTS])
    ref_t = trimesh.load(ref_path)
    mine_t = trimesh.util.concatenate([trimesh.load(f"stl/{p}.stl") for p in PARTS])

    print("Phone-clash test (overlap volume, mm^3)")
    for plan_r in (g.CAV_R - 0.5, 13.5, 15.0):
        print(f"  phone plan corner radius {plan_r:.2f} mm")
        print("    grow    reference      this case")
        for d in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
            p = phone(d, plan_r)
            print(f"    {d:4.2f}  {(p ^ ref_m).volume():10.3f}   {(p ^ mine_m).volume():10.3f}")

    print("\nCorner-lip intrusion past the pocket wall at the 45-degree point of each arc (mm)")
    A, B, R = g.CAV_W / 2, g.CAV_L / 2, g.CAV_R
    zs = (8.0, 8.4, 8.8, 9.4, 10.2, 10.8)
    print("    z:      " + " ".join(f"{z:6}" for z in zs))
    for name, t in (("reference", ref_t), ("this case", mine_t)):
        rows = []
        for sx, sy in ((1, 1), (-1, -1)):
            n = np.array([sx, sy]) / np.sqrt(2)
            p = np.array([sx * (A - R), sy * (B - R)]) + R * n
            rows.append(" ".join(f"{wall_intrusion(t, z, p, n):6.2f}" for z in zs))
        print(f"    {name:10s} TR {rows[0]}\n    {'':10s} BL {rows[1]}")

    print("\nCamera opening bounds [x0, y0, x1, y1] at z = -2")
    print("    reference", camera_hole(ref_t), "\n    this case", camera_hole(mine_t))


if __name__ == "__main__":
    main()
