#!/usr/bin/env python3
"""Parametric iPhone 17 Pro Max "exoskeleton" case (two-colour, 5 parts).

Parts (all in mm, back of case faces -Z, phone screen faces +Z, camera at +Y):
  shell      - blue back shell: back plate, full-height top/bottom walls,
               low side lips (phone buttons stay exposed), raised camera bezel,
               MagSafe ring groove. Four corner windows with tongues.
  corner_*   - orange wrap-around corner caps (4x). Front lip retains the phone.

Outer size, cavity size, back-plate thickness and overall height are taken from
the reference STL (82.6 x 168.0 x 13.95 mm, cavity ~79 x 164, plate z -3..0).

Usage: python3 generate_case.py   ->  writes ./stl/*.stl
Requires: numpy, trimesh, manifold3d
"""
import os
import numpy as np
import trimesh
from manifold3d import Manifold, CrossSection, JoinType

SEG = 96

# ---- dimensions (from reference STL) --------------------------------------
OUT_W, OUT_L = 82.6, 168.0      # outer footprint
OUT_R = 13.6                    # outer plan corner radius
CAV_W, CAV_L = 79.0, 164.0      # phone cavity (phone is 78.0 x 163.4 x 8.75)
CAV_R = 11.6
Z_BACK = -3.0                   # outside face of back plate
Z_TOP = 10.95                   # top of corner caps
PHONE_H = 8.9                   # cavity height (z 0 .. PHONE_H)

# ---- design choices matched to the photos ---------------------------------
SIDE_LIP_H = 4.0                # low lip along the long edges
CORNER_X, CORNER_Y = 15.0, 19.0 # corner cap extent along short / long edge
FRONT_LIP = 2.0                 # corner cap lip overlapping the screen edge
GAP = 0.15                      # clearance shell <-> corner cap
CAM_W, CAM_H = 71.0, 45.0       # camera opening
CAM_R = 9.0
CAM_TOP_MARGIN = 5.0            # from top edge of case to opening
BEZEL_W, BEZEL_H = 1.6, 1.6     # raised camera rim (width, protrusion)
MAGSAFE_Y = -4.0
MAGSAFE_OD, MAGSAFE_ID, MAGSAFE_DEPTH = 56.0, 53.0, 0.4


def rrect(w, l, r, cx=0.0, cy=0.0):
    cs = CrossSection.square((w - 2 * r, l - 2 * r), center=True)
    cs = cs.offset(r, JoinType.Round, 2.0, SEG)
    return cs.translate((cx, cy))


def prism(cs, z0, z1):
    return Manifold.extrude(cs, z1 - z0).translate((0, 0, z0))


def box(x0, x1, y0, y1, z0, z1):
    return Manifold.cube((x1 - x0, y1 - y0, z1 - z0)).translate((x0, y0, z0))


def to_mesh(m):
    mm = m.to_mesh()
    return trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3],
                           np.asarray(mm.tri_verts), process=True)


def build():
    outer = prism(rrect(OUT_W, OUT_L, OUT_R), Z_BACK, Z_TOP)
    cavity = prism(rrect(CAV_W, CAV_L, CAV_R), 0.0, PHONE_H)

    hx, hy = OUT_W / 2, OUT_L / 2
    cx0, cy0 = hx - CORNER_X, hy - CORNER_Y       # inner edge of corner region

    # ---- corner cap (built at +X,+Y, mirrored for the others) -------------
    region = box(cx0, hx + 1, cy0, hy + 1, Z_BACK - 1, Z_TOP + 1)
    cap = outer ^ region
    cap = cap - cavity
    window = prism(rrect(CAV_W - 2 * FRONT_LIP, CAV_L - 2 * FRONT_LIP, CAV_R - FRONT_LIP), 0.0, Z_TOP + 1)
    cap = cap - window
    # tongue pockets (receive shell tongues): top wall + side wall
    t_top = box(cx0 - 0.5, cx0 + 3.0 + GAP, hy - 1.5, hy - 0.5, 2.0 - GAP, 7.0 + GAP)
    t_side = box(hx - 1.5, hx - 0.5, cy0 - 0.5, cy0 + 3.0 + GAP, 1.0 - GAP, 3.5 + GAP)
    cap = cap - t_top - t_side

    # ---- shell -------------------------------------------------------------
    shell = outer - cavity
    # lower the long-edge walls to a low lip (keeps buttons / sides exposed)
    side_cut = box(-hx - 1, hx + 1, -hy - 1, hy + 1, SIDE_LIP_H, Z_TOP + 1) \
        - box(-cx0, cx0, -hy - 1, hy + 1, SIDE_LIP_H - 1, Z_TOP + 2)   # keep top/bottom walls
    shell = shell - side_cut
    # top / bottom walls stop at phone height (no lip on the shell)
    shell = shell - box(-hx - 1, hx + 1, -hy - 1, hy + 1, PHONE_H, Z_TOP + 1)

    # camera opening + raised bezel
    cam_cy = hy - CAM_TOP_MARGIN - CAM_H / 2
    bezel = prism(rrect(CAM_W + 2 * BEZEL_W, CAM_H + 2 * BEZEL_W, CAM_R + BEZEL_W, 0, cam_cy),
                  Z_BACK - BEZEL_H, 0.0)
    shell = shell + bezel
    shell = shell - prism(rrect(CAM_W, CAM_H, CAM_R, 0, cam_cy), Z_BACK - BEZEL_H - 1, 1.0)

    # MagSafe ring groove on the inside floor
    ring = prism(CrossSection.circle(MAGSAFE_OD / 2, SEG) - CrossSection.circle(MAGSAFE_ID / 2, SEG),
                 0.0 - MAGSAFE_DEPTH, 0.01).translate((0, MAGSAFE_Y, 0))
    pill = prism(rrect(6.0, 14.0, 3.0), -MAGSAFE_DEPTH, 0.01).translate((0, MAGSAFE_Y - 36.0, 0))
    shell = shell - ring - pill

    # open the four corner regions (with clearance) and add tongues
    for sx in (1, -1):
        for sy in (1, -1):
            r = box(cx0 - GAP, hx + 1, cy0 - GAP, hy + 1, Z_BACK - 1, Z_TOP + 1)
            r = r.mirror((1, 0, 0)) if sx < 0 else r
            r = r.mirror((0, 1, 0)) if sy < 0 else r
            shell = shell - r
    for sx in (1, -1):
        for sy in (1, -1):
            tt = box(cx0 - GAP - 0.01, cx0 + 3.0, hy - 1.4, hy - 0.6, 2.0, 7.0)
            ts = box(hx - 1.4, hx - 0.6, cy0 - GAP - 0.01, cy0 + 3.0, 1.0, 3.5)
            for t in (tt, ts):
                t = t.mirror((1, 0, 0)) if sx < 0 else t
                t = t.mirror((0, 1, 0)) if sy < 0 else t
                shell = shell + t

    caps = {}
    for name, sx, sy in (("top_right", 1, 1), ("top_left", -1, 1),
                         ("bottom_right", 1, -1), ("bottom_left", -1, -1)):
        c = cap
        if sx < 0:
            c = c.mirror((1, 0, 0))
        if sy < 0:
            c = c.mirror((0, 1, 0))
        caps[name] = c
    return shell, caps


def main():
    shell, caps = build()
    os.makedirs("stl", exist_ok=True)
    meshes = {"shell_blue": to_mesh(shell)}
    for k, c in caps.items():
        meshes[f"corner_{k}_orange"] = to_mesh(c)
    for k, m in meshes.items():
        m.export(f"stl/{k}.stl")
        print(f"{k:28s} watertight={m.is_watertight} vol={m.volume:9.1f} mm3 "
              f"size={np.round(m.extents, 2)}")
    asm = trimesh.util.concatenate(list(meshes.values()))
    asm.export("stl/iphone17_promax_exoskeleton_assembled.stl")
    print("assembled size", np.round(asm.extents, 2))


if __name__ == "__main__":
    main()
