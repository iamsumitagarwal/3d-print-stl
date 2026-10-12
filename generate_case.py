#!/usr/bin/env python3
"""Parametric iPhone 17 Pro Max "exoskeleton" case - single printable piece.

Shape (mm; case back faces -Z, phone screen faces +Z, camera end at +Y):
  * flat back plate with chamfered edges and only a low lip along all four
    edges (sides, buttons, ports stay exposed - as in the reference photos)
  * four full-height wrap-around corner bumpers, slightly proud of the plate
    outline, with a front lip that holds the phone
  * tall raised rim around a large rounded camera opening
  * MagSafe ring + alignment pill grooves on the inside floor
  * bold debossed LOGO_TEXT on the back, reading bottom-to-top

Outer size, cavity size, back-plate thickness and overall height are taken from
the reference STL (82.6 x 168.0 mm, cavity ~79 x 164, plate z -3..0, top 10.95).

Usage: python3 generate_case.py   ->  writes ./stl/iphone17_promax_exoskeleton_case.stl
Requires: numpy, trimesh, manifold3d, shapely, matplotlib (font outlines)
"""
import os
import numpy as np
import trimesh
from manifold3d import Manifold, CrossSection, JoinType

SEG = 96

# ---- dimensions (from reference STL) --------------------------------------
OUT_W, OUT_L = 82.6, 168.0      # outer footprint (at the corner bumpers)
OUT_R = 13.6                    # outer plan corner radius
CAV_W, CAV_L = 79.0, 164.0      # phone cavity (phone is 78.0 x 163.4 x 8.75)
CAV_R = 11.6
Z_BACK = -3.0                   # outside face of back plate
Z_TOP = 10.95                   # top of corner bumpers
PHONE_H = 8.9                   # cavity height under the corner lips

# ---- design choices matched to the photos ---------------------------------
EDGE_LIP_H = 1.2                # low lip along the edges between the corners
PLATE_INSET = 0.6               # plate outline sits inside the bumper outline
CHAMFER = 0.8                   # edge chamfer on plate and bumpers
CORNER_X, CORNER_Y = 15.0, 21.0 # bumper extent along short / long edge
FRONT_LIP = 2.0                 # bumper lip overlapping the screen edge
CAM_W, CAM_H = 67.0, 41.0       # camera opening
CAM_R = 8.0
CAM_TOP_MARGIN = 8.0            # case top edge -> opening
RIM_W, RIM_H = 2.5, 3.5         # raised camera rim (width, height above back)
LOGO_TEXT = "SUMIT"
LOGO_HEIGHT, LOGO_DEPTH = 8.0, 1.2   # cap height, deboss depth (plate is 3 mm)
LOGO_POS = (-29.0, -44.0)       # text centre; -X is the right side seen from the back
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


def chamfered(w, l, r, z0, z1, c_bot=CHAMFER, c_top=CHAMFER, cx=0.0, cy=0.0):
    """Rounded-rect slab with 45-degree chamfers on the bottom and/or top edge."""
    parts = [prism(rrect(w, l, r, cx, cy), z0 + c_bot, z1 - c_top)]
    if c_bot:
        parts.append(prism(rrect(w - 2 * c_bot, l - 2 * c_bot, r - c_bot, cx, cy), z0, z0 + 0.01))
    if c_top:
        parts.append(prism(rrect(w - 2 * c_top, l - 2 * c_top, r - c_top, cx, cy), z1 - 0.01, z1))
    return Manifold.batch_hull(parts)


def text_solid(text, height, depth, cx, cy):
    """Debossing tool: text as seen from the back (-Z), rotated to read upward."""
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    import shapely.geometry as sg
    import shapely.affinity as sa
    tp = TextPath((0, 0), text, size=height,
                  prop=FontProperties(family="DejaVu Sans", weight="bold"))
    polys = [sg.Polygon(p) for p in tp.to_polygons()]
    geom = polys[0]
    for p in polys[1:]:                                  # even-odd: letter holes
        geom = geom.symmetric_difference(p)
    geom = sa.rotate(geom, 90, origin=(0, 0))            # read bottom-to-top
    minx, miny, maxx, maxy = geom.bounds
    geom = sa.translate(geom, -(minx + maxx) / 2, -(miny + maxy) / 2)
    geom = sa.scale(geom, xfact=-1, yfact=1, origin=(0, 0))  # back view is X-mirrored
    tool = None
    for g in (geom.geoms if hasattr(geom, "geoms") else [geom]):
        g = sg.polygon.orient(g, 1.0)                    # exterior CCW, holes CW
        cs = CrossSection([np.array(g.exterior.coords)[:-1]] +
                          [np.array(i.coords)[:-1] for i in g.interiors])
        m = prism(cs, Z_BACK - 0.01, Z_BACK + depth)
        tool = m if tool is None else tool + m
    return tool.translate((cx, cy, 0))


def to_mesh(m):
    mm = m.to_mesh()
    return trimesh.Trimesh(np.asarray(mm.vert_properties)[:, :3],
                           np.asarray(mm.tri_verts), process=True)


def build():
    hx, hy = OUT_W / 2, OUT_L / 2
    cavity = prism(rrect(CAV_W, CAV_L, CAV_R), 0.0, PHONE_H)
    window = prism(rrect(CAV_W - 2 * FRONT_LIP, CAV_L - 2 * FRONT_LIP, CAV_R - FRONT_LIP),
                   0.0, Z_TOP + 1)

    # back plate + low edge lip (slightly inset from the bumper outline)
    pw, pl, pr = OUT_W - 2 * PLATE_INSET, OUT_L - 2 * PLATE_INSET, OUT_R - PLATE_INSET
    body = chamfered(pw, pl, pr, Z_BACK, EDGE_LIP_H, CHAMFER, 0.4)

    # full-height corner bumpers
    bumper_shell = chamfered(OUT_W, OUT_L, OUT_R, Z_BACK, Z_TOP)
    for sx in (1, -1):
        for sy in (1, -1):
            region = box(hx - CORNER_X, hx + 1, hy - CORNER_Y, hy + 1, Z_BACK - 1, Z_TOP + 1)
            if sx < 0:
                region = region.mirror((1, 0, 0))
            if sy < 0:
                region = region.mirror((0, 1, 0))
            body = body + (bumper_shell ^ region)

    body = body - cavity - window

    # camera opening with tall raised rim on the back
    cam_cy = hy - CAM_TOP_MARGIN - CAM_H / 2
    rim = chamfered(CAM_W + 2 * RIM_W, CAM_H + 2 * RIM_W, CAM_R + RIM_W,
                    Z_BACK - RIM_H, Z_BACK + 0.5, 0.6, 0.0, 0, cam_cy)
    body = body + rim
    body = body - prism(rrect(CAM_W, CAM_H, CAM_R, 0, cam_cy), Z_BACK - RIM_H - 1, 1.0)

    # MagSafe ring + alignment pill grooves on the inside floor
    ring = prism(CrossSection.circle(MAGSAFE_OD / 2, SEG) - CrossSection.circle(MAGSAFE_ID / 2, SEG),
                 -MAGSAFE_DEPTH, 0.01).translate((0, MAGSAFE_Y, 0))
    pill_out = rrect(7.0, 16.0, 3.5, 0, MAGSAFE_Y - 38.0)
    pill = prism(pill_out - rrect(5.0, 14.0, 2.5, 0, MAGSAFE_Y - 38.0), -MAGSAFE_DEPTH, 0.01)
    body = body - ring - pill

    # bold debossed logo on the back
    body = body - text_solid(LOGO_TEXT, LOGO_HEIGHT, LOGO_DEPTH, *LOGO_POS)
    return body


def main():
    solid = build()
    case = to_mesh(solid)
    os.makedirs("stl", exist_ok=True)
    out = "stl/iphone17_promax_exoskeleton_case.stl"
    case.export(out)
    print(f"{out}: watertight={case.is_watertight} bodies={len(solid.decompose())} "
          f"vol={case.volume:.1f} mm3 size={np.round(case.extents, 2)}")


if __name__ == "__main__":
    main()
