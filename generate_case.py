#!/usr/bin/env python3
"""Parametric iPhone 17 Pro Max "exoskeleton" case: skeleton + 4 clip-on corners.

Construction follows the reference product photos (contours measured from them):
  skeleton  - flat back plate with chamfered edges and a low 1.2 mm edge lip;
              tall raised rim around a large rounded camera opening; MagSafe
              ring + pill grooves inside; bold debossed LOGO_TEXT on the back;
              at both ends of every corner a full-height clip post with a
              rectangular window; plate corners notched so the caps wrap onto
              the back face.
  corner_*  - four curved L-shaped corner caps (16 x 22 mm in plan), full case
              height, front lip over the screen, back flange that fills the
              plate notch; each arm slides over a post and a snap bump clicks
              into the post window.

Mm units; case back faces -Z, screen faces +Z, camera end at +Y. All fit-critical
geometry is measured from the reference STL: outer 82.6 x 168.0 mm, phone pocket
78.97 x 164.43 mm R 14.72, plate z -3..0, top 10.95, corner-arc screen lip (45 deg
from z 8.2 to 2 mm), camera opening 75.57 x 45.99 mm R 12. validate_fit.py
checks the result against the reference.

Usage: python3 generate_case.py   ->  writes ./stl/*.stl
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
# Phone pocket, fitted to the reference at every height (max deviation 0.03 mm):
# 78.97 x 164.43 mm, R 14.72 -> ~0.5 mm clearance around a 78.0 x 163.4 mm phone.
CAV_W, CAV_L = 78.97, 164.43
CAV_R = 14.72
Z_BACK = -3.0                   # outside face of back plate (pocket floor is z = 0)
Z_TOP = 10.95                   # top of corner bumpers
# Screen lip, as in the reference: only along the four corner arcs, a 45-degree
# chamfer starting at LIP_Z0 that reaches LIP_DEPTH at LIP_Z0 + LIP_DEPTH.
LIP_Z0, LIP_DEPTH = 8.2, 2.0
POST_H = LIP_Z0 - 0.25          # clip post height (relief stays below the lip)
PORT_HALF_W = 24.9              # bottom edge kept free for |x| < this (ports / speakers)

# ---- design choices matched to the photos ---------------------------------
EDGE_LIP_H = 1.2                # low lip along the edges between the corners
CHAMFER = 0.8                   # edge chamfer on plate and bumpers
CORNER_X, CORNER_Y = 16.0, 22.0 # corner cap extent along short / long edge (photo contours)
GAP = 0.15                      # print clearance skeleton <-> corner cap
POST_T = 0.9                    # post thickness where the cap slides over it
POST_OUT, POST_IN = 4.0, 6.0    # post length visible / hidden under the cap arm
WIN_L, WIN_H, WIN_Z0 = 4.0, 4.0, 2.2   # clip window in each post
SEAT_W = 2.5                    # width of the cap band that wraps onto the back face
BUMP_C = 0.25                   # snap bump clearance inside the window
CAM_W, CAM_H = 75.57, 45.99     # camera opening (from reference)
CAM_R = 12.0
CAM_CY = 57.80                  # opening centre (y 34.80 .. 80.79)
CAM_RELIEF, CAM_RELIEF_D = 1.2, 1.0   # inside relief around the opening (width, depth)
RIM_W, RIM_H = 2.0, 3.5         # raised camera rim (width, height above back)
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


def phone_void():
    """Space for the phone (z >= 0): the pocket, closed in by the corner-arc lip."""
    hw, hl = CAV_W / 2 - CAV_R, CAV_L / 2 - CAV_R      # arc centres
    top = Z_TOP + 1
    void = box(-hw, hw, -CAV_L / 2, CAV_L / 2, 0, top) + box(-CAV_W / 2, CAV_W / 2, -hl, hl, 0, top)
    r0, r1 = CAV_R, CAV_R - LIP_DEPTH
    z1 = LIP_Z0 + LIP_DEPTH
    arc = (Manifold.cylinder(LIP_Z0, r0, r0, SEG)
           + Manifold.cylinder(z1 - LIP_Z0, r0, r1, SEG).translate((0, 0, LIP_Z0))
           + Manifold.cylinder(top - z1, r1, r1, SEG).translate((0, 0, z1)))
    for sx in (1, -1):
        for sy in (1, -1):
            void = void + arc.translate((sx * hw, sy * hl, 0))
    return void


def corner_box(sx, sy, x0, x1, y0, y1, z0, z1):
    """Box given in +X,+Y corner coordinates, mirrored into corner (sx, sy)."""
    m = box(x0, x1, y0, y1, z0, z1)
    if sx < 0:
        m = m.mirror((1, 0, 0))
    if sy < 0:
        m = m.mirror((0, 1, 0))
    return m


def build():
    hx, hy = OUT_W / 2, OUT_L / 2
    cx0, cy0 = hx - CORNER_X, hy - CORNER_Y          # where the cap arms end
    cav_cs = rrect(CAV_W, CAV_L, CAV_R)
    cavity = phone_void()

    def band(off0, off1, z0, z1):
        """Ring between cavity outline offset by off0 and off1 (mm outward)."""
        cs = rrect(CAV_W + 2 * off1, CAV_L + 2 * off1, CAV_R + off1)
        cs = cs - rrect(CAV_W + 2 * off0, CAV_L + 2 * off0, CAV_R + off0)
        return prism(cs, z0, z1)

    wall = chamfered(OUT_W, OUT_L, OUT_R, 0.0 - 1, Z_TOP, 0.0, CHAMFER) - cavity  # full wall ring
    wall_s = wall ^ box(-hx - 1, hx + 1, -hy - 1, hy + 1, -0.5, Z_TOP + 1)  # skeleton: overlaps plate
    wall = wall ^ box(-hx - 1, hx + 1, -hy - 1, hy + 1, 0.0, Z_TOP + 1)

    # ---- skeleton: back plate + low edge lip + clip posts -----------------
    skel = chamfered(OUT_W, OUT_L, OUT_R, Z_BACK, 0.0, CHAMFER, 0.0)
    lip = wall_s ^ box(-hx - 1, hx + 1, -hy - 1, hy + 1, -0.5, EDGE_LIP_H)
    for sx in (1, -1):
        for sy in (1, -1):
            lip = lip - corner_box(sx, sy, cx0, hx + 1, cy0, hy + 1, -1, Z_TOP + 1)
    lip = lip - box(-PORT_HALF_W, PORT_HALF_W, -hy - 1, -hy + 6, -0.4, Z_TOP + 1)
    skel = skel + lip

    thin = band(-0.01, POST_T, -0.5, POST_H)           # inner part of the wall under the cap
    cap_clear = band(-0.05, POST_T + GAP, -0.5, POST_H + GAP) \
        + band(-SEAT_W - 1, POST_T + GAP, Z_BACK - 1, 0.0)
    for sx in (1, -1):
        for sy in (1, -1):
            for long_edge in (True, False):
                if long_edge:   # post on the long (side) edge, around y = cy0
                    exposed = corner_box(sx, sy, hx - 6, hx + 1, cy0 - POST_OUT, cy0, -1, POST_H)
                    under = corner_box(sx, sy, hx - 6, hx + 1, cy0, cy0 + POST_IN, -1, POST_H)
                    hole = corner_box(sx, sy, hx - 6, hx + 1, cy0 + 1.0, cy0 + 1.0 + WIN_L,
                                      WIN_Z0, WIN_Z0 + WIN_H)
                else:           # post on the short (top/bottom) edge, around x = cx0
                    exposed = corner_box(sx, sy, cx0 - POST_OUT, cx0, hy - 6, hy + 1, -1, POST_H)
                    under = corner_box(sx, sy, cx0, cx0 + POST_IN, hy - 6, hy + 1, -1, POST_H)
                    hole = corner_box(sx, sy, cx0 + 1.0, cx0 + 1.0 + WIN_L, hy - 6, hy + 1,
                                      WIN_Z0, WIN_Z0 + WIN_H)
                if not long_edge and sy < 0:
                    exposed = None      # keep the bottom edge clear for the ports
                post = (thin ^ under) if exposed is None else (wall_s ^ exposed) + (thin ^ under)
                skel = skel + (post - hole)

    # zones where the cap arms slide over the posts (+X,+Y corner coordinates)
    post_zone = (box(hx - 6, hx + 1, cy0 - 1, cy0 + POST_IN + GAP, Z_BACK - 1, Z_TOP + 1)
                 + box(cx0 - 1, cx0 + POST_IN + GAP, hy - 6, hy + 1, Z_BACK - 1, Z_TOP + 1))
    foot = POST_T + 0.05
    post_foot = prism(rrect(CAV_W + 2 * foot, CAV_L + 2 * foot, CAV_R + foot),
                      Z_BACK - 1, 0.5) ^ post_zone            # plate kept under the posts
    seat_cs = rrect(OUT_W - 2 * SEAT_W, OUT_L - 2 * SEAT_W, OUT_R - SEAT_W)

    # notch the plate corners so the caps wrap onto the back face
    notch = (box(cx0, hx + 1, cy0, hy + 1, Z_BACK - 1, 0.2)
             - prism(seat_cs.offset(-GAP, JoinType.Round, 2.0, SEG), Z_BACK - 2, 1)
             - post_foot)
    for sx in (1, -1):
        for sy in (1, -1):
            n = notch.mirror((1, 0, 0)) if sx < 0 else notch
            skel = skel - (n.mirror((0, 1, 0)) if sy < 0 else n)

    # ---- corner caps ------------------------------------------------------
    cap_box = box(cx0 + GAP, hx + 1, cy0 + GAP, hy + 1, Z_BACK - 1, Z_TOP + 1)
    cap0 = wall ^ cap_box
    flange_slab = chamfered(OUT_W, OUT_L, OUT_R, Z_BACK, 0.3, CHAMFER, 0.0)  # overlaps the wall
    flange = (flange_slab ^ cap_box) - prism(seat_cs, Z_BACK - 2, 0.0) - cavity
    cap0 = cap0 + flange
    # relieve the cap where it slides over the posts, add snap bumps for the windows
    relief = cap_clear ^ post_zone
    cap0 = cap0 - relief
    bump_ring = band(0.3, POST_T + GAP + 0.01, WIN_Z0 + BUMP_C, WIN_Z0 + WIN_H - BUMP_C)
    bumps = bump_ring ^ (box(hx - 6, hx + 1, cy0 + 1.0 + BUMP_C, cy0 + 1.0 + WIN_L - BUMP_C, -1, Z_TOP)
                         + box(cx0 + 1.0 + BUMP_C, cx0 + 1.0 + WIN_L - BUMP_C, hy - 6, hy + 1, -1, Z_TOP))
    cap0 = cap0 + bumps
    caps = {}
    for name, sx, sy in (("top_right", 1, 1), ("top_left", -1, 1),
                         ("bottom_right", 1, -1), ("bottom_left", -1, -1)):
        c = cap0.mirror((1, 0, 0)) if sx < 0 else cap0
        caps[name] = c.mirror((0, 1, 0)) if sy < 0 else c

    # ---- camera opening with tall raised rim ------------------------------
    cam_cy = CAM_CY
    rim_cs = rrect(CAM_W + 2 * RIM_W, CAM_H + 2 * RIM_W, CAM_R + RIM_W, 0, cam_cy)
    rim = chamfered(CAM_W + 2 * RIM_W, CAM_H + 2 * RIM_W, CAM_R + RIM_W,
                    Z_BACK - RIM_H, Z_BACK, 0.6, 0.0, 0, cam_cy)
    # fuse into the plate only where the plate is (not under the corner caps)
    rim = rim + prism(rim_cs.offset(-0.2, JoinType.Round, 2.0, SEG) ^ seat_cs.offset(-GAP - 0.3, JoinType.Round, 2.0, SEG), Z_BACK - 0.3, Z_BACK + 0.5)
    skel = skel + rim
    skel = skel - prism(rrect(CAM_W, CAM_H, CAM_R, 0, cam_cy), Z_BACK - RIM_H - 1, 1.0)
    skel = skel - prism(rrect(CAM_W + 2 * CAM_RELIEF, CAM_H + 2 * CAM_RELIEF, CAM_R + CAM_RELIEF,
                              0, cam_cy), -CAM_RELIEF_D, 1.0)

    # ---- MagSafe grooves + logo -------------------------------------------
    ring = prism(CrossSection.circle(MAGSAFE_OD / 2, SEG) - CrossSection.circle(MAGSAFE_ID / 2, SEG),
                 -MAGSAFE_DEPTH, 0.01).translate((0, MAGSAFE_Y, 0))
    pill = prism(rrect(7.0, 16.0, 3.5, 0, MAGSAFE_Y - 38.0) - rrect(5.0, 14.0, 2.5, 0, MAGSAFE_Y - 38.0),
                 -MAGSAFE_DEPTH, 0.01)
    skel = skel - ring - pill
    skel = skel - text_solid(LOGO_TEXT, LOGO_HEIGHT, LOGO_DEPTH, *LOGO_POS)
    return skel, caps


def to_trimesh_all(skel, caps):
    meshes = {"skeleton": to_mesh(skel)}
    for k, c in caps.items():
        meshes[f"corner_{k}"] = to_mesh(c)
    return meshes


def main():
    skel, caps = build()
    os.makedirs("stl", exist_ok=True)
    for f in os.listdir("stl"):
        os.remove(os.path.join("stl", f))
    for k, solid in [("skeleton", skel)] + [(f"corner_{k}", c) for k, c in caps.items()]:
        m = to_mesh(solid)
        m.export(f"stl/{k}.stl")
        print(f"{k:22s} watertight={m.is_watertight} bodies={len(solid.decompose())} "
              f"vol={m.volume:8.1f} mm3 size={np.round(m.extents, 2)}")
    clash = sum((skel ^ c).volume() for c in caps.values())
    print("skeleton/corner overlap volume:", round(clash, 4))
    asm = trimesh.util.concatenate([to_mesh(skel)] + [to_mesh(c) for c in caps.values()])
    asm.export("stl/assembled_preview.stl")


if __name__ == "__main__":
    main()
