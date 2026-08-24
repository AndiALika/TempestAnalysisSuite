"""
tempest_room.py
==============
Room-level TEMPEST assessment for the analysis suite.  Builds on the pure
primitives in ``tempest_physics`` (emission field, aperture SE, SE combination)
to add the facility-assessment layer:

* **Per-wall shielding** with **window/door apertures** — each of the four walls
  has its own SE, and a window or doorway is treated as an aperture that combines
  with (and usually dominates) that wall's bulk shielding.
* **Inspectable space / eavesdropper analysis** — the field that escapes to a
  controlled-perimeter standoff or to an explicit adversary position, with the
  correct wall attenuation along the device→receiver path, and a PASS/FAIL
  against the detectability floor.
* **Design solver** — inverts the model to give the extra wall SE, the minimum
  controlled-perimeter standoff, and a shielding-material recommendation needed
  to make the room pass.

GUI-free and unit-tested, like the other core modules.

References
----------
[1] NATO SDIP-27 / former AMSG 720B — emanation-security zoning & inspectable
    space concept.
[2] H. W. Ott, *Electromagnetic Compatibility Engineering*, Wiley, 2009.
"""

from __future__ import annotations

import numpy as np

import tempest_physics as tp

WALLS = ("W", "E", "S", "N")          # x=0, x=W, y=0, y=L
WALL_NAMES = {"W": "West", "E": "East", "S": "South", "N": "North"}


# ─────────────────────────────────────────────────────────────────────────────
#  Wall shielding (bulk + aperture)
# ─────────────────────────────────────────────────────────────────────────────
def wall_effective_se(wall_se_db, aperture=None, freq_hz=1e8):
    """Effective SE of a wall, combining its bulk SE with a window/door aperture.

    ``aperture`` (or ``None``) is a dict with keys ``length``, ``width`` [m],
    optional ``depth`` (default 0.1 m, a typical wall), ``count`` and ``shape``.
    The aperture SE is evaluated at ``freq_hz`` and combined as a parallel path.
    """
    if not aperture:
        return float(wall_se_db)
    ap = tp.aperture_se_db(freq_hz, aperture["length"], aperture["width"],
                           aperture.get("depth", 0.10), aperture.get("count", 1),
                           aperture.get("shape", "rectangular"))
    return float(tp.combine_se_db(np.array([float(wall_se_db)]),
                                  np.array([float(ap)]))[0])


def effective_walls(wall_se, apertures=None, freq_hz=1e8):
    """Map side → effective SE for all four walls.

    ``wall_se`` maps side → bulk SE [dB]; ``apertures`` maps side → aperture spec.
    """
    apertures = apertures or {}
    return {s: wall_effective_se(wall_se.get(s, 0.0), apertures.get(s), freq_hz)
            for s in WALLS}


# ─────────────────────────────────────────────────────────────────────────────
#  Geometry
# ─────────────────────────────────────────────────────────────────────────────
def exit_wall(room_w, room_l, inside, outside):
    """Which wall side the segment inside→outside crosses, or ``None``."""
    x0, y0 = inside
    x1, y1 = outside
    dx, dy = x1 - x0, y1 - y0
    cands = []
    if dx > 0:
        t = (room_w - x0) / dx
        if t > 0 and 0 <= y0 + t * dy <= room_l:
            cands.append((t, "E"))
    elif dx < 0:
        t = (0.0 - x0) / dx
        if t > 0 and 0 <= y0 + t * dy <= room_l:
            cands.append((t, "W"))
    if dy > 0:
        t = (room_l - y0) / dy
        if t > 0 and 0 <= x0 + t * dx <= room_w:
            cands.append((t, "N"))
    elif dy < 0:
        t = (0.0 - y0) / dy
        if t > 0 and 0 <= x0 + t * dx <= room_w:
            cands.append((t, "S"))
    return min(cands)[1] if cands else None


def _perimeter_points(room_w, room_l, standoff, n=240):
    """Points sampled around the controlled perimeter (rectangle ``standoff`` m
    outside the room walls)."""
    s = float(standoff)
    x0, y0, x1, y1 = -s, -s, room_w + s, room_l + s
    nx = max(2, n // 4)
    ny = max(2, n // 4)
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    pts = [(x, y0) for x in xs] + [(x, y1) for x in xs]
    pts += [(x0, y) for y in ys] + [(x1, y) for y in ys]
    return pts


# ─────────────────────────────────────────────────────────────────────────────
#  Field escaping the room
# ─────────────────────────────────────────────────────────────────────────────
def field_at_point(devices, eff_walls, room_w, room_l, point):
    """Combined leaked field [dBµV/m] at an external ``point``.

    Each device's free-space field at the point is attenuated by the SE of the
    wall its path crosses; contributions add in the linear (field) domain.
    Returns ``(field_dbuv, contributions)``.
    """
    leak = 0.0
    contributions = []
    for d in devices:
        dist = float(np.hypot(point[0] - d["x"], point[1] - d["y"]))
        e_free = float(tp.field_at_distance_dbuv(d["type"], d["power"], dist))
        wall = exit_wall(room_w, room_l, (d["x"], d["y"]), point)
        se = eff_walls.get(wall, 0.0) if wall else 0.0
        leaked = e_free - se
        leak += 10.0 ** (leaked / 20.0)
        contributions.append({**d, "dist": dist, "wall": wall,
                              "e_free": e_free, "leaked": leaked})
    field = 20.0 * np.log10(leak) if leak > 0 else float("-inf")
    return field, contributions


def perimeter_worst_case(devices, eff_walls, room_w, room_l, standoff, n=240):
    """Worst (highest) leaked field on the controlled perimeter and its point."""
    best_f, best_p = float("-inf"), None
    for p in _perimeter_points(room_w, room_l, standoff, n):
        f, _ = field_at_point(devices, eff_walls, room_w, room_l, p)
        if f > best_f:
            best_f, best_p = f, p
    return best_f, best_p


# ─────────────────────────────────────────────────────────────────────────────
#  Assessment
# ─────────────────────────────────────────────────────────────────────────────
def assess(devices, wall_se, room_w, room_l, standoff, apertures=None,
           freq_hz=1e8, eavesdropper=None):
    """Full inspectable-space assessment.

    Returns a dict with ``eff_walls``, ``worst_field``/``worst_point`` on the
    perimeter, the detectability ``floor``, ``margin`` (dB above floor),
    ``passed``, a 0–100 ``risk`` score, and — if ``eavesdropper`` is given —
    ``eaves_field``/``eaves_margin``/``eaves_contribs`` at that point.
    """
    eff = effective_walls(wall_se, apertures, freq_hz)
    worst_field, worst_pt = perimeter_worst_case(
        devices, eff, room_w, room_l, standoff) if devices else (float("-inf"), None)
    floor = tp.DETECTION_FLOOR_DBUV
    margin = worst_field - floor
    out = {"eff_walls": eff, "worst_field": worst_field, "worst_point": worst_pt,
           "floor": floor, "margin": margin, "passed": worst_field <= floor,
           "risk": float(np.clip((margin + 20.0) / 40.0 * 100.0, 0.0, 100.0))}
    if eavesdropper is not None:
        ef, contribs = field_at_point(devices, eff, room_w, room_l, eavesdropper)
        out["eaves_field"] = ef
        out["eaves_margin"] = ef - floor
        out["eaves_contribs"] = contribs
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Design solver
# ─────────────────────────────────────────────────────────────────────────────
def required_standoff(devices, wall_se, room_w, room_l, apertures=None,
                      freq_hz=1e8, max_standoff=300.0):
    """Smallest controlled-perimeter standoff [m] at which the worst leaked field
    drops to the detectability floor, or ``None`` if unattainable."""
    eff = effective_walls(wall_se, apertures, freq_hz)
    floor = tp.DETECTION_FLOOR_DBUV
    worst = lambda s: perimeter_worst_case(devices, eff, room_w, room_l, s)[0]
    if not devices or worst(0.1) <= floor:
        return 0.1
    if worst(max_standoff) > floor:
        return None
    lo, hi = 0.1, max_standoff
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if worst(mid) <= floor:
            hi = mid
        else:
            lo = mid
    return hi


def recommend_material(target_se_db, f_lo=1e6, f_hi=1e9, t_max_mm=6.0):
    """Cheapest material (by real cost per m²) meeting ``target_se_db`` across the
    band within ``t_max_mm``, or ``None``."""
    best = None
    for name, m in tp.MATERIALS.items():
        t = tp.min_thickness_for_se(m["sigma"], m["mu_r"], target_se_db,
                                    f_lo, f_hi, t_max_mm / 1000.0)
        if t is None:
            continue
        cand = {"material": name, "thickness_mm": t * 1000.0,
                "cost_per_m2": m["price_per_kg"] * m["density"] * t}   # USD/m²
        if best is None or cand["cost_per_m2"] < best["cost_per_m2"]:
            best = cand
    return best


def design_recommendations(devices, wall_se, room_w, room_l, standoff,
                           apertures=None, freq_hz=1e8):
    """Actionable design output: extra wall SE, required standoff and a material
    recommendation to bring the room into compliance."""
    res = assess(devices, wall_se, room_w, room_l, standoff, apertures, freq_hz)
    rec = {"passed": res["passed"], "worst_field": res["worst_field"],
           "worst_point": res["worst_point"], "margin": res["margin"]}
    if res["passed"] or res["worst_point"] is None:
        return rec

    add_se = res["margin"]                       # raising all walls by margin passes
    _, contribs = field_at_point(devices, res["eff_walls"], room_w, room_l,
                                 res["worst_point"])
    dom = max(contribs, key=lambda c: c["leaked"])
    wall = dom["wall"]
    cur_eff = res["eff_walls"].get(wall, 0.0)
    rec.update(offending_wall=wall, offending_device=dom["type"],
               current_wall_se=cur_eff, target_wall_se=cur_eff + add_se,
               add_wall_se_db=add_se,
               aperture_limited=bool(apertures and apertures.get(wall)),
               required_standoff_m=required_standoff(
                   devices, wall_se, room_w, room_l, apertures, freq_hz),
               material=recommend_material(cur_eff + add_se))
    return rec


__all__ = [
    "WALLS", "WALL_NAMES",
    "wall_effective_se", "effective_walls", "exit_wall",
    "field_at_point", "perimeter_worst_case",
    "assess", "required_standoff", "recommend_material", "design_recommendations",
]
