# annotate3d.py — Manual 3D cattle box annotator (no detector output needed)
# Run:  pip install flask numpy opencv-python    # opencv only for MP4 export
#       MMCows (4 cams, floorplan available):
#       python annotate3d.py --dataset /path/to/mmcows_images \
#           --calibration_json output_barn_multi/camera_calibration.json \
#           --floorplan_json output_barn_multi/floorplan.json \
#           --review_dir output_annotations/mmcows
#       JerCCows (10 cams, NO floorplan -- it is optional now):
#       python annotate3d.py --dataset /path/to/JerCCows_socialmixing \
#           --calibration_json output_jerccows/camera_calibration.json \
#           --review_dir output_annotations/jerccows \
#           --panel_cameras cam_5,cam_9,cam_11,cam_12
#       open http://127.0.0.1:5000
#
# The frame list is built from the IMAGES (dataset/cam_X/<tag>.jpg), not from
# box JSONs. The first frame starts empty; every later frame pre-fills by
# carrying forward the most recently SAVED annotation (cows move slowly).

# Run:  pip install flask numpy opencv-python      # opencv only needed for
#                                                  # the MP4 video export
...
import os, json, glob, re, math, argparse, colorsys
import shutil, subprocess, tempfile
import numpy as np
from flask import Flask, jsonify, request, send_file, render_template_string, abort

# ----------------------------------------------------------------------------
CFG = {}
app = Flask(__name__)

# Legacy fallback used only until startup has loaded the calibration file.
CAM_LIST = ["cam_1", "cam_2", "cam_3", "cam_4"]
N_CAMERA_PANELS = 4


def _camera_sort_key(name):
    """Natural camera order: cam_2 < cam_10; unknown names sort last."""
    s = str(name)
    m = re.search(r"(\d+)$", s)
    return (0, int(m.group(1)), s) if m else (1, 0, s)


def camera_names():
    """All calibrated cameras for the current dataset, in display order."""
    cams = CFG.get("all_cams")
    return list(cams) if cams else list(CAM_LIST)


def normalize_panel_cameras(cams):
    """Validate the four camera names assigned to the on-screen panels."""
    if not isinstance(cams, (list, tuple)) or len(cams) != N_CAMERA_PANELS:
        raise ValueError("exactly %d panel cameras are required" % N_CAMERA_PANELS)
    cams = [str(c).strip() for c in cams]
    known = set(camera_names())
    unknown = [c for c in cams if c not in known]
    if unknown:
        raise ValueError("unknown panel camera(s): %s (available: %s)"
                         % (", ".join(unknown), ", ".join(camera_names())))
    if len(set(cams)) != len(cams):
        raise ValueError("panel cameras must be unique")
    return cams


def panel_cameras():
    """The current four panel cameras (default: first four calibrated cams)."""
    requested = CFG.get("panel_cams")
    if requested:
        try:
            return normalize_panel_cameras(requested)
        except ValueError as e:
            print("[cameras] invalid panel selection %s: %s; using defaults"
                  % (requested, e))
    return camera_names()[:N_CAMERA_PANELS]

POSTURE_PRIORS = {
    "standing": {"length": 220.0, "width": 70.0,  "height": 145.0},
    "lying":    {"length": 180.0, "width": 80.0,  "height": 85.0},
}

# ----------------------------------------------------------------------------
# Filesystem helpers
# ----------------------------------------------------------------------------
def _parse_tag(fname):
    tag = os.path.splitext(os.path.basename(fname))[0]

    # JerCCows-style numbered frames: frame_000001.jpg -> (1, "frame_000001")
    m = re.match(r"^frame_(\d+)$", tag, re.IGNORECASE)
    if m:
        return int(m.group(1)), tag

    # MMCows-style timestamped frames: 1690271846_02-57-26.jpg
    m = re.match(r"^(\d+)_\d{2}-\d{2}-\d{2}$", tag)
    if m:
        return int(m.group(1)), tag

    # Plain numeric frame/timestamp names: 000123.jpg or 1690271846.jpg
    m = re.match(r"^(\d+)$", tag)
    if m:
        return int(m.group(1)), tag
    return None, tag


def _image_ts_index():
    """ts -> tag derived from the dataset IMAGES (dataset/cam_X/<tag>.jpg).

    Annotation mode has no detector output, so the frame universe is defined
    by the images on disk. A frame counts if ANY camera has its jpg; a camera
    missing that frame just shows a black panel (handled downstream).
    """
    idx = {}
    root = CFG.get("dataset")
    if not root:
        return idx
    for cam in camera_names():
        for f in glob.glob(os.path.join(root, cam, "*.jpg")):
            ts, tag = _parse_tag(f)
            if ts is not None:
                idx.setdefault(ts, tag)
    return idx


def build_ts_index():
    """ts -> tag: images ALONE define the frame universe. JSONs in
    boxes_dir/review_dir may only refine the tag spelling of a frame that
    already has an image -- a stale annotation file can never CREATE a
    frame (this is what kept old sessions' frames out of the UI)."""
    idx = _image_ts_index()
    for d in (CFG["boxes_dir"], CFG["review_dir"]):
        if not d or not os.path.isdir(d):
            continue
        for f in glob.glob(os.path.join(d, "*.json")):
            if os.path.basename(f).startswith("_"):
                continue
            ts, tag = _parse_tag(f)
            if ts is not None and ts in idx:   # only refine known frames
                idx[ts] = tag
    return dict(sorted(idx.items()))


def review_path(tag):
    return os.path.join(CFG["review_dir"], f"{tag}.json")


def orig_path(tag):
    return os.path.join(CFG["boxes_dir"], f"{tag}.json")


def _read_frame_file(path, ts):
    """Load a single box-json file and normalize its box flags."""
    with open(path) as f:
        data = json.load(f)
    data.setdefault("timestamp", ts)
    data.setdefault("boxes", {})
    for b in data["boxes"].values():
        b.setdefault("reviewed", False)
        b.setdefault("edited", False)
        b["z_base"] = 0.0
    return data


def load_original(tag, ts):
    """This frame's raw automated detections, or None if absent."""
    op = orig_path(tag)
    if not os.path.exists(op):
        return None
    return _read_frame_file(op, ts)


def _file_mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return -1.0


def most_recent_reviewed_prior(ts, ts_idx):
    """Nearest EARLIER timestamp that already has a saved review file.

    Returns (prior_ts, prior_data, prior_mtime), or (None, None, -1.0).
    """
    for t in sorted((t for t in ts_idx if t < ts), reverse=True):
        rp = review_path(ts_idx[t])
        if os.path.exists(rp):
            return t, _read_frame_file(rp, t), _file_mtime(rp)
    return None, None, -1.0


def load_frame(ts, tag, ts_idx=None):
    """Build the starting boxes for frame `ts`.

    Two mutually exclusive cases, decided purely by whether this frame has a
    SAVED review file (i.e. whether a human has already reviewed it):

      1. FROZEN -- this frame already has a review file. It has been reviewed
         and saved by a human, so it is returned verbatim and is NEVER
         re-propagated over. This is the key guarantee: editing (and re-saving)
         an EARLIER frame can no longer clobber the annotations of a later
         frame that was already reviewed. Fixing one cow in the past stays
         isolated to the past.

      2. NOT-YET-REVIEWED -- this frame has no review file. Build the starting
         boxes from this frame's ORIGINAL detections, carrying forward (by cow
         ID) the boxes from the most-recently-reviewed earlier frame:
           - ID in both prior-reviewed and this frame's originals -> carry the
             reviewed box forward (posture is still taken per-frame from the
             detection).
           - ID only in this frame's originals -> keep the raw detection.
           - ID only in the prior reviewed frame -> dropped.

    Because carry-forward only ever runs for frames WITHOUT a review file, an
    unreviewed frame automatically picks up upstream corrections when you visit
    it (the old "great when there are no future annotations" behaviour), while
    already-reviewed frames are left untouched.
    """
    if ts_idx is None:
        ts_idx = build_ts_index()

    # Case 1: already reviewed & saved -> EXISTING cows are frozen and returned
    # verbatim. We still merge in any explicitly copied ("pasted") cows whose ID
    # is MISSING here: injecting an absent cow can't clobber an existing
    # annotation, and is the whole point of copy-paste. Existing cows are never
    # touched, so the frozen guarantee still holds for anything already present.
    rp = review_path(tag)
    if os.path.exists(rp):
        data = _read_frame_file(rp, ts)
        overlay = load_copied_cows().get(str(ts), {})
        for ocid, ob in overlay.items():
            if ocid in data["boxes"]:
                continue
            b = dict(ob)
            b["source"] = "copied"
            b["reviewed"] = False  # pasted guess: confirm it in THIS frame
            b["edited"] = False
            b["z_base"] = 0.0
            data["boxes"][ocid] = b
        return data, "review"

    # Case 2: not annotated yet -> carry EVERY cow forward from the most
    # recently saved earlier frame. There are no detector boxes to merge:
    # cows move slowly, so the previous annotation is the starting guess,
    # and a cow only disappears when the user explicitly deletes it.
    _, prior_data, _ = most_recent_reviewed_prior(ts, ts_idx)
    prior_boxes = prior_data["boxes"] if prior_data else {}

    boxes = {}
    for cid, pb in prior_boxes.items():
        b = dict(pb)                      # center, yaw, L/W/H, posture, ...
        b["source"] = "carried_forward"
        b["reviewed"] = False             # user hasn't checked THIS frame yet
        b["edited"] = False
        b["z_base"] = 0.0
        b.pop("interpolated", None)       # a carried guess is not interp output
        boxes[cid] = b

    # ---- Manually copied ("pasted") cows (logic unchanged) -----------------
    overlay = load_copied_cows().get(str(ts), {})
    for ocid, ob in overlay.items():
        if ocid in boxes:
            continue
        b = dict(ob)
        b["source"] = "copied"
        b["reviewed"] = False
        b["edited"] = False
        b["z_base"] = 0.0
        boxes[ocid] = b

    return {"timestamp": ts, "boxes": boxes}, ("carried" if prior_boxes else "empty")


def save_frame(tag, data):
    os.makedirs(CFG["review_dir"], exist_ok=True)
    for b in data.get("boxes", {}).values():
        b["z_base"] = 0.0     # enforce snap-to-floor
        b["reviewed"] = True  # saving implies the whole frame was checked
    # A human save is a GROUND-TRUTH ANCHOR for forward re-interpolation.
    # reinterpolate_forward() writes top-level "interpolated": True, so the
    # explicit False here is what lets classify_frame_file() tell a hand-made
    # anchor apart from an interpolated in-between frame.
    data["interpolated"] = False
    with open(review_path(tag), "w") as f:
        json.dump(data, f, indent=2)

    # Once saved, the review file is authoritative for this frame (loaded
    # verbatim, never re-derived, overlay ignored). Any pasted cow is now baked
    # into that file, so drop the overlay entry -- otherwise a later deletion of
    # that cow could be undone by stale re-injection.
    ts = data.get("timestamp")
    if ts is not None:
        store = load_copied_cows()
        if str(ts) in store:
            del store[str(ts)]
            save_copied_cows(store)


def verified_file():
    return os.path.join(CFG["review_dir"], "_verified.json")


def load_verified():
    p = verified_file()
    if os.path.exists(p):
        try:
            return set(json.load(open(p)))
        except Exception:
            return set()
    return set()


def save_verified(s):
    os.makedirs(CFG["review_dir"], exist_ok=True)
    json.dump(sorted(s), open(verified_file(), "w"), indent=2)


# ----------------------------------------------------------------------------
# Copied ("pasted") cows overlay
# ----------------------------------------------------------------------------
# Cows a human explicitly copied from one timestep into another (e.g. a cow the
# detector missed in a range of frames) live here, keyed by timestamp then cow
# ID. This is deliberately SEPARATE from review files so that inserting a cow
# into an unreviewed frame does not freeze that frame or stop its other cows
# from carrying forward.
def copied_cows_file():
    return os.path.join(CFG["review_dir"], "_copied_cows.json")


def load_copied_cows():
    p = copied_cows_file()
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            return {}
    return {}


def save_copied_cows(store):
    os.makedirs(CFG["review_dir"], exist_ok=True)
    json.dump(store, open(copied_cows_file(), "w"), indent=2)

# ----------------------------------------------------------------------------
# WildTrack-style export (purely additive: reads review files, writes exports)
# ----------------------------------------------------------------------------
# Grid constants + positionID formula are copied verbatim from
# wildtrack_annotations.generate_wildtrack_json so the exported positionIDs stay
# byte-compatible with the legacy dataset loader.
EXPORT_X_MIN_CM = -879
EXPORT_X_MAX_CM = 1042
EXPORT_Y_MIN_CM = -646
EXPORT_Y_MAX_CM = 533
EXPORT_GRID_CELL_CM = 10
EXPORT_GRID_W = int((EXPORT_X_MAX_CM - EXPORT_X_MIN_CM) / EXPORT_GRID_CELL_CM)
EXPORT_GRID_H = int((EXPORT_Y_MAX_CM - EXPORT_Y_MIN_CM) / EXPORT_GRID_CELL_CM)
EMPTY_VIEW = {"xmin": -1, "ymin": -1, "xmax": -1, "ymax": -1}
EXPORT_SEQ_NUM = 1


def export_grid_bounds(floorplan=None, cams_cfg=None):
    """Return (x_min, x_max, y_min, y_max) for WildTrack position IDs.

    Precedence:
      1. --export_bounds xmin,xmax,ymin,ymax (explicit fixed grid)
      2. floorplan perimeter, when a floorplan is supplied
      3. calibrated camera-centre extents + --export_margin_cm (no floorplan)
      4. the legacy MMCows constants above
    """
    raw = CFG.get("export_bounds")
    if raw not in (None, ""):
        try:
            if isinstance(raw, str):
                vals = [float(v) for v in raw.replace(" ", "").split(",")]
            else:
                vals = [float(v) for v in raw]
            if len(vals) != 4 or vals[1] <= vals[0] or vals[3] <= vals[2]:
                raise ValueError
            return tuple(vals)
        except (TypeError, ValueError):
            print("[export] invalid --export_bounds %r; falling back" % (raw,))

    per = (floorplan or {}).get("perimeter")
    if per:
        try:
            vals = (float(per["xmin"]), float(per["xmax"]),
                    float(per["ymin"]), float(per["ymax"]))
            if vals[1] > vals[0] and vals[3] > vals[2]:
                return vals
        except (KeyError, TypeError, ValueError):
            pass

    centers = []
    for c in (cams_cfg or {}).values():
        C = c.get("center") or c.get("center_cm")
        if C and len(C) >= 2:
            try:
                centers.append((float(C[0]), float(C[1])))
            except (TypeError, ValueError):
                pass
    if centers:
        margin = float(CFG.get("export_margin_cm") or 500.0)
        xs = [p[0] for p in centers]
        ys = [p[1] for p in centers]
        return (min(xs) - margin, max(xs) + margin,
                min(ys) - margin, max(ys) + margin)

    return (float(EXPORT_X_MIN_CM), float(EXPORT_X_MAX_CM),
            float(EXPORT_Y_MIN_CM), float(EXPORT_Y_MAX_CM))


def world_to_position_id(x, y, ctx="", bounds=None):
    """(world cm) -> (positionID, was_out_of_bounds).

    Same math (and same warn-then-clamp behaviour) as the legacy script:
      grid_x = int((x - x_min)/cell), grid_y = int((y - y_min)/cell)
      position_id = grid_x * grid_height + grid_y
    `bounds` = (x_min, x_max, y_min, y_max); defaults to the legacy MMCows
    constants so any other caller is unaffected.
    """
    x_min, x_max, y_min, y_max = bounds or (
        EXPORT_X_MIN_CM, EXPORT_X_MAX_CM, EXPORT_Y_MIN_CM, EXPORT_Y_MAX_CM)
    grid_w = int((x_max - x_min) / EXPORT_GRID_CELL_CM)
    grid_h = int((y_max - y_min) / EXPORT_GRID_CELL_CM)
    gx = int((x - x_min) / EXPORT_GRID_CELL_CM)
    gy = int((y - y_min) / EXPORT_GRID_CELL_CM)
    oob = not (0 <= gx < grid_w and 0 <= gy < grid_h)
    if oob:
        print("[export] WARNING out of bounds %s: world=(%.1f, %.1f) "
              "grid=(%d, %d) bounds X=[0,%d) Y=[0,%d) -> clamped"
              % (ctx, x, y, gx, gy, grid_w, grid_h))
    gx = max(0, min(grid_w - 1, gx))
    gy = max(0, min(grid_h - 1, gy))
    return gx * grid_h + gy, oob


def extract_bbox_from_json_file(json_file_path):
    """labelme rectangle JSON -> {cow_id:int -> {xmin,ymin,xmax,ymax}}.

    Reused (almost) as-is from wildtrack_annotations.py. Missing file or a
    malformed shape is NOT an error: it simply yields no entry, which the caller
    turns into a {-1,-1,-1,-1} view.
    """
    if not os.path.exists(json_file_path):
        return {}
    try:
        with open(json_file_path, "r") as f:
            data = json.load(f)

        bbox_by_cow = {}
        for shape in data.get("shapes", []):
            if shape.get("shape_type", "") != "rectangle":
                continue
            parts = str(shape.get("label", "")).split("_")
            if len(parts) != 2:
                continue
            _action, cow_str = parts
            try:
                cow_id = int(cow_str)
            except ValueError:
                continue
            pts = shape.get("points", [])
            if len(pts) != 2:
                continue
            pt1, pt2 = pts
            bbox_by_cow[cow_id] = {
                "xmin": min(pt1[0], pt2[0]),
                "ymin": min(pt1[1], pt2[1]),
                "xmax": max(pt1[0], pt2[0]),
                "ymax": max(pt1[1], pt2[1]),
            }
        return bbox_by_cow
    except Exception as e:
        print(f"[export] error reading labelme {json_file_path}: {e}")
        return {}


def project_box_to_2d(b, cam_cfg):
    """Project one annotated 3D box into a camera -> {xmin,ymin,xmax,ymax}.

    Annotation mode has no labelme 2D boxes, so the WildTrack 'views' are
    derived from the 3D annotation itself: project the 8 cuboid corners with
    the camera's P_cm and take their 2D bounding rectangle, clamped to the
    calibrated image bounds. Returns None if the box projects to nothing
    usable (behind the camera / fully off-frame) -> caller writes -1s.
    """
    if not cam_cfg:
        return None
    pts = [project_point(cam_cfg["P_cm"], c) for c in box_corners_world(b)]
    pts = [p for p in pts if p is not None]
    if not pts:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    res = cam_cfg.get("image_res") or [None, None]   # [H, W]
    if res[1]:
        xs = [min(max(x, 0.0), float(res[1] - 1)) for x in xs]
    if res[0]:
        ys = [min(max(y, 0.0), float(res[0] - 1)) for y in ys]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return None
    return {"xmin": x0, "ymin": y0, "xmax": x1, "ymax": y1}


def labelme_path(tag, cam):
    """dataset/<cam>/<tag>.json -- same folder/tag convention as api_image."""
    return os.path.join(CFG["dataset"], cam, f"{tag}.json")


def _resolve_export_ts(name, ts_idx):
    """Accept either a frame tag ('1690271846_02-57-26') or a raw ts."""
    if name is None:
        return None
    s = str(name).strip()
    if not s:
        return None
    for ts, tag in ts_idx.items():          # exact tag match first
        if tag == s:
            return ts
    try:
        ts = int(s)
    except ValueError:
        return None
    return ts if ts in ts_idx else None

def _prepare_export_range(start, end, interval):
    """Shared, side-effect-free validation for every range export.

    Used by BOTH export_wildtrack_range() and export_video_range() so the two
    exports agree exactly on which frames are in scope and on the wording/shape
    of every error. Nothing is created or written here: callers only touch the
    filesystem after this returns success.

    Returns (ctx, None) on success, where ctx = {
        "ts_idx":   {ts -> tag},
        "ts0","ts1": resolved inclusive range endpoints,
        "interval": positive int,
        "expected": [ts, ...] on the interval grid,
        "warnings": [str, ...],
    }
    or (None, error_dict) on failure -- error_dict is JSON-ready and carries
    "missing_timestamps" / "unreviewed" exactly as before.

    Validation order (all fatal):
      1. interval must be a positive int (user-supplied: 1 or 15).
      2. start/end must resolve to known frames.
      3. every expected timestamp on the interval grid must exist in the index.
      4. every expected frame must have a SAVED review file.
    """
    ts_idx = build_ts_index()
    if not ts_idx:
        return None, {"ok": False,
                      "error": "no timestamps found in boxes_dir/review_dir"}

    try:
        interval = int(interval)
    except (TypeError, ValueError):
        interval = 0
    if interval <= 0:
        return None, {"ok": False,
                      "error": "interval must be a positive integer (e.g. 1 or 15)"}

    ts0 = _resolve_export_ts(start, ts_idx)
    ts1 = _resolve_export_ts(end, ts_idx)
    if ts0 is None:
        return None, {"ok": False, "error": f"start frame '{start}' not found"}
    if ts1 is None:
        return None, {"ok": False, "error": f"end frame '{end}' not found"}
    if ts1 < ts0:
        ts0, ts1 = ts1, ts0

    expected = list(range(ts0, ts1 + 1, interval))
    warnings = []
    if (ts1 - ts0) % interval != 0:
        warnings.append("end timestamp %d is not on the %ds grid starting at %d; "
                        "last exported timestamp is %d"
                        % (ts1, interval, ts0, expected[-1]))

    missing = [t for t in expected if t not in ts_idx]
    if missing:
        return None, {"ok": False,
                      "error": "%d expected timestamp(s) are missing from the "
                               "frame index" % len(missing),
                      "missing_timestamps": missing,
                      "n_expected": len(expected), "warnings": warnings}

    unreviewed = [{"ts": t, "tag": ts_idx[t]} for t in expected
                  if not os.path.exists(review_path(ts_idx[t]))]
    if unreviewed:
        return None, {"ok": False,
                      "error": "%d frame(s) in range have no saved review file"
                               % len(unreviewed),
                      "unreviewed": unreviewed,
                      "n_expected": len(expected), "warnings": warnings}

    return {"ts_idx": ts_idx, "ts0": ts0, "ts1": ts1, "interval": interval,
            "expected": expected, "warnings": warnings}, None


def export_wildtrack_range(start, end, interval,
                           export_root=None, subfolder=None):
    """Export WildTrack JSONs + MOTA/MODA gt for an inclusive frame range.

    Range/interval/missing-frame/unreviewed-frame validation lives in the shared
    helper _prepare_export_range() (also used by the video export), so both
    exports abort identically and nothing is written before the checks pass.

    Box inclusion is frame-level: every box in a reviewed frame is exported,
    regardless of its own `reviewed` flag or the timestamp's `verified` flag.
    """
    ctx, err = _prepare_export_range(start, end, interval)
    if err:
        return err
    ts_idx = ctx["ts_idx"]
    ts0, ts1 = ctx["ts0"], ctx["ts1"]
    interval = ctx["interval"]
    expected = ctx["expected"]
    warnings = list(ctx["warnings"])
    cams_cfg = load_calibration()  # for 3D-box -> 2D-view projection fallback
    export_cams = list(cams_cfg.keys())  # every calibrated camera
    floorplan = load_floorplan()  # None for jerccows
    grid_bounds = export_grid_bounds(floorplan, cams_cfg)

    root = (export_root or CFG.get("export_dir") or "output_3dboxes/exports")
    sub = (subfolder or "").strip() or f"{ts0}-{ts1}_2Dannotations"
    dest = os.path.join(root, sub)
    ann_dir = os.path.join(dest, "annotations_positions")
    eval_dir = os.path.join(dest, "evaluations")
    os.makedirs(ann_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)

    moda_lines, mota_lines = [], []
    oob, missing_labelme = [], []
    n_entries = 0

    for t in expected:
        tag = ts_idx[t]
        data = _read_frame_file(review_path(tag), t)
        boxes = data.get("boxes", {}) or {}

        # per-camera labelme 2D boxes for this exact tag
        cam_bboxes = {}
        for cam in export_cams:
            lp = labelme_path(tag, cam)
            if not os.path.exists(lp):
                missing_labelme.append(f"{cam}/{tag}.json")
                cam_bboxes[cam] = {}
            else:
                cam_bboxes[cam] = extract_bbox_from_json_file(lp)

        entries = []
        for cid, b in boxes.items():
            c = b.get("center") or [0.0, 0.0]
            x, y = float(c[0]), float(c[1])
            pid, was_oob = world_to_position_id(x, y, f"cow {cid} @ ts {t}", bounds=grid_bounds)
            if was_oob:
                oob.append({"ts": t, "tag": tag, "cow_id": cid,
                            "x": round(x, 1), "y": round(y, 1)})

            raw_id = b.get("cow_id", cid)
            try:
                person_id = int(raw_id)
            except (TypeError, ValueError):
                person_id = int(cid) if str(cid).isdigit() else cid

            views = []
            for cam in export_cams:
                # Prefer a real labelme 2D box when one exists; otherwise
                # derive the view by projecting the annotated 3D box into
                # this camera (annotation mode has no labelme files).
                bb = (cam_bboxes[cam].get(person_id)
                      or project_box_to_2d(b, cams_cfg.get(cam)))
                views.append({"xmin": bb["xmin"], "ymin": bb["ymin"],
                              "xmax": bb["xmax"], "ymax": bb["ymax"]}
                             if bb else dict(EMPTY_VIEW))

            entries.append({"personID": person_id,
                            "positionID": int(pid),
                            "views": views})
            n_entries += 1

            ix, iy = int(round(x)), int(round(y))
            # Frame number == raw UNIX timestamp (no sequential remapping).
            moda_lines.append(f"{t},{ix},{iy}\n")
            mota_lines.append(
                f"{EXPORT_SEQ_NUM},{t},{person_id},-1,-1,-1,-1,1,{ix},{iy},-1\n")

        entries.sort(key=lambda e: (0, e["personID"])
                     if isinstance(e["personID"], int) else (1, str(e["personID"])))
        with open(os.path.join(ann_dir, f"{tag}.json"), "w") as f:
            json.dump(entries, f, indent=2)

    with open(os.path.join(eval_dir, "gt_moda.txt"), "w") as f:
        f.writelines(moda_lines)
    with open(os.path.join(eval_dir, "gt_mota.txt"), "w") as f:
        f.writelines(mota_lines)

    print("[export] %s: %d frame(s), %d box(es) -> %s"
          % (sub, len(expected), n_entries, dest))

    return {"ok": True,
            "dest": dest,
            "annotations_dir": ann_dir,
            "evaluations_dir": eval_dir,
            "start_ts": ts0, "end_ts": ts1, "interval": interval,
            "n_frames": len(expected), "n_boxes": n_entries,
            "n_out_of_bounds": len(oob), "out_of_bounds": oob[:20],
            "n_missing_labelme": len(missing_labelme),
            "missing_labelme": missing_labelme[:20],
            "grid": {"width": int((grid_bounds[1] - grid_bounds[0]) / EXPORT_GRID_CELL_CM),
                     "height": int((grid_bounds[3] - grid_bounds[2]) / EXPORT_GRID_CELL_CM),
                     "cell_cm": EXPORT_GRID_CELL_CM,
                     "bounds_cm": grid_bounds},
            "warnings": warnings}

# ----------------------------------------------------------------------------
# Server-side renderer + MP4 video export
# ----------------------------------------------------------------------------
# Python port of the browser canvas overlay (cowColor / boxCorners / projCam /
# renderCam / computeBevBounds / renderBEV in PAGE) so that a reviewer can watch
# the reviewed ground truth as a video. Rendering rules -- colour per cow ID,
# dashed outline for `lying`, ID labels, BEV footprint + yaw arrow -- are kept
# identical to what the reviewer sees on screen.
#
# Deliberate (documented) deviation from the JS: the BEV world->pixel transform
# is computed ONCE over every sampled frame in the range instead of per frame,
# so the BEV does not pan/zoom between frames of the video.

VIDEO_DEFAULT_W = 1920          # explicit default resolution, never "auto"
VIDEO_DEFAULT_H = 1080
VIDEO_DEFAULT_FPS = 1.0         # one encoded frame per sampled timestamp
VIDEO_CAMS_FLEX = 1.15          # mirrors #cams{flex:1.15}
VIDEO_RIGHT_FLEX = 1.0          # mirrors #right{flex:1}

# --- analytics panels (right column is split pie / heatmap / BEV) -----------
# Fractions of the TOTAL frame height. The BEV keeps whatever is left, and
# video_layout() guarantees it never drops below 160 px.
VIDEO_ANALYTICS_PIE_FRAC  = 0.20
VIDEO_ANALYTICS_HEAT_FRAC = 0.40
VIDEO_ANALYTICS_MIN_BEV_H = 160

# BGR colours matching the CSS palette used by the page.
COL_BG        = (24, 20, 17)     # --bg      #111418
COL_BEV_BG    = (19, 15, 12)     # #bevwrap  #0c0f13
COL_PANEL_EDGE= (60, 50, 42)     # .camwrap border #2a323c
COL_CAMLBL    = (221, 255, 153)  # .camlbl   #9fd
COL_PERIMETER = (115, 102, 90)   # #5a6673
COL_BED       = (62, 65, 176)    # #b0413e
COL_BED_DIV   = (45, 45, 110)    # rgba(176,65,62,.6) over the BEV background
COL_CAMMARK   = (176, 164, 154)  # #9aa4b0
COL_TEXT      = (255, 255, 255)
COL_STANDING  = (110, 210, 130)  # pie wedge: standing  (green)
COL_LYING     = (235, 165,  70)  # pie wedge: lying     (blue/orange)
COL_MUT       = (163, 149, 139)  # --mut #8b95a3

CUBOID_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7)]

_CV2 = None


def _require_cv2():
    """Import OpenCV on first use so the JSON export still works without it."""
    global _CV2
    if _CV2 is None:
        try:
            import cv2 as _c
        except ImportError as e:
            raise RuntimeError(
                "video export requires OpenCV -- install it with "
                "`pip install opencv-python`") from e
        _CV2 = _c
    return _CV2


def _cid_sort_key(cid):
    s = str(cid)
    return (0, int(s), "") if s.lstrip("-").isdigit() else (1, 0, s)


def cow_color_bgr(cid):
    """Port of JS cowColor(): hsl((id*67)%360, 75%, 60%) -> OpenCV BGR."""
    s = str(cid)
    n = int(s) if s.lstrip("-").isdigit() else abs(hash(s))
    h = (n * 67) % 360
    r, g, b = colorsys.hls_to_rgb(h / 360.0, 0.60, 0.75)   # (hue, light, sat)
    return (int(round(b * 255)), int(round(g * 255)), int(round(r * 255)))


def box_corners_world(b):
    """Port of JS boxCorners(): 8 world-cm cuboid corners (base 0-3, top 4-7)."""
    c = b.get("center") or [0.0, 0.0]
    cx, cy = float(c[0]), float(c[1])
    yaw = float(b.get("yaw", 0.0) or 0.0)
    L = float(b.get("length", 0.0) or 0.0)
    W = float(b.get("width", 0.0) or 0.0)
    H = float(b.get("height", 0.0) or 0.0)
    u = (math.cos(yaw), math.sin(yaw))
    p = (-math.sin(yaw), math.cos(yaw))
    hl, hw = L / 2.0, W / 2.0
    base = [(cx + hl * u[0] + hw * p[0], cy + hl * u[1] + hw * p[1]),
            (cx + hl * u[0] - hw * p[0], cy + hl * u[1] - hw * p[1]),
            (cx - hl * u[0] - hw * p[0], cy - hl * u[1] - hw * p[1]),
            (cx - hl * u[0] + hw * p[0], cy - hl * u[1] + hw * p[1])]
    return ([(q[0], q[1], 0.0) for q in base] +
            [(q[0], q[1], H) for q in base])


def project_point(P, pt):
    """Port of JS projCam(): world cm -> calibration pixels, or None if behind."""
    x = P[0][0] * pt[0] + P[0][1] * pt[1] + P[0][2] * pt[2] + P[0][3]
    y = P[1][0] * pt[0] + P[1][1] * pt[1] + P[1][2] * pt[2] + P[1][3]
    w = P[2][0] * pt[0] + P[2][1] * pt[1] + P[2][2] * pt[2] + P[2][3]
    if w <= 1e-6:
        return None
    return (x / w, y / w)


def _ok_pt(p, lim=1.0e5):
    return p is not None and abs(p[0]) < lim and abs(p[1]) < lim


def _ipt(p):
    return (int(round(p[0])), int(round(p[1])))


def _line(img, p0, p1, color, thickness=1, dashed=False, dash=6, gap=4):
    """Solid or dashed segment; dash pattern mirrors ctx.setLineDash([6,4])."""
    cv2 = _require_cv2()
    if not (_ok_pt(p0) and _ok_pt(p1)):
        return
    if not dashed:
        cv2.line(img, _ipt(p0), _ipt(p1), color, thickness, cv2.LINE_AA)
        return
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return
    ux, uy = dx / dist, dy / dist
    s = 0.0
    while s < dist:
        e = min(s + dash, dist)
        cv2.line(img, _ipt((p0[0] + ux * s, p0[1] + uy * s)),
                 _ipt((p0[0] + ux * e, p0[1] + uy * e)),
                 color, thickness, cv2.LINE_AA)
        s = e + gap


def _rect(img, p0, p1, color, thickness=1, dashed=False):
    a = (p0[0], p0[1]); b = (p1[0], p0[1])
    c = (p1[0], p1[1]); d = (p0[0], p1[1])
    for s, e in ((a, b), (b, c), (c, d), (d, a)):
        _line(img, s, e, color, thickness, dashed)


def draw_cam_panel(cam_name, cam_cfg, tag, boxes, w, h):
    """Port of JS renderCam(): letterboxed image + projected 3D wireframes.

    Returns (panel_bgr, image_found).
    """
    cv2 = _require_cv2()
    panel = np.zeros((h, w, 3), np.uint8)
    panel[:] = (0, 0, 0)
    if not cam_cfg:
        cv2.putText(panel, "no calib", (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (90, 90, 90), 1, cv2.LINE_AA)
        return panel, False

    res = cam_cfg.get("image_res") or [None, None]
    calib_h, calib_w = res[0], res[1]

    ip = os.path.join(CFG["dataset"], cam_name, f"{tag}.jpg")
    img = cv2.imread(ip) if os.path.exists(ip) else None
    found = img is not None
    if found:
        nat_h, nat_w = img.shape[:2]
    else:
        nat_w = int(calib_w or w)
        nat_h = int(calib_h or h)

    # exactly the JS fit: sc = min(cw/natW, ch/natH), centred
    sc = min(w / float(nat_w), h / float(nat_h))
    dw, dh = max(1, int(round(nat_w * sc))), max(1, int(round(nat_h * sc)))
    ox, oy = (w - dw) // 2, (h - dh) // 2
    if found:
        panel[oy:oy + dh, ox:ox + dw] = cv2.resize(img, (dw, dh),
                                                   interpolation=cv2.INTER_AREA)

    rx = (nat_w / float(calib_w)) if calib_w else 1.0
    ry = (nat_h / float(calib_h)) if calib_h else 1.0
    P = cam_cfg["P_cm"]

    def to_panel(pt3):
        q = project_point(P, pt3)
        if q is None:
            return None
        return (ox + q[0] * rx * sc, oy + q[1] * ry * sc)

    for cid in sorted(boxes, key=_cid_sort_key):
        b = boxes[cid]
        col = cow_color_bgr(cid)
        dashed = (b.get("posture") == "lying")
        pts = [to_panel(c) for c in box_corners_world(b)]
        for a, bb in CUBOID_EDGES:
            _line(panel, pts[a], pts[bb], col, 2, dashed)
        lab = pts[4]                      # top-front corner, as in the JS
        if _ok_pt(lab):
            cv2.putText(panel, str(cid), (int(lab[0]) + 2, int(lab[1]) - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    cv2.putText(panel, cam_name, (7, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, COL_CAMLBL, 1, cv2.LINE_AA)
    if not found:
        cv2.putText(panel, "image missing", (7, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (120, 140, 255), 1, cv2.LINE_AA)
    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), COL_PANEL_EDGE, 1)
    return panel, found


def compute_bev_bounds(floorplan, cams_cfg, centers):
    """Port of JS computeBevBounds(), fed with the centres of ALL sampled frames
    so the BEV transform is constant for the whole video."""
    xs, ys = [], []
    fp = floorplan or {}
    per = fp.get("perimeter")
    if per:
        xs += [per["xmin"], per["xmax"]]
        ys += [per["ymin"], per["ymax"]]
    for c in (cams_cfg or {}).values():
        C = c.get("center") or [0, 0, 0]
        xs.append(C[0]); ys.append(C[1])
    for (x, y) in centers:
        xs.append(x); ys.append(y)
    if not xs:
        xs, ys = [-1000, 1000], [-800, 800]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    mx = (x1 - x0) * 0.08 + 50
    my = (y1 - y0) * 0.08 + 50
    return {"x0": x0 - mx, "x1": x1 + mx, "y0": y0 - my, "y1": y1 + my}


def _draw_floorplan(panel, floorplan, w2c):
    """Perimeter + beds + bed dividers, using any world->pixel transform.

    Shared by the BEV panel and the heatmap panel so both show the identical
    barn geometry.
    """
    fp = floorplan or {}
    per = fp.get("perimeter")
    if per:
        _rect(panel, w2c(per["xmin"], per["ymax"]),
              w2c(per["xmax"], per["ymin"]), COL_PERIMETER, 2)

    for bed in (fp.get("beds") or []):
        _rect(panel, w2c(bed["xmin"], bed["ymax"]),
              w2c(bed["xmax"], bed["ymin"]), COL_BED, 2, dashed=True)
        nl = int(fp.get("bed_divider_lines") or 0)
        ncol = int(fp.get("bed_columns") or 1)
        for i in range(1, nl + 1):
            x = bed["xmin"] + (bed["xmax"] - bed["xmin"]) * i / float(nl + 1)
            _line(panel, w2c(x, bed["ymin"]), w2c(x, bed["ymax"]), COL_BED_DIV, 1)
        for i in range(1, ncol):
            y = bed["ymin"] + (bed["ymax"] - bed["ymin"]) * i / float(ncol)
            _line(panel, w2c(bed["xmin"], y), w2c(bed["xmax"], y), COL_BED_DIV, 1)


def _draw_cam_marks(panel, cams_cfg, w2c, font_scale=0.38):
    """Camera position crosses + names."""
    cv2 = _require_cv2()
    for cn, c in (cams_cfg or {}).items():
        C = c.get("center") or [0, 0, 0]
        q = w2c(C[0], C[1])
        if not _ok_pt(q):
            continue
        _line(panel, (q[0] - 6, q[1] - 6), (q[0] + 6, q[1] + 6), COL_CAMMARK, 2)
        _line(panel, (q[0] + 6, q[1] - 6), (q[0] - 6, q[1] + 6), COL_CAMMARK, 2)
        cv2.putText(panel, cn, (int(q[0]) + 8, int(q[1]) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, COL_CAMMARK, 1, cv2.LINE_AA)


def draw_bev_panel(boxes, cams_cfg, floorplan, bd, w, h, label=None):
    """Port of JS renderBEV(): floorplan + camera markers + footprints/yaw."""
    cv2 = _require_cv2()
    panel = np.zeros((h, w, 3), np.uint8)
    panel[:] = COL_BEV_BG

    pad = 20
    sx = (w - 2 * pad) / max(1e-6, (bd["x1"] - bd["x0"]))
    sy = (h - 2 * pad) / max(1e-6, (bd["y1"] - bd["y0"]))
    sc = min(sx, sy)

    def w2c(x, y):
        return (pad + (x - bd["x0"]) * sc, h - pad - (y - bd["y0"]) * sc)

    _draw_floorplan(panel, floorplan, w2c)
    _draw_cam_marks(panel, cams_cfg, w2c)

    for cid in sorted(boxes, key=_cid_sort_key):
        b = boxes[cid]
        col = cow_color_bgr(cid)
        dashed = (b.get("posture") == "lying")
        base = [w2c(c[0], c[1]) for c in box_corners_world(b)[:4]]
        for i in range(4):
            _line(panel, base[i], base[(i + 1) % 4], col, 2, dashed)
        c = b.get("center") or [0.0, 0.0]
        yaw = float(b.get("yaw", 0.0) or 0.0)
        L = float(b.get("length", 0.0) or 0.0)
        c0 = w2c(float(c[0]), float(c[1]))
        tip = w2c(float(c[0]) + math.cos(yaw) * L * 0.5,
                  float(c[1]) + math.sin(yaw) * L * 0.5)
        _line(panel, c0, tip, col, 2)
        if _ok_pt(tip):
            cv2.circle(panel, _ipt(tip), 3, col, -1, cv2.LINE_AA)
        if _ok_pt(c0):
            cv2.putText(panel, str(cid), (int(c0[0]) + 4, int(c0[1]) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_TEXT, 1, cv2.LINE_AA)

    if label:
        cv2.putText(panel, label, (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, COL_TEXT, 1, cv2.LINE_AA)
    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), COL_PANEL_EDGE, 1)
    return panel


# ---------------------------------------------------------------------------
# Analytics: posture pie chart + spatial-usage heatmap
# ---------------------------------------------------------------------------
def _posture_of(b):
    """Normalise the free-text posture field to 'lying' | 'standing'."""
    p = str(b.get("posture", "") or "").strip().lower()
    return "lying" if p.startswith("ly") else "standing"


def posture_counts(boxes):
    """{'standing': n, 'lying': n, 'total': n} for one frame's boxes."""
    n_ly = sum(1 for b in (boxes or {}).values() if _posture_of(b) == "lying")
    n_tot = len(boxes or {})
    return {"standing": n_tot - n_ly, "lying": n_ly, "total": n_tot}


class HeatAccumulator:
    """Accumulates oriented-Gaussian cow dwell time on a fixed BEV raster.

    The raster IS the heatmap panel (same size, same pad/scale rule as
    draw_bev_panel), so `colorize()` output can be blitted straight into the
    frame and lines up with the BEV panel below it.

    Each `add(boxes)` call evaluates an anisotropic Gaussian aligned with
    every box's yaw. Gaussian sigma is one sixth of the projected box length
    or width, so the approximately +/-3-sigma visible support corresponds to
    the original cow footprint. With weight = sampling interval, the centre
    accumulates "cow-seconds" and the surrounding area falls off smoothly.
    """

    def __init__(self, bd, w, h, pad=20, weight=1.0):
        self.bd = bd
        self.w, self.h, self.pad = int(w), int(h), int(pad)
        self.weight = float(weight)
        self.acc = np.zeros((self.h, self.w), np.float32)
        sx = (self.w - 2 * self.pad) / max(1e-6, (bd["x1"] - bd["x0"]))
        sy = (self.h - 2 * self.pad) / max(1e-6, (bd["y1"] - bd["y0"]))
        self.sc = min(sx, sy)
        self.n_frames = 0
        self.locked = False          # True => 'total' mode, pre-accumulated

    def w2c(self, x, y):
        return (self.pad + (x - self.bd["x0"]) * self.sc,
                self.h - self.pad - (y - self.bd["y0"]) * self.sc)

    def _oriented_gaussian(self, b):
        """Return (x0, x1, y0, y1, gaussian_patch) for one box, or None.

        The Gaussian is aligned with the box yaw. Sigma is expressed in heatmap
        pixels: one sixth of the projected length/width, making the +/-3-sigma
        support approximately match the original box footprint.
        """
        c = b.get("center") or [0.0, 0.0]
        try:
            wx, wy = float(c[0]), float(c[1])
            yaw = float(b.get("yaw", 0.0) or 0.0)
            length = float(b.get("length", 0.0) or 0.0)
            width = float(b.get("width", 0.0) or 0.0)
        except (TypeError, ValueError, IndexError):
            return None

        values = np.asarray([wx, wy, yaw, length, width], dtype=np.float64)
        if length <= 0.0 or width <= 0.0 or not np.all(np.isfinite(values)):
            return None

        px, py = self.w2c(wx, wy)
        if not np.all(np.isfinite((px, py))):
            return None

        # length / 6 means 3*sigma is half the projected box length. Likewise,
        # width / 6 means 3*sigma is half the projected box width.
        sigma_l = max(0.5, float(length * self.sc / 6.0))
        sigma_w = max(0.5, float(width * self.sc / 6.0))

        radius = int(np.ceil(3.0 * max(sigma_l, sigma_w)))
        cx, cy = int(round(px)), int(round(py))

        x0 = max(cx - radius, 0)
        x1 = min(cx + radius + 1, self.w)
        y0 = max(cy - radius, 0)
        y1 = min(cy + radius + 1, self.h)
        if x0 >= x1 or y0 >= y1:
            return None

        ys = (np.arange(y0, y1, dtype=np.float32).reshape(-1, 1)
              - float(cy))
        xs = (np.arange(x0, x1, dtype=np.float32).reshape(1, -1)
              - float(cx))

        co, si = float(np.cos(yaw)), float(np.sin(yaw))

        # World Y points upward, but canvas Y points downward. These signs are
        # therefore intentionally different from a training heatmap whose
        # coordinates already use image-style Y.
        u = co * xs - si * ys       # distance along the cow's length
        v = -si * xs - co * ys      # distance along the cow's width

        g = np.exp(
            -(u * u) / (2.0 * sigma_l * sigma_l)
            -(v * v) / (2.0 * sigma_w * sigma_w)
        ).astype(np.float32, copy=False)

        # Same numerical tail removal idea as draw_oriented_gaussian().
        cutoff = float(np.finfo(np.float32).eps) * float(g.max())
        g[g < cutoff] = 0.0

        return x0, x1, y0, y1, g

    def add(self, boxes):
        """Add one sampled frame of oriented-Gaussian occupancy."""
        self.n_frames += 1
        for b in (boxes or {}).values():
            patch = self._oriented_gaussian(b)
            if patch is None:
                continue
            x0, x1, y0, y1, g = patch
            self.acc[y0:y1, x0:x1] += self.weight * g

    def total_seconds(self):
        return float(self.acc.max()) if self.acc.size else 0.0

    def colorize(self, floorplan=None, cams_cfg=None, label=None, gamma=0.6):
        """Render the accumulator as a panel: colormap + floorplan + colourbar."""
        cv2 = _require_cv2()
        panel = np.zeros((self.h, self.w, 3), np.uint8)
        panel[:] = COL_BEV_BG

        mx = self.total_seconds()
        if mx > 0:
            # gamma < 1 lifts the low-occupancy tail so lightly used areas stay
            # visible next to a few very hot lying spots.
            norm = np.clip(self.acc / mx, 0.0, 1.0) ** float(gamma)
            heat = cv2.applyColorMap((norm * 255.0).astype(np.uint8),
                                     cv2.COLORMAP_INFERNO)
            m = self.acc > 0
            panel[m] = heat[m]

        _draw_floorplan(panel, floorplan, self.w2c)
        _draw_cam_marks(panel, cams_cfg, self.w2c, font_scale=0.33)

        # colour bar (0 .. max dwell)
        bw, bh = min(120, max(40, self.w // 4)), 8
        bx, by = self.w - bw - 16, self.h - 24
        if bx > 4 and by > 4:
            ramp = np.linspace(0, 255, bw).astype(np.uint8)[None, :].repeat(bh, 0)
            panel[by:by + bh, bx:bx + bw] = cv2.applyColorMap(ramp,
                                                              cv2.COLORMAP_INFERNO)
            cv2.rectangle(panel, (bx - 1, by - 1), (bx + bw, by + bh),
                          COL_PANEL_EDGE, 1)
            cv2.putText(panel, "0", (bx - 9, by + bh + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, COL_MUT, 1, cv2.LINE_AA)
            cv2.putText(panel, "%.0fs" % mx, (bx + bw - 22, by + bh + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.34, COL_MUT, 1, cv2.LINE_AA)

        if label:
            cv2.putText(panel, label, (10, 19), cv2.FONT_HERSHEY_SIMPLEX,
                        0.44, COL_CAMLBL, 1, cv2.LINE_AA)
        cv2.rectangle(panel, (0, 0), (self.w - 1, self.h - 1), COL_PANEL_EDGE, 1)
        return panel


def draw_pie_panel(counts, w, h, label=None):
    """Standing-vs-lying pie for ONE frame, with a numeric legend."""
    cv2 = _require_cv2()
    panel = np.zeros((h, w, 3), np.uint8)
    panel[:] = COL_BEV_BG

    n_st = int(counts.get("standing", 0))
    n_ly = int(counts.get("lying", 0))
    tot = n_st + n_ly

    cx, cy = int(w * 0.28), int(h * 0.56)
    r = int(max(16, min(w * 0.22, h * 0.34)))

    if tot == 0:
        cv2.circle(panel, (cx, cy), r, (70, 70, 70), 2, cv2.LINE_AA)
        cv2.putText(panel, "no cows", (cx - 26, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_MUT, 1, cv2.LINE_AA)
    else:
        start = -90.0                            # 12 o'clock, clockwise
        for n, col in ((n_st, COL_STANDING), (n_ly, COL_LYING)):
            if n <= 0:
                continue
            sweep = 360.0 * n / float(tot)
            if n == tot:
                cv2.circle(panel, (cx, cy), r, col, -1, cv2.LINE_AA)
            else:
                cv2.ellipse(panel, (cx, cy), (r, r), 0.0, start, start + sweep,
                            col, -1, cv2.LINE_AA)
            start += sweep
        cv2.circle(panel, (cx, cy), r, (25, 25, 25), 1, cv2.LINE_AA)

    lx, ly, dy = int(w * 0.55), int(h * 0.40), 24
    for i, (nm, n, col) in enumerate((("standing", n_st, COL_STANDING),
                                      ("lying", n_ly, COL_LYING))):
        y = ly + i * dy
        cv2.rectangle(panel, (lx, y - 10), (lx + 14, y + 2), col, -1)
        pct = (100.0 * n / tot) if tot else 0.0
        cv2.putText(panel, "%s  %d  (%.0f%%)" % (nm, n, pct), (lx + 21, y + 1),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_TEXT, 1, cv2.LINE_AA)
    cv2.putText(panel, "total %d cow(s)" % tot, (lx, ly + 2 * dy + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, COL_MUT, 1, cv2.LINE_AA)

    cv2.putText(panel, label or "posture split (this frame)", (10, 19),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, COL_CAMLBL, 1, cv2.LINE_AA)
    cv2.rectangle(panel, (0, 0), (w - 1, h - 1), COL_PANEL_EDGE, 1)
    return panel


def video_layout(W, H, analytics=False,
                 pie_frac=VIDEO_ANALYTICS_PIE_FRAC,
                 heat_frac=VIDEO_ANALYTICS_HEAT_FRAC,
                 cameras=None):
    """Single source of truth for the video frame layout.

    Called by BOTH export_video_range() (to size the heatmap accumulator before
    encoding starts) and render_review_frame() (to place the panels), so the
    accumulator raster can never disagree with the panel it is blitted into.
    """
    cam_w = int(round(W * VIDEO_CAMS_FLEX / (VIDEO_CAMS_FLEX + VIDEO_RIGHT_FLEX)))
    cam_w = max(4, cam_w - (cam_w % 2))
    right_w = W - cam_w
    top_h = H // 2
    bot_h = H - top_h
    left_w = cam_w // 2
    rcam_w = cam_w - left_w
    cams = (normalize_panel_cameras(cameras) if cameras is not None
            else panel_cameras())
    if len(cams) != N_CAMERA_PANELS:
        raise ValueError("video export requires %d calibrated cameras"
                         % N_CAMERA_PANELS)
    slots = [(cams[0], 0, 0, left_w, top_h),
             (cams[1], left_w, 0, rcam_w, top_h),
             (cams[2], 0, top_h, left_w, bot_h),
             (cams[3], left_w, top_h, rcam_w, bot_h)]

    if analytics:
        pie_h = max(110, int(round(H * float(pie_frac))))
        heat_h = max(130, int(round(H * float(heat_frac))))
        room = H - VIDEO_ANALYTICS_MIN_BEV_H
        if pie_h + heat_h > room and (pie_h + heat_h) > 0:
            k = max(0.0, room) / float(pie_h + heat_h)
            pie_h, heat_h = int(pie_h * k), int(heat_h * k)
        bev_h = H - pie_h - heat_h
    else:
        pie_h = heat_h = 0
        bev_h = H

    return {"cam_w": cam_w, "right_x": cam_w, "right_w": right_w,
            "cam_slots": slots, "pie_h": pie_h, "heat_h": heat_h,
            "bev_h": bev_h}


def render_review_frame(t, tag, boxes, cams_cfg, floorplan, bd, size,
                        analytics=None, cameras=None):
    """Composite one video frame: 2x2 camera grid (left) + right analytics column.

    Right column, top to bottom:
        posture pie (this frame) -> spatial-usage heatmap -> BEV floorplan.
    With analytics=None the layout is byte-identical to the old cameras+BEV one.

    `analytics` (when given) is
        {"heat": HeatAccumulator, "counts": {...},
         "pie_label": str|None, "heat_label": str|None,
         "pie_frac": float|None, "heat_frac": float|None}

    Returns (frame_bgr, [missing camera image names]).
    """
    W, H = size
    canvas = np.zeros((H, W, 3), np.uint8)
    canvas[:] = COL_BG

    an = analytics or {}
    lay = video_layout(W, H, analytics is not None,
                       an.get("pie_frac") or VIDEO_ANALYTICS_PIE_FRAC,
                       an.get("heat_frac") or VIDEO_ANALYTICS_HEAT_FRAC,
                       cameras=cameras)

    missing = []
    for cam, x, y, pw, ph in lay["cam_slots"]:
        panel, found = draw_cam_panel(cam, (cams_cfg or {}).get(cam),
                                      tag, boxes, pw, ph)
        canvas[y:y + ph, x:x + pw] = panel
        if not found:
            missing.append(f"{cam}/{tag}.jpg")

    rx, rw = lay["right_x"], lay["right_w"]
    y = 0

    if analytics is not None:
        counts = an.get("counts") or posture_counts(boxes)
        if lay["pie_h"] > 0:
            canvas[y:y + lay["pie_h"], rx:rx + rw] = draw_pie_panel(
                counts, rw, lay["pie_h"], an.get("pie_label"))
            y += lay["pie_h"]
        heat = an.get("heat")
        if heat is not None and lay["heat_h"] > 0:
            canvas[y:y + lay["heat_h"], rx:rx + rw] = heat.colorize(
                floorplan, cams_cfg, an.get("heat_label"))
            y += lay["heat_h"]

    label = "%s   (ts %d)   %d cow(s)" % (tag, t, len(boxes))
    bev = draw_bev_panel(boxes, cams_cfg, floorplan, bd, rw, lay["bev_h"], label)
    canvas[y:y + lay["bev_h"], rx:rx + rw] = bev
    return canvas, missing


def _parse_resolution(res):
    """None/'' -> default; '1920x1080' | [w,h] | (w,h) -> (even w, even h)."""
    if res in (None, "", "auto"):
        w, h = VIDEO_DEFAULT_W, VIDEO_DEFAULT_H
    elif isinstance(res, (list, tuple)) and len(res) == 2:
        w, h = int(res[0]), int(res[1])
    else:
        s = str(res).strip().lower().replace("*", "x").replace(",", "x")
        parts = [p for p in s.split("x") if p]
        if len(parts) != 2:
            raise ValueError("resolution must look like '1920x1080'")
        w, h = int(parts[0]), int(parts[1])
    if w < 320 or h < 240 or w > 7680 or h > 4320:
        raise ValueError("resolution out of range (320x240 .. 7680x4320)")
    return w - (w % 2), h - (h % 2)      # H.264 needs even dimensions


# ---------------------------------------------------------------------------
# Video encoder backends
# ---------------------------------------------------------------------------
# Two sinks, one interface (.write(bgr_img) / .close(success)):
#   * _FfmpegSink : raw BGR frames piped to an ffmpeg binary -> real H.264
#                   (yuv420p + faststart, i.e. plays everywhere). Works even if
#                   OpenCV was built without video support.
#   * _CvSink     : cv2.VideoWriter, kept as a fallback.
#
# IMPORTANT: cv2.VideoWriter chooses its muxer from the FILE EXTENSION, so every
# temp/working path below must still end in .mp4 (or .avi for MJPG). Writing to
# "foo.mp4.part" makes *every* codec fail to open -- that was the bug behind
# "OpenCV could not open an MP4 VideoWriter".

def _ffmpeg_bin():
    """Path to a usable ffmpeg executable, or None.

    Order: --ffmpeg CLI arg / $FFMPEG_BINARY -> PATH -> imageio-ffmpeg's
    bundled binary (pip install imageio-ffmpeg) .
    """
    cand = CFG.get("ffmpeg") or os.environ.get("FFMPEG_BINARY") or "ffmpeg"
    if os.path.isabs(cand):
        return cand if os.path.exists(cand) else None
    p = shutil.which(cand)
    if p:
        return p
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def video_backend_report():
    """What encoders this machine offers (for diagnostics / error messages)."""
    rep = {"preference": (CFG.get("video_encoder") or "auto"),
           "ffmpeg_bin": _ffmpeg_bin(),
           "cv2": None, "cv2_ffmpeg": None, "cv2_gstreamer": None}
    try:
        cv2 = _require_cv2()
    except RuntimeError as e:
        rep["cv2_error"] = str(e)
        return rep
    rep["cv2"] = cv2.__version__
    try:
        for line in cv2.getBuildInformation().splitlines():
            s = line.strip()
            up = s.upper()
            if up.startswith("FFMPEG:"):
                rep["cv2_ffmpeg"] = s.split(":", 1)[1].strip()
            elif up.startswith("GSTREAMER:"):
                rep["cv2_gstreamer"] = s.split(":", 1)[1].strip()
    except Exception:
        pass
    return rep


class _CvSink:
    """cv2.VideoWriter wrapper."""
    kind = "opencv"

    def __init__(self, fourcc, path, fps, size):
        cv2 = _require_cv2()
        self.name = "opencv:%s" % fourcc
        self.path = path
        self._vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc),
                                   float(fps), (int(size[0]), int(size[1])))
        if not self._vw.isOpened():
            self._vw.release()
            raise RuntimeError("cv2.VideoWriter(%s) would not open %s"
                               % (fourcc, os.path.basename(path)))

    def write(self, img):
        self._vw.write(img)

    def close(self, success=True):
        self._vw.release()


class _FfmpegSink:
    """Raw BGR frames -> ffmpeg stdin -> encoded file."""
    kind = "ffmpeg"

    def __init__(self, exe, vcodec, path, fps, size):
        self.name = "ffmpeg:%s" % vcodec
        self.path = path
        w, h = int(size[0]), int(size[1])
        self._err = tempfile.TemporaryFile()
        cmd = [exe, "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", "%dx%d" % (w, h), "-r", "%g" % float(fps),
               "-i", "-", "-an",
               "-c:v", vcodec]
        if vcodec == "libx264":
            cmd += ["-preset", "medium", "-crf", "20"]
        else:
            cmd += ["-q:v", "3"]
        cmd += ["-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
        self._p = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                   stdout=subprocess.DEVNULL, stderr=self._err)

    def _stderr(self):
        try:
            self._err.seek(0)
            return self._err.read().decode("utf-8", "replace").strip()[-800:]
        except Exception:
            return ""

    def write(self, img):
        try:
            self._p.stdin.write(np.ascontiguousarray(img).tobytes())
        except (BrokenPipeError, OSError):
            raise RuntimeError("ffmpeg exited early: %s" % (self._stderr() or "?"))

    def close(self, success=True):
        try:
            if not success:
                self._p.kill()
            else:
                try:
                    self._p.stdin.close()
                except Exception:
                    pass
                rc = self._p.wait(timeout=120)
                if rc != 0:
                    raise RuntimeError("ffmpeg failed (exit %d): %s"
                                       % (rc, self._stderr() or "?"))
        finally:
            try:
                self._p.stdin.close()
            except Exception:
                pass
            try:
                self._err.close()
            except Exception:
                pass


def _make_sink(spec, path, fps, size):
    """spec = ('ffmpeg', (exe, vcodec)) | ('cv', fourcc)."""
    kind, ident = spec[0], spec[1]
    if kind == "ffmpeg":
        return _FfmpegSink(ident[0], ident[1], path, fps, size)
    return _CvSink(ident, path, fps, size)


def _encoder_candidates():
    """Ordered [(spec, ext, is_h264, label)] honouring --video_encoder."""
    pref = (CFG.get("video_encoder") or "auto").lower()
    exe = _ffmpeg_bin()
    out = []
    if pref in ("auto", "ffmpeg") and exe:
        out.append((("ffmpeg", (exe, "libx264")), ".mp4", True, "ffmpeg libx264"))
    if pref in ("auto", "opencv"):
        out.append((("cv", "avc1"), ".mp4", True, "OpenCV avc1 (H.264)"))
        out.append((("cv", "H264"), ".mp4", True, "OpenCV H264"))
        out.append((("cv", "mp4v"), ".mp4", False, "OpenCV mp4v (MPEG-4 pt.2)"))
    if pref in ("auto", "ffmpeg") and exe:
        out.append((("ffmpeg", (exe, "mpeg4")), ".mp4", False, "ffmpeg mpeg4"))
    if pref in ("auto", "opencv"):
        out.append((("cv", "MJPG"), ".avi", False, "OpenCV MJPG (AVI)"))
    return out


def probe_video_encoder(fps, size):
    """Pick the best encoder that actually works, WITHOUT touching the export
    folder: every candidate is opened in a temp dir and fed one real frame, then
    the output size is checked (catches 'opens fine, writes a 0-byte file').

    Returns (choice_or_None, diag) where choice = (spec, ext, is_h264, label).
    """
    diag = video_backend_report()
    cands = _encoder_candidates()
    diag["candidates"] = [c[3] for c in cands]
    diag["tried"] = []
    blank = np.zeros((int(size[1]), int(size[0]), 3), np.uint8)
    tmpd = tempfile.mkdtemp(prefix="rv3d_probe_")
    try:
        for spec, ext, is_h264, label in cands:
            p = os.path.join(tmpd, "probe" + ext)
            why = "ok"
            try:
                sink = _make_sink(spec, p, fps, size)
            except Exception as e:
                diag["tried"].append({"encoder": label, "result": "open failed: %s" % e})
                continue
            try:
                sink.write(blank)
                sink.close(True)
                sz = os.path.getsize(p) if os.path.exists(p) else 0
                if sz <= 0:
                    why = "produced an empty file"
            except Exception as e:
                why = "write/close failed: %s" % e
                try:
                    sink.close(False)
                except Exception:
                    pass
            diag["tried"].append({"encoder": label, "result": why})
            if why == "ok":
                diag["chosen"] = label
                return (spec, ext, is_h264, label), diag
    finally:
        shutil.rmtree(tmpd, ignore_errors=True)
    return None, diag


def export_video_range(start, end, interval, output_fps=None, resolution=None,
                       export_root=None, subfolder=None,
                       analytics=True, heatmap_mode="cumulative", cameras=None):
    """Render the reviewed annotations of an inclusive frame range to one MP4.

    One sampled timestamp == one encoded video frame, held for 1/output_fps
    seconds (straight hold, no interpolation between samples). Validation is the
    shared _prepare_export_range(); the encoder is probed in a temp dir BEFORE
    the export folder is created, and any mid-encode failure deletes the
    partially written file, so a failed export never leaves output behind.

    analytics=True adds two panels to the right column:
      * a posture pie chart recomputed for EVERY frame (standing vs lying), and
      * a spatial-usage heatmap of cow footprints, in cow-seconds.
    heatmap_mode:
      'cumulative' -- the heatmap grows as the video plays (final frame == total)
      'total'      -- the whole-range heatmap is pre-computed and shown static.
    Both also get written next to the video as analytics/heatmap.png and
    analytics/posture_timeline.csv.
    """
    try:
        _require_cv2()                 # needed for rendering, not just encoding
    except RuntimeError as e:
        return {"ok": False, "error": str(e),
                "hint": "pip install opencv-python"}

    ctx, err = _prepare_export_range(start, end, interval)
    if err:
        return err
    ts_idx = ctx["ts_idx"]
    ts0, ts1 = ctx["ts0"], ctx["ts1"]
    interval = ctx["interval"]
    expected = ctx["expected"]
    warnings = list(ctx["warnings"])

    try:
        video_cams = (normalize_panel_cameras(cameras) if cameras is not None
                      else panel_cameras())
    except ValueError as e:
        return {"ok": False, "error": str(e)}

    try:
        fps = float(VIDEO_DEFAULT_FPS if output_fps in (None, "") else output_fps)
    except (TypeError, ValueError):
        return {"ok": False, "error": "output_fps must be a number (e.g. 1 or 2)"}
    if not (0.1 <= fps <= 120.0):
        return {"ok": False, "error": "output_fps must be between 0.1 and 120"}

    try:
        W, H = _parse_resolution(resolution)
    except ValueError as e:
        return {"ok": False, "error": str(e)}

    want_analytics = bool(analytics)
    hm = str(heatmap_mode or "cumulative").strip().lower()
    if hm in ("static", "full", "range"):
        hm = "total"
    if hm not in ("cumulative", "total"):
        return {"ok": False,
                "error": "heatmap_mode must be 'cumulative' or 'total'"}

    # ---- pick a working encoder BEFORE writing anything --------------------
    choice, backends = probe_video_encoder(fps, (W, H))
    if choice is None:
        hint = ("No usable video encoder. Easiest fixes: (a) install ffmpeg and "
                "make sure it is on PATH -- `conda install -c conda-forge ffmpeg`, "
                "`apt install ffmpeg`, `brew install ffmpeg`; or (b) "
                "`pip install imageio-ffmpeg` (ships its own ffmpeg binary, "
                "auto-detected); or (c) pass --ffmpeg /full/path/to/ffmpeg; or "
                "(d) reinstall OpenCV with video support: "
                "`pip uninstall opencv-python opencv-python-headless && "
                "pip install opencv-python` (cv2.getBuildInformation() should "
                "report 'FFMPEG: YES').")
        return {"ok": False,
                "error": "no working video encoder found on this machine",
                "hint": hint, "video_backends": backends,
                "n_expected": len(expected), "warnings": warnings}
    spec, ext, is_h264, enc_label = choice
    if not is_h264:
        warnings.append("H.264 unavailable; encoded with %s instead. Install "
                        "ffmpeg (or imageio-ffmpeg) for true H.264 MP4."
                        % enc_label)
    if ext != ".mp4":
        warnings.append("no MP4-capable encoder available; wrote an %s container "
                        "instead." % ext.upper().lstrip("."))

    cams_cfg = load_calibration()
    floorplan = load_floorplan()

    # Pre-scan the sampled frames: gives a stable BEV transform for the whole
    # video (boxes themselves are re-read lazily to keep memory flat).
    frames, centers = [], []
    for t in expected:
        tag = ts_idx[t]
        boxes = _read_frame_file(review_path(tag), t).get("boxes", {}) or {}
        frames.append((t, tag, boxes))
        for b in boxes.values():
            c = b.get("center") or [0.0, 0.0]
            centers.append((float(c[0]), float(c[1])))
    bd = compute_bev_bounds(floorplan, cams_cfg, centers)

    # ---- analytics setup ---------------------------------------------------
    # The accumulator raster is sized from the SAME layout the renderer uses.
    lay = video_layout(W, H, want_analytics, cameras=video_cams)
    heat = None
    if want_analytics:
        heat = HeatAccumulator(bd, lay["right_w"], lay["heat_h"],
                               weight=float(interval))
        if hm == "total":
            for (_t, _tag, _bx) in frames:
                heat.add(_bx)
            heat.locked = True
    timeline = []

    root = (export_root or CFG.get("export_dir") or "output_3dboxes/exports")
    sub = (subfolder or "").strip() or f"{ts0}-{ts1}_video"
    dest = os.path.join(root, sub)
    os.makedirs(dest, exist_ok=True)
    video_path = os.path.join(dest, f"annotated_{ts0}-{ts1}{ext}")
    # Working file MUST keep the real extension: OpenCV derives the container
    # from it (".part" made every codec fail to open).
    work_path = os.path.join(dest, f".writing_annotated_{ts0}-{ts1}{ext}")
    if os.path.exists(work_path):
        os.remove(work_path)

    try:
        sink = _make_sink(spec, work_path, fps, (W, H))
    except Exception as e:
        if os.path.exists(work_path):
            os.remove(work_path)
        return {"ok": False,
                "error": "could not open the %s encoder for %s: %s"
                         % (enc_label, os.path.basename(work_path), e),
                "video_backends": backends, "warnings": warnings}

    missing_images = []
    t = None
    try:
        for (t, tag, boxes) in frames:
            counts = posture_counts(boxes)
            timeline.append((t, tag, counts))

            an = None
            if heat is not None:
                if not heat.locked:
                    heat.add(boxes)          # cumulative: grows frame by frame
                an = {"heat": heat, "counts": counts,
                      "pie_label": "posture split @ %s" % tag,
                      "heat_label": ("spatial usage - %s (%ds/sample)"
                                     % ("whole range" if heat.locked
                                        else "cumulative", interval))}

            img, miss = render_review_frame(t, tag, boxes, cams_cfg, floorplan,
                                            bd, (W, H), an, cameras=video_cams)
            missing_images.extend(miss)
            sink.write(img)
        sink.close(True)
    except Exception as e:
        try:
            sink.close(False)
        except Exception:
            pass
        if os.path.exists(work_path):
            os.remove(work_path)      # never leave a partial video behind
        return {"ok": False,
                "error": "encoding failed%s: %s"
                         % (("" if t is None else " at ts %d" % t), e),
                "video_backends": backends, "warnings": warnings}

    if not (os.path.exists(work_path) and os.path.getsize(work_path) > 0):
        if os.path.exists(work_path):
            os.remove(work_path)
        return {"ok": False,
                "error": "%s produced an empty file" % enc_label,
                "video_backends": backends, "warnings": warnings}
    os.replace(work_path, video_path)

    # ---- analytics sidecars (only after the video itself succeeded) --------
    ana_summary = None
    if want_analytics:
        try:
            cv2 = _require_cv2()
            adir = os.path.join(dest, "analytics")
            os.makedirs(adir, exist_ok=True)
            csv_path = os.path.join(adir, "posture_timeline.csv")
            with open(csv_path, "w") as f:
                f.write("timestamp,tag,standing,lying,total\n")
                for (tt, tg, c) in timeline:
                    f.write("%d,%s,%d,%d,%d\n"
                            % (tt, tg, c["standing"], c["lying"], c["total"]))
            png_path = os.path.join(adir, "heatmap.png")
            cv2.imwrite(png_path,
                        heat.colorize(floorplan, cams_cfg,
                                      "spatial usage - whole range (cow-seconds)"))
            n_st = sum(c["standing"] for _a, _b, c in timeline)
            n_ly = sum(c["lying"] for _a, _b, c in timeline)
            n_obs = max(1, n_st + n_ly)
            ana_summary = {
                "heatmap_mode": hm,
                "heatmap_png": png_path,
                "posture_csv": csv_path,
                "max_dwell_sec": round(heat.total_seconds(), 1),
                "cow_frames_standing": n_st,
                "cow_frames_lying": n_ly,
                "pct_lying": round(100.0 * n_ly / n_obs, 1),
                "mean_cows_per_frame": round(n_obs / float(len(timeline) or 1), 2),
            }
        except Exception as e:
            warnings.append("video written, but the analytics sidecars failed: %s" % e)

    print("[export] video %s: %d frame(s) @ %.3g fps (%dx%d, %s, analytics=%s) -> %s"
          % (sub, len(frames), fps, W, H, enc_label,
             (hm if want_analytics else "off"), video_path))

    return {"ok": True,
            "dest": dest,
            "video_path": video_path,
            "start_ts": ts0, "end_ts": ts1, "interval": interval,
            "n_frames": len(frames),
            "output_fps": fps,
            "duration_sec": round(len(frames) / fps, 3),
            "resolution": [W, H],
            "panel_cameras": video_cams,
            "codec": enc_label,
            "encoder": enc_label,
            "analytics": ana_summary,
            "n_missing_images": len(missing_images),
            "missing_images": missing_images[:20],
            "warnings": warnings}

# ----------------------------------------------------------------------------
# Forward re-interpolation on Save
# ----------------------------------------------------------------------------
# The math below is an exact mirror of interpolate_3d_annotations.py so that a
# re-interpolated frame is indistinguishable from one produced by the offline
# batch script (same fields, same nearest-in-time rule, same angle short-path).

NUMERIC_FIELDS = ["height", "length", "width", "z_base", "confidence"]
ANGLE_FIELDS   = ["yaw"]
NEAREST_FIELDS = ["cams", "cow_id", "n_hull_voxels", "n_views",
                  "posture", "source"]

# If the next ground-truth frame has already been reviewed/corrected by a human,
# use the corrected boxes as the right anchor (they are strictly better than the
# raw file). Anchor *detection* still uses boxes_dir, per spec.
USE_REVIEWED_ANCHOR = True


def _lerp(a, b, alpha):
    return a + (b - a) * alpha


def _lerp_angle(a0, a1, alpha):
    """Interpolate angles (radians) along the shortest path."""
    diff = (a1 - a0 + math.pi) % (2.0 * math.pi) - math.pi
    return a0 + diff * alpha


def interpolate_box(b0, b1, alpha):
    """Interpolate a single box dict between b0 (left) and b1 (right)."""
    src = b0 if alpha < 0.5 else b1  # nearest-in-time source for categoricals
    box = {}

    c0, c1 = b0["center"], b1["center"]
    box["center"] = [_lerp(c0[i], c1[i], alpha) for i in range(len(c0))]

    for f in NUMERIC_FIELDS:
        if f in b0 and f in b1:
            box[f] = _lerp(b0[f], b1[f], alpha)
        elif f in src:
            box[f] = src[f]

    for f in ANGLE_FIELDS:
        if f in b0 and f in b1:
            box[f] = _lerp_angle(b0[f], b1[f], alpha)
        elif f in src:
            box[f] = src[f]

    for f in NEAREST_FIELDS:
        if f in src:
            box[f] = src[f]

    box["z_base"] = 0.0
    box["reviewed"] = False
    box["edited"] = False
    box["interpolated"] = True
    return box


def constant_box(b0):
    """No right anchor (cow added by the user): hold the box constant."""
    box = dict(b0)
    box.pop("_meta", None)
    box["center"] = list(b0.get("center", [0.0, 0.0]))
    box["z_base"] = 0.0
    box["reviewed"] = False
    box["edited"] = False
    box["interpolated"] = True
    return box


# ---------------------------------------------------------------------------
# Anchor resolution: index by TIMESTAMP, never by filename.
# ---------------------------------------------------------------------------
def build_dir_index(d):
    """ts -> full path for every *.json in `d` (ignoring _private files).

    Keyed on the timestamp parsed out of the name, so "<ts>.json" and
    "<ts>_HH-MM-SS.json" resolve identically. This is what makes anchor lookup
    immune to boxes_dir / review_dir using different naming conventions.
    """
    idx = {}
    if not d or not os.path.isdir(d):
        return idx
    for f in glob.glob(os.path.join(d, "*.json")):
        if os.path.basename(f).startswith("_"):
            continue
        ts, _tag = _parse_tag(f)
        if ts is not None:
            idx[ts] = f
    return idx


def gt_dir():
    """Annotation mode: anchors are the SAVED annotations themselves.

    There is no detector output to interpolate towards; a Save bridges
    forward to the NEXT hand-annotated frame in review_dir. (The --gt_dir
    override is kept in case you ever want anchors elsewhere.)
    """
    return CFG.get("gt_dir") or CFG["review_dir"]


_CLASS_CACHE = {}   # path -> (mtime, verdict)


def classify_frame_file(path):
    """'gt' | 'interp' | 'unknown' | 'missing' for one box-json file.

    Tolerant on purpose: different stages of the pipeline write different
    metadata, and requiring an exact top-level `"interpolated": false` silently
    breaks anchor detection.
    """
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return "missing"
    hit = _CLASS_CACHE.get(path)
    if hit and hit[0] == mt:
        return hit[1]

    try:
        with open(path) as f:
            d = json.load(f)
    except Exception:
        d = None

    verdict = "unknown"
    if isinstance(d, dict):
        top = d.get("interpolated", None)
        if isinstance(top, bool) or isinstance(top, (int, float)):
            verdict = "interp" if bool(top) else "gt"
        elif isinstance(top, str):
            verdict = "gt" if top.strip().lower() in ("false", "0", "no") else "interp"
        else:
            st = d.get("source_timestamps")
            if isinstance(st, list) and len(st) == 1:
                verdict = "gt"
            elif isinstance(st, list) and len(st) >= 2 and st[0] != st[-1]:
                verdict = "interp"
            else:
                flags = [b.get("interpolated")
                         for b in (d.get("boxes") or {}).values()
                         if isinstance(b, dict) and "interpolated" in b]
                if flags:
                    verdict = "interp" if all(bool(x) for x in flags) else "gt"

    _CLASS_CACHE[path] = (mt, verdict)
    return verdict


def find_next_ground_truth(ts, ts_idx=None, max_scan=None):
    """Nearest FUTURE anchor. Returns (gt_ts, gt_path, diag_dict).

    Modes (--gt_mode):
      flag    : only frames explicitly classified 'gt'
      present : the next frame that exists in gt_dir at all (sparse-anchor dir)
      spacing : the next multiple of --gt_every from the first anchor ts
      auto    : 'flag'; if the forward scan finds NO flagged GT and every
                candidate is 'unknown' (i.e. the data carries no interpolation
                metadata), fall back to 'present'.
    """
    gidx = build_dir_index(gt_dir())
    mode = (CFG.get("gt_mode") or "auto").lower()
    every = int(CFG.get("gt_every") or 0)
    max_scan = int(max_scan or CFG.get("gt_max_scan") or 600)

    future = sorted(t for t in gidx if t > ts)[:max_scan]
    diag = {"mode": mode, "gt_dir": gt_dir(), "n_files_in_gt_dir": len(gidx),
            "n_future_candidates": len(future), "scanned": []}

    if not future:
        diag["why"] = ("no json files with a timestamp > %d in %s"
                       % (ts, gt_dir()))
        return None, None, diag

    if mode == "spacing":
        if every <= 0:
            diag["why"] = "--gt_mode spacing requires --gt_every > 0"
            return None, None, diag
        base = min(gidx)
        for t in future:
            if (t - base) % every == 0:
                diag["why"] = "spacing anchor"
                return t, gidx[t], diag
        diag["why"] = "no timestamp on the %ds grid ahead" % every
        return None, None, diag

    for t in future:
        v = classify_frame_file(gidx[t])
        diag["scanned"].append([t, v, os.path.basename(gidx[t])])
        if mode == "present":
            diag["why"] = "first file present in gt_dir"
            return t, gidx[t], diag
        if v == "gt":
            diag["why"] = "explicit interpolated=false"
            return t, gidx[t], diag

    kinds = {v for _t, v, _n in diag["scanned"]}
    if mode == "auto" and kinds and kinds <= {"unknown"}:
        t = diag["scanned"][0][0]
        diag["why"] = ("FALLBACK: no interpolation metadata anywhere in %s, "
                       "treating every file there as an anchor" % gt_dir())
        diag["fallback"] = "present"
        return t, gidx[t], diag

    diag["why"] = ("scanned %d future file(s) in %s, none had "
                   "interpolated=false (kinds seen: %s)"
                   % (len(diag["scanned"]), gt_dir(), ",".join(sorted(kinds)) or "-"))
    return None, None, diag


def _load_boxes_for(ts, prefer_review=True, ridx=None, gidx=None):
    """Boxes for a timestamp, resolved by ts (naming-agnostic)."""
    ridx = build_dir_index(CFG["review_dir"]) if ridx is None else ridx
    gidx = build_dir_index(gt_dir()) if gidx is None else gidx
    for idx in ((ridx, gidx) if prefer_review else (gidx, ridx)):
        p = idx.get(ts)
        if p and os.path.exists(p):
            return _read_frame_file(p, ts).get("boxes", {})
    return {}


def reinterpolate_forward(saved_ts, saved_boxes, deleted_ids, ts_idx=None):
    """Forward-only re-interpolation triggered by a Save.

    * Left anchor  : the just-saved frame (only cows with edited == True).
    * Right anchor : the next ground truth ahead of saved_ts.
    * Targets      : every frame strictly between the two anchors, OVERWRITTEN
                     in review_dir even if it was already reviewed.
    * Untouched cows in those frames are carried over byte-for-byte.
    """
    if ts_idx is None:
        ts_idx = build_ts_index()

    changed = {cid: dict(b) for cid, b in (saved_boxes or {}).items()
               if b.get("edited")}
    deleted = set(str(c) for c in (deleted_ids or []))

    if not changed and not deleted:
        return {"ran": False, "reason": "no edited or deleted cows",
                "written": []}

    gt_ts, gt_path, diag = find_next_ground_truth(saved_ts, ts_idx)
    if gt_ts is None:
        print("[reinterp] %d: NO ANCHOR -- %s" % (saved_ts, diag.get("why")))
        return {"ran": False, "reason": "no future ground truth",
                "detail": diag.get("why"), "diag": diag, "written": []}

    ridx = build_dir_index(CFG["review_dir"])
    gidx = build_dir_index(gt_dir())

    gt_boxes = _load_boxes_for(gt_ts, USE_REVIEWED_ANCHOR, ridx, gidx)
    mids = sorted(t for t in ts_idx if saved_ts < t < gt_ts)

    os.makedirs(CFG["review_dir"], exist_ok=True)
    written, const_ids, missing_tag = [], set(), []

    for t in mids:
        tag = ts_idx.get(t)
        if tag is None:
            missing_tag.append(t)
            continue
        # Base = this frame's CURRENT EFFECTIVE state: its review file if one
        # exists (cows we are not touching survive verbatim), otherwise the
        # carry-forward chain built by load_frame(). With no detector output,
        # _load_boxes_for() would return {} for a never-visited frame and
        # every cow you did NOT edit would silently vanish from the mid
        # frames. Note the frames written earlier in this same loop are
        # already on disk, so the chain stays complete.
        boxes = dict(load_frame(t, tag, ts_idx)[0]["boxes"])
        alpha = (t - saved_ts) / float(gt_ts - saved_ts)

        for cid in deleted:                      # deleted cows disappear
            boxes.pop(cid, None)

        for cid, b0 in changed.items():          # edited/added cows re-fit
            b1 = gt_boxes.get(cid)
            if b1 is not None:
                boxes[cid] = interpolate_box(b0, b1, alpha)
            else:
                boxes[cid] = constant_box(b0)
                const_ids.add(cid)

        out = {
            "timestamp": t,
            "interpolated": True,
            "reviewed": False,
            "edited": False,
            "source_timestamps": [saved_ts, gt_ts],
            "alpha": alpha,
            "boxes": boxes,
        }
        with open(review_path(tag), "w") as f:
            json.dump(out, f, indent=2)
        written.append(t)

    print("[reinterp] %d -> GT %d (%s): wrote %d frame(s), cows=%s%s"
          % (saved_ts, gt_ts, diag.get("why"), len(written),
             sorted(changed.keys()),
             (" deleted=%s" % sorted(deleted)) if deleted else ""))

    return {"ran": True, "next_gt": gt_ts, "gt_file": os.path.basename(gt_path),
            "written": written, "n_between": len(mids),
            "changed": sorted(changed.keys()),
            "deleted": sorted(deleted),
            "constant": sorted(const_ids),
            "missing_tag": missing_tag,
            "why": diag.get("why")}

# ----------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------
def load_calibration():
    """Load every calibrated camera, not just the legacy four-camera list.

    The returned per-camera dict keeps the UI-facing keys `P_cm`, `image_res`,
    and `center`. Calibration files may supply the camera centre as either
    `center_cm` (JerCCows) or `center`; otherwise it is recovered from P.
    """
    with open(CFG["calibration_json"]) as f:
        calib = json.load(f)

    cams = {}
    entries = calib.get("cameras", {}) or {}
    for cam in sorted(entries, key=_camera_sort_key):
        e = entries[cam] or {}
        P_raw = e.get("P_cm", e.get("P_rect"))
        if P_raw is None:
            print("[calibration] %s has no P_cm/P_rect; skipped" % cam)
            continue
        try:
            P = np.asarray(P_raw, dtype=float)
        except (TypeError, ValueError):
            print("[calibration] %s has a non-numeric projection matrix; skipped" % cam)
            continue
        if P.shape != (3, 4):
            print("[calibration] %s projection matrix is %s, expected 3x4; skipped"
                  % (cam, P.shape))
            continue

        C = e.get("center_cm") or e.get("center")
        try:
            C = [float(C[0]), float(C[1]), float(C[2])]
        except (TypeError, ValueError, IndexError):
            M = P[:, :3]
            p4 = P[:, 3]
            try:
                C = (-np.linalg.inv(M) @ p4).tolist()
            except np.linalg.LinAlgError:
                C = [0.0, 0.0, 0.0]

        res = e.get("image_res", [None, None])  # [H, W]
        cams[cam] = {"P_cm": P.tolist(), "image_res": res, "center": C}
    return cams


def load_floorplan():
    p = CFG.get("floorplan_json")
    if p and os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:
            return None
    return None


# ----------------------------------------------------------------------------
# API
# ----------------------------------------------------------------------------
@app.route("/api/config")
def api_config():
    ts_idx = build_ts_index()
    verified = load_verified()
    timestamps = [{"ts": ts, "tag": tag, "verified": str(ts) in verified}
                  for ts, tag in ts_idx.items()]
    cameras = load_calibration()
    CFG["all_cams"] = list(cameras.keys())   # picks up a changed calibration file
    return jsonify({
        "cameras": cameras,
        "cam_order": list(cameras.keys()),   # all calibrated cams, for the dropdowns
        "panel_cameras": panel_cameras(),    # the four currently displayed cams
        "floorplan": load_floorplan(),       # None when no floorplan is supplied
        "timestamps": timestamps,
        "posture_priors": POSTURE_PRIORS,
        "export_dir": CFG.get("export_dir"),
    })


@app.route("/api/frame/<int:ts>")
def api_frame(ts):
    ts_idx = build_ts_index()
    if ts not in ts_idx:
        abort(404)
    tag = ts_idx[ts]
    data, src = load_frame(ts, tag, ts_idx)
    cams = {}
    for cam in camera_names():
        ip = os.path.join(CFG["dataset"], cam, f"{tag}.jpg")
        if os.path.exists(ip):
            cams[cam] = {"url": f"/api/image/{tag}/{cam}"}
    verified = str(ts) in load_verified()
    return jsonify({"timestamp": ts, "tag": tag, "boxes": data["boxes"],
                    "source": src, "cams": cams, "verified": verified})


@app.route("/api/image/<tag>/<cam>")
def api_image(tag, cam):
    if cam not in set(camera_names()):
        abort(404)
    ip = os.path.join(CFG["dataset"], cam, f"{tag}.jpg")
    if not os.path.exists(ip):
        abort(404)
    return send_file(ip, mimetype="image/jpeg")


@app.route("/api/save/<int:ts>", methods=["POST"])
def api_save(ts):
    ts_idx = build_ts_index()
    payload = request.json or {}
    tag = ts_idx.get(ts) or payload.get("tag")
    if not tag:
        abort(404)
    boxes = payload.get("boxes", {})

    # --- detect deletions BEFORE writing: compare the effective pre-save state
    #     of this frame (what the user was shown) with the submitted payload.
    try:
        prev_data, _ = load_frame(ts, tag, ts_idx)
        prev_ids = set(prev_data.get("boxes", {}).keys())
    except Exception:
        prev_ids = set()
    deleted_ids = prev_ids - set(boxes.keys())

    save_frame(tag, {"timestamp": ts, "boxes": boxes})

    # --- forward-only re-interpolation, exclusively on Save -----------------
    ts_idx = build_ts_index()          # the save may have created a new file
    try:
        info = reinterpolate_forward(ts, boxes, deleted_ids, ts_idx)
    except Exception as e:             # never let this break the save itself
        info = {"ran": False, "reason": f"error: {e}", "written": []}

    return jsonify({"ok": True, "path": review_path(tag), "reinterp": info})


@app.route("/api/verify/<int:ts>", methods=["POST"])
def api_verify(ts):
    v = load_verified()
    on = bool(request.json.get("verified", True))
    if on:
        v.add(str(ts))
    else:
        v.discard(str(ts))
    save_verified(v)
    return jsonify({"ok": True, "verified": on})

@app.route("/api/copy_cow", methods=["POST"])
def api_copy_cow():
    """Copy one cow's box from a source frame into a set of target frames.

    Never overwrites: a target keeps whatever it already has for that cow ID.
    Reviewed target frames are treated as authoritative and left alone.
    """
    ts_idx = build_ts_index()
    payload = request.json or {}
    cid = str(payload.get("cow_id", "")).strip()
    src_ts = payload.get("source_ts")
    targets = payload.get("target_ts", []) or []
    if not cid or src_ts is None:
        return jsonify({"ok": False, "error": "cow_id and source_ts required"}), 400
    src_ts = int(src_ts)
    if src_ts not in ts_idx:
        return jsonify({"ok": False, "error": "source frame unknown"}), 404

    # Grab the source cow's current effective box.
    src_data, _ = load_frame(src_ts, ts_idx[src_ts], ts_idx)
    src_box = src_data["boxes"].get(cid)
    if src_box is None:
        return jsonify({"ok": False, "error": f"cow {cid} not in source frame"}), 404

    template = {k: src_box[k] for k in
                ("center", "yaw", "length", "width", "height", "posture")
                if k in src_box}
    template["center"] = list(template.get("center", [0.0, 0.0]))
    template["cow_id"] = int(cid) if cid.isdigit() else cid
    template["n_views"] = 0

    store = load_copied_cows()
    added, added_into_reviewed, skipped_exists, skipped_unknown = [], [], [], []
    for t in targets:
        t = int(t)
        if t == src_ts:
            continue
        if t not in ts_idx:
            skipped_unknown.append(t)
            continue
        tag = ts_idx[t]
        eff, _ = load_frame(t, tag, ts_idx)
        # "Don't overwrite" is a PER-COW-ID rule, not per-frame. Only skip if
        # this exact cow is already present (detected / carried-forward /
        # reviewed / previously pasted). A cow that is simply MISSING can always
        # be injected -- even into a frame that already has a review file --
        # because adding a new ID never clobbers an existing annotation.
        if cid in eff["boxes"]:
            skipped_exists.append(t)
            continue
        store.setdefault(str(t), {})[cid] = dict(template)
        added.append(t)
        if os.path.exists(review_path(tag)):
            added_into_reviewed.append(t)  # informational: user should re-check

    save_copied_cows(store)
    return jsonify({"ok": True, "added": added,
                    "added_into_reviewed": added_into_reviewed,
                    "skipped_exists": skipped_exists,
                    "skipped_unknown": skipped_unknown})


@app.route("/api/uncopy_cow", methods=["POST"])
def api_uncopy_cow():
    """Remove a pasted cow from some/all target frames (undo a copy)."""
    payload = request.json or {}
    cid = str(payload.get("cow_id", "")).strip()
    targets = payload.get("target_ts")
    store = load_copied_cows()
    removed = []
    keys = [str(int(t)) for t in targets] if targets else list(store.keys())
    for k in keys:
        if k in store and cid in store[k]:
            del store[k][cid]
            if not store[k]:
                del store[k]
            removed.append(int(k))
    save_copied_cows(store)
    return jsonify({"ok": True, "removed": removed})

@app.route("/api/export_wildtrack", methods=["POST"])
def api_export_wildtrack():
    """Export WildTrack JSON + MOTA/MODA gt for an inclusive tag/ts range."""
    p = request.json or {}
    try:
        res = export_wildtrack_range(
            p.get("start"), p.get("end"), p.get("interval"),
            export_root=(p.get("export_dir") or None),
            subfolder=(p.get("subfolder") or None))
    except Exception as e:
        return jsonify({"ok": False, "error": f"export failed: {e}"}), 500
    return jsonify(res), (200 if res.get("ok") else 400)

@app.route("/api/export_video", methods=["POST"])
def api_export_video():
    """Render the reviewed annotations of a tag/ts range to an annotated MP4."""
    p = request.json or {}
    try:
        # accept "1"/"0"/"true"/false from the UI as well as real booleans
        raw_an = p.get("analytics", True)
        want_an = (bool(raw_an) if isinstance(raw_an, bool)
                   else str(raw_an).strip().lower() not in
                        ("0", "false", "no", "off", ""))
        res = export_video_range(
            p.get("start"), p.get("end"), p.get("interval"),
            output_fps=p.get("output_fps"),
            resolution=p.get("resolution"),
            export_root=(p.get("export_dir") or None),
            subfolder=(p.get("subfolder") or None),
            analytics=want_an,
            heatmap_mode=(p.get("heatmap_mode") or "cumulative"),
            cameras=p.get("panel_cameras"))
    except Exception as e:
        return jsonify({"ok": False, "error": f"video export failed: {e}"}), 500
    return jsonify(res), (200 if res.get("ok") else 400)


@app.route("/api/video_debug")
def api_video_debug():
    """Which video encoders this machine can actually use. Open in a browser."""
    w = int(request.args.get("w", VIDEO_DEFAULT_W))
    h = int(request.args.get("h", VIDEO_DEFAULT_H))
    fps = float(request.args.get("fps", VIDEO_DEFAULT_FPS))
    try:
        choice, diag = probe_video_encoder(fps, (w - w % 2, h - h % 2))
    except RuntimeError as e:          # cv2 missing entirely
        return jsonify({"ok": False, "error": str(e),
                        "hint": "pip install opencv-python"})
    return jsonify({"ok": choice is not None,
                    "probe_size": [w, h], "probe_fps": fps,
                    "chosen": (choice[3] if choice else None),
                    "container": (choice[1] if choice else None),
                    "backends": diag})


@app.route("/api/reinterp_debug")
@app.route("/api/reinterp_debug/<int:ts>")
def api_reinterp_debug(ts=None):
    """Explain what the re-interpolator sees. Open in a browser tab."""
    gidx = build_dir_index(gt_dir())
    ridx = build_dir_index(CFG["review_dir"])
    ts_idx = build_ts_index()
    if ts is None:
        ts = min(ts_idx) if ts_idx else 0

    counts = {}
    sample = []
    for t in sorted(gidx)[:400]:
        v = classify_frame_file(gidx[t])
        counts[v] = counts.get(v, 0) + 1
        if len(sample) < 40:
            sample.append({"ts": t, "file": os.path.basename(gidx[t]), "class": v})

    gt_ts, gt_path, diag = find_next_ground_truth(ts, ts_idx)
    return jsonify({
        "query_ts": ts,
        "gt_dir": gt_dir(), "gt_mode": CFG.get("gt_mode"),
        "gt_every": CFG.get("gt_every"),
        "files_in_gt_dir": len(gidx), "files_in_review_dir": len(ridx),
        "class_counts_first400": counts,
        "sample": sample,
        "next_gt_ts": gt_ts,
        "next_gt_file": os.path.basename(gt_path) if gt_path else None,
        "diag": diag,
        "frames_between": (sorted(t for t in ts_idx if ts < t < gt_ts)
                           if gt_ts else []),
    })

@app.route("/")
def index():
    return render_template_string(PAGE)


# ============================================================================
# Frontend (single page)
# ============================================================================
PAGE = r"""
<!doctype html><html><head><meta charset="utf-8">
<title>3D Cattle Box Annotator</title>
<style>
  :root{--bg:#111418;--panel:#1b2027;--fg:#e5e9ef;--mut:#8b95a3;--acc:#4ea3ff;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--fg);font:13px/1.4 system-ui,sans-serif;height:100vh;overflow:hidden}
  #top{display:flex;align-items:center;gap:10px;padding:6px 10px;background:var(--panel);border-bottom:1px solid #000}
  button{background:#2a323c;color:var(--fg);border:1px solid #3a444f;border-radius:5px;padding:5px 9px;cursor:pointer}
  button:hover{background:#354150}
  button.acc{background:var(--acc);color:#06121f;border-color:var(--acc);font-weight:600}
  button.warn{background:#b0413e;border-color:#b0413e}
  #main{display:flex;height:calc(100vh - 44px)}
  #cams{display:grid;grid-template-columns:1fr 1fr;grid-template-rows:1fr 1fr;gap:4px;flex:1.15;padding:4px}
  .camwrap{position:relative;background:#000;border:1px solid #2a323c;overflow:hidden}
  .camwrap canvas{position:absolute;top:0;left:0}
    .camsel{position:absolute;top:3px;left:5px;z-index:3;max-width:calc(100% - 10px);background:rgba(12,15,19,.88);color:#9fd;border:1px solid #3a444f;border-radius:4px;padding:2px 4px;font-weight:600}
  #right{flex:1;display:flex;flex-direction:column;border-left:1px solid #000}
  #bevwrap{flex:1;position:relative;background:#0c0f13}
  #bev{position:absolute;top:0;left:0;cursor:crosshair}
  #side{height:190px;overflow:auto;background:var(--panel);border-top:1px solid #000;padding:6px 8px;display:flex;gap:14px}
  .col{flex:1;min-width:0}
  .lbl{color:var(--mut);font-size:11px;text-transform:uppercase;letter-spacing:.05em;margin:2px 0}
  #cowlist{display:flex;flex-wrap:wrap;gap:4px}
  .chip{padding:3px 7px;border-radius:12px;border:1px solid #3a444f;cursor:pointer;font-weight:600;display:inline-flex;align-items:center;gap:5px}
  .chip.sel{outline:2px solid var(--acc)}
  .chip.rev::after{content:" ✓";color:#5fd18a}
  .chip.hid{opacity:.4}
  .eye{cursor:pointer;font-size:12px;line-height:1;filter:grayscale(0)}
  .eye.off{opacity:.6;filter:grayscale(1)}
  .visctl{margin-left:8px}
  .mini{padding:1px 6px;font-size:10px;border-radius:4px}
  table{width:100%;border-collapse:collapse}
  td{padding:1px 4px}
  td.k{color:var(--mut)}
  kbd{background:#2a323c;border:1px solid #3a444f;border-radius:3px;padding:0 4px;font-size:11px}
  #status{color:var(--mut)}
  #help{position:absolute;inset:40px;background:rgba(10,12,16,.97);z-index:50;padding:24px 30px;overflow:auto;display:none;border:1px solid #3a444f;border-radius:8px}
  #help h3{margin:12px 0 4px;color:var(--acc)}
  .dirty{color:#ffb454}
</style></head><body>

<div id="top">
  <button id="prevTs">◀ prev</button>
  <span id="tsinfo" style="min-width:230px;text-align:center"></span>
  <button id="nextTs">next ▶</button>
  <span style="width:8px"></span>
  <input id="gotoInput" placeholder="frame name / ts…"
         style="background:#0c0f13;color:var(--fg);border:1px solid #3a444f;border-radius:5px;padding:5px 7px;width:130px">
  <button id="gotoBtn">Go</button>
  <span style="width:14px"></span>
  <button id="acceptBtn" class="acc">Accept box (Enter)</button>
  <button id="addBtn">＋ Add (N)</button>
  <button id="delBtn" class="warn">Delete (Del)</button>
  <button id="postureBtn">Posture (P)</button>
  <span style="width:14px"></span>
  <button id="copyBtn">Copy cow (Ctrl+C)</button>
  <button id="pasteBtn">Paste (Ctrl+V)</button>
  <button id="copyRangeBtn">Copy → frames…</button>
  <span style="width:14px"></span>
  <button id="exportBtn">Export 2D annots…</button>
  <button id="videoBtn">Export video…</button>
  <span style="width:14px"></span>
  <button id="saveBtn">Save (Ctrl+S)</button>
  <button id="verifyBtn">Verify TS (V)</button>
  <button id="helpBtn">? Help</button>
  <span style="flex:1"></span>
  <span id="status"></span>
</div>

<div id="main">
  <div id="cams">
    <div class="camwrap" data-slot="0"><select class="camsel" data-slot="0" aria-label="Camera for panel 1"></select><canvas></canvas></div>
    <div class="camwrap" data-slot="1"><select class="camsel" data-slot="1" aria-label="Camera for panel 2"></select><canvas></canvas></div>
    <div class="camwrap" data-slot="2"><select class="camsel" data-slot="2" aria-label="Camera for panel 3"></select><canvas></canvas></div>
    <div class="camwrap" data-slot="3"><select class="camsel" data-slot="3" aria-label="Camera for panel 4"></select><canvas></canvas></div>
  </div>
  <div id="right">
    <div id="bevwrap"><canvas id="bev"></canvas></div>
    <div id="side">
      <div class="col">
        <div class="lbl">Cows (click to select)
          <span class="visctl">
            <button id="showAllBtn" class="mini">show all</button>
            <button id="hideAllBtn" class="mini">hide all</button>
            <button id="soloBtn" class="mini">solo sel</button>
          </span>
        </div>
        <div id="cowlist"></div>
      </div>
      <div class="col" style="max-width:280px">
        <div class="lbl">Selected box</div>
        <table id="boxtable"></table>
      </div>
    </div>
  </div>
</div>

<div id="help">
  <button style="float:right" onclick="document.getElementById('help').style.display='none'">close</button>
  <h3>Navigation</h3>
  <kbd>[</kbd>/<kbd>]</kbd> prev/next timestamp &nbsp; <kbd>Tab</kbd>/<kbd>Shift+Tab</kbd> next/prev cow &nbsp; click a chip / BEV footprint / camera box to select
  <br>Type a frame name (tag) or timestamp in the top-bar box and press <kbd>Enter</kbd> to jump directly to that frame.
  <h3>Move (BEV, world cm)</h3>
  <kbd>←→↑↓</kbd> translate X/Y (5cm; <kbd>Shift</kbd>=1cm fine, <kbd>Ctrl</kbd>=25cm coarse). Or drag the footprint in the BEV.
  <h3>Rotate yaw</h3>
  <kbd>Q</kbd>/<kbd>E</kbd> −/+ 5° (<kbd>Shift</kbd>=1°). Or drag the yaw-arrow handle in the BEV.
  <h3>Resize</h3>
  <kbd>Z</kbd>/<kbd>X</kbd> length −/+ &nbsp; <kbd>C</kbd>/<kbd>V</kbd> width −/+ &nbsp; <kbd>G</kbd>/<kbd>T</kbd> height −/+ (5cm; Shift=1cm)
  <h3>Workflow</h3>
  <kbd>Enter</kbd> accept/toggle reviewed &nbsp; <kbd>N</kbd> add box &nbsp; <kbd>Del</kbd> delete &nbsp; <kbd>P</kbd> toggle posture
  <kbd>Ctrl+S</kbd> save &nbsp; <kbd>V</kbd> mark timestamp verified
  <h3>Camera overlay visibility</h3>
  Click the 👁 icon on a cow chip to hide/show its 3D box on the camera views (BEV always shows all).
  <kbd>H</kbd> toggles the selected cow. Use <b>show all / hide all / solo sel</b> in the COWS section for bulk control.
  <p style="color:#8b95a3">Frames start empty until you save: every box from the most recently saved frame is carried forward when you open the next timestamp. Annotate frame 1 by hand (<kbd>N</kbd> per cow), Save, then walk forward — only nudge boxes that moved. Saving a frame re-interpolates your edits forward to the next annotated frame.</p>
  <h3>Export annotated video</h3>
  <p style="color:#8b95a3"><b>Export video…</b> renders the reviewed boxes of a frame range into an
  MP4 with the same 5 panels you see here (4 cameras + BEV). Pick the interval (1s or 15s) and an
  output FPS: each sampled timestamp becomes exactly one video frame held for 1/FPS seconds, so
  e.g. 5 samples at 2 fps → a 2.5 s clip. Every frame in the range must already be saved/reviewed,
  otherwise the export aborts and lists the offending frames.</p>
  <h3>Auto re-interpolation on Save</h3>
  <p style="color:#8b95a3">Pressing <kbd>Ctrl+S</kbd> re-interpolates every cow you <b>edited</b> in this frame
  forward to the next ground-truth frame (the next file with <code>interpolated:false</code>).
  Intermediate frames are overwritten <b>even if already reviewed</b>; cows you did not touch are left exactly as they were.
  Deleted cows are removed from those frames; cows you added and that don't exist in the next ground truth are held constant.
  Nothing is interpolated backwards, and nothing happens while you edit or navigate.</p>
  <p style="color:#8b95a3">Snap-to-floor (Z=0 base) is always enforced. Height edits raise the top only.</p>
</div>

<div id="copybox" style="display:none;position:absolute;left:50%;top:80px;transform:translateX(-50%);z-index:60;background:#1b2027;border:1px solid #3a444f;border-radius:8px;padding:18px 20px;width:360px;max-width:90vw;box-shadow:0 8px 30px rgba(0,0,0,.6)">
  <div style="font-weight:600;margin-bottom:8px">Copy cow to a range of frames</div>
  <div id="copyInfo" style="color:#8b95a3;margin-bottom:10px"></div>
  <div class="lbl">From frame (tag or ts)</div>
  <input id="copyFrom" style="width:100%;background:#0c0f13;color:#e5e9ef;border:1px solid #3a444f;border-radius:5px;padding:5px 7px;margin-bottom:8px">
  <div class="lbl">To frame (tag or ts, inclusive)</div>
  <input id="copyTo" style="width:100%;background:#0c0f13;color:#e5e9ef;border:1px solid #3a444f;border-radius:5px;padding:5px 7px;margin-bottom:12px">
  <div style="display:flex;gap:8px;justify-content:flex-end">
    <button onclick="document.getElementById('copybox').style.display='none'">Cancel</button>
    <button class="acc" id="copyApply">Copy cow</button>
  </div>
  <div id="copyResult" style="color:#8b95a3;margin-top:10px"></div>
</div>
<div id="exportbox" style="display:none;position:absolute;left:50%;top:70px;transform:translateX(-50%);z-index:60;background:#1b2027;border:1px solid #3a444f;border-radius:8px;padding:18px 20px;width:460px;max-width:92vw;box-shadow:0 8px 30px rgba(0,0,0,.6)">
  <div style="font-weight:600;margin-bottom:8px">Export WildTrack-style 2D annotations + GT</div>
  <div style="color:#8b95a3;margin-bottom:10px">
    Exports <code>annotations_positions/&lt;tag&gt;.json</code> (positionID from each
    reviewed 3D box centre) and <code>evaluations/gt_mota.txt</code> /
    <code>gt_moda.txt</code> for the inclusive frame range.
  </div>
  <div class="lbl">Start frame (tag or ts)</div>
  <input id="expFrom" style="width:100%;background:#0c0f13;color:#e5e9ef;border:1px solid #3a444f;border-radius:5px;padding:5px 7px;margin-bottom:8px">
  <div class="lbl">End frame (tag or ts, inclusive)</div>
  <input id="expTo" style="width:100%;background:#0c0f13;color:#e5e9ef;border:1px solid #3a444f;border-radius:5px;padding:5px 7px;margin-bottom:8px">
  <div class="lbl">Sampling interval (seconds)</div>
  <select id="expInterval" style="width:100%;background:#0c0f13;color:#e5e9ef;border:1px solid #3a444f;border-radius:5px;padding:5px 7px;margin-bottom:8px">
    <option value="1">1 s</option>
    <option value="15">15 s</option>
  </select>
  <div class="lbl">Export root (blank = CLI --export_dir)</div>
  <input id="expRoot" style="width:100%;background:#0c0f13;color:#e5e9ef;border:1px solid #3a444f;border-radius:5px;padding:5px 7px;margin-bottom:8px">
  <div class="lbl">Subfolder name (blank = &lt;start&gt;-&lt;end&gt;_2Dannotations)</div>
  <input id="expSub" style="width:100%;background:#0c0f13;color:#e5e9ef;border:1px solid #3a444f;border-radius:5px;padding:5px 7px;margin-bottom:8px">
  <div class="lbl">Destination</div>
  <div id="expDest" style="color:#9fd;word-break:break-all;margin-bottom:12px"></div>
  <div style="display:flex;gap:8px;justify-content:flex-end">
    <button onclick="document.getElementById('exportbox').style.display='none'">Cancel</button>
    <button class="acc" id="expApply">Export</button>
  </div>
  <div id="expResult" style="color:#8b95a3;margin-top:10px;max-height:220px;overflow:auto"></div>
</div>
<div id="videobox" style="display:none;position:absolute;left:50%;top:70px;transform:translateX(-50%);z-index:60;background:#1b2027;border:1px solid #3a444f;border-radius:8px;padding:18px 20px;width:460px;max-width:92vw;box-shadow:0 8px 30px rgba(0,0,0,.6)">
  <div style="font-weight:600;margin-bottom:8px">Export annotated video (MP4)</div>
  <div style="color:#8b95a3;margin-bottom:10px">
    Renders one video frame per sampled timestamp: the 4 camera views with the
    reviewed 3D boxes projected on them, plus the BEV floorplan panel — the same
    5-panel layout you see here. Each sample is <b>held</b> for 1/FPS seconds
    (no interpolation). Every frame in the range must already be saved/reviewed.
  </div>
  <div class="lbl">Start frame (tag or ts)</div>
  <input id="vidFrom" style="width:100%;background:#0c0f13;color:#e5e9ef;border:1px solid #3a444f;border-radius:5px;padding:5px 7px;margin-bottom:8px">
  <div class="lbl">End frame (tag or ts, inclusive)</div>
  <input id="vidTo" style="width:100%;background:#0c0f13;color:#e5e9ef;border:1px solid #3a444f;border-radius:5px;padding:5px 7px;margin-bottom:8px">
  <div class="lbl">Sampling interval (seconds)</div>
  <select id="vidInterval" style="width:100%;background:#0c0f13;color:#e5e9ef;border:1px solid #3a444f;border-radius:5px;padding:5px 7px;margin-bottom:8px">
    <option value="1">1 s</option>
    <option value="15">15 s</option>
  </select>
  <div class="lbl">Output FPS (video frames per second)</div>
  <input id="vidFps" value="1" style="width:100%;background:#0c0f13;color:#e5e9ef;border:1px solid #3a444f;border-radius:5px;padding:5px 7px;margin-bottom:8px">
  <div class="lbl">Analytics panels (pie + heatmap)</div>
  <select id="vidAnalytics" style="width:100%;background:#0c0f13;color:#e5e9ef;border:1px solid #3a444f;border-radius:5px;padding:5px 7px;margin-bottom:8px">
    <option value="1" selected>on — pie chart + spatial heatmap + BEV</option>
    <option value="0">off — cameras + BEV only (legacy layout)</option>
  </select>
  <div class="lbl">Heatmap accumulation</div>
  <select id="vidHeatMode" style="width:100%;background:#0c0f13;color:#e5e9ef;border:1px solid #3a444f;border-radius:5px;padding:5px 7px;margin-bottom:8px">
    <option value="cumulative" selected>cumulative (builds up as the video plays)</option>
    <option value="total">total (whole range, static)</option>
  </select>
  <div class="lbl">Resolution (WxH)</div>
  <input id="vidRes" value="1920x1080" style="width:100%;background:#0c0f13;color:#e5e9ef;border:1px solid #3a444f;border-radius:5px;padding:5px 7px;margin-bottom:8px">
  <div class="lbl">Export root (blank = CLI --export_dir)</div>
  <input id="vidRoot" style="width:100%;background:#0c0f13;color:#e5e9ef;border:1px solid #3a444f;border-radius:5px;padding:5px 7px;margin-bottom:8px">
  <div class="lbl">Subfolder name (blank = &lt;start&gt;-&lt;end&gt;_video)</div>
  <input id="vidSub" style="width:100%;background:#0c0f13;color:#e5e9ef;border:1px solid #3a444f;border-radius:5px;padding:5px 7px;margin-bottom:8px">
  <div class="lbl">Destination</div>
  <div id="vidDest" style="color:#9fd;word-break:break-all;margin-bottom:12px"></div>
  <div style="display:flex;gap:8px;justify-content:flex-end">
    <button onclick="document.getElementById('videobox').style.display='none'">Cancel</button>
    <button class="acc" id="vidApply">Render video</button>
  </div>
  <div id="vidResult" style="color:#8b95a3;margin-top:10px;max-height:220px;overflow:auto"></div>
</div>

<script>
// ---------- state ----------
const S = {
  cfg:null, tsList:[], tsIdx:0, ts:null, tag:null,
  boxes:{}, sel:null, dirty:false, verified:false,
  panelCams:[],         // slot(0-3) -> camera name currently shown in that panel
  frameCams:{},         // cam -> {url} for the current frame (from /api/frame)
  camImgs:{},           // slot -> {img, natW, natH, cam}
  camView:{},           // slot -> layout for hit testing (incl. camName)
  bev:null,             // bev transform
  camHidden:new Set(),  // cow IDs whose 3D box is hidden on the 2D camera views
  clipboard:null,       // {cow_id, box} copied for quick single-frame paste
};
const $ = s=>document.querySelector(s);
const EDGES=[[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];

// ---------- color ----------
function cowColor(id){ const h=(id*67)%360; return `hsl(${h},75%,60%)`; }

// ---------- geometry ----------
function boxCorners(b){
  const cx=b.center[0], cy=b.center[1], y=b.yaw, L=b.length, W=b.width, H=b.height;
  const u=[Math.cos(y),Math.sin(y)], p=[-Math.sin(y),Math.cos(y)];
  const hl=L/2, hw=W/2;
  const base=[
    [cx+hl*u[0]+hw*p[0], cy+hl*u[1]+hw*p[1]],
    [cx+hl*u[0]-hw*p[0], cy+hl*u[1]-hw*p[1]],
    [cx-hl*u[0]-hw*p[0], cy-hl*u[1]-hw*p[1]],
    [cx-hl*u[0]+hw*p[0], cy-hl*u[1]+hw*p[1]],
  ];
  const c=[]; for(const q of base) c.push([q[0],q[1],0]);
  for(const q of base) c.push([q[0],q[1],H]);
  return c;
}
function projCam(cam, P){ // P world(cm) point -> {x,y,valid} in calib px
  const M=cam.P; const p=[
    M[0][0]*P[0]+M[0][1]*P[1]+M[0][2]*P[2]+M[0][3],
    M[1][0]*P[0]+M[1][1]*P[1]+M[1][2]*P[2]+M[1][3],
    M[2][0]*P[0]+M[2][1]*P[1]+M[2][2]*P[2]+M[2][3]];
  if(p[2]<=1e-6) return {x:0,y:0,valid:false};
  return {x:p[0]/p[2], y:p[1]/p[2], valid:true};
}

// ---------- camera rendering ----------
function fitCanvas(cv){ const w=cv.parentElement.clientWidth, h=cv.parentElement.clientHeight;
  cv.width=w; cv.height=h; return [w,h]; }

function panelSelect(slot){ return document.querySelector(`.camsel[data-slot="${slot}"]`); }

function populateCameraPanels(){
  const all=S.cfg.cam_order||Object.keys(S.cfg.cameras||{});
  let panels=(S.cfg.panel_cameras||[]).filter(c=>all.includes(c));
  for(const c of all){ if(panels.length>=4) break; if(!panels.includes(c)) panels.push(c); }
  S.panelCams=panels.slice(0,4);
  for(let slot=0; slot<4; slot++){
    const sel=panelSelect(slot); if(!sel) continue;
    sel.innerHTML='';
    for(const c of all){
      const o=document.createElement('option');
      o.value=c; o.textContent=c; sel.appendChild(o);
    }
    sel.value=S.panelCams[slot]||all[0]||'';
    sel.onchange=()=>selectPanelCamera(slot, sel.value);
  }
}

function loadPanelImage(slot){
  const cam=S.panelCams[slot];
  delete S.camImgs[slot]; delete S.camView[slot];
  if(!cam){ renderCam(slot); return; }
  const info=S.frameCams[cam];
  if(!info){ renderCam(slot); return; }   // this camera has no image for the frame
  const img=new Image();
  const rec={img,natW:0,natH:0,cam};
  img.onload=()=>{
    if(S.camImgs[slot]!==rec || S.panelCams[slot]!==cam) return;  // stale load
    rec.natW=img.naturalWidth; rec.natH=img.naturalHeight; renderCam(slot);
  };
  img.src=info.url;
  S.camImgs[slot]=rec;
  renderCam(slot);   // draw the black/calib panel now; the image pops in onload
}

function selectPanelCamera(slot, camName){
  if(!camName || S.panelCams[slot]===camName) return;
  const other=S.panelCams.indexOf(camName);
  if(other>=0 && other!==slot){
    // keep panels unique: swap the two panel assignments
    S.panelCams[other]=S.panelCams[slot];
    const otherSel=panelSelect(other); if(otherSel) otherSel.value=S.panelCams[other];
    S.panelCams[slot]=camName;
    loadPanelImage(other);
  } else {
    S.panelCams[slot]=camName;
  }
  loadPanelImage(slot);
}

function renderCam(slot){
  const wrap=document.querySelector(`.camwrap[data-slot="${slot}"]`);
  const cv=wrap.querySelector('canvas');
  const [cw,ch]=fitCanvas(cv);
  const ctx=cv.getContext('2d'); ctx.clearRect(0,0,cw,ch);
  const camName=S.panelCams[slot];
  const camCfg=camName&&S.cfg.cameras[camName];
  if(!camCfg){ ctx.fillStyle='#333';ctx.fillText('no calib',10,20); return; }
  const rec=S.camImgs[slot];
  const recOk=rec&&rec.cam===camName&&rec.img.complete&&rec.natW;
  let sc=1, ox=0, oy=0, natW=camCfg.image_res?camCfg.image_res[1]:cw, natH=camCfg.image_res?camCfg.image_res[0]:ch;
  if(recOk){
    natW=rec.natW; natH=rec.natH;
    sc=Math.min(cw/natW, ch/natH); const dw=natW*sc, dh=natH*sc;
    ox=(cw-dw)/2; oy=(ch-dh)/2;
    ctx.drawImage(rec.img, ox,oy,dw,dh);
  } else {
    ctx.fillStyle='#000'; ctx.fillRect(0,0,cw,ch);
    sc=Math.min(cw/natW, ch/natH); ox=(cw-natW*sc)/2; oy=(ch-natH*sc)/2;
  }
  const calibW=camCfg.image_res?camCfg.image_res[1]:natW;
  const calibH=camCfg.image_res?camCfg.image_res[0]:natH;
  const cam={P:camCfg.P_cm};
  // store view transform for hit-testing
  const toCanvas=(pt)=>({x: ox + pt.x*(natW/calibW)*sc, y: oy + pt.y*(natH/calibH)*sc, valid:pt.valid});
  S.camView[slot]={toCanvas, cam, camName};

  for(const cid in S.boxes){
    // Hidden on the 2D camera views (reduces occlusion clutter),
    // but the currently-selected cow is always drawn so editing stays visible.
    if(S.camHidden.has(cid) && S.sel!==cid) continue;
    const b=S.boxes[cid];
    const corners=boxCorners(b).map(c=>toCanvas(projCam(cam,c)));
    const col=cowColor(+cid);
    const seld=(S.sel===cid);
    ctx.lineWidth=seld?3:1.6;
    ctx.strokeStyle=col;
    ctx.setLineDash(b.posture==='lying'?[6,4]:[]);
    for(const [a,bb] of EDGES){
      const pa=corners[a], pb=corners[bb];
      if(!pa.valid||!pb.valid) continue;
      ctx.beginPath(); ctx.moveTo(pa.x,pa.y); ctx.lineTo(pb.x,pb.y); ctx.stroke();
    }
    ctx.setLineDash([]);
    // label at top-front corner
    const lab=corners[4];
    if(lab.valid){ ctx.fillStyle=col; ctx.font=(seld?'bold ':'')+'14px sans-serif';
      ctx.fillText(cid, lab.x+2, lab.y-2); }
  }
}

// ---------- BEV ----------
function computeBevBounds(){
  const fp=S.cfg.floorplan; let xs=[],ys=[];
  if(fp&&fp.perimeter){const p=fp.perimeter;xs.push(p.xmin,p.xmax);ys.push(p.ymin,p.ymax);}
  for(const c in S.cfg.cameras){const C=S.cfg.cameras[c].center;xs.push(C[0]);ys.push(C[1]);}
  for(const id in S.boxes){const c=S.boxes[id].center;xs.push(c[0]);ys.push(c[1]);}
  if(!xs.length){xs=[-1000,1000];ys=[-800,800];}
  let x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
  const mx=(x1-x0)*0.08+50, my=(y1-y0)*0.08+50;
  return {x0:x0-mx,x1:x1+mx,y0:y0-my,y1:y1+my};
}
function renderBEV(){
  const cv=$('#bev'); const [cw,ch]=fitCanvas(cv);
  const ctx=cv.getContext('2d'); ctx.clearRect(0,0,cw,ch);
  ctx.fillStyle='#0c0f13'; ctx.fillRect(0,0,cw,ch);
  const bd=computeBevBounds();
  const pad=20;
  const sc=Math.min((cw-2*pad)/(bd.x1-bd.x0),(ch-2*pad)/(bd.y1-bd.y0));
  const w2c=(x,y)=>[pad+(x-bd.x0)*sc, ch-pad-(y-bd.y0)*sc];
  const c2w=(px,py)=>[ (px-pad)/sc+bd.x0, (ch-pad-py)/sc+bd.y0 ];
  S.bev={w2c,c2w,sc};

  // floorplan
  const fp=S.cfg.floorplan;
  if(fp){
    ctx.strokeStyle='#5a6673'; ctx.lineWidth=2;
    if(fp.perimeter){const p=fp.perimeter; const a=w2c(p.xmin,p.ymin),b=w2c(p.xmax,p.ymax);
      ctx.strokeRect(a[0],b[1],b[0]-a[0],a[1]-b[1]);}
    for(const bed of (fp.beds||[])){
      const a=w2c(bed.xmin,bed.ymin),b=w2c(bed.xmax,bed.ymax);
      ctx.strokeStyle='#b0413e'; ctx.setLineDash([6,4]); ctx.lineWidth=1.4;
      ctx.strokeRect(a[0],b[1],b[0]-a[0],a[1]-b[1]); ctx.setLineDash([]);
      const nl=fp.bed_divider_lines||0, ncol=fp.bed_columns||1;
      ctx.strokeStyle='rgba(176,65,62,.6)'; ctx.lineWidth=.8;
      for(let i=1;i<=nl;i++){const x=bed.xmin+(bed.xmax-bed.xmin)*i/(nl+1);
        const s=w2c(x,bed.ymin),e=w2c(x,bed.ymax);ctx.beginPath();ctx.moveTo(s[0],s[1]);ctx.lineTo(e[0],e[1]);ctx.stroke();}
      for(let i=1;i<ncol;i++){const y=bed.ymin+(bed.ymax-bed.ymin)*i/ncol;
        const s=w2c(bed.xmin,y),e=w2c(bed.xmax,y);ctx.beginPath();ctx.moveTo(s[0],s[1]);ctx.lineTo(e[0],e[1]);ctx.stroke();}
    }
  }
  // cameras
  for(const cn in S.cfg.cameras){const C=S.cfg.cameras[cn].center; const q=w2c(C[0],C[1]);
    ctx.strokeStyle='#9aa4b0'; ctx.lineWidth=2;
    ctx.beginPath();ctx.moveTo(q[0]-6,q[1]-6);ctx.lineTo(q[0]+6,q[1]+6);ctx.moveTo(q[0]+6,q[1]-6);ctx.lineTo(q[0]-6,q[1]+6);ctx.stroke();
    ctx.fillStyle='#9aa4b0';ctx.font='10px sans-serif';ctx.fillText(cn,q[0]+8,q[1]-4);}
  // boxes
  for(const cid in S.boxes){
    const b=S.boxes[cid]; const corners=boxCorners(b);
    const base=corners.slice(0,4).map(c=>w2c(c[0],c[1]));
    const col=cowColor(+cid); const seld=(S.sel===cid);
    ctx.strokeStyle=col; ctx.lineWidth=seld?3:1.8;
    ctx.setLineDash(b.posture==='lying'?[6,4]:[]);
    ctx.beginPath();ctx.moveTo(base[0][0],base[0][1]);
    for(let i=1;i<4;i++)ctx.lineTo(base[i][0],base[i][1]);
    ctx.closePath();ctx.stroke();ctx.setLineDash([]);
    // yaw arrow
    const u=[Math.cos(b.yaw),Math.sin(b.yaw)];
    const c0=w2c(b.center[0],b.center[1]);
    const tip=w2c(b.center[0]+u[0]*b.length*0.5, b.center[1]+u[1]*b.length*0.5);
    ctx.strokeStyle=col;ctx.lineWidth=seld?3:2;
    ctx.beginPath();ctx.moveTo(c0[0],c0[1]);ctx.lineTo(tip[0],tip[1]);ctx.stroke();
    ctx.fillStyle=col;ctx.beginPath();ctx.arc(tip[0],tip[1],seld?5:3,0,7);ctx.fill();
    ctx.fillStyle='#fff';ctx.font=(seld?'bold ':'')+'12px sans-serif';
    ctx.fillText(cid,c0[0]+4,c0[1]-4);
  }
}

// ---------- render all ----------
function renderAll(){ cameraPanelsRender(); renderBEV(); renderCowList(); renderBoxTable(); updateTop(); }
function cameraPanelsRender(){ for(let slot=0; slot<4; slot++) renderCam(slot); }

function renderCowList(){
  const el=$('#cowlist'); el.innerHTML='';
  const ids=Object.keys(S.boxes).sort((a,b)=>+a-+b);
  for(const cid of ids){
    const b=S.boxes[cid];
    const hidden=S.camHidden.has(cid);
    const d=document.createElement('div');
    d.className='chip'+(S.sel===cid?' sel':'')+(b.reviewed?' rev':'')+(hidden?' hid':'');
    d.style.borderColor=cowColor(+cid); d.style.color=cowColor(+cid);

    // eye toggle: controls 3D-overlay visibility on the 2D camera views only
    const eye=document.createElement('span');
    eye.className='eye'+(hidden?' off':'');
    eye.textContent=hidden?'🙈':'👁';
    eye.title=hidden?'Show 3D box on camera views (H)':'Hide 3D box on camera views (H)';
    eye.onclick=(ev)=>{ ev.stopPropagation(); toggleVis(cid); };

    const lbl=document.createElement('span');
    lbl.textContent=cid+' ('+(b.posture==='lying'?'L':'S')+')';

    d.appendChild(eye);
    d.appendChild(lbl);
    d.onclick=()=>{ S.sel=cid; renderAll(); };
    el.appendChild(d);
  }
}
function renderBoxTable(){
  const t=$('#boxtable');
  if(!S.sel||!S.boxes[S.sel]){t.innerHTML='<tr><td class="k">none selected</td></tr>';return;}
  const b=S.boxes[S.sel];
  const row=(k,v)=>`<tr><td class="k">${k}</td><td>${v}</td></tr>`;
  t.innerHTML=
    row('cow_id',S.sel)+
    row('posture',b.posture)+
    row('center',`${b.center[0].toFixed(1)}, ${b.center[1].toFixed(1)}`)+
    row('yaw°',(b.yaw*180/Math.PI).toFixed(1))+
    row('L / W / H',`${b.length.toFixed(0)} / ${b.width.toFixed(0)} / ${b.height.toFixed(0)}`)+
    row('n_views',b.n_views||'-')+
    row('source',b.source||'-')+
    row('reviewed',b.reviewed?'✓ yes':'no')+
    row('edited',b.edited?'yes':'no');
}
function updateTop(){
  const rc=Object.values(S.boxes).filter(b=>b.reviewed).length;
  const tot=Object.keys(S.boxes).length;
  $('#tsinfo').innerHTML=`TS ${S.ts} &nbsp;[${S.tsIdx+1}/${S.tsList.length}]&nbsp; `+
    `<b>${rc}/${tot} reviewed</b>${S.verified?' <span style="color:#5fd18a">✔verified</span>':''}`;
  $('#status').innerHTML=S.dirty?'<span class="dirty">● unsaved</span>':'saved';
  $('#verifyBtn').textContent=S.verified?'Unverify TS (V)':'Verify TS (V)';
}

// ---------- loading ----------
async function loadConfig(){
  S.cfg=await (await fetch('/api/config')).json();
  S.tsList=S.cfg.timestamps;
  populateCameraPanels();
}
async function loadFrame(idx){
  if(S.dirty && !confirm('Unsaved changes will be lost. Continue?')) return;
  idx=Math.max(0,Math.min(S.tsList.length-1,idx));
  S.tsIdx=idx; const t=S.tsList[idx]; S.ts=t.ts; S.tag=t.tag;
  const fr=await (await fetch('/api/frame/'+S.ts)).json();
  S.boxes=fr.boxes; S.sel=Object.keys(S.boxes).sort((a,b)=>+a-+b)[0]||null;
  S.verified=fr.verified; S.dirty=false;
  // load images for the four selected panel cameras only
  S.frameCams=fr.cams||{};
  S.camImgs={}; S.camView={};
  for(let slot=0; slot<4; slot++) loadPanelImage(slot);
  renderAll();
}

function gotoFrame(name){
  if(name==null) return;
  name=String(name).trim();
  if(!name) return;
  // match by tag (frame name) first
  let idx=S.tsList.findIndex(t=>t.tag===name);
  // fall back to matching by numeric timestamp
  if(idx<0){
    const n=Number(name);
    if(!Number.isNaN(n)) idx=S.tsList.findIndex(t=>t.ts===n);
  }
  if(idx<0){
    $('#status').innerHTML='<span class="dirty">frame "'+name+'" not found</span>';
    return;
  }
  loadFrame(idx);
  $('#gotoInput').value='';
  $('#gotoInput').blur();
}

// ---------- editing ops ----------
function mark(){ if(S.sel){S.boxes[S.sel].edited=true;} S.dirty=true; }
function translate(dx,dy){ if(!S.sel)return; const b=S.boxes[S.sel]; b.center[0]+=dx;b.center[1]+=dy; mark(); renderAll(); }
function rotate(drad){ if(!S.sel)return; S.boxes[S.sel].yaw+=drad; mark(); renderAll(); }
function resize(dim,delta){ if(!S.sel)return; const b=S.boxes[S.sel]; b[dim]=Math.max(10,b[dim]+delta); mark(); renderAll(); }
function togglePosture(){
  if(!S.sel) return;
  const b=S.boxes[S.sel];
  b.posture = (b.posture==='lying') ? 'standing' : 'lying';
  const pr = S.cfg.posture_priors[b.posture];
  if(pr){
    b.length = pr.length;
    b.width  = pr.width;
    b.height = pr.height;
  }
  mark();
  renderAll();
}
function toggleVis(cid){
  cid = cid || S.sel; if(!cid) return;
  if(S.camHidden.has(cid)) S.camHidden.delete(cid); else S.camHidden.add(cid);
  renderAll();
}
function showAllCams(){ S.camHidden.clear(); renderAll(); }
function hideAllCams(){ S.camHidden = new Set(Object.keys(S.boxes)); renderAll(); }
function soloSelected(){
  if(!S.sel){ return; }
  S.camHidden = new Set(Object.keys(S.boxes).filter(c=>c!==S.sel));
  renderAll();
}
function toggleReviewed(){ if(!S.sel)return; const b=S.boxes[S.sel]; b.reviewed=!b.reviewed; S.dirty=true; renderAll(); }
function copyCow(){
  if(!S.sel||!S.boxes[S.sel]) return;
  S.clipboard={cow_id:S.sel, box:JSON.parse(JSON.stringify(S.boxes[S.sel]))};
  $('#status').innerHTML='copied cow '+S.sel;
}
function pasteCow(){
  if(!S.clipboard){ $('#status').innerHTML='<span class="dirty">clipboard empty</span>'; return; }
  const cid=S.clipboard.cow_id;
  if(S.boxes[cid] && !confirm('Cow '+cid+' already exists here. Overwrite its box?')) return;
  const b=JSON.parse(JSON.stringify(S.clipboard.box));
  b.source='copied'; b.reviewed=false; b.edited=true; b.z_base=0;
  S.boxes[cid]=b; S.sel=cid; S.dirty=true; renderAll();
}
function resolveIdx(name){
  name=String(name).trim(); if(!name) return -1;
  let idx=S.tsList.findIndex(t=>t.tag===name);
  if(idx<0){ const n=Number(name); if(!Number.isNaN(n)) idx=S.tsList.findIndex(t=>t.ts===n); }
  return idx;
}
function openCopyModal(){
  if(!S.sel){ $('#status').innerHTML='<span class="dirty">select a cow first</span>'; return; }
  $('#copyInfo').textContent='Cow '+S.sel+'  •  source frame '+S.tag;
  $('#copyFrom').value=''; $('#copyTo').value=''; $('#copyResult').textContent='';
  $('#copybox').style.display='block';
}
async function applyCopyRange(){
  if(!S.sel) return;
  const i0=resolveIdx($('#copyFrom').value), i1=resolveIdx($('#copyTo').value);
  if(i0<0||i1<0){ $('#copyResult').innerHTML='<span class="dirty">frame(s) not found</span>'; return; }
  const lo=Math.min(i0,i1), hi=Math.max(i0,i1);
  const targets=[]; for(let i=lo;i<=hi;i++) targets.push(S.tsList[i].ts);
  const r=await (await fetch('/api/copy_cow',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({cow_id:S.sel, source_ts:S.ts, target_ts:targets})})).json();
  if(!r.ok){ $('#copyResult').innerHTML='<span class="dirty">'+(r.error||'failed')+'</span>'; return; }
  const intoRev=(r.added_into_reviewed||[]).length;
  const unk=(r.skipped_unknown||[]).length;
  $('#copyResult').innerHTML='Added to '+r.added.length+' frame(s)'+
    (intoRev?(' ('+intoRev+' of them already-reviewed — cow added as unreviewed, please re-check)'):'')+
    '. Skipped '+r.skipped_exists.length+' (cow already present)'+
    (unk?(', '+unk+' (unknown frame)'):'')+'. '+
    'Open a target frame to check/adjust the pasted cow, then Save.';
}
// ---------- WildTrack export ----------
function exportDestPreview(){
  const root=($('#expRoot').value.trim()||(S.cfg&&S.cfg.export_dir)||'output_3dboxes/exports');
  let sub=$('#expSub').value.trim();
  if(!sub){
    const i0=resolveIdx($('#expFrom').value), i1=resolveIdx($('#expTo').value);
    if(i0>=0&&i1>=0){
      let a=S.tsList[i0].ts, b=S.tsList[i1].ts;
      if(b<a){const t=a;a=b;b=t;}
      sub=a+'-'+b+'_2Dannotations';
    } else sub='<start_ts>-<end_ts>_2Dannotations';
  }
  $('#expDest').textContent=root.replace(/\/+$/,'')+'/'+sub;
}
function openExportModal(){
  $('#expFrom').value=S.tag||''; $('#expTo').value=S.tag||'';
  $('#expRoot').value=''; $('#expSub').value=''; $('#expResult').innerHTML='';
  exportDestPreview();
  $('#exportbox').style.display='block';
}
async function applyExport(){
  const body={start:$('#expFrom').value, end:$('#expTo').value,
              interval:$('#expInterval').value,
              export_dir:$('#expRoot').value.trim(),
              subfolder:$('#expSub').value.trim()};
  $('#expResult').innerHTML='exporting…';
  let r=null;
  try{
    r=await (await fetch('/api/export_wildtrack',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)})).json();
  }catch(err){ $('#expResult').innerHTML='<span class="dirty">request failed</span>'; return; }

  const lim=(a,n)=>a.slice(0,n).join(', ')+(a.length>n?(' … (+'+(a.length-n)+' more)'):'');
  if(!r.ok){
    let h='<span class="dirty">'+(r.error||'export failed')+'</span>';
    if(r.missing_timestamps&&r.missing_timestamps.length){
      h+='<div style="margin-top:6px"><b>Missing timestamps ('+r.missing_timestamps.length+'):</b><br>'
        +r.missing_timestamps.join(', ')+'</div>';
    }
    if(r.unreviewed&&r.unreviewed.length){
      h+='<div style="margin-top:6px"><b>Frames without a saved review file ('+r.unreviewed.length+'):</b><br>'
        +r.unreviewed.map(u=>u.tag+' ('+u.ts+')').join(', ')+'</div>';
    }
    if(r.warnings&&r.warnings.length) h+='<div style="margin-top:6px">'+r.warnings.join('<br>')+'</div>';
    $('#expResult').innerHTML=h;
    return;
  }
  let h='<span style="color:#5fd18a">✔ exported '+r.n_frames+' frame(s), '+r.n_boxes+' box(es)</span>'
      +'<div style="margin-top:6px">'+r.dest+'</div>'
      +'<div>annotations_positions/ • evaluations/gt_mota.txt • evaluations/gt_moda.txt</div>';
  if(r.n_out_of_bounds) h+='<div class="dirty" style="margin-top:6px">'+r.n_out_of_bounds
      +' box centre(s) outside the grid bounds (clamped)</div>';
  if(r.n_missing_labelme) h+='<div class="dirty">'+r.n_missing_labelme
      +' frame(s) had no labelme file → 2D views projected from the 3D box: '+lim(r.missing_labelme,6)+'</div>';
  if(r.warnings&&r.warnings.length) h+='<div class="dirty">'+r.warnings.join('<br>')+'</div>';
  $('#expResult').innerHTML=h;
}
// ---------- annotated video export ----------
function videoDestPreview(){
  const root=($('#vidRoot').value.trim()||(S.cfg&&S.cfg.export_dir)||'output_3dboxes/exports');
  let sub=$('#vidSub').value.trim(), a=null, b=null;
  const i0=resolveIdx($('#vidFrom').value), i1=resolveIdx($('#vidTo').value);
  if(i0>=0&&i1>=0){ a=S.tsList[i0].ts; b=S.tsList[i1].ts; if(b<a){const t=a;a=b;b=t;} }
  if(!sub) sub=(a!==null)?(a+'-'+b+'_video'):'<start_ts>-<end_ts>_video';
  const file=(a!==null)?('annotated_'+a+'-'+b+'.mp4'):'annotated_<start_ts>-<end_ts>.mp4';
  $('#vidDest').textContent=root.replace(/\/+$/,'')+'/'+sub+'/'+file;
}
function openVideoModal(){
  $('#vidFrom').value=S.tag||''; $('#vidTo').value=S.tag||'';
  $('#vidRoot').value=''; $('#vidSub').value=''; $('#vidResult').innerHTML='';
  videoDestPreview();
  $('#videobox').style.display='block';
}
async function applyVideoExport(){
    const body={start:$('#vidFrom').value, end:$('#vidTo').value,
              interval:$('#vidInterval').value,
              output_fps:$('#vidFps').value,
              resolution:$('#vidRes').value.trim(),
              analytics:$('#vidAnalytics').value,
              heatmap_mode:$('#vidHeatMode').value,
              export_dir:$('#vidRoot').value.trim(),
              subfolder:$('#vidSub').value.trim(),
              panel_cameras:S.panelCams.slice(0,4)};
  $('#vidResult').innerHTML='rendering… (this can take a while: 5 panels per frame)';
  $('#vidApply').disabled=true;
  let r=null;
  try{
    r=await (await fetch('/api/export_video',{method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)})).json();
  }catch(err){ $('#vidApply').disabled=false;
    $('#vidResult').innerHTML='<span class="dirty">request failed</span>'; return; }
  $('#vidApply').disabled=false;

  const lim=(a,n)=>a.slice(0,n).join(', ')+(a.length>n?(' … (+'+(a.length-n)+' more)'):'');
  if(!r.ok){
    let h='<span class="dirty">'+(r.error||'export failed')+'</span>';
    if(r.missing_timestamps&&r.missing_timestamps.length){
      h+='<div style="margin-top:6px"><b>Missing timestamps ('+r.missing_timestamps.length+'):</b><br>'
        +r.missing_timestamps.join(', ')+'</div>';
    }
    if(r.unreviewed&&r.unreviewed.length){
      h+='<div style="margin-top:6px"><b>Frames without a saved review file ('+r.unreviewed.length+'):</b><br>'
        +r.unreviewed.map(u=>u.tag+' ('+u.ts+')').join(', ')+'</div>';
    }
    if(r.warnings&&r.warnings.length) h+='<div style="margin-top:6px">'+r.warnings.join('<br>')+'</div>';
    $('#vidResult').innerHTML=h;
    return;
  }
  let h='<span style="color:#5fd18a">✔ rendered '+r.n_frames+' frame(s), '
      +r.duration_sec+'s @ '+r.output_fps+' fps, '+r.resolution[0]+'×'+r.resolution[1]
      +' ('+r.codec+')</span>'
      +'<div style="margin-top:6px">'+r.video_path+'</div>';
  if(r.analytics){
    const a=r.analytics;
    h+='<div style="margin-top:6px">analytics ('+a.heatmap_mode+'): '
      +a.pct_lying+'% lying, mean '+a.mean_cows_per_frame+' cow(s)/frame, '
      +'peak dwell '+a.max_dwell_sec+'s</div>'
      +'<div>'+a.heatmap_png+'</div><div>'+a.posture_csv+'</div>';
  }
  if(r.n_missing_images) h+='<div class="dirty">'+r.n_missing_images
      +' missing camera image(s) → black panel: '+lim(r.missing_images,6)+'</div>';
  if(r.warnings&&r.warnings.length) h+='<div class="dirty">'+r.warnings.join('<br>')+'</div>';
  $('#vidResult').innerHTML=h;
}
function addBox(){
  const existing=Object.keys(S.boxes).map(Number);
  let nid=1; while(existing.includes(nid)) nid++;
  const id=String(nid);
  const bd=computeBevBounds();
  const pr=S.cfg.posture_priors.standing;
  S.boxes[id]={center:[(bd.x0+bd.x1)/2,(bd.y0+bd.y1)/2],yaw:0,
    length:pr.length,width:pr.width,height:pr.height,cow_id:nid,
    posture:'standing',n_views:0,cams:[],z_base:0,source:'manual',
    n_hull_voxels:0,confidence:0.5,reviewed:false,edited:true};
  S.sel=id; S.dirty=true; renderAll();
}
function delBox(){ if(!S.sel)return; if(!confirm('Delete cow '+S.sel+'?'))return;
  delete S.boxes[S.sel]; S.sel=Object.keys(S.boxes)[0]||null; S.dirty=true; renderAll(); }
function nextCow(dir){
  const ids=Object.keys(S.boxes).sort((a,b)=>+a-+b); if(!ids.length)return;
  let i=ids.indexOf(S.sel); i=(i+dir+ids.length)%ids.length; S.sel=ids[i]; renderAll();
}
async function saveFrame(){
  const payload={tag:S.tag,boxes:S.boxes};
  let r=null;
  try{
    r=await (await fetch('/api/save/'+S.ts,{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})).json();
  }catch(err){ $('#status').innerHTML='<span class="dirty">save failed</span>'; return; }
  S.dirty=false; updateTop();
  const ri = r && r.reinterp;
  if(ri && ri.ran){
    $('#status').innerHTML='saved • re-interpolated '+ri.written.length+
      ' frame(s) → GT '+ri.next_gt+
      ' (cows '+(ri.changed.join(',')||'-')+
      (ri.deleted.length?('; removed '+ri.deleted.join(',')):'')+')';
  } else if(ri){
    const det = ri.detail ? (' — '+ri.detail) : '';
    $('#status').innerHTML='<span class="dirty">saved • no re-interp: '+
      (ri.reason||'?')+'</span>';
    $('#status').title=(ri.reason||'')+det;
    console.warn('[reinterp] skipped:', ri.reason, det, ri.diag||'');
  }
}
async function verifyTs(){
  const nv=!S.verified;
  await fetch('/api/verify/'+S.ts,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({verified:nv})});
  S.verified=nv; S.tsList[S.tsIdx].verified=nv; updateTop();
}

// ---------- BEV mouse interaction ----------
let drag=null;
$('#bev').addEventListener('mousedown',e=>{
  const r=e.target.getBoundingClientRect(); const px=e.clientX-r.left, py=e.clientY-r.top;
  const [wx,wy]=S.bev.c2w(px,py);
  // check yaw handle of selected
  if(S.sel){
    const b=S.boxes[S.sel]; const u=[Math.cos(b.yaw),Math.sin(b.yaw)];
    const tip=S.bev.w2c(b.center[0]+u[0]*b.length*0.5,b.center[1]+u[1]*b.length*0.5);
    if(Math.hypot(px-tip[0],py-tip[1])<10){drag={mode:'rot'};return;}
  }
  // hit test footprints (topmost = highest id)
  const ids=Object.keys(S.boxes).sort((a,b)=>+b-+a);
  for(const cid of ids){
    const b=S.boxes[cid];
    const dx=wx-b.center[0], dy=wy-b.center[1];
    const u=[Math.cos(b.yaw),Math.sin(b.yaw)], p=[-Math.sin(b.yaw),Math.cos(b.yaw)];
    const a=dx*u[0]+dy*u[1], bb=dx*p[0]+dy*p[1];
    if(Math.abs(a)<=b.length/2 && Math.abs(bb)<=b.width/2){
      S.sel=cid; drag={mode:'trans',lwx:wx,lwy:wy}; renderAll(); return;
    }
  }
  S.sel=null; renderAll();
});
window.addEventListener('mousemove',e=>{
  if(!drag)return;
  const cv=$('#bev'); const r=cv.getBoundingClientRect(); const px=e.clientX-r.left,py=e.clientY-r.top;
  const [wx,wy]=S.bev.c2w(px,py); const b=S.boxes[S.sel];
  if(drag.mode==='trans'){ b.center[0]+=wx-drag.lwx; b.center[1]+=wy-drag.lwy; drag.lwx=wx;drag.lwy=wy; mark(); renderAll(); }
  else if(drag.mode==='rot'){ b.yaw=Math.atan2(wy-b.center[1],wx-b.center[0]); mark(); renderAll(); }
});
window.addEventListener('mouseup',()=>{drag=null;});

// ---------- camera-view click to select ----------
document.querySelectorAll('.camwrap canvas').forEach(cv=>{
  cv.addEventListener('click',e=>{
    const slot=+e.target.parentElement.dataset.slot; const v=S.camView[slot]; if(!v)return;
    const r=e.target.getBoundingClientRect(); const px=e.clientX-r.left, py=e.clientY-r.top;
    let best=null,bd=1e9;
    for(const cid in S.boxes){
      const b=S.boxes[cid]; const c=[b.center[0],b.center[1],b.height/2];
      const q=v.toCanvas(projCam(v.cam,c)); if(!q.valid)continue;
      const d=Math.hypot(px-q.x,py-q.y); if(d<bd){bd=d;best=cid;}
    }
    if(best&&bd<50){S.sel=best;renderAll();}
  });
});

// ---------- keyboard ----------
window.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;
  const fine=e.shiftKey, coarse=e.ctrlKey||e.metaKey&&e.key!=='s';
  const step=fine?1:(e.ctrlKey?25:5);
  const yaw=(fine?1:5)*Math.PI/180;
  if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='s'){e.preventDefault();saveFrame();return;}
  if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='c'){e.preventDefault();copyCow();return;}
  if((e.ctrlKey||e.metaKey)&&e.key.toLowerCase()==='v'){e.preventDefault();pasteCow();return;}
  switch(e.key){
    case 'ArrowLeft': e.preventDefault(); translate(-step,0); break;
    case 'ArrowRight':e.preventDefault(); translate(step,0); break;
    case 'ArrowUp':   e.preventDefault(); translate(0,step); break;
    case 'ArrowDown': e.preventDefault(); translate(0,-step); break;
    case 'q': case 'Q': rotate(-yaw); break;
    case 'e': case 'E': rotate(yaw); break;
    case 'z': case 'Z': resize('length',-(fine?1:5)); break;
    case 'x': case 'X': resize('length',(fine?1:5)); break;
    case 'c': case 'C': resize('width',-(fine?1:5)); break;
    case 'v': case 'V': if(e.key==='V'&&!fine){/*fallthrough handled below*/} resize('width',(fine?1:5)); break;
    case 'g': case 'G': resize('height',-(fine?1:5)); break;
    case 't': case 'T': resize('height',(fine?1:5)); break;
    case 'p': case 'P': togglePosture(); break;
    case 'h': case 'H': toggleVis(); break;
    case 'Enter': toggleReviewed(); break;
    case 'n': case 'N': addBox(); break;
    case 'Delete': case 'Backspace': delBox(); break;
    case '[': loadFrame(S.tsIdx-1); break;
    case ']': loadFrame(S.tsIdx+1); break;
    case 'Tab': e.preventDefault(); nextCow(e.shiftKey?-1:1); break;
  }
},{passive:false});
// dedicated V (verify) key without collision: use uppercase handled here
window.addEventListener('keydown',e=>{
  if(e.target.tagName==='INPUT')return;
  if(e.key==='V'&&e.shiftKey){ /* Shift+V reserved */ }
});

// buttons
$('#prevTs').onclick=()=>loadFrame(S.tsIdx-1);
$('#nextTs').onclick=()=>loadFrame(S.tsIdx+1);
$('#gotoBtn').onclick=()=>gotoFrame($('#gotoInput').value);
$('#gotoInput').addEventListener('keydown',e=>{
  if(e.key==='Enter'){ e.preventDefault(); gotoFrame(e.target.value); }
});
$('#acceptBtn').onclick=toggleReviewed;
$('#addBtn').onclick=addBox;
$('#delBtn').onclick=delBox;
$('#postureBtn').onclick=togglePosture;
$('#copyBtn').onclick=copyCow;
$('#pasteBtn').onclick=pasteCow;
$('#copyRangeBtn').onclick=openCopyModal;
$('#copyApply').onclick=applyCopyRange;
$('#copyTo').addEventListener('keydown',e=>{ if(e.key==='Enter'){ e.preventDefault(); applyCopyRange(); }});
$('#exportBtn').onclick=openExportModal;
$('#expApply').onclick=applyExport;
['#expFrom','#expTo','#expRoot','#expSub'].forEach(s=>
  $(s).addEventListener('input',exportDestPreview));
$('#expSub').addEventListener('keydown',e=>{ if(e.key==='Enter'){ e.preventDefault(); applyExport(); }});
$('#videoBtn').onclick=openVideoModal;
$('#vidApply').onclick=applyVideoExport;
['#vidFrom','#vidTo','#vidRoot','#vidSub'].forEach(s=>
  $(s).addEventListener('input',videoDestPreview));
$('#vidSub').addEventListener('keydown',e=>{ if(e.key==='Enter'){ e.preventDefault(); applyVideoExport(); }});
$('#showAllBtn').onclick=showAllCams;
$('#hideAllBtn').onclick=hideAllCams;
$('#soloBtn').onclick=soloSelected;
$('#saveBtn').onclick=saveFrame;
$('#verifyBtn').onclick=verifyTs;
$('#helpBtn').onclick=()=>{const h=$('#help');h.style.display=h.style.display==='block'?'none':'block';};
window.addEventListener('resize',()=>{ if(S.cfg) renderAll(); });
window.addEventListener('beforeunload',e=>{ if(S.dirty){e.preventDefault();e.returnValue='';} });

// go
(async()=>{ await loadConfig(); if(S.tsList.length) await loadFrame(0);
    else document.body.insertAdjacentHTML('beforeend','<p style="padding:20px">No .jpg frames found under --dataset (expected dataset/cam_X/&lt;tag&gt;.jpg).</p>'); })();
</script>
</body></html>
"""

# For the V-key verify shortcut we bind it cleanly in JS above via button; add key:
PAGE = PAGE.replace(
    "case 'Tab': e.preventDefault(); nextCow(e.shiftKey?-1:1); break;",
    "case 'Tab': e.preventDefault(); nextCow(e.shiftKey?-1:1); break;\n"
    "    case 'y': case 'Y': verifyTs(); break;"
)

# ----------------------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Image root (dataset/cam_X/<tag>.jpg)")
    ap.add_argument("--calibration_json", default="output_barn_multi/camera_calibration.json")
    ap.add_argument("--floorplan_json", default=None,
                    help="Optional. MMCows: pass the barn floorplan.json. "
                         "JerCCows: omit (or pass 'none') -- the BEV bounds "
                         "are then derived from camera centres + boxes.")
    ap.add_argument("--boxes_dir", default=None,
                    help="Unused in annotation mode (kept for CLI "
                         "compatibility with review3d.py).")
    ap.add_argument("--review_dir", required=True,
                    help="Folder your 3D annotation JSONs are read from AND "
                         "written to. Point at an existing folder to "
                         "CONTINUE a session, or at a new/empty folder to "
                         "start fresh. Use one folder per dataset.")
    ap.add_argument("--export_dir", default="output_3dboxes/exports",
                    help="Root folder for WildTrack exports; each run writes "
                         "<export_dir>/<start_ts>-<end_ts>_2Dannotations/")
    ap.add_argument("--panel_cameras", default=None,
                    help="Comma-separated initial cameras for the 4 panels, "
                         "e.g. cam_5,cam_9,cam_11,cam_12 (default: first 4 "
                         "calibrated cameras).")
    ap.add_argument("--export_bounds", default=None,
                    help="WildTrack grid as xmin,xmax,ymin,ymax in cm. "
                         "Default: floorplan perimeter, else camera-centre "
                         "extents + --export_margin_cm.")
    ap.add_argument("--export_margin_cm", type=float, default=500.0,
                    help="Margin added around camera centres when deriving "
                         "the export grid without a floorplan.")
    ap.add_argument("--gt_dir", default=None,
                    help="Folder searched for ground-truth anchors "
                         "(default: --boxes_dir). Point this at your manual "
                         "15s annotation folder if boxes_dir has no "
                         "'interpolated' metadata.")
    ap.add_argument("--gt_mode", default="auto",
                    choices=["auto", "flag", "present", "spacing"],
                    help="How to recognise a ground-truth anchor.")
    ap.add_argument("--gt_every", type=int, default=0,
                    help="Anchor spacing in seconds for --gt_mode spacing.")
    ap.add_argument("--gt_max_scan", type=int, default=600,
                    help="Max future files inspected when hunting an anchor.")
    ap.add_argument("--video_encoder", default="auto",
                    choices=["auto", "ffmpeg", "opencv"],
                    help="Backend used by the MP4 video export. 'auto' prefers "
                         "an ffmpeg binary (true H.264) and falls back to "
                         "cv2.VideoWriter.")
    ap.add_argument("--ffmpeg", default=None,
                    help="Path to an ffmpeg executable (default: $FFMPEG_BINARY, "
                         "then PATH, then imageio-ffmpeg's bundled binary).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5000)
    args = ap.parse_args()
    CFG.update(vars(args))

    if str(CFG.get("floorplan_json") or "").strip().lower() in ("", "none", "null"):
        CFG["floorplan_json"] = None

    # The calibration file defines the dataset's camera set (4 for mmcows,
    # 10 for jerccows) -- nothing about cameras is hardcoded any more.
    startup_cams = load_calibration()
    CFG["all_cams"] = list(startup_cams.keys())
    if not CFG["all_cams"]:
        ap.error("no usable cameras found in --calibration_json")
    if args.panel_cameras:
        try:
            CFG["panel_cams"] = normalize_panel_cameras(
                [c.strip() for c in args.panel_cameras.split(",")])
        except ValueError as e:
            ap.error(str(e))
    else:
        CFG["panel_cams"] = camera_names()[:N_CAMERA_PANELS]
    if len(CFG["panel_cams"]) != N_CAMERA_PANELS:
        ap.error("calibration must contain at least %d cameras"
                 % N_CAMERA_PANELS)

    os.makedirs(CFG["review_dir"], exist_ok=True)
    print(f"[annotator] annotations (read+write) = {CFG['review_dir']}")
    print(f"[annotator] calibrated cameras = {', '.join(CFG['all_cams'])}")
    print(f"[annotator] initial panels = {', '.join(CFG['panel_cams'])}")
    print(f"[annotator] floorplan = {CFG.get('floorplan_json') or 'none (auto BEV bounds)'}")
    print(f"[reviewer] export root={CFG['export_dir']}")

    # ---- annotation-mode sanity check -------------------------------------
    _imgs = _image_ts_index()
    print(f"[annotator] frames from images: {len(_imgs)}  "
          f"(dataset={CFG['dataset']})")
    if not _imgs:
        print("[annotator] !! no dataset/cam_X/<tag>.jpg found -> "
              "the UI will show an empty frame list.")
    _g = build_dir_index(gt_dir())
    print(f"[annotator] anchors = saved files in {gt_dir()} "
          f"({len(_g)} present); saving a frame re-interpolates forward "
          f"to the next saved frame.")
    # ---- video encoder sanity check --------------------------------------
    try:
        _vc, _vd = probe_video_encoder(VIDEO_DEFAULT_FPS,
                                       (VIDEO_DEFAULT_W, VIDEO_DEFAULT_H))
        if _vc:
            print("[video] encoder=%s  (cv2=%s FFMPEG=%s, ffmpeg bin=%s)"
                  % (_vc[3], _vd.get("cv2"), _vd.get("cv2_ffmpeg"),
                     _vd.get("ffmpeg_bin")))
        else:
            print("[video] !! no working encoder: %s" % _vd.get("tried"))
            print("[video]    install ffmpeg (or `pip install imageio-ffmpeg`) "
                  "to enable the video export.")
    except RuntimeError as e:
        print("[video] video export disabled: %s" % e)

    print(f"[reviewer] debug: http://{args.host}:{args.port}/api/reinterp_debug")
    print(f"[reviewer] video debug: http://{args.host}:{args.port}/api/video_debug")
    print(f"[reviewer] open http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)