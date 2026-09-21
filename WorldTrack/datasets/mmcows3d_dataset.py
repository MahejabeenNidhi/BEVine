# WorldTrack/datasets/mmcows3d_dataset.py
import os
import json
import numpy as np

from datasets.mmcows_dataset import MmCows


class MmCows3D(MmCows):
    """MmCows variant that can additionally load 3D bounding-box annotations.

    ``annotation_mode`` decides how much of the 3D data is exposed, WITHOUT
    touching the filesystem:

      '3d'        full 3D boxes            -> has_3d=True
      '2d'        no boxes, but the metric centres from centers_3d.json are
                  still used, so the BEV centre GT is BIT-IDENTICAL to a 3D
                  run                      -> has_3d=False
      '2d_strict' nothing 3D at all; centres come from positionID (the true
                  legacy 2D baseline)      -> has_3d=False
    """

    VALID_MODES = ('3d', '2d', '2d_strict')

    def __init__(self, root, annotation_mode='3d'):
        annotation_mode = str(annotation_mode).lower()
        if annotation_mode not in self.VALID_MODES:
            raise ValueError(
                f"annotation_mode must be one of {self.VALID_MODES}, "
                f"got {annotation_mode!r}"
            )
        self.annotation_mode = annotation_mode
        self._use_center_cache = (annotation_mode != '2d_strict')

        super().__init__(root)                      # calib, frame_list, centre cache
        self.__name__ = 'MmCows3D'

        self.annotations_3d = {}                    # frame_key -> {cow_id(int): box}
        ann3d_dir = os.path.join(self.root, '3D_annotations')
        self.has_3d_on_disk = os.path.isdir(ann3d_dir)
        self.has_3d = (annotation_mode == '3d') and self.has_3d_on_disk

        if self.has_3d:
            self._load_3d_annotations(ann3d_dir)
            print(f"  MmCows3D[{annotation_mode}]: loaded 3D annotations for "
                  f"{len(self.annotations_3d)} frames from {ann3d_dir}")
            self._verify_center_cache()
        elif annotation_mode == '3d':
            print(f"  MmCows3D[3d]: ⚠ no 3D_annotations/ in {self.root} "
                  f"— falling back to 2D-only behaviour.")
            if not getattr(self, 'has_gt', True):
                print(f"  MmCows3D[3d]: sequence has NO annotations at all "
                      f"-> inference-only (metrics will skip it).")
        else:
            if (annotation_mode == '2d' and not self._center_overrides
                    and getattr(self, 'has_gt', True)):
                raise FileNotFoundError(
                    f"annotation_mode='2d' is DEFINED as \"no 3D boxes, but "
                    f"the metric centres of the 3D run\", so it needs a "
                    f"usable centers_3d.json — none was found (or it was "
                    f"empty/unreadable) in {self.root}.\n"
                    f"  Create it ONCE from the 3D annotations:\n"
                    f"      from datasets.mmcows3d_dataset import MmCows3D\n"
                    f"      MmCows3D(r'{self.root}', annotation_mode='3d')"
                    f".export_center_cache()\n"
                    f"  Use annotation_mode='2d_strict' only if you really "
                    f"want the legacy positionID centres (NOT comparable "
                    f"with a 3D run)."
                )
            src = ('centers_3d.json' if self._center_overrides
                   else 'positionID (legacy 2D)')
            skipped = ' (3D_annotations/ present but IGNORED)' \
                if self.has_3d_on_disk else ''
            print(f"  MmCows3D[{annotation_mode}]: 3D boxes disabled{skipped}; "
                  f"BEV centres from {src}.")

    # ------------------------------------------------------------------
    def _load_3d_annotations(self, ann3d_dir):
        for fname in sorted(os.listdir(ann3d_dir)):
            if not fname.endswith('.json'):
                continue
            frame_key = self._parse_frame_key(fname)
            with open(os.path.join(ann3d_dir, fname)) as f:
                data = json.load(f)
            boxes = data.get('boxes', {})
            # key by int cow_id so it matches the 2D personID space
            self.annotations_3d[frame_key] = {
                int(cow_id): box for cow_id, box in boxes.items()
            }

        # ------------------------------------------------------------------
    def _verify_center_cache(self, tol_cm=1.0):
        """Check centers_3d.json really equals the 3D box centroids.

        A stale cache (exported before the 3D annotations were fixed, or
        with a different key convention) would make the 2D run train on
        DIFFERENT centres than the 3D run without any error message.
        """
        if not self._center_overrides:
            print("  MmCows3D: no centers_3d.json in this sequence — a "
                  "future annotation_mode='2d' run would NOT match this "
                  "3D run. Run export_center_cache() once.")
            return

        deltas, n = [], 0
        for fk, boxes in self.annotations_3d.items():
            cached = self._center_overrides.get(fk)
            if not cached:
                continue
            for cid, b in boxes.items():
                c = cached.get(int(cid))
                if c is None:
                    continue
                deltas.append(max(abs(float(b['center'][0]) - c[0]),
                                  abs(float(b['center'][1]) - c[1])))
                n += 1

        missing_f = [f for f in self.annotations_3d
                     if f not in self._center_overrides]
        if not n:
            print("  MmCows3D: ⚠ centre cache shares NO (frame, cow) key "
                  "with 3D_annotations/ — key mismatch. Re-export it.")
            return

        worst = max(deltas)
        if worst > tol_cm or missing_f:
            print(f"  MmCows3D: ⚠ centre cache looks STALE "
                  f"(max |Δ| = {worst:.2f} cm over {n} centres, "
                  f"{len(missing_f)} annotated frame(s) missing). "
                  f"Re-run export_center_cache().")
        else:
            print(f"  MmCows3D: centre cache verified against the 3D boxes "
                  f"({n} centres, max |Δ| = {worst:.4f} cm).")

    # ------------------------------------------------------------------
    def get_3d_boxes(self, frame_key):
        """Return {cow_id: box_dict} for a frame, or {} if none exist."""
        return self.annotations_3d.get(frame_key, {})

    # get_worldgrid_from_center() moved to MmCows (base class).

    def get_center_overrides(self, frame_key):
        """Metric body centres. 3D boxes win per-cow; cache fills the gaps.

        The old version was all-or-nothing per FRAME: if a frame had a
        single 3D box, every other animal in that frame lost its cached
        centre and silently reverted to the positionID point — which the
        2D run (cache only) would NOT do. Merging per cow keeps the two
        runs bit-identical.
        """
        merged = dict(super().get_center_overrides(frame_key))
        boxes = self.annotations_3d.get(frame_key)
        if boxes:
            merged.update(
                {int(cid): [float(b['center'][0]), float(b['center'][1])]
                 for cid, b in boxes.items()}
            )
        return merged

    # ------------------------------------------------------------------
    def export_center_cache(self, path=None):
        """Write {frame: {cow_id: [x_cm, y_cm]}} next to the annotations.

        Run this ONCE on the full (3D) dataset.  Keeping the resulting
        centers_3d.json in the sequence directory lets a 2D-only run be
        supervised at *exactly* the same centres as the 3D run.
        """
        path = path or os.path.join(self.root, 'centers_3d.json')
        out = {str(k): {str(cid): [float(b['center'][0]), float(b['center'][1])]
                        for cid, b in boxes.items()}
               for k, boxes in self.annotations_3d.items()}
        with open(path, 'w') as f:
            json.dump(out, f)
        print(f"  MmCows3D: wrote {len(out)} frames of metric centres -> {path}")
        return path

    # ------------------------------------------------------------------
    def infer_position_encoding(self,
                                candidates=(1, 2, 2.5, 5, 10, 20, 25),
                                axes=('height', 'width')):
        """Recover the grid that was used to ENCODE positionID.

        Decodes every positionID with each candidate (cell_size, axis) and
        compares it against the 3D centroid of the SAME cow.  The winner is
        the one with the smallest median distance; copy it into
        MmCows.ANN_GRID_CELL_SIZE / ANN_STRIDE_AXIS.  A small median with a
        large *mean signed* offset means only the origin is off -> subtract
        it from ANN_X_MIN_CM / ANN_Y_MIN_CM.
        """
        if not self.has_3d:
            print("  infer_position_encoding: needs 3D_annotations/ — skipped.")
            return None

        pairs = []
        ann_dir = os.path.join(self.root, 'annotations_positions')
        for fname in sorted(os.listdir(ann_dir)):
            if not fname.endswith('.json'):
                continue
            boxes = self.annotations_3d.get(self._parse_frame_key(fname), {})
            if not boxes:
                continue
            with open(os.path.join(ann_dir, fname)) as f:
                for ped in json.load(f):
                    b = boxes.get(int(ped['personID']))
                    if b is None:
                        continue
                    pairs.append((int(ped['positionID']),
                                  float(b['center'][0]), float(b['center'][1])))
        if not pairs:
            print("  infer_position_encoding: no matching 2D/3D ids.")
            return None

        print(f"\n  positionID encoding search  ({len(pairs)} paired samples)")
        print(f"  {'cell(cm)':>9}{'axis':>8}{'median(cm)':>12}"
              f"{'mean dx':>10}{'mean dy':>10}")
        best = None
        for cell in candidates:
            W = int((self.x_max_cm - self.ANN_X_MIN_CM) / cell)
            H = int((self.y_max_cm - self.ANN_Y_MIN_CM) / cell)
            for axis in axes:
                dx, dy, d = [], [], []
                for pos, cx, cy in pairs:
                    gx, gy = ((pos // H, pos % H) if axis == 'height'
                              else (pos % W, pos // W))
                    if gx >= W or gy >= H:
                        d.append(1e9); dx.append(0.0); dy.append(0.0); continue
                    x = self.ANN_X_MIN_CM + (gx + 0.5) * cell
                    y = self.ANN_Y_MIN_CM + (gy + 0.5) * cell
                    dx.append(x - cx); dy.append(y - cy)
                    d.append(float(np.hypot(x - cx, y - cy)))
                med = float(np.median(d))
                print(f"  {cell:>9}{axis:>8}{med:>12.1f}"
                      f"{np.mean(dx):>10.1f}{np.mean(dy):>10.1f}")
                if best is None or med < best[0]:
                    best = (med, cell, axis, float(np.mean(dx)), float(np.mean(dy)))
        print(f"  -> ANN_GRID_CELL_SIZE = {best[1]}, "
              f"ANN_STRIDE_AXIS = '{best[2]}'  (median {best[0]:.1f} cm, "
              f"residual mean offset {best[3]:+.1f}, {best[4]:+.1f} cm)\n")
        return best