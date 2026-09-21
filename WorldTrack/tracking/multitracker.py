from collections import OrderedDict
from collections import deque
from typing import List

import numpy as np
import torch

from tracking import matching
from tracking.kalman_filter import KalmanFilter
from tracking.attr3d import Attr3DState


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack(object):
    _count = 0

    track_id = 0
    is_activated = False
    state = TrackState.New

    history = OrderedDict()
    features = []
    curr_feature = None
    score = 0
    start_frame = 0
    frame_id = 0
    time_since_update = 0

    # multi-camera
    location = (np.inf, np.inf)

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def mark_lost(self):
        self.state = TrackState.Lost

    def mark_removed(self):
        self.state = TrackState.Removed


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, xy, xy_prev, score, buffer_size=30,
                 attrs=None, attr_cfg=None, freeze_when_lying=False):
        # Convert at the tensor/NumPy boundary. The entire tracker is NumPy-based
        # so bf16 tensors must be cast before they touch any NumPy code.
        if isinstance(xy, torch.Tensor):
            xy = xy.float().cpu().detach().numpy()
        if isinstance(xy_prev, torch.Tensor):
            xy_prev = xy_prev.float().cpu().detach().numpy()
        if isinstance(score, torch.Tensor):
            score = score.float().cpu().detach().item()

        # wait activate
        self._xy = xy
        self._xy_prev = xy_prev
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        # self.update_features(temp_feat)
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

        # 3D attribute state (yaw / size / posture)
        # `attrs` is the RAW per-detection measurement for THIS frame,
        # `attr_cfg` is the smoothing configuration. When attr_cfg is None
        # the whole 3D path is inert -> bit-identical legacy behaviour.
        self._attrs = attrs
        self.attr3d = Attr3DState(**attr_cfg) if attr_cfg else None

        # LYING-FREEZE: a lying cow does not translate. When enabled,
        # the track's Kalman velocity is held at zero (and its position
        # frozen in predict) for as long as the FUSED posture state says
        # "lying". Default False -> bit-identical legacy behaviour.
        self.freeze_when_lying = bool(freeze_when_lying)
        self.n_lying_zeroed = 0  # diagnostic: velocity zeroings

        self.det_index = None

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def predict(self):
        mean_state = self.mean.copy()
        if getattr(self, 'is_lying', False):
            # Zero velocity BEFORE the constant-velocity step, so
            # x += v*dt leaves the position unchanged. The covariance
            # still inflates with process noise, which keeps the
            # re-association gate sane.
            mean_state[2:] = 0.0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            # LYING-FREEZE: zero the velocity of lying tracks BEFORE the
            # constant-velocity propagation, so x += v*dt leaves their
            # position unchanged (F keeps v constant, so it stays zero).
            for i, st in enumerate(stracks):
                if getattr(st, 'is_lying', False):
                    multi_mean[i, 2:] = 0.0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov
            # advance each track's yaw filter by one frame, so a
            # coasting track keeps rotating at its estimated yaw rate
            # instead of freezing (or reading background).
            for st in stracks:
                if st.attr3d is not None:
                    st.attr3d.predict()

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.xy)
        self._absorb_attrs(self)          # seed the 3D state from frame 0

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        # self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xy
        )
        # self.update_features(new_track.curr_feat)
        self._absorb_attrs(new_track)
        if self.is_lying:
            # Posture is fused BEFORE this check, so the freshest
            # estimate decides: a lying cow's detections jitter around
            # its stall and must not integrate into phantom velocity.
            self.mean[2:] = 0.0
            self.n_lying_zeroed += 1
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        self.det_index = new_track.det_index
        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=False):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, new_track.xy)
        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score
        self._absorb_attrs(new_track)
        if self.is_lying:
            # Same reasoning as in re_activate: no phantom velocity for
            # a motionless (lying) animal.
            self.mean[2:] = 0.0
            self.n_lying_zeroed += 1
        self.det_index = new_track.det_index
        if update_feature:
            self.update_features(new_track.curr_feat)

    # ------------------------------------------------------------------
    # 3D attribute helpers (no-ops unless attr_cfg was supplied)
    # ------------------------------------------------------------------
    def _absorb_attrs(self, src_track):
        """Fuse the raw per-frame yaw/size/posture of `src_track` into
        THIS track's running 3D estimate."""
        if self.attr3d is None:
            return
        a = getattr(src_track, '_attrs', None)
        if not a:
            return
        self.attr3d.update(yaw=a.get('yaw'),
                           dims=a.get('dims'),
                           posture_prob=a.get('posture_prob'))

    @property
    def has_3d(self):
        """True once this track has fused at least one 3D measurement."""
        return (self.attr3d is not None
                and self.attr3d.n_obs > 0
                and self.attr3d.has_size
                and self.attr3d.has_yaw)

    @property
    def yaw3d(self):
        return None if self.attr3d is None else self.attr3d.yaw

    @property
    def dims3d(self):
        """(length, width, height) in CENTIMETRES."""
        return None if self.attr3d is None else self.attr3d.dims

    @property
    def posture3d(self):
        return 0 if self.attr3d is None else self.attr3d.posture_class

    @property
    def posture3d_prob(self):
        return None if self.attr3d is None else self.attr3d.posture_prob

    @property
    def is_lying(self):
        """True only when lying-freeze is enabled AND the fused,
        hysteresis-smoothed posture state says LYING.

        Guards: attr3d must exist AND must have fused at least one
        posture measurement (`posture_prob is not None`). Without the
        second guard the posture-class DEFAULT (0 = lying) would freeze
        every track that has no posture evidence yet.
        """
        # Require CLEAR, REPEATED evidence of lying: enough fused
        # observations AND the fused probability below the LOW hysteresis
        # edge. The sticky posture_class defaults to 0 (lying) and the
        # dead band keeps it there, so an uncertain head (~0.5 outputs)
        # used to freeze tracks indefinitely -> lost tracks, re-births,
        # ID switches.
        return (self.freeze_when_lying
                and self.attr3d is not None
                and self.attr3d.posture_prob is not None
                and self.attr3d.n_obs >= 3
                and self.attr3d.posture_prob
                    < (0.5 - self.attr3d.posture_hysteresis))

    def track_box_bev(self, cell_cm):
        """(cx, cy, length, width, yaw) in BEV cells from this track's
        SMOOTHED 3D state, or None if no 3D observation was fused yet."""
        if not self.has_3d:
            return None
        d = self.dims3d                                # cm
        return (float(self.xy[0]), float(self.xy[1]),
                float(d[0]) / cell_cm, float(d[1]) / cell_cm,
                float(self.yaw3d))

    def det_box_bev(self, cell_cm):
        """(cx, cy, length, width, yaw) in BEV cells from this
        DETECTION's raw per-frame attributes, or None."""
        a = getattr(self, '_attrs', None)
        if not a or a.get('yaw') is None or a.get('dims') is None:
            return None
        d = a['dims']                                  # cm
        return (float(self._xy[0]), float(self._xy[1]),
                float(d[0]) / cell_cm, float(d[1]) / cell_cm,
                float(a['yaw']))

    @property
    def xy(self):
        # if self.state == TrackState.Lost:
        #     return self.mean[:2]
        if self.mean is None:
            return self._xy
        return self.mean[:2]

    @property
    def xy_prev(self):
        return self._xy_prev

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class JDETracker:
    def __init__(self, conf_thres=0.1, track_buffer=5, dup_dist=10.0,
                 assoc_dist=75.0, use_3d_attrs=False, attr_cfg=None,
                 track_writeback_to_detections=False,
                 assoc_iou_weight=0.0, cell_cm=10.0,
                 freeze_when_lying=False):
        # use_3d_attrs=False -> `_attr_cfg` stays None and every STrack is
        # created exactly as before (ablation baseline).
        self._attr_cfg = dict(attr_cfg) if (use_3d_attrs and attr_cfg) else None
        self.tracked_stracks: List[STrack] = []
        self.lost_stracks: List[STrack] = []
        self.removed_stracks: List[STrack] = []
        self.track_writeback_to_detections = track_writeback_to_detections
        self.assoc_iou_weight = float(assoc_iou_weight)
        self.cell_cm = float(cell_cm)


        self.frame_id = 0
        self.det_thresh = conf_thres
        self.max_time_lost = track_buffer

        self.kalman_filter = KalmanFilter()

        self.dup_dist = dup_dist
        # association gate in BEV CELLS; the old hard-coded 75/100 were
        # tuned for Wildtrack's 2.5 cm cells (= 187/250 cm) and are far
        # too loose for the 10 cm mmCows grid.
        self.assoc_dist = float(assoc_dist)
        self.assoc_dist_unconfirmed = float(assoc_dist) * 4.0 / 3.0

        # Lying-freeze is forwarded to every STrack. It is INERT unless
        # the 3D attribute stream is on (posture comes from the fused
        # Attr3DState), and defaults to False -> legacy behaviour.
        self.freeze_when_lying = bool(freeze_when_lying)
        # diagnostics: track-frames where prediction was frozen
        self.n_lying_frozen = 0

    def lying_diagnostics(self):
        """Aggregate lying-freeze counters over all track lists."""
        all_tracks = (self.tracked_stracks + self.lost_stracks
                      + self.removed_stracks)
        return {
            'lying_frozen_trackframes': int(self.n_lying_frozen),
            'lying_velocity_zeroings': int(sum(
                getattr(t, 'n_lying_zeroed', 0) for t in all_tracks)),
        }

    # ------------------------------------------------------------------
    def _prepare_attrs(self, attrs, remain_inds):
        """Slice the per-detection 3D attributes with the SAME mask that
        was applied to `dets`, and return a list of plain dicts.

        `attrs` = {'yaw': (K,), 'dims': (K,3), 'posture_prob': (K,)}
        (numpy or torch). Returns None when 3D tracking is disabled.
        """
        if self._attr_cfg is None or not attrs:
            return None

        m = remain_inds
        if isinstance(m, torch.Tensor):
            m = m.detach().cpu().numpy()
        m = np.asarray(m).reshape(-1).astype(bool)

        def _np(v):
            if v is None:
                return None
            if isinstance(v, torch.Tensor):
                v = v.detach().float().cpu().numpy()
            v = np.asarray(v)
            return v[m] if v.shape[0] == m.shape[0] else None

        yaw = _np(attrs.get('yaw'))
        dims = _np(attrs.get('dims'))
        post = _np(attrs.get('posture_prob'))

        n = int(m.sum())
        out = []
        for i in range(n):
            out.append({
                'yaw': (float(yaw[i]) if yaw is not None else None),
                'dims': (np.asarray(dims[i], dtype=np.float64)
                         if dims is not None else None),
                'posture_prob': (float(post[i]) if post is not None else None),
            })
        return out

    def update(self, dets, dets_prev, score, attrs=None):
        # frame_tag is the DATASET frame number; the online tracker
        # ignores it (it counts frames internally). OfflineTracker uses
        # it to key its per-sequence export buffer.
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        score_all = (
            score.detach().float().cpu().numpy()
            if isinstance(score, torch.Tensor)
            else np.asarray(score)
        )
        n_input = int(score_all.size)
        n_tracker_pool = int(np.sum(
            score_all > self.det_thresh - 0.1
        ))
        n_birth_eligible = int(np.sum(
            score_all >= self.det_thresh
        ))

        remain_inds = score > self.det_thresh - 0.1
        dets = dets[remain_inds]
        dets_prev = dets_prev[remain_inds]
        # id_feature = id_feature[remain_inds]
        attr_list = self._prepare_attrs(attrs, remain_inds)
        # UNCONDITIONAL: decode.decoder() zeroes NMS-suppressed peaks, so
        # `score` is not sorted and remain_inds is not a prefix -> the old
        # zip() handed surviving detections the WRONG scores.
        score = score[remain_inds]

        if len(dets) > 0:
            """Detections"""
            detections = [
                STrack(xy, xy_prev, s, self.max_time_lost,
                       attrs=(attr_list[i] if attr_list is not None else None),
                       attr_cfg=self._attr_cfg,
                       freeze_when_lying=self.freeze_when_lying)
                for i, (xy, xy_prev, s) in enumerate(
                    zip(dets, dets_prev, score))
            ]
            for i, d in enumerate(detections):
                d.det_index = i  # index into the MASKED arrays
        else:
            detections = []

        """ Add newly detected tracklets to tracked_stracks"""
        unconfirmed = []
        tracked_stracks: List[STrack] = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        '''association'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        # Predict the current location with KF
        STrack.multi_predict(strack_pool)
        # diagnostic: how many track-frames had their prediction frozen
        # because the fused posture says "lying"
        self.n_lying_frozen += sum(
            1 for t in strack_pool if getattr(t, 'is_lying', False))

        strack_pool_xy = [track.xy for track in strack_pool]
        detections_xy_prev = [det.xy_prev for det in detections]

        dists = matching.center_distance(strack_pool_xy, detections_xy_prev)
        if self.assoc_iou_weight > 0 and self._attr_cfg is not None:
            iou_cost = matching.box_iou_cost(
                [t.track_box_bev(self.cell_cm) for t in strack_pool],
                [d.det_box_bev(self.cell_cm) for d in detections],
            )
            # BONUS form: overlap can only LOWER the cost. The penalty
            # form added up to assoc_iou_weight*assoc_dist cells to EVERY
            # pair, and maximally penalised pairs with a missing box
            # (box_iou_cost returns 1.0 there), which vetoes valid
            # long-distance matches whenever the yaw/size head is noisy.
            iou_bonus = 1.0 - iou_cost      # 0.0 where a box is missing
            dists = dists - self.assoc_iou_weight * self.assoc_dist * iou_bonus
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.assoc_dist)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = strack_pool[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        detections_xy = [det.xy_prev for det in detections]
        unconfirmed_xy = [track.xy for track in unconfirmed]
        # dists = matching.iou_distance(unconfirmed, detections)
        dists = matching.center_distance(unconfirmed_xy, detections_xy)
        if self.assoc_iou_weight > 0 and self._attr_cfg is not None:
            iou_cost = matching.box_iou_cost(
                [t.track_box_bev(self.cell_cm) for t in unconfirmed],
                [d.det_box_bev(self.cell_cm) for d in detections],
            )
            dists = dists + self.assoc_iou_weight * self.assoc_dist * iou_cost
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=self.assoc_dist_unconfirmed)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)
        """Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks, self.dup_dist)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        if self.frame_id <= 20 or self.frame_id % 100 == 0:
            print(
                f"[TRACK-IN] frame={self.frame_id} "
                f"input={n_input} "
                f"pool>{self.det_thresh - 0.1:.2f}={n_tracker_pool} "
                f"birth>={self.det_thresh:.2f}={n_birth_eligible} "
                f"tracked={len(self.tracked_stracks)} "
                f"lost={len(self.lost_stracks)} "
                f"removed={len(self.removed_stracks)} "
                f"output={len(output_stracks)}"
            )

        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb, dist_thresh=10.0):
    track_a = [t.xy_prev for t in stracksa]
    track_b = [t.xy for t in stracksb]
    pdist = matching.center_distance(track_a, track_b)
    pairs = np.where(pdist < dist_thresh)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
