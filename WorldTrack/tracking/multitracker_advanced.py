# WorldTrack/tracking/multitracker_advanced.py

import torch
from collections import OrderedDict, deque
from typing import List
import numpy as np
from tracking import matching_extended as matching
from tracking.kalman_filter_extended import ExtendedKalmanFilter, MotionModel


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


class STrackAdvanced(BaseTrack):
    """Enhanced track with trajectory history and adaptive motion"""

    shared_kalman = ExtendedKalmanFilter(motion_model=MotionModel.ADAPTIVE)

    def __init__(self, xy, xy_prev, score, buffer_size=30, history_size=30):
        # Position - convert to numpy immediately
        if isinstance(xy, torch.Tensor):
            self._xy = xy.cpu().numpy() if xy.is_cuda else xy.numpy()
        else:
            self._xy = np.array(xy) if not isinstance(xy, np.ndarray) else xy

        if isinstance(xy_prev, torch.Tensor):
            self._xy_prev = xy_prev.cpu().numpy() if xy_prev.is_cuda else xy_prev.numpy()
        else:
            self._xy_prev = np.array(xy_prev) if not isinstance(xy_prev, np.ndarray) else xy_prev

        # Kalman filter
        self.kalman_filter = None
        self.mean, self.covariance = None, None

        # State
        self.is_activated = False
        self.score = score
        self.tracklet_len = 0

        # Trajectory history for buffered matching
        self.position_history = deque(maxlen=history_size)
        self.position_history.append(self._xy.flatten())

        # Velocity history for motion analysis
        self.velocity_history = deque(maxlen=10)

        # Predicted future positions
        self.predicted_positions = []

        # Motion characteristics
        self.avg_speed = 0.0
        self.is_stationary = False
        self.is_turning = False

        # Features (for appearance if needed)
        self.smooth_feat = None
        self.features = deque([], maxlen=buffer_size)
        self.alpha = 0.9

    def update_motion_characteristics(self):
        """Analyze motion to choose appropriate model"""
        if len(self.velocity_history) < 3:
            return

        velocities = np.array(list(self.velocity_history))

        # Compute average speed
        speeds = np.linalg.norm(velocities, axis=1)
        self.avg_speed = np.mean(speeds)

        # Detect if stationary (cattle move slower than people)
        self.is_stationary = self.avg_speed < 10.0  # Increased from 5.0 cm/frame

        # Detect turning (change in direction)
        if len(velocities) >= 3:
            angles = []
            for i in range(len(velocities) - 1):
                v1 = velocities[i]
                v2 = velocities[i + 1]
                if np.linalg.norm(v1) > 1e-6 and np.linalg.norm(v2) > 1e-6:
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    angle = np.arccos(np.clip(cos_angle, -1, 1))
                    angles.append(angle)

            if len(angles) > 0:
                avg_angle_change = np.mean(angles)
                # Cattle turn more gradually
                self.is_turning = avg_angle_change > 0.3

    def choose_motion_model(self):
        """Choose best motion model based on recent behavior"""
        if self.is_stationary:
            return 'stationary'
        elif self.is_turning:
            return 'curvilinear'
        elif len(self.velocity_history) >= 3:
            # Check if accelerating
            velocities = np.array(list(self.velocity_history))
            speeds = np.linalg.norm(velocities, axis=1)
            if len(speeds) >= 3:
                acceleration = np.diff(speeds)
                if np.mean(np.abs(acceleration)) > 2.0:  # Significant acceleration
                    return 'acceleration'

        return 'constant_velocity'

    def predict(self):
        """Predict with adaptive motion model"""
        mean_state = self.mean.copy()

        # Choose appropriate motion model
        motion_model = self.choose_motion_model()

        if motion_model == 'stationary':
            # Don't move - just increase uncertainty
            self.covariance *= 1.1
        else:
            # Normal prediction
            self.mean, self.covariance = self.kalman_filter.predict(
                mean_state, self.covariance)

        # Store predicted position for trajectory matching
        self.predicted_positions = [self.mean[:2].copy()]

        # Predict multiple steps ahead (for lookahead matching)
        temp_mean = self.mean.copy()
        temp_cov = self.covariance.copy()
        for _ in range(3):  # Predict 3 frames ahead
            temp_mean, temp_cov = self.kalman_filter.predict(temp_mean, temp_cov)
            self.predicted_positions.append(temp_mean[:2].copy())

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            for st in stracks:
                st.predict()

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.xy)

        self.tracklet_len = 0
        self.state = TrackState.Tracked

        if frame_id == 1:
            self.is_activated = True

        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xy)

        # Update history - ensure numpy array
        new_xy = new_track.xy
        if isinstance(new_xy, torch.Tensor):
            new_xy_np = new_xy.cpu().numpy() if new_xy.is_cuda else new_xy.numpy()
        else:
            new_xy_np = np.array(new_xy) if not isinstance(new_xy, np.ndarray) else new_xy
        self.position_history.append(new_xy_np.flatten())

        # Update velocity
        if len(self.mean) >= 4:
            velocity = self.mean[2:4]
            self.velocity_history.append(velocity)

        self.update_motion_characteristics()

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id

        if new_id:
            self.track_id = self.next_id()

    def update(self, new_track, frame_id, update_feature=False):
        """Update a matched track"""
        self.frame_id = frame_id
        self.tracklet_len += 1

        # Kalman update
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xy)

        # Update position history - ensure it's numpy array
        new_xy = new_track.xy
        if isinstance(new_xy, torch.Tensor):
            new_xy_np = new_xy.cpu().numpy() if new_xy.is_cuda else new_xy.numpy()
        else:
            new_xy_np = np.array(new_xy) if not isinstance(new_xy, np.ndarray) else new_xy
        self.position_history.append(new_xy_np.flatten())

        # Update velocity history
        if len(self.mean) >= 4:
            velocity = self.mean[2:4]
            self.velocity_history.append(velocity)

        # Analyze motion
        self.update_motion_characteristics()

        self.state = TrackState.Tracked
        self.is_activated = True
        self.score = new_track.score

        if update_feature:
            self.update_features(new_track.curr_feat)

    def update_features(self, feat):
        """Update appearance features (if available)"""
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    @property
    def xy(self):
        if self.mean is None:
            # Return as numpy array
            if isinstance(self._xy, torch.Tensor):
                return self._xy.cpu().numpy() if self._xy.is_cuda else self._xy.numpy()
            return np.array(self._xy) if not isinstance(self._xy, np.ndarray) else self._xy
        # Return position from Kalman state (already numpy)
        return self.mean[:2]

    @property
    def xy_prev(self):
        # Return as numpy array
        if isinstance(self._xy_prev, torch.Tensor):
            return self._xy_prev.cpu().numpy() if self._xy_prev.is_cuda else self._xy_prev.numpy()
        return np.array(self._xy_prev) if not isinstance(self._xy_prev, np.ndarray) else self._xy_prev

    def to_xyah(self):
        """Convert to [x, y, aspect, height] for compatibility"""
        if self.mean is None:
            return np.r_[self._xy, 1.0, 1.0]
        return np.r_[self.mean[:2], 1.0, 1.0]

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame,
                                      self.end_frame)


class CascadedTracker:
    """
    Advanced tracker with cascaded matching and buffered IoU in BEV.

    Matching cascade (from high to low confidence):
    1. High-confidence tracks + IoU matching
    2. Medium-confidence tracks + trajectory matching
    3. Low-confidence tracks + center distance
    4. New detections
    """

    def __init__(self, conf_thres=0.1, track_buffer=30,
                 motion_model=MotionModel.ADAPTIVE,
                 use_trajectory_matching=True,
                 use_bev_iou=True,
                 cattle_size=30):
        """
        Parameters
        ----------
        conf_thres : float
            Detection confidence threshold
        track_buffer : int
            Frames to keep lost tracks
        motion_model : MotionModel
            Type of motion model to use
        use_trajectory_matching : bool
            Enable buffered trajectory matching
        use_bev_iou : bool
            Use BEV IoU instead of center distance
        cattle_size : float
            Approximate cattle size (radius in cm)
        """
        self.tracked_stracks: List[STrackAdvanced] = []
        self.lost_stracks: List[STrackAdvanced] = []
        self.removed_stracks: List[STrackAdvanced] = []

        self.frame_id = 0
        self.det_thresh = conf_thres
        self.max_time_lost = track_buffer

        self.kalman_filter = ExtendedKalmanFilter(motion_model=motion_model)

        self.use_trajectory_matching = use_trajectory_matching
        self.use_bev_iou = use_bev_iou
        self.cattle_size = cattle_size

        # Cascade parameters
        self.cascade_depth = 3  # Number of cascade stages
        self.high_thresh = 0.5
        self.med_thresh = 0.3

    def update(self, dets, dets_prev, score):
        """
        Update tracker with new detections using cascaded matching.

        Parameters
        ----------
        dets : ndarray
            Current detections [N, 2] (x, y)
        dets_prev : ndarray
            Previous frame positions [N, 2]
        score : ndarray
            Detection scores [N]

        Returns
        -------
        output_stracks : list[STrackAdvanced]
            Active tracks
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        # Filter detections
        remain_inds = score > self.det_thresh - 0.1
        dets = dets[remain_inds]
        dets_prev = dets_prev[remain_inds]
        score = score[remain_inds]

        if len(dets) > 0:
            detections = [STrackAdvanced(xy, xy_prev, s,
                                         history_size=self.max_time_lost)
                          for (xy, xy_prev, s) in zip(dets, dets_prev, score)]
        else:
            detections = []

        # Separate tracks by confidence
        unconfirmed = []
        tracked_stracks_high = []
        tracked_stracks_med = []
        tracked_stracks_low = []

        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                # Categorize by score and track age
                age_factor = min(track.tracklet_len / 10.0, 1.0)
                effective_score = track.score * (0.5 + 0.5 * age_factor)

                if effective_score >= self.high_thresh:
                    tracked_stracks_high.append(track)
                elif effective_score >= self.med_thresh:
                    tracked_stracks_med.append(track)
                else:
                    tracked_stracks_low.append(track)

        # Combine with lost tracks
        strack_pool = joint_stracks(
            tracked_stracks_high + tracked_stracks_med + tracked_stracks_low,
            self.lost_stracks
        )

        # Predict all tracks
        STrackAdvanced.multi_predict(strack_pool)

        # ============ CASCADE MATCHING ============

        unmatched_dets = list(range(len(detections)))

        # STAGE 1: High-confidence tracks with BEV IoU
        print(f"[Cascade] Frame {self.frame_id}: Stage 1 - High confidence ({len(tracked_stracks_high)} tracks)")

        if len(tracked_stracks_high) > 0 and len(unmatched_dets) > 0:
            track_pool = tracked_stracks_high
            det_pool = [detections[i] for i in unmatched_dets]

            if self.use_bev_iou:
                # Use BEV IoU for spatial overlap
                dists = matching.iou_distance(track_pool, det_pool,
                                              box_size=self.cattle_size)
                matches, u_track, u_detection = matching.linear_assignment(
                    dists, thresh=0.5)  # IoU threshold
            else:
                # Fallback to center distance
                track_locs = [t.xy for t in track_pool]
                det_locs = [d.xy_prev for d in det_pool]
                dists = matching.center_distance(track_locs, det_locs)
                matches, u_track, u_detection = matching.linear_assignment(
                    dists, thresh=75)

            for itracked, idet in matches:
                track = track_pool[itracked]
                det = det_pool[idet]
                track.update(det, self.frame_id)
                activated_starcks.append(track)

            # Update unmatched detections
            matched_det_indices = set([idet for _, idet in matches])
            unmatched_dets = [unmatched_dets[i] for i in u_detection]

            print(f"  Matched: {len(matches)}, Unmatched tracks: {len(u_track)}, Unmatched dets: {len(unmatched_dets)}")

        # STAGE 2: Medium-confidence tracks with trajectory matching
        print(f"[Cascade] Stage 2 - Medium confidence ({len(tracked_stracks_med)} tracks)")

        if len(tracked_stracks_med) > 0 and len(unmatched_dets) > 0:
            track_pool = tracked_stracks_med
            det_pool = [detections[i] for i in unmatched_dets]

            if self.use_trajectory_matching:
                # Buffered matching using trajectory history
                dists = matching.trajectory_distance(track_pool, det_pool,
                                                     buffer_size=10)
                matches, u_track, u_detection = matching.linear_assignment(
                    dists, thresh=100)
            else:
                track_locs = [t.xy for t in track_pool]
                det_locs = [d.xy_prev for d in det_pool]
                dists = matching.center_distance(track_locs, det_locs)
                matches, u_track, u_detection = matching.linear_assignment(
                    dists, thresh=100)

            for itracked, idet in matches:
                track = track_pool[itracked]
                det = det_pool[idet]
                track.update(det, self.frame_id)
                activated_starcks.append(track)

            unmatched_dets = [unmatched_dets[i] for i in u_detection]
            print(f"  Matched: {len(matches)}, Unmatched dets: {len(unmatched_dets)}")

        # STAGE 3: Low-confidence and lost tracks
        print(
            f"[Cascade] Stage 3 - Low confidence + lost ({len(tracked_stracks_low)} + {len(self.lost_stracks)} tracks)")

        remaining_tracks = tracked_stracks_low + self.lost_stracks

        if len(remaining_tracks) > 0 and len(unmatched_dets) > 0:
            track_pool = remaining_tracks
            det_pool = [detections[i] for i in unmatched_dets]

            track_locs = [t.xy for t in track_pool]
            det_locs = [d.xy_prev for d in det_pool]
            dists = matching.center_distance(track_locs, det_locs)

            # More lenient threshold for lost tracks
            matches, u_track, u_detection = matching.linear_assignment(
                dists, thresh=150)

            for itracked, idet in matches:
                track = track_pool[itracked]
                det = det_pool[idet]

                if track.state == TrackState.Tracked:
                    track.update(det, self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            # Mark unmatched tracks as lost
            for it in u_track:
                track = track_pool[it]
                if not track.state == TrackState.Lost:
                    track.mark_lost()
                    lost_stracks.append(track)

            unmatched_dets = [unmatched_dets[i] for i in u_detection]
            print(f"  Matched: {len(matches)}, Unmatched dets: {len(unmatched_dets)}")

        # STAGE 4: Unconfirmed tracks
        print(f"[Cascade] Stage 4 - Unconfirmed ({len(unconfirmed)} tracks)")

        if len(unconfirmed) > 0 and len(unmatched_dets) > 0:
            det_pool = [detections[i] for i in unmatched_dets]

            unconf_locs = [t.xy for t in unconfirmed]
            det_locs = [d.xy_prev for d in det_pool]
            dists = matching.center_distance(unconf_locs, det_locs)

            matches, u_unconfirmed, u_detection = matching.linear_assignment(
                dists, thresh=100)

            for itracked, idet in matches:
                unconfirmed[itracked].update(det_pool[idet], self.frame_id)
                activated_starcks.append(unconfirmed[itracked])

            for it in u_unconfirmed:
                track = unconfirmed[it]
                track.mark_removed()
                removed_stracks.append(track)

            unmatched_dets = [unmatched_dets[i] for i in u_detection]
            print(f"  Matched: {len(matches)}, Unmatched dets: {len(unmatched_dets)}")

        # STAGE 5: Initialize new tracks
        print(f"[Cascade] Stage 5 - New tracks ({len(unmatched_dets)} detections)")

        for inew in unmatched_dets:
            track = detections[inew]
            if track.score < self.det_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        # Remove old lost tracks
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        # Update state
        self.tracked_stracks = [t for t in self.tracked_stracks
                                if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks,
                                             activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks,
                                             refind_stracks)

        self.lost_stracks = sub_stracks(self.lost_stracks,
                                        self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks,
                                        self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)

        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
            self.tracked_stracks, self.lost_stracks)

        output_stracks = [track for track in self.tracked_stracks
                          if track.is_activated]

        print(f"[Cascade] Output: {len(output_stracks)} tracks\n")

        return output_stracks


# Helper functions (same as original)

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


def remove_duplicate_stracks(stracksa, stracksb):
    track_a = [t.xy_prev if hasattr(t, 'xy_prev') else t.xy for t in stracksa]
    track_b = [t.xy for t in stracksb]

    pdist = matching.center_distance(track_a, track_b)
    pairs = np.where(pdist < 6)

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