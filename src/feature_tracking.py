import rospy
import cv2
import numpy as np

from utils_py2 import open_fifo, read_np, send_np, read_bytes

def visualize_tracking(frame, xy0, xy1, index):
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for kp0, kp1 in zip(xy0, xy1):
        cv2.line(vis, tuple(kp0), tuple(kp1), (0, 255, 0), 1)
    for kp in xy1:
        cv2.circle(vis, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), 1)

    cv2.imshow("Tracking {index}".format(index=index), vis)
    cv2.waitKey(3)

class FeatureTrackingExternal(object):
    def __init__(self, config, index):
        self.config = config
        self.index = index

    def track(
            self, frame0, frame1, xy0, xy1, scores0, scores1, scales0, scales1,
            descriptors0, descriptors1, track_ids0):
        # Encode images.
        cv_success0, cv_binary0 = cv2.imencode('.png', frame0)
        cv_success1, cv_binary1 = cv2.imencode('.png', frame1)
        assert(cv_success0 and cv_success1)

        # Lock thread for transmission duration.
        self.config.lock_tracking.acquire()

        # Send images and features to external module.
        send_np(self.config.fifo_tracking_out, cv_binary0)
        send_np(self.config.fifo_tracking_out, cv_binary1)

        send_np(self.config.fifo_tracking_out, xy0)
        send_np(self.config.fifo_tracking_out, scores0)
        send_np(self.config.fifo_tracking_out, descriptors0)

        send_np(self.config.fifo_tracking_out, xy1)
        send_np(self.config.fifo_tracking_out, scores1)
        send_np(self.config.fifo_tracking_out, descriptors1)

        # Receive matches (where -1 means to match found).
        matches = read_np(self.config.fifo_tracking_in, np.int32)
        valid = matches > -1

        if self.config.debug_tracking:
            visualize_tracking(
                frame1, xy0[valid], xy1[matches[valid]], self.index)

        self.config.lock_tracking.release()

        # Filter out invalid matches (i.e. points that do not have a
        # correspondence in the previous frame)
        xy1 = xy1[matches[valid]]
        scores1 = scores1[matches[valid]]
        scales1 = scales1[matches[valid]]
        descriptors1 = descriptors1[matches[valid]]

        # Keep consistent track ids
        track_ids1 = track_ids0[valid]

        return xy1, scores1, scales1, descriptors1, track_ids1

class FeatureTrackingLK(object):
    def __init__(self, config, index):
        self.config = config
        self.index = index

        # LK tracker settings
        self.lk_params = dict(
            winSize  = (self.config.lk_window_size, self.config.lk_window_size),
            maxLevel = self.config.lk_max_level,
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                self.config.lk_stop_criteria_steps,
                self.config.lk_stop_criteria_eps))

    def track(
            self, frame0, frame1, xy0, xy1, scores0, scores1, scales0, scales1,
            descriptors0, descriptors1, track_ids0):
        # Use optical from to determine keypoint movement
        p0 = xy0.reshape(-1, 1, 2).astype(np.float32)
        p1, _, _ = cv2.calcOpticalFlowPyrLK(
            frame0, frame1, p0, None, **self.lk_params)
        p0r, _, _ = cv2.calcOpticalFlowPyrLK(
            frame1, frame0, p1, None, **self.lk_params)

        # Determine quality based on:
        #   1) distance between the optical flow predictions
        #   2) eliminate tracks that have converged to the same point
        #      (prefer keeping smaller track ids which means the track is
        #      is longer)
        d = abs(p0 - p0r).reshape(-1, 2).max(-1)
        mask = np.ones((frame1.shape[0], frame1.shape[1]))
        idxs = np.argsort(track_ids0)

        keep = []
        for i in idxs:
            x, y = xy0[i].astype(np.int32)

            if d[i] >= self.config.lk_max_step_error_px:
                continue

            if (y < 0 or y >= frame1.shape[0] or
                x < 0 or x >= frame1.shape[1]):
                continue

            if mask[y, x]:
                keep.append(i)
                cv2.circle(mask, (x, y), self.config.lk_merge_tracks_thr_px, 0,
                    cv2.FILLED)
        keep = np.array(keep).astype(np.int)

        if self.config.debug_tracking:
            self.config.lock_tracking.acquire()
            visualize_tracking(frame1, xy0[keep].astype(np.int),
                p1[keep].reshape(-1, 2).astype(np.int), self.index)
            self.config.lock_tracking.release()

        # Drop bad keypoints
        if keep.size != 0:
            xy1 = p1[keep].reshape(-1, 2)
            scores1 = scores0[keep]
            scales1 = scales0[keep]
            descriptors1 = descriptors0[keep]
            track_ids1 = track_ids0[keep]
        else:
            xy1 = []
            scores1 = []
            scales1 = []
            descriptors1 = []
            track_ids1 = []

        return xy1, scores1, scales1, descriptors1, track_ids1
