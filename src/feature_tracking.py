import rospy
import cv2
import numpy as np

from utils_py2 import open_fifo, read_np, send_np, read_bytes

def visualize_tracking(frame, xy0, xy1):
    vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    for kp0, kp1 in zip(xy0, xy1):
        cv2.line(vis, tuple(kp0), tuple(kp1), (0, 255, 0), 1)
    for kp in xy1:
        cv2.circle(vis, (int(kp[0]), int(kp[1])), 3, (0, 255, 0), 1)

    cv2.imshow("Tracking", vis)
    cv2.waitKey(3)

class FeatureTrackingExternal(object):
    def __init__(self, config):
        self.config = config

        # Pipe for transferring images
        self.fifo_images = open_fifo('/tmp/maplab_tracking_images', 'wb')
        self.fifo_matches = open_fifo('/tmp/maplab_tracking_matches', 'rb')

        self.debug = True

    def track(
            self, frame0, frame1, xy0, xy1, scores0, scores1, scales0, scales1,
            descriptors0, descriptors1, track_ids0):
        # Transmit images for processing
        cv_success0, cv_binary0 = cv2.imencode('.png', frame0)
        cv_success1, cv_binary1 = cv2.imencode('.png', frame1)
        assert(cv_success0 and cv_success1)
        send_np(self.fifo_images, cv_binary0)
        send_np(self.fifo_images, cv_binary1)

        send_np(self.fifo_images, xy0)
        send_np(self.fifo_images, scores0)
        send_np(self.fifo_images, descriptors0)

        send_np(self.fifo_images, xy1)
        send_np(self.fifo_images, scores1)
        send_np(self.fifo_images, descriptors1)

        matches = read_np(self.fifo_matches, np.int32)
        valid = matches > -1

        if self.debug:
            visualize_tracking(frame1, xy0[valid], xy1[matches[valid]])

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
    def __init__(self, config):
        self.config = config

        # LK tracker settings
        self.lk_params = dict(
            winSize  = (15, 15),
            maxLevel = 2,
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01))
        self.lk_max_step_px = 2.0
        self.lk_merge_tracks_thr_px = 3

        self.debug = True

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

            if d[i] >= self.lk_max_step_px:
                continue

            if (y < 0 or y >= frame1.shape[0] or
                x < 0 or x >= frame1.shape[1]):
                continue

            if mask[y, x]:
                keep.append(i)
                cv2.circle(mask, (x, y), self.lk_merge_tracks_thr_px, 0,
                    cv2.FILLED)
        keep = np.array(keep).astype(np.int)

        if self.debug:
            visualize_tracking(frame0, xy0[keep].astype(np.int),
                p1[keep].reshape(-1, 2).astype(np.int))

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
