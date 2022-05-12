import marshal
import pickle
import sys
from math import cos, sin
from pathlib import Path
from string import Template

import cv2
import depthai as dai
import numpy as np
from numpy.core.fromnumeric import trace

import mediapipe_utils as mpu
from FPS import FPS, now
from o3d_utils import Visu3D
import RLSEstimator


class PoseEstimatorRunner:
    def __init__(self, poses_path):
        self.skeleton = (
            (11, 13),
            (12, 14),
            (13, 15),
            (14, 16),
            (23, 25),
            (24, 26),
            (25, 27),
            (26, 28),
            (27, 29),
            (28, 30),
            (27, 31),
            (28, 32),
            (29, 31),
            (30, 32),
        )
        self.pairs = (
            (0, 2),
            (1, 3),
            (4, 6),
            (5, 7),
            (6, 8),
            (6, 10),
            (7, 9),
            (7, 11),
            (8, 12),
            (9, 13),
            (10, 12),
            (11, 13),
        )
        self.load_poses(poses_path, 20, 80)
        self.estimator = RLSEstimator()

    def load_poses(self, poses_path, first=0, last=-1):
        with open(poses_path, "rb") as file:
            self.poses = pickle.load(file)[first:last]
        self.keypoints = self._extract_keypoints()
        self.distances = self.calc_distances()

    def save_poses(self, poses_path):
        with open(poses_path, "wb") as file:
            pickle.dump(self.poses, file, protocol=pickle.HIGHEST_PROTOCOL)

    def _extract_keypoints(self):
        keypoints_list = []
        for pose in self.poses:
            if pose is not None:
                keypoints_list.append(pose.landmarks_world)
        keypoints = np.array(keypoints_list)
        return keypoints

    def calc_distances(self):
        distances_list = []
        for keypoints in self.keypoints:
            distances = []
            for bone in self.skeleton:
                joints = np.take(keypoints, bone, axis=0)
                distance = np.linalg.norm(joints[0] - joints[-1])
                distances.append(distance)
            distances_list.append(distances)
        return np.array(distances_list)


if __name__ == "__main__":
    poses_path = "/home/locobot/Documents/Repos/depthai_blazepose/outputs/moving_no_smoothing.pickle"
    runner = PoseEstimatorRunner(poses_path)
