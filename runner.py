import marshal
import pickle
import sys
from math import cos, sin
from pathlib import Path
from string import Template
from math import sqrt

import cv2
import depthai as dai
import numpy as np
from numpy.core.fromnumeric import trace

import mediapipe_utils as mpu
from FPS import FPS, now
from o3d_utils import Visu3D
from RLSEstimator import RLSEstimator


class PoseEstimatorRunner:
    def __init__(self):
        pass

    def load_poses(self, poses_path, first=0, last=-1):
        with open(poses_path, "rb") as file:
            return pickle.load(file)[first:last]

    def save_poses(self, poses, poses_path):
        with open(poses_path, "wb") as file:
            pickle.dump(poses, file, protocol=pickle.HIGHEST_PROTOCOL)

    def extract_keypoints(self, poses):
        keypoints_list = []
        for pose in poses:
            if pose is not None:
                keypoints_list.append(pose.landmarks_world)
        keypoints = np.array(keypoints_list)
        return keypoints

    def calc_distances(self, keypoints_arr, skeleton):
        distances_list = []
        for keypoints in keypoints_arr:
            distances = []
            for bone in skeleton:
                joints = np.take(keypoints, bone, axis=0)
                distance = np.linalg.norm(joints[0] - joints[-1])
                distances.append(distance)
            distances_list.append(distances)
        return np.array(distances_list)

    def estimate_distances(self, distances_arr):
        distances_t = distances_arr.T
        estimated_list = []
        for distances in distances_t:
            estimated = self._estimate_distance(distances)
            estimated_list.append(estimated)
        return np.array(estimated_list).T

    def _estimate_distance(self, measurements):
        estimator = RLSEstimator()
        estimates = []
        for measurement in measurements:
            estimate = estimator.estimate(measurement)
            estimates.append(estimate)
        return estimates

    def refine_keypoints(self, keypoints_arr, estimates_arr, skeleton, pairs):
        refined_keypoints = keypoints_arr
        for i, (keypoints, estimates) in enumerate(zip(keypoints_arr, estimates_arr)):
            for pair in pairs:
                quad = np.take(skeleton, pair, axis=0)
                mid = np.bincount(quad.flatten()).argmax()
                lr = quad[quad != mid]
                left = lr[0]
                right = lr[1]
                A = keypoints[left]
                B = keypoints[right]
                C = keypoints[mid]
                dA = estimates[pair[0]]
                dB = estimates[pair[1]]
                newC = self._refine_distance(A, B, C, dA, dB)
                refined_keypoints[i][mid] = newC
        return refined_keypoints

    def _refine_distance(self, A, B, C, dAC, dBC):
        dAB = np.linalg.norm(A - B)
        AB = B - A
        AB = AB / np.linalg.norm(AB)  # x axis
        AC = C - A
        AC = AC / np.linalg.norm(AC)
        N = np.cross(AB, AC)
        N = N / np.linalg.norm(N)  # z axis
        V = np.cross(AB, N)
        V = V / np.linalg.norm(V)  # Y axis
        u = A + AB
        v = A + V
        n = A + N
        S = np.array(
            [
                [A[0], u[0], v[0], n[0]],
                [A[1], u[1], v[1], n[1]],
                [A[2], u[2], v[2], n[2]],
                [1, 1, 1, 1],
            ]
        )
        D = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 1, 1, 1]])
        Sinv = np.linalg.inv(S)
        M = D @ Sinv
        Minv = np.linalg.inv(M)
        Aaug = np.append(A, [1])
        Baug = np.append(B, [1])
        Caug = np.append(C, [1])
        # Enter 2D plane
        # A2D = (M @ Aaug)[0:2]
        # B2D = (M @ Baug)[0:2]
        C2D = (M @ Caug)[0:2]
        num = (-(dBC**2)) + (dAB**2) + (dAC**2)
        new_Cx = num / (2 * dAB)
        new_Cy = sqrt((dAC**2) - (new_Cx**2))
        if C2D[1] < 0:
            new_Cy = -new_Cy
        new_C2D = np.array([new_Cx, new_Cy, 0, 1])
        # Back to 3D
        new_C = Minv @ new_C2D
        return new_C[0:3]


def main():
    runner = PoseEstimatorRunner()
    poses_path = "/home/locobot/Documents/Repos/depthai_blazepose/outputs/moving_no_smoothing.pickle"
    skeleton = (
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
    pairs = (
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
    poses = runner.load_poses(poses_path, 20, 80)
    keypoints = runner.extract_keypoints(poses)
    distances = runner.calc_distances(keypoints, skeleton)
    estimates = runner.estimate_distances(distances)
    refined_keypoints = runner.refine_keypoints(keypoints, estimates, skeleton, pairs)

    



if __name__ == "__main__":
    main()

