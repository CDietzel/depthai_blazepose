import marshal
import pickle
import sys
from math import cos, sin, sqrt
from pathlib import Path
from string import Template

import cv2
import depthai as dai
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.fromnumeric import trace
from tslearn.barycenters import softdtw_barycenter
from tslearn.metrics import soft_dtw_alignment
from tslearn.preprocessing import TimeSeriesResampler

import mediapipe_utils as mpu
from BlazeposeRenderer import LINES_BODY, BlazeposeRenderer
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
                # keypoints_list.append(pose.landmarks_world)
                keypoints_list.append(pose.norm_landmarks)
                # keypoints_list.append(pose.landmarks)
        keypoints = np.array(keypoints_list)
        return keypoints



def main():
    runner = PoseEstimatorRunner()
    poses_path = "/home/locobot/Documents/Repos/depthai_blazepose/outputs/6.pickle"

    poses = runner.load_poses(poses_path)
    keypoints = runner.extract_keypoints(poses)

    r_shoulder_key_orig = keypoints[:, 14]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection="3d")
    fig.set_tight_layout(True)
    ax.plot(
        r_shoulder_key_orig[:, 0], r_shoulder_key_orig[:, 1], r_shoulder_key_orig[:, 2]
    )
    x = ax.get_xlim()
    y = ax.get_ylim()
    z = ax.get_zlim()
    ax.set_xlabel("X-axis", fontweight="bold")
    ax.set_ylabel("Y-axis", fontweight="bold")
    ax.set_zlabel("Z-axis", fontweight="bold")
    ax.set_title("Initial Right Shoulder Locations (m)")
    ax.view_init(elev=0, azim=0)
    # plt.savefig("1.png", bbox_inches="tight")

    plt.show()
    pass


if __name__ == "__main__":
    main()
