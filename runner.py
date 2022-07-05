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
import tensorflow as tf


class TrajectoryPreprocessor:
    def __init__(self):
        pass

    def load_pickle(self, pickle_path):
        with open(pickle_path, "rb") as file:
            return pickle.load(file)

    def save_pickle(self, pickle_data, pickle_path):
        with open(pickle_path, "wb") as file:
            pickle.dump(pickle_data, file, protocol=pickle.HIGHEST_PROTOCOL)

    def extract_human_keypoints(self, poses):
        keypoints_list = []
        for pose in poses:
            if pose is not None:
                # keypoints_list.append(pose.landmarks_world)
                keypoints_list.append(pose.norm_landmarks)
                # keypoints_list.append(pose.landmarks)
        keypoints = np.array(keypoints_list)
        return keypoints


def main():
    runner = TrajectoryPreprocessor()
    human_poses_path = (
        "/home/locobot/Documents/Repos/depthai_blazepose/outputs/6.pickle"
    )
    robot_joint_path = (
        "/home/locobot/Documents/Repos/depthai_blazepose/6DoF/6.recording"
    )

    human_poses = runner.load_pickle(human_poses_path)
    robot_data = np.array(runner.load_pickle(robot_joint_path))[0, :, :]
    human_data = runner.extract_human_keypoints(human_poses)[:, 0:33, :]
    human_data = human_data.reshape(-1, 99)

    # THIS CODE IS TO FIX THE FACT THAT THE ROBOT ARM DATA IS 5x OVERSAMPLED
    # TODO: Change motion recording code to sample at only 10Hz
    robot_data = robot_data[2::5]
    # ===================================================================

    robot_grad = np.gradient(robot_data, axis=0)
    human_grad = np.gradient(human_data, axis=0)

    robot_grad_mag = np.linalg.norm(robot_grad, axis=1)
    human_grad_mag = np.linalg.norm(human_grad, axis=1)

    alignment, _ = soft_dtw_alignment(human_grad_mag, robot_grad_mag)
    row_sum = np.sum(alignment, axis=1)
    aligned_unscaled = alignment @ robot_data
    robot_data = aligned_unscaled / row_sum[:, None]

    # plt.gray()
    # plt.imshow(alignment, interpolation='nearest')
    # plt.show()

    pass


if __name__ == "__main__":
    main()
