import glob
import itertools
import marshal
import os
import pickle
import sys
from itertools import tee
from math import cos, sin, sqrt
from pathlib import Path
from string import Template

import cv2
import depthai as dai
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy.core.fromnumeric import trace
from skimage import filters
from tf_agents.trajectories.trajectory import Trajectory
from tf_agents.utils import example_encoding_dataset
from tslearn.barycenters import softdtw_barycenter
from tslearn.metrics import soft_dtw_alignment
from tslearn.preprocessing import TimeSeriesResampler

import mediapipe_utils as mpu
from BlazeposeRenderer import LINES_BODY, BlazeposeRenderer
from FPS import FPS, now
from o3d_utils import Visu3D
from RLSEstimator import RLSEstimator


class TrajectoryPreprocessor:
    def __init__(self):
        self.output_spec = Trajectory(
            discount=tf.TensorSpec(shape=(), dtype=tf.float32, name="discount"),
            step_type=tf.TensorSpec(shape=(), dtype=tf.int64, name="step_type"),
            next_step_type=tf.TensorSpec(shape=(), dtype=tf.int64, name="step_type"),
            observation={
                "human_pose": tf.TensorSpec(
                    shape=(36,), dtype=tf.float32, name="observation/human_pose"
                ),
                "action": tf.TensorSpec(
                    shape=(5,), dtype=tf.float32, name="observation/action"
                ),
            },
            action=tf.TensorSpec(shape=(5,), dtype=tf.float32, name="action"),
            reward=tf.TensorSpec(shape=(), dtype=tf.float32, name="reward"),
            policy_info=(),
        )

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
                keypoints_list.append(pose.landmarks_world)  # Only use this,
                # keypoints_list.append(pose.norm_landmarks)  # Not these.
                # keypoints_list.append(pose.landmarks)  # They are image-relative
        keypoints = np.array(keypoints_list)[:, 0:33, :]
        return keypoints

    def _bytes_feature(self, values):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

    def _float_feature(self, values):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))

    def _int64_feature(self, values):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    def serialize_example(
        self, step_type, next_step_type, obs, prev_act, act, reward, discount
    ):
        features = {
            "discount": self._float_feature([discount]),
            "step_type": self._int64_feature([step_type]),
            "next_step_type": self._int64_feature([next_step_type]),
            "observation/human_pose": self._float_feature(obs),
            "observation/action": self._float_feature(prev_act),
            "action": self._float_feature(act),
            "reward": self._float_feature([reward]),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
        return example_proto.SerializeToString()


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def main():
    runner = TrajectoryPreprocessor()
    # human_poses_folder = "/home/locobot/Documents/Repos/depthai_blazepose/outputs/"
    robot_joint_folder = "/home/locobot/Documents/Repos/depthai_blazepose/broken/"
    # tfrecord_folder = (
    #     "/home/locobot/Documents/Repos/ibc/ibc/data/interbotix_data/oracle_interbotix_"
    # )

    human_poses_ext = ".pickle"
    robot_joint_ext = ".modulated"
    robot_joint_fixed_ext = ".recording"
    tfrecord_ext = ".tfrecord"
    tfrecord_spec_ext = ".tfrecord.spec"
    num_recordings = 15
    num_samples_per_recording = 6

    # human_poses_paths = glob.glob(human_poses_folder + "*" + human_poses_ext)
    # human_poses_files = [
    #     os.path.splitext(os.path.basename(path))[0] for path in human_poses_paths
    # ]
    robot_joint_paths = glob.glob(robot_joint_folder + "*" + robot_joint_ext)
    robot_joint_files = [
        os.path.splitext(os.path.basename(path))[0] for path in robot_joint_paths
    ]

    # assert set(human_poses_files) == set(
    #     robot_joint_files
    # ), "must have the same human pose files and robot joint files"

    for (
        recording_sample_name,
        robot_joint_path,
    ) in zip(robot_joint_files, robot_joint_paths):

        robot_data = runner.load_pickle(robot_joint_path)
        robot_data = [robot_data]
        with open(
            robot_joint_folder + str(recording_sample_name + ".recording"),
            "wb",
        ) as f:
            pickle.dump(robot_data, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
