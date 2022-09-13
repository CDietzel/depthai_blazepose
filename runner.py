import glob
import marshal
import os
import pickle
import sys
from math import cos, sin, sqrt
from pathlib import Path
from string import Template

import cv2
import depthai as dai
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from numpy.core.fromnumeric import trace
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
                    shape=(99,), dtype=tf.float32, name="observation/human_pose"
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

    def save_tfrecord_dataset_sequence(
        self,
        tfrecord_path,
        obs_df,
        acts_df,
    ):
        pass

    def _bytes_feature(self, values):
        """Returns a bytes_list from a string / byte."""
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

    def _float_feature(self, values):
        """Returns a float_list from a float / double."""
        return tf.train.Feature(float_list=tf.train.FloatList(value=values))

    def _int64_feature(self, values):
        """Returns an int64_list from a bool / enum / int / uint."""
        return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

    def serialize_example(self, step_type, next_step_type, obs, act, reward, discount):
        features = {
            "discount": self._float_feature([discount]),
            "step_type": self._int64_feature([step_type]),
            "next_step_type": self._int64_feature([next_step_type]),
            "observation/human_pose": self._float_feature(obs),
            "observation/action": self._float_feature(act),
            "action": self._float_feature(act),
            "reward": self._float_feature([reward]),
        }
        example_proto = tf.train.Example(features=tf.train.Features(feature=features))
        return example_proto.SerializeToString()


def main():
    runner = TrajectoryPreprocessor()
    human_poses_folder = "/home/locobot/Documents/Repos/depthai_blazepose/outputs/"
    robot_joint_folder = "/home/locobot/Documents/Repos/depthai_blazepose/5DoF/"
    tfrecord_folder = (
        "/home/locobot/Documents/Repos/ibc/ibc/data/interbotix_data/oracle_interbotix_"
    )

    human_poses_ext = ".pickle"
    robot_joint_ext = ".recording"
    tfrecord_ext = ".tfrecord"
    tfrecord_spec_ext = ".tfrecord.spec"
    num_recordings = 15
    num_samples_per_recording = 6

    num_shards = num_recordings
    for recording_sample_num in range(num_shards * num_samples_per_recording):

        recording_num = recording_sample_num % num_recordings
        sample_num = recording_sample_num // num_recordings
        recording_sample_name = str(recording_num + 1) + "_" + str(sample_num + 1)

        def generator():
            human_poses_path = (
                human_poses_folder + recording_sample_name + human_poses_ext
            )
            robot_joint_path = (
                robot_joint_folder + recording_sample_name + robot_joint_ext
            )

            human_poses = runner.load_pickle(human_poses_path)
            robot_data = runner.load_pickle(robot_joint_path)[0]
            human_data = runner.extract_human_keypoints(human_poses)
            # Ignore first few samples (noisy data)
            human_data = human_data.reshape(-1, 99)[2:]

            # THIS CODE IS TO FIX THE FACT THAT THE ROBOT ARM DATA IS 5x OVERSAMPLED
            # TODO: Change robot motion recording code to sample at only 10Hz
            robot_data = robot_data[::5]
            # ===================================================================

            robot_grad = np.gradient(robot_data, axis=0)
            human_grad = np.gradient(human_data, axis=0)

            robot_grad_mag = np.linalg.norm(robot_grad, axis=1)
            human_grad_mag = np.linalg.norm(human_grad, axis=1)

            alignment, _ = soft_dtw_alignment(human_grad_mag, robot_grad_mag)
            row_sum = np.sum(alignment, axis=1)
            aligned_unscaled = alignment @ robot_data
            aligned_robot_data = aligned_unscaled / row_sum[:, None]

            dataset_len, _ = human_data.shape

            for i, (obs, act) in enumerate(zip(human_data, aligned_robot_data)):
                if i == 0:
                    yield runner.serialize_example(
                        step_type=0,
                        next_step_type=1,
                        obs=obs,
                        act=act,
                        reward=0.00001,
                        discount=1,
                    )
                elif i >= (dataset_len - 1):
                    yield runner.serialize_example(
                        step_type=2,
                        next_step_type=0,
                        obs=obs,
                        act=act,
                        reward=0,
                        discount=1,
                    )
                elif i == (dataset_len - 2):
                    yield runner.serialize_example(
                        step_type=1,
                        next_step_type=2,
                        obs=obs,
                        act=act,
                        reward=1,
                        discount=0,
                    )
                else:
                    yield runner.serialize_example(
                        step_type=1,
                        next_step_type=1,
                        obs=obs,
                        act=act,
                        reward=0.00001,
                        discount=1,
                    )

        # test_val = next(generator())

        serialized_features_dataset = tf.data.Dataset.from_generator(
            generator, output_types=tf.string, output_shapes=()
        )

        # dataset_shard = serialized_features_dataset.shard(
        #     num_shards=num_shards, index=i
        # )
        tfrecord_path = tfrecord_folder + recording_sample_name + tfrecord_ext
        tfrecord_spec_path = tfrecord_folder + recording_sample_name + tfrecord_spec_ext
        # example_encoding_dataset.encode_spec_to_file(spec_filename, dataset_shard.element_spec)
        example_encoding_dataset.encode_spec_to_file(
            tfrecord_spec_path, runner.output_spec
        )
        # print(dataset_shard.element_spec)
        # writer = tf.data.experimental.TFRecordWriter(filename)
        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            for record in serialized_features_dataset:
                writer.write(record.numpy())

    # plt.gray()
    # plt.imshow(alignment, interpolation='nearest')
    # plt.show()

    # pass


if __name__ == "__main__":
    main()
