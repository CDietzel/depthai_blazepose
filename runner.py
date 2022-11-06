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
    human_poses_folder = "/home/locobot/Documents/Repos/depthai_blazepose/outputs/"
    robot_joint_folder = "/home/locobot/Documents/Repos/depthai_blazepose/5DoF/"
    tfrecord_folder = (
        "/home/locobot/Documents/Repos/ibc/ibc/data/interbotix_data/oracle_interbotix_"
    )

    human_poses_ext = ".pickle"
    robot_joint_ext = ".recording"
    tfrecord_ext = ".tfrecord"
    tfrecord_spec_ext = ".tfrecord.spec"

    human_poses_paths = glob.glob(human_poses_folder + "*" + human_poses_ext)
    human_poses_files = [
        os.path.splitext(os.path.basename(path))[0] for path in human_poses_paths
    ]
    robot_joint_paths = glob.glob(robot_joint_folder + "*" + robot_joint_ext)
    robot_joint_files = [
        os.path.splitext(os.path.basename(path))[0] for path in robot_joint_paths
    ]

    assert set(human_poses_files) == set(
        robot_joint_files
    ), "must have the same human pose files and robot joint files"

    inverse_list = [False, True]
    # inverse_list = [False]
    pre_swap = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
    ]
    post_swap = [
        0,
        4,
        5,
        6,
        1,
        2,
        3,
        8,
        7,
        10,
        9,
        12,
        11,
        14,
        13,
        16,
        15,
        18,
        17,
        20,
        19,
        22,
        21,
        24,
        23,
        26,
        25,
        28,
        27,
        30,
        29,
        32,
        31,
    ]

    for (
        recording_sample_name,
        human_poses_path,
        robot_joint_path,
    ), inverse in itertools.product(
        zip(human_poses_files, human_poses_paths, robot_joint_paths), inverse_list
    ):

        def generator():
            human_poses = runner.load_pickle(human_poses_path)
            robot_data = runner.load_pickle(robot_joint_path)[0]
            human_data = runner.extract_human_keypoints(human_poses)

            if inverse:
                human_data[:, :, 0] = human_data[:, :, 0] * -1
                human_data[:, pre_swap, :] = human_data[:, post_swap, :]
                robot_data[:, 0] = robot_data[:, 0] * -1
                robot_data[:, -1] = robot_data[:, -1] * -1

            # fig = plt.figure()
            # ax = plt.axes(projection="3d")
            # ax.scatter3D(
            #     human_data[20, :, 0], human_data[20, :, 1], human_data[20, :, 2]
            # )
            # for i in range(len(human_data[20, :, 0])):
            #     ax.text(
            #         human_data[20, :, 0][i],
            #         human_data[20, :, 1][i],
            #         human_data[20, :, 2][i],
            #         "%s" % (str(i)),
            #         size=15,
            #         zorder=1,
            #         color="k",
            #     )
            #     ax.view_init(elev=-90, azim=-90)
            # print(recording_sample_name)
            # print(inverse)
            # plt.show()

            # Only include arm keypoints (11-22)
            human_data = human_data[:, 11:23, :]

            # Reshape to flatten xyz coordinates for all keypoints for each frame
            human_data = human_data.reshape(-1, 36)

            # Ignore first few samples (noisy data)
            human_data = human_data[3:]

            # THIS CODE IS TO FIX THE FACT THAT THE ROBOT ARM DATA IS 5x OVERSAMPLED
            # TODO: Change robot motion recording code to sample at only 10Hz
            robot_data = robot_data[::5]
            # ===================================================================

            robot_grad = np.gradient(robot_data, axis=0)
            human_grad = np.gradient(human_data, axis=0)

            robot_grad_mag = np.linalg.norm(robot_grad, axis=1)
            human_grad_mag = np.linalg.norm(human_grad, axis=1)

            # TODO: Figure out how to make this edge trimmer work well

            # r_hist, r_bin_edges = np.histogram(robot_grad_mag, bins="auto")
            # h_hist, h_bin_edges = np.histogram(human_grad_mag, bins="auto")

            # r_thresh = filters.threshold_isodata(hist=r_hist)
            # h_thresh = filters.threshold_isodata(hist=h_hist)

            # r_thresh_val = (r_bin_edges[r_thresh] + r_bin_edges[r_thresh + 1]) / 2
            # h_thresh_val = (h_bin_edges[h_thresh] + h_bin_edges[h_thresh + 1]) / 2

            # plt.rcParams["figure.figsize"] = (20, 10)
            # plt.subplot(1, 2, 1)
            # plt.plot(range(len(human_grad_mag)), human_grad_mag)
            # plt.axhline(y=h_thresh_val)
            # plt.subplot(1, 2, 2)
            # plt.plot(range(len(robot_grad_mag)), robot_grad_mag)
            # plt.axhline(y=r_thresh_val)
            # print(recording_sample_name)
            # print(inverse)
            # plt.show()

            # plt.rcParams["figure.figsize"] = (20, 10)
            # plt.plot(robot_data)
            # print(recording_sample_name)
            # plt.show()

            alignment, _ = soft_dtw_alignment(robot_grad_mag, human_grad_mag)
            row_sum = np.sum(alignment, axis=1)
            aligned_unscaled = alignment @ human_data
            aligned_human_data = aligned_unscaled / row_sum[:, None]

            dataset_len, _ = human_data.shape

            for i, ((_, prev_act), (obs, act)) in enumerate(
                pairwise(zip(aligned_human_data, robot_data))
            ):
                if i == 0:
                    yield runner.serialize_example(
                        step_type=0,
                        next_step_type=1,
                        obs=obs,
                        prev_act=prev_act,
                        act=act,
                        reward=0.00001,
                        discount=1,
                    )
                elif i >= (dataset_len - 1):
                    yield runner.serialize_example(
                        step_type=2,
                        next_step_type=0,
                        obs=obs,
                        prev_act=prev_act,
                        act=act,
                        reward=0,
                        discount=1,
                    )
                elif i == (dataset_len - 2):
                    yield runner.serialize_example(
                        step_type=1,
                        next_step_type=2,
                        obs=obs,
                        prev_act=prev_act,
                        act=act,
                        reward=1,
                        discount=0,
                    )
                else:
                    yield runner.serialize_example(
                        step_type=1,
                        next_step_type=1,
                        obs=obs,
                        prev_act=prev_act,
                        act=act,
                        reward=0.00001,
                        discount=1,
                    )

        # for g in generator():
        #     pass
        # test_val = next(generator())

        serialized_features_dataset = tf.data.Dataset.from_generator(
            generator,
            output_types=tf.string,
            output_shapes=(),
        )

        if inverse:
            tag = "_i"
        else:
            tag = "_n"

        # dataset_shard = serialized_features_dataset.shard(
        #     num_shards=num_shards, index=i
        # )
        tfrecord_path = tfrecord_folder + recording_sample_name + tag + tfrecord_ext
        tfrecord_spec_path = (
            tfrecord_folder + recording_sample_name + tag + tfrecord_spec_ext
        )
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
