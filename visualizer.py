import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tf_agents.trajectories.trajectory import Trajectory


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


def main():
    runner = TrajectoryPreprocessor()
    robot_joint_folder = "/home/locobot/Documents/Repos/depthai_blazepose/for_viz/"
    other_folder = (
        "/home/locobot/Documents/Repos/depthai_blazepose/5DoF/test3.recording"
    )

    human_poses_ext = ".pickle"
    robot_joint_ext = ".modulated"
    robot_joint_fixed_ext = ".recording"
    image_ext = ".eps"
    motion_num = 5
    motion_name = "test"
    orig_file = "orig_" + motion_name + str(motion_num) + robot_joint_ext
    lang_file = "lang_" + motion_name + str(motion_num) + robot_joint_ext
    mse_file = "mse_" + motion_name + str(motion_num) + robot_joint_ext

    orig_path = robot_joint_folder + orig_file
    lang_path = robot_joint_folder + lang_file
    mse_path = robot_joint_folder + mse_file

    orig_data = runner.load_pickle(orig_path)[::5]
    lang_data = runner.load_pickle(lang_path)
    mse_data = runner.load_pickle(mse_path)

    joint_names = ["Waist", "shoulder", "elbow", "wrist_angle", "wrist_rotate"]
    orig_title = "Human Motion (Ground Truth)"
    lang_title = "Generated Motion (Dancing From Demonstration)"
    mse_title = "Generated Motion (Mean Square Error Baseline)"
    over_title = "Motion Comparison for " + motion_name + str(motion_num)
    # x_axis = "Sample Time (10Hz)"
    y_axis = "Joint Angle (Radians)"

    orig_df = pd.DataFrame(orig_data, columns=joint_names)
    lang_df = pd.DataFrame(lang_data, columns=joint_names)
    mse_df = pd.DataFrame(mse_data, columns=joint_names)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 6))
    fig.suptitle(over_title, fontsize=14)
    orig_df.plot(ax=axes[0], title=orig_title, ylabel=y_axis)
    lang_df.plot(ax=axes[1], title=lang_title, ylabel=y_axis)
    mse_df.plot(ax=axes[2], title=mse_title, ylabel=y_axis)
    plt.savefig(robot_joint_folder + motion_name + str(motion_num) + image_ext)


if __name__ == "__main__":
    main()
