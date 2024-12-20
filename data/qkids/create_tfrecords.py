#!/usr/bin/env python
# encoding: utf-8
"""
@author: coffee
@license: (C) Copyright 2022-2032, Node Supply Chain Manager Corporation Limited.
@contact: leonhe0119@gmail.com
@file: create_tfrecords.py
@time: 2024/12/5 17:43
@desc:
"""
import glob
import pickle

import tensorflow as tf
import h5py
import os
import fnmatch
import cv2
import numpy as np


def decode_img(img):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def _bool_feature(value):
    """Returns a bool_list from a boolean."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(action, qpos, cam_high, cam_left_wrist, cam_right_wrist, terminate_episode):
    feature = {
        'action': _bytes_feature(tf.io.serialize_tensor(action)),
        'qpos': _bytes_feature(tf.io.serialize_tensor(qpos)),
        # 'qvel': _bytes_feature(tf.io.serialize_tensor(qvel)),
        'cam_high': _bytes_feature(tf.io.serialize_tensor(cam_high)),
        'cam_right_wrist': _bytes_feature(tf.io.serialize_tensor(cam_right_wrist)),
        # 'cam_low': _bytes_feature(tf.io.serialize_tensor(cam_low)),
        'terminate_episode': _bool_feature(terminate_episode)
    }
    if cam_left_wrist:
        feature['cam_left_wrist'] = bytes_feature(tf.io.serialize_tensor(cam_left_wrist)),
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def handle_file(filepath, out_dir, root_dir='datasets/qkids'):
    with h5py.File(filepath, 'r') as f:
        output_dir = os.path.join(out_dir, os.path.relpath(root, root_dir))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        print(f"Writing TFRecords to {output_dir}")
        tfrecord_path = os.path.join(output_dir, filename.replace('.hdf5', '.tfrecord'))

        with tf.io.TFRecordWriter(tfrecord_path) as writer:
            num_episodes = f['action'].shape[0]
            for i in range(num_episodes):
                action = f['action'][i]

                base_action = None
                qpos = f['observations']['qpos'][i]
                qvel = f['observations']['qvel'][i]
                cam_high = decode_img(f['observations']['images']['cam_high'][i])
                cam_left_wrist = decode_img(f['observations']['images']['cam_left_wrist'][i])
                cam_right_wrist = decode_img(f['observations']['images']['cam_right_wrist'][i])

                cam_low = None
                instruction = f['instruction'][()]
                terminate_episode = i == num_episodes - 1
                serialized_example = serialize_example(action, base_action, qpos, qvel, cam_high, cam_left_wrist,
                                                       cam_right_wrist, cam_low, instruction, terminate_episode)
                writer.write(serialized_example)


def _preprocess(raw):
    master = []
    puppet = []
    image_top = []
    image_right = []
    data = raw['data']
    for e in data:
        master.append(e['right_master'])
        puppet.append(e['right_puppet'])
        image_right.append(e['camera']['RIGHT'])
        image_top.append(e['camera']['TOP'])

    image_right = np.stack(image_right)
    # image_right = np.moveaxis(image_right, -1, 1)

    image_top = np.stack(image_top)
    # image_top = np.moveaxis(image_top, -1, 1)
    return np.stack(master), np.stack(puppet), image_top, image_right


def process_episode(episode, tf_writer):
    act, state, img_top, img_right = _preprocess(episode)
    num_episodes = len(act)

    for i in range(num_episodes):
        action = act[i]

        qpos = state[i]
        cam_high =  cv2.cvtColor(img_top[i], cv2.COLOR_BGR2RGB)
        cam_left_wrist = None
        cam_right_wrist = cv2.cvtColor(img_right[i], cv2.COLOR_BGR2RGB)

        terminate_episode = i == num_episodes - 1
        serialized_example = serialize_example(action, qpos, cam_high, cam_left_wrist,
                                               cam_right_wrist, terminate_episode)
        tf_writer.write(serialized_example)


def process_task(path: str):
    out = []
    print(path)
    tfrecord_path = os.path.join(output_dir, os.path.basename(path))
    os.makedirs(tfrecord_path, exist_ok=True)
    for epi in glob.glob(os.path.join(path, "*.pkl")):
        out_path = os.path.join(tfrecord_path, os.path.basename(epi).replace('.pkl', '.tfrecord'))
        if os.path.exists(out_path):
            continue
        data = pickle.load(open(epi, 'rb'))
        print(epi)
        with tf.io.TFRecordWriter(out_path) as tf_writer:
            process_episode(data, tf_writer)


def run():
    for i in glob.glob(os.path.join(raw_data_path, "*")):
        task_data = process_task(i)


if __name__ == '__main__':
    raw_data_path = "/mnt/d4t/data/lerobot"
    output_dir = "/mnt/d4t/code/RoboticsDiffusionTransformer/data/datasets/qkids/tfrecords"
    run()
