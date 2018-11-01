import tensorflow as tf
import numpy as np
import pickle
import os
from tqdm import tqdm
import joblib


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def load_cifar10_write_tfrecord(data_dir):
    
    def write_to_tfrecord(images, labels, save_file):
        writer = tf.python_io.TFRecordWriter(save_file)
        for i in tqdm(range(images.shape[0])):
            image_raw = images[i].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                "height": _int64_feature(32),
                "width": _int64_feature(32),
                "depth": _int64_feature(3),
                "label": _int64_feature(labels[i]),
                "image_raw": _bytes_feature(image_raw)
            }))
            writer.write(example.SerializeToString())


    train_images = np.zeros((50000, 3072), dtype=np.uint8)
    train_labels = np.zeros((50000, ), dtype=np.int32)
    for i in range(5):
        data_batch_file = os.path.join(data_dir, "data_batch_%d" % (i+1))
        with open(data_batch_file, "rb") as f:
            data_batch_dict = pickle.load(f, encoding="bytes")
        train_images[10000*i : 10000*(i+1)] = data_batch_dict[b"data"]
        train_labels[10000*i : 10000*(i+1)] = np.array(
            data_batch_dict[b"labels"], dtype=np.int32)
    train_images = np.reshape(train_images, [50000, 3, 32, 32])
    train_images = np.transpose(train_images, [0, 2, 3, 1])

    # Convert training data into tfrecord format
    write_to_tfrecord(train_images, train_labels, "./data/cifar10_train.tfrecord")

    # mean and std
    image_mean = np.mean(train_images.astype(np.float32), axis=(0, 1, 2))
    image_std = np.std(train_images.astype(np.float32), axis=(0, 1, 2))
    joblib.dump({"mean": image_mean, "std": image_std}, "./data/meanstd.pkl")

    test_batch_file = os.path.join(data_dir, "test_batch")
    with open(test_batch_file, "rb") as f:
        data_batch_dict = pickle.load(f, encoding="bytes")
        test_images = data_batch_dict[b"data"].astype(np.uint8)
        test_labels = np.array(data_batch_dict[b"labels"], dtype=np.int32)
    test_images = np.reshape(test_images, [10000, 3, 32, 32])
    test_images = np.transpose(test_images, [0, 2, 3, 1])

    # Convert test data into tfrecord
    write_to_tfrecord(test_images, test_labels, "./data/cifar10_test.tfrecord")


if __name__ == "__main__":
    load_cifar10_write_tfrecord("./data/cifar-10-batches-py")

