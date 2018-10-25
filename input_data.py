import tensorflow as tf


def train_input_fn():
    filenames = ["./data/cifar10_train.tfrecord"]
    dataset = tf.data.TFRecordDataset(filenames)

    def parser(record):
        keys_to_features = {
            "image_raw": tf.FixedLenFeature([], tf.string, default_value=""),
            "label": tf.FixedLenFeature([], tf.int64)
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        image = tf.decode_raw(parsed["image_raw"], tf.uint8)
        image = tf.reshape(image, [32, 32, 3])
        image = tf.cast(image, tf.float32)
        label = tf.cast(parsed["label"], tf.int32)
        return {"image": image}, label

    dataset = dataset.map(parser) 
    dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.batch(32)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()

    feature_dict, labels = iterator.get_next()
    return feature_dict, labels


def eval_input_fn():
    filenames = ["./data/cifar10_test.tfrecord"]
    dataset = tf.data.TFRecordDataset(filenames)

    def parser(record):
        keys_to_features = {
            "image_raw": tf.FixedLenFeature([], tf.string, default_value=""),
            "label": tf.FixedLenFeature([], tf.int64)
        }
        parsed = tf.parse_single_example(record, keys_to_features)
        image = tf.decode_raw(parsed["image_raw"], tf.uint8)
        image = tf.reshape(image, [32, 32, 3])
        label = tf.cast(parsed["label"], tf.int32)
        return {"image": image}, label

    dataset = dataset.map(parser)
    dataset = dataset.batch(32)
    dataset = dataset.repeat(1)
    iterator = dataset.make_one_shot_iterator()

    feature_dict, labels = iterator.get_next()
    return feature_dict, labels


if __name__ == "__main__":

    with tf.Session() as sess:
        sess.run([
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        ])

        feature_dict, labels = sess.run(train_input_fn())
        print(feature_dict["image"].shape)
        print(labels.shape)

