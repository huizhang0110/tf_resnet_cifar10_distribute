import tensorflow as tf
import joblib


meanstd_dict = joblib.load("./data/meanstd.pkl")


def normalize_image(image):
    mean, std = meanstd_dict["mean"], meanstd_dict["std"]
    normed_image = (image - mean) / std
    return normed_image


def image_augmentation(image):
    image = tf.image.pad_to_bounding_box(image, 4, 4, 40, 40)
    image = tf.random_crop(image, [32, 32, 3])
    image = tf.image.random_flip_left_right(image)
    return image


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
        image = normalize_image(image)
        image = image_augmentation(image)
        label = tf.cast(parsed["label"], tf.int32)
        return {"image": image}, label

    dataset = dataset.map(parser) 
    dataset = dataset.shuffle(buffer_size=10000)
    dataset = dataset.batch(64)
    dataset = dataset.repeat()
    iterator = dataset.make_one_shot_iterator()

    # feature_dict, labels = iterator.get_next()
    return dataset


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
        image = tf.cast(image, tf.float32)
        image = normalize_image(image)
        label = tf.cast(parsed["label"], tf.int32)
        return {"image": image}, label

    dataset = dataset.map(parser)
    dataset = dataset.batch(64)
    dataset = dataset.repeat(1)
    iterator = dataset.make_one_shot_iterator()

    return dataset


if __name__ == "__main__":

    with tf.Session() as sess:
        sess.run([
            tf.global_variables_initializer(),
            tf.local_variables_initializer()
        ])

        feature_dict, labels = sess.run(train_input_fn())
        print(feature_dict["image"].shape)
        print(labels.shape)

