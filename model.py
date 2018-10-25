import tensorflow as tf
from utils import layer_tools


flags = tf.app.flags
flags.DEFINE_string("exp_dir", "experiments/cifar10_152", "")
flags.DEFINE_string("data_dir", "data/cifar-10-batches-py", "")
flags.DEFINE_integer("n_classes", 10, "")
flags.DEFINE_integer("batch_size", 32, "")
flags.DEFINE_float("weight_decay", 0.1, "")
FLAGS = flags.FLAGS



def model_fn(features, labels, mode, params):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    images = features["image"]
    logits = layer_tools.resnet50_cifar10(images, FLAGS.n_classes, is_training)
    predictions = {"pred_classes": tf.argmax(input=logits, axis=1, name="pred_classes")}

    # PREDICT Mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss_tensor = tf.losses.softmax_cross_entropy(onehot_labels=tf.one_hot(labels, depth=10), 
            logits=logits, scope="loss")
    accuracy, update_op = tf.metrics.accuracy(labels=labels, 
            predictions=predictions["pred_classes"], name="accuracy")

    # For Training mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizer = tf.train.AdamOptimizer()
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss_tensor, tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss_tensor, train_op=train_op)

    # For EVAL mode
    eval_metric_ops = {"accuracy": (accuracy, update_op)}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss_tensor, eval_metric_ops=eval_metric_ops)

