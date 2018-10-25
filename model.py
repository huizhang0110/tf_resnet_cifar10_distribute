import tensorflow as tf
from utils import layer_tools


flags = tf.app.flags
flags.DEFINE_string("exp_dir", "experiments/cifar10_152", "")
flags.DEFINE_string("data_dir", "data/cifar-10-batches-py", "")
flags.DEFINE_integer("n_classes", 10, "")
flags.DEFINE_integer("batch_size", 32, "")
flags.DEFINE_float("weight_decay", 0.1, "")
FLAGS = flags.FLAGS


def get_loss(logits, labels, scope="loss"):
    with tf.variable_scope(scope):
        entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits,
            labels=labels,
        )
        entropy_loss = tf.reduce_mean(entropy_loss, name="entropy_loss")
        tf.add_to_collection("losses", entropy_loss)
        # Weight l2 decay loss 
#         weight_l2_losses = [tf.nn.l2_loss(x) for x in tf.get_collection("weights")]
        # weight_decay_loss = tf.multiply(
            # FLAGS.weight_decay, tf.add_n(weight_l2_losses),
            # name="weight_decay_loss"
        # ) 
        # tf.add_to_collection("losses", weight_decay_loss)

        for var in tf.get_collection("losses"):
            tf.summary.scalar("losses/" + var.op.name, var)
        return tf.add_n(tf.get_collection("losses"), name="total_loss")


def model_fn(features, labels, mode, params):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    logits = layer_tools.resnet50_cifar10(
            features["image"], FLAGS.n_classes, is_training)

    predictions = {
        "pred_classes": tf.argmin(input=logits, axis=1, name="pred_classes"),
    }
    # PREDICT Mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
                mode=mode, predictions=predictions)

    # For TRAIN and EVAL mode
    loss_tensor = get_loss(logits, labels)
    accuracy, update_op = tf.metrics.accuracy(
        labels=labels, 
        predictions=predictions["pred_classes"], 
        name="accuracy"
    )
    batch_acc = tf.reduce_mean(tf.cast(
        tf.equal(tf.cast(labels, tf.int64), predictions["pred_classes"]),
        tf.float32
    ))
    tf.summary.scalar("batch_acc", batch_acc)
    tf.summary.scalar("streaming_acc", update_op)
    if mode == tf.estimator.ModeKeys.TRAIN:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(
                loss_tensor,
                tf.train.get_global_step(),
            )
        return tf.estimator.EstimatorSpec(
                mode=mode, 
                loss=loss_tensor, 
                train_op=train_op
        )

    # For EVAL mode
    eval_metric_ops = {
        "accuracy": (accuracy, update_op)
    }
    return tf.estimator.EstimatorSpec(
            mode=mode, 
            loss=loss_tensor, 
            eval_metric_ops=eval_metric_ops
    )

